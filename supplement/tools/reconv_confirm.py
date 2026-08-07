"""Stage B confirmation: separate real divergence from tactic NONDETERMINISM.

Stage B flags a class when two rendering-identical members disagree on some probe. That test
cannot by itself distinguish two very different causes:

  (a) the states really do differ executably              -- the finding
  (b) the probe is nondeterministic, and would disagree
      with ITSELF on a single fixed state                 -- an artifact

Search tactics are a live risk for (b): `aesop` explores a priority queue and `exact?`/`simp_all`
consult mutable caches, so repeated invocations need not agree. A five-member class where aesop
yields five different successors is exactly what pure nondeterminism would also produce.

So every flagged probe is re-run REPEATS times on each member, in a FRESH Dojo session:

  self-inconsistent on any single member   -> NONDETERMINISTIC, probe discarded
  self-consistent everywhere, members differ -> CONFIRMED
  members agree on re-run                    -> NOT REPRODUCED

The fresh session also re-tests reproduction: the histories are replayed from scratch and the
members' renderings are re-checked for byte equality before anything is probed.

Usage:
    python tools/reconv_confirm.py --run runs/reconv_b2 --states runs/killtest/states.jsonl \\
        --repeats 3
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.reconvergence import _h  # noqa: E402


def _log(m: str) -> None:
    print(f"[confirm] {m}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dojo-timeout", type=int, default=600)
    args = ap.parse_args()

    from lean_dojo import Dojo, Theorem
    import lean_dojo as ldj
    from audit.replay import lean_git_repo, kill_orphan_lean, deadline

    run = Path(args.run)
    cand = [json.loads(l) for f in glob.glob(str(run / "stageb.shard*.jsonl"))
            for l in open(f) if l.strip()]
    cand = [c for c in cand if c["divergent_probes"]]
    _log(f"{len(cand)} candidate divergent classes to confirm")

    states = [json.loads(l) for l in Path(args.states).read_text().splitlines()]
    prov = {}
    for s in states:
        p = s["provenance"]
        prov[(p["file_path"], p["theorem_full_name"], p["tactic_index"])] = s

    def outcome(r) -> str:
        if isinstance(r, ldj.ProofFinished):
            return "COMPLETE"
        if isinstance(r, ldj.TacticState):
            return "SUCC:" + _h(r.pp)[:12]
        msg = (getattr(r, "error", "") or str(r) or "")
        return "TIMEOUT" if "timeout" in msg.lower() else "FAIL"

    results = []
    for ci, c in enumerate(cand, 1):
        key = (c["file"], c["theorem"], c["tactic_index"])
        s = prov.get(key)
        if s is None:
            _log(f"[{ci}/{len(cand)}] {c['theorem']}: root not found; skip")
            continue
        t0 = time.time()
        p = s["provenance"]
        try:
            thm = Theorem(lean_git_repo(ldj, p["repo_url"], p["repo_commit"]),
                          p["file_path"], p["theorem_full_name"])
            ctx = Dojo(thm, timeout=args.dojo_timeout)
            dojo, st0 = ctx.__enter__()
        except BaseException:
            kill_orphan_lean()
            continue
        # tactic_index identifies WHICH step of the theorem is the root. Joining downstream on
        # (file, theorem) alone silently picks a different step, a different prefix, and a
        # different root state.
        rec = {"theorem": c["theorem"], "file": c["file"], "pp_hash": c["pp_hash"],
               "tactic_index": c["tactic_index"],
               "histories": c["histories"], "flagged": c["divergent_probes"]}
        try:
            with deadline(300, "prefix"):
                for pre in s["proof_prefix"]:
                    st0 = dojo.run_tac(st0, pre)
                    if not isinstance(st0, ldj.TacticState):
                        raise RuntimeError("prefix failed")

            # Replay each recorded history from scratch in this fresh session.
            members, pps = {}, {}
            for hist in c["histories"]:
                st = st0
                ok = True
                for tac in hist:
                    with deadline(60, tac):
                        st = dojo.run_tac(st, tac)
                    if not isinstance(st, ldj.TacticState):
                        ok = False
                        break
                if ok:
                    members[" ; ".join(hist)] = st
                    pps[" ; ".join(hist)] = st.pp
            rec["members_restored"] = len(members)
            rec["renderings_identical"] = (len(set(pps.values())) == 1
                                           if len(pps) >= 2 else None)
            if len(members) < 2 or not rec["renderings_identical"]:
                rec["verdict"] = "NOT_REPRODUCED"
                _log(f"[{ci}/{len(cand)}] {c['theorem'][:44]}: NOT REPRODUCED "
                     f"({len(members)} restored, identical={rec['renderings_identical']})")
                results.append(rec)
                continue

            # Repeat each flagged probe on each member. A CRASH means our watchdog interrupted
            # a request, which desynchronises the pipe: from then on `run_tac` returns the
            # PREVIOUS request's answer. Nothing measured after a crash on a live session is
            # trustworthy, so tear the session down, replay every history, and mark the probe
            # unusable rather than recording a shifted result as a divergence.
            def rebuild():
                nonlocal ctx, dojo
                try:
                    ctx.__exit__(None, None, None)
                except Exception:
                    pass
                ctx = Dojo(thm, timeout=args.dojo_timeout)
                dojo, s0 = ctx.__enter__()
                for pre in s["proof_prefix"]:
                    s0 = dojo.run_tac(s0, pre)
                m = {}
                for hist in c["histories"]:
                    st_ = s0
                    for tac in hist:
                        st_ = dojo.run_tac(st_, tac)
                        if not isinstance(st_, ldj.TacticState):
                            st_ = None
                            break
                    if st_ is not None:
                        m[" ; ".join(hist)] = st_
                return m

            table, nondet, poisoned = {}, set(), set()
            for a in c["divergent_probes"]:
                table[a] = {}
                for name in list(members):
                    obs = []
                    for _ in range(args.repeats):
                        try:
                            with deadline(90, f"probe {a}"):
                                obs.append(outcome(dojo.run_tac(members[name], a)))
                        except BaseException:
                            obs.append("CRASH")
                            poisoned.add(a)
                            members = rebuild()      # pipe is desynced; discard and restart
                            break
                    table[a][name] = obs
                    if len(set(obs)) > 1:            # disagrees with ITSELF
                        nondet.add(a)
                    if name not in members:
                        poisoned.add(a)
                        break
            rec["repeat_table"] = table
            rec["nondeterministic_probes"] = sorted(nondet)
            rec["poisoned_probes"] = sorted(poisoned)
            stable = [a for a in c["divergent_probes"]
                      if a not in nondet and a not in poisoned]
            confirmed = [a for a in stable
                         if len({table[a][n][0] for n in table[a]}) > 1
                         and not any(table[a][n][0].startswith(("TIMEOUT", "CRASH"))
                                     for n in table[a])]
            rec["confirmed_probes"] = confirmed
            rec["verdict"] = "CONFIRMED" if confirmed else (
                "NONDETERMINISTIC" if nondet else
                "CRASH_CONTAMINATED" if poisoned else "NOT_REPRODUCED")

            # Capture what actually differs, for the mechanism write-up.
            if confirmed:
                a = confirmed[0]
                rec["successor_pp"] = {}
                for name, st in members.items():
                    try:
                        with deadline(60, a):
                            r = dojo.run_tac(st, a)
                        rec["successor_pp"][name] = (
                            r.pp if isinstance(r, ldj.TacticState)
                            else ("COMPLETE" if isinstance(r, ldj.ProofFinished)
                                  else "FAIL: " + (getattr(r, "error", "") or "")[:300]))
                    except BaseException:
                        rec["successor_pp"][name] = "CRASH"
            _log(f"[{ci}/{len(cand)}] {c['theorem'][:44]}: {rec['verdict']} "
                 f"confirmed={confirmed} nondet={sorted(nondet)} "
                 f"poisoned={sorted(poisoned)} ({time.time()-t0:.0f}s)")
        except BaseException as e:
            rec["verdict"] = f"ERROR:{type(e).__name__}"
            _log(f"[{ci}/{len(cand)}] {c['theorem'][:44]}: ERROR {type(e).__name__}")
        finally:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
        results.append(rec)

    out = run / "confirmation.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    n_c = sum(1 for r in results if r.get("verdict") == "CONFIRMED")
    n_n = sum(1 for r in results if r.get("verdict") == "NONDETERMINISTIC")
    n_x = sum(1 for r in results if r.get("verdict") == "NOT_REPRODUCED")
    n_p = sum(1 for r in results if r.get("verdict") == "CRASH_CONTAMINATED")
    print("\n" + "=" * 78)
    print(f"CONFIRMED (real executable divergence) : {n_c}")
    print(f"NONDETERMINISTIC (probe artifact)      : {n_n}")
    print(f"NOT REPRODUCED                         : {n_x}")
    print(f"CRASH-CONTAMINATED (unusable)          : {n_p}")
    print(f"wrote {out}")
    kill_orphan_lean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
