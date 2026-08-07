"""Control for the visibility-escape result: is `contrapose!` escape real and reproducible?

The lockstep rollout found that 19 of 59 confirmed alias classes escape — the members' renderings
diverge under a shared continuation — and that 15 of those escapes are driven by one tactic,
`contrapose!`. That contradicts the earlier "observationally confined / self-loop dominance"
reading, which was measured with a probe battery containing no contraposition or negation tactic.
Before that correction is believed it has to survive the same controls everything else did:

  * fresh Dojo session, members replayed from scratch;
  * renderings re-checked byte-identical BEFORE the probe;
  * the escaping tactic run REPEATS times per member (`contrapose!` should be deterministic, but
    `aesop` was "obviously deterministic" too until it wasn't);
  * escape counted only when every member is self-consistent and at least two disagree.

It also records the successor renderings so the escape can be characterised, not merely counted.

Usage:
    python tools/verify_escape.py --contwit runs/contwit --states runs/killtest/states.jsonl \\
        --repeats 3
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _log(m: str) -> None:
    print(f"[verify] {m}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contwit", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dojo-timeout", type=int, default=900)
    args = ap.parse_args()

    from lean_dojo import Dojo, Theorem
    import lean_dojo as ldj
    from audit.replay import lean_git_repo, kill_orphan_lean, deadline

    cw = Path(args.contwit)
    rows = [json.loads(l) for f in glob.glob(str(cw / "witness.shard*.jsonl"))
            for l in open(f) if l.strip()]
    esc = [r for r in rows if r.get("escape_depth")]
    _log(f"{len(esc)} escaping classes to verify")

    idx = {}
    for run in ("runs/reconv_b2", "runs/hunt_enn", "runs/hunt_rare"):
        for f in glob.glob(str(Path(run) / "stageb.shard*.jsonl")):
            for l in open(f):
                if l.strip():
                    d = json.loads(l)
                    idx[(d["file"], d["theorem"], d["pp_hash"])] = d["tactic_index"]
    for r in esc:
        if "tactic_index" not in r:
            r["tactic_index"] = idx.get((r["file"], r["theorem"], r["pp_hash"]))
    wanted = {(r["file"], r["theorem"], r["tactic_index"]) for r in esc}
    prov = {}
    with open(args.states, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            pv = d["provenance"]
            k = (pv["file_path"], pv["theorem_full_name"], pv["tactic_index"])
            if k in wanted:
                prov[k] = d
    _log(f"indexed {len(prov)} candidate roots")

    def outcome(r):
        if isinstance(r, ldj.ProofFinished):
            return "COMPLETE"
        if isinstance(r, ldj.TacticState):
            return "S:" + _h(r.pp)
        msg = (getattr(r, "error", "") or str(r) or "")
        return "TIMEOUT" if "timeout" in msg.lower() else "FAIL"

    results = []
    for i, r in enumerate(esc, 1):
        # The escaping step is the last one in the recorded rollout; everything before it is the
        # shared prefix that must be replayed to reach the escape point.
        roll = r.get("rollout") or []
        if not roll:
            continue
        prefix = [st["tactic"] for st in roll[:-1]]
        probe = roll[-1]["tactic"]
        s = prov.get((r["file"], r["theorem"], r.get("tactic_index")))
        if s is None:
            _log(f"[{i}/{len(esc)}] {r['theorem'][:40]}: root not found")
            continue
        p = s["provenance"]
        t0 = time.time()
        rec = {"theorem": r["theorem"], "file": r["file"], "probe": probe,
               "shared_prefix": prefix, "recorded_escape_depth": r["escape_depth"],
               "histories": r["histories"]}
        try:
            thm = Theorem(lean_git_repo(ldj, p["repo_url"], p["repo_commit"]),
                          p["file_path"], p["theorem_full_name"])
            ctx = Dojo(thm, timeout=args.dojo_timeout)
            dojo, st0 = ctx.__enter__()
        except BaseException:
            kill_orphan_lean()
            continue
        try:
            for pre in s["proof_prefix"]:
                st0 = dojo.run_tac(st0, pre)
            members = {}
            for hist in r["histories"]:
                st = st0
                ok = True
                for tac in list(hist) + prefix:
                    with deadline(90, tac[:30]):
                        st = dojo.run_tac(st, tac)
                    if not isinstance(st, ldj.TacticState):
                        ok = False
                        break
                if ok:
                    members[" ; ".join(hist)] = st
            rec["members_restored"] = len(members)
            pps = {n: m.pp for n, m in members.items()}
            rec["identical_before_probe"] = (len(set(pps.values())) == 1
                                             if len(pps) >= 2 else None)
            if len(members) < 2 or not rec["identical_before_probe"]:
                rec["verdict"] = "NOT_REPRODUCED"
                _log(f"[{i}/{len(esc)}] {r['theorem'][:40]}: NOT REPRODUCED "
                     f"({len(members)} restored, identical={rec['identical_before_probe']})")
                results.append(rec)
                continue

            table, pp_after = {}, {}
            for n, st in members.items():
                obs = []
                for _ in range(args.repeats):
                    try:
                        with deadline(90, probe[:30]):
                            rr = dojo.run_tac(st, probe)
                    except BaseException:
                        obs.append("CRASH")
                        break
                    obs.append(outcome(rr))
                    if isinstance(rr, ldj.TacticState):
                        pp_after[n] = rr.pp
                table[n] = obs
            rec["repeat_table"] = table
            selfconsistent = all(len(set(v)) == 1 and "CRASH" not in v
                                 for v in table.values())
            firsts = {n: v[0] for n, v in table.items()}
            rec["self_consistent"] = selfconsistent
            rec["first_outcomes"] = firsts
            rec["verdict"] = ("CONFIRMED_ESCAPE"
                              if selfconsistent and len(set(firsts.values())) > 1
                              else "NONDETERMINISTIC" if not selfconsistent
                              else "NO_ESCAPE_ON_RERUN")
            if rec["verdict"] == "CONFIRMED_ESCAPE" and len(pp_after) >= 2:
                vals = list(pp_after.values())
                rec["successor_bytes"] = {n: len(v) for n, v in pp_after.items()}
                import difflib
                a, b = vals[0].split(), vals[1].split()
                rec["successor_diff"] = [x for x in difflib.unified_diff(a, b, n=3, lineterm="")
                                         if x[:1] in "+-" and x[:3] not in ("+++", "---")][:24]
            _log(f"[{i}/{len(esc)}] {r['theorem'][:40]}: {rec['verdict']} "
                 f"probe={probe[:20]!r} ({time.time()-t0:.0f}s)")
        except BaseException as e:
            rec["verdict"] = f"ERROR:{type(e).__name__}"
            _log(f"[{i}/{len(esc)}] {r['theorem'][:40]}: ERROR {type(e).__name__}")
        finally:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
        results.append(rec)

    dest = Path(args.out) if args.out else (cw / "escape_verification.json")
    dest.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    import collections
    print("\n" + "=" * 78)
    print(dict(collections.Counter(r.get("verdict") for r in results)))
    print(dict(collections.Counter(r["probe"] for r in results
                                   if r.get("verdict") == "CONFIRMED_ESCAPE")))
    print(f"wrote {dest}")
    kill_orphan_lean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
