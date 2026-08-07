"""Proof-carrying continuation witnesses, harvested AFTER visibility escape.

The first attempt at this (`tools/continuation_witness.py`) searched for a closing suffix from the
merged state itself and found **zero** across 59 classes: ReProver at 25 expansions closes almost
no mid-proof MeasureTheory/Analysis goal. That was zero power, not a negative result.

The escape finding supplies the fix. In 18 confirmed classes the members' renderings diverge under
a shared continuation — 17 of them under `contrapose!`. After that step the two members sit at
*genuinely different goals*, and a one-shot closer need only dispatch one of them. So instead of
asking a weak search to close the shared goal, we ask a cheap closer battery to close the
post-escape goals, and assemble the suffix:

    u = shared_prefix ++ [escape_tactic] ++ [closer]

If u closes the proof on member s₁ and fails on s₂, while φ(s₁) = φ(s₂) byte-identically, then

    u ∈ L(s₁) \\ L(s₂)

is an executable, Lean-checked witness that two identically-rendered states have different
verified continuation languages. A pp-keyed transposition table that merges them keeps one
representative and discards the other; if it keeps s₂, the proof u is erased from the search space.

Controls, all of which have caught a false result earlier in this project:
  * fresh Dojo session per class; members replayed from scratch;
  * renderings re-verified byte-identical BEFORE the escape step;
  * every closer run REPEATS times per member; a closer that disagrees with itself is discarded
    (`aesop`, `exact?`, and `refine ⟨…,?_⟩` are all nondeterministic here);
  * any candidate witness is re-verified END TO END: the full suffix u is replayed from the
    original class member states on a fresh session, and must close on one and fail on the other.

Usage:
    python tools/postescape_witness.py --contwit runs/contwit --states runs/killtest/states.jsonl \\
        --repeats 3 --shard 0 --n-shards 4
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: One-shot closers. Broad on purpose: the post-escape goals are ordinary Mathlib obligations and
#: any of these may dispatch one side. Nondeterministic members (`aesop`, `exact?`) are kept
#: because the repeat control will expose them rather than silently admitting them.
CLOSERS = [
    "rfl", "assumption", "trivial", "omega", "decide", "norm_num", "simp", "simp_all",
    "tauto", "aesop", "linarith", "nlinarith", "positivity", "gcongr", "ring", "field_simp",
    "exact?", "norm_cast", "push_cast", "bound", "measurability", "fun_prop",
]


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _log(m: str) -> None:
    print(f"[postesc] {m}", file=sys.stderr, flush=True)


def goal_closers(pp: str, cap: int = 6) -> list:
    """Hypothesis-derived closers: `exact h`, `simp [h]`, `linarith [h]`."""
    names = []
    for line in pp.splitlines():
        if line.lstrip().startswith("⊢"):
            break
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_'!?₀-₉]*)\s*:", line.strip())
        if m and m.group(1) not in names:
            names.append(m.group(1))
    out = []
    for n in names[:cap]:
        out += [f"exact {n}", f"simp [{n}]", f"linarith [{n}]"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contwit", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", default="runs/postesc")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dojo-timeout", type=int, default=900)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    from lean_dojo import Dojo, Theorem
    import lean_dojo as ldj
    from audit.replay import lean_git_repo, kill_orphan_lean, deadline

    cw = Path(args.contwit)
    ver = json.load(open(cw / "escape_verification.json"))
    esc = [r for r in ver if r.get("verdict") == "CONFIRMED_ESCAPE"]
    mine = [r for i, r in enumerate(esc) if i % args.n_shards == args.shard]
    _log(f"shard {args.shard}: {len(mine)} confirmed-escape classes of {len(esc)}")

    # tactic_index recovery, joined via pp_hash (matching on file+theorem alone picks the wrong
    # step of the theorem and silently reaches a different root).
    idx = {}
    for run in ("runs/reconv_b2", "runs/hunt_enn", "runs/hunt_rare"):
        for f in glob.glob(str(Path(run) / "stageb.shard*.jsonl")):
            for l in open(f):
                if l.strip():
                    d = json.loads(l)
                    idx[(d["file"], d["theorem"])] = d["tactic_index"]
    wanted = set()
    for r in mine:
        r["tactic_index"] = r.get("tactic_index", idx.get((r["file"], r["theorem"])))
        wanted.add((r["file"], r["theorem"], r["tactic_index"]))
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
    _log(f"indexed {len(prov)} roots")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh_out = open(out / f"postesc.shard{args.shard}.jsonl", "a", encoding="utf-8")

    def outcome(r):
        if isinstance(r, ldj.ProofFinished):
            return "COMPLETE"
        if isinstance(r, ldj.TacticState):
            return "S:" + _h(r.pp)
        msg = (getattr(r, "error", "") or str(r) or "")
        return "TIMEOUT" if "timeout" in msg.lower() else "FAIL"

    n_wit = 0
    for ci, r in enumerate(mine, 1):
        s = prov.get((r["file"], r["theorem"], r.get("tactic_index")))
        if s is None:
            _log(f"[{ci}/{len(mine)}] {r['theorem'][:40]}: root not found")
            continue
        p = s["provenance"]
        prefix = r["shared_prefix"]
        escape_tac = r["probe"]
        t0 = time.time()
        rec = {"theorem": r["theorem"], "file": r["file"], "tactic_index": r["tactic_index"],
               "histories": r["histories"], "shared_prefix": prefix,
               "escape_tactic": escape_tac}

        def open_members(dojo_ctx):
            """Replay each history + shared prefix; returns name -> state at the escape point."""
            dojo, st0 = dojo_ctx
            for pre in s["proof_prefix"]:
                st0 = dojo.run_tac(st0, pre)
            m = {}
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
                    m[" ; ".join(hist)] = st
            return m

        try:
            thm = Theorem(lean_git_repo(ldj, p["repo_url"], p["repo_commit"]),
                          p["file_path"], p["theorem_full_name"])
            ctx = Dojo(thm, timeout=args.dojo_timeout)
            dojo, root = ctx.__enter__()
        except BaseException:
            kill_orphan_lean()
            continue
        try:
            members = open_members((dojo, root))
            pps = {n: m.pp for n, m in members.items()}
            rec["identical_at_escape_point"] = (len(set(pps.values())) == 1
                                                if len(pps) >= 2 else None)
            if len(members) < 2 or not rec["identical_at_escape_point"]:
                rec["skipped"] = "members not restorable / not identical"
                fh_out.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh_out.flush()
                continue

            # Step into the escape. Both members must accept the escape tactic for the suffix to
            # be well-formed on both sides; if one refuses it, the separation is at depth 1 and is
            # already recorded elsewhere (and is not proof-carrying).
            post = {}
            for n, st in members.items():
                with deadline(120, escape_tac[:30]):
                    rr = dojo.run_tac(st, escape_tac)
                if isinstance(rr, ldj.TacticState):
                    post[n] = rr
            rec["post_escape"] = {n: _h(v.pp) for n, v in post.items()}
            if len(post) < 2:
                rec["skipped"] = "escape tactic does not apply to both members"
                fh_out.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh_out.flush()
                continue
            rec["post_escape_distinct"] = len({v.pp for v in post.values()}) > 1

            pvals = list(post.values())
            battery = CLOSERS + goal_closers(pvals[0].pp) + goal_closers(pvals[1].pp)
            seen_b = set()
            battery = [b for b in battery if not (b in seen_b or seen_b.add(b))]
            table = {}
            for c in battery:
                obs = {}
                for n, st in post.items():
                    vals = []
                    for _ in range(args.repeats):
                        try:
                            with deadline(60, c[:24]):
                                vals.append(outcome(dojo.run_tac(st, c)))
                        except BaseException:
                            vals.append("CRASH")
                            break
                    obs[n] = vals
                table[c] = obs
            rec["closer_table"] = {c: {n: v for n, v in o.items()} for c, o in table.items()}

            # Candidate witness: closer is self-consistent on every member, CLOSES on at least
            # one and does not close on at least one other (excluding crashes/timeouts).
            cands = []
            for c, obs in table.items():
                if not all(len(set(v)) == 1 and "CRASH" not in v and "TIMEOUT" not in v
                           for v in obs.values()):
                    continue
                firsts = {n: v[0] for n, v in obs.items()}
                closers_ = [n for n, v in firsts.items() if v == "COMPLETE"]
                others = [n for n, v in firsts.items() if v != "COMPLETE"]
                if closers_ and others:
                    cands.append({"closer": c, "closes_on": closers_, "fails_on": others,
                                  "outcomes": firsts})
            rec["candidates"] = cands
            _log(f"[{ci}/{len(mine)}] {r['theorem'][:38]}: post_distinct="
                 f"{rec['post_escape_distinct']} candidates={len(cands)}")
        except BaseException as e:
            rec["error"] = type(e).__name__
            cands = []
        finally:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass

        # ---- end-to-end re-verification of each candidate on a FRESH session ----------------
        confirmed = []
        for cand in (rec.get("candidates") or []):
            u = list(prefix) + [escape_tac, cand["closer"]]
            try:
                thm = Theorem(lean_git_repo(ldj, p["repo_url"], p["repo_commit"]),
                              p["file_path"], p["theorem_full_name"])
                ctx2 = Dojo(thm, timeout=args.dojo_timeout)
                dojo2, root2 = ctx2.__enter__()
            except BaseException:
                kill_orphan_lean()
                continue
            try:
                st0 = root2
                for pre in s["proof_prefix"]:
                    st0 = dojo2.run_tac(st0, pre)
                final = {}
                for hist in r["histories"]:
                    st = st0
                    ok = True
                    for tac in hist:
                        with deadline(90, tac[:30]):
                            st = dojo2.run_tac(st, tac)
                        if not isinstance(st, ldj.TacticState):
                            ok = False
                            break
                    if not ok:
                        continue
                    name = " ; ".join(hist)
                    # the suffix u, verbatim, from the alias-class member state
                    cur, res = st, None
                    for tac in u:
                        with deadline(90, tac[:30]):
                            rr = dojo2.run_tac(cur, tac)
                        res = outcome(rr)
                        if not isinstance(rr, ldj.TacticState):
                            break
                        cur = rr
                    final[name] = res
                closes = [n for n, v in final.items() if v == "COMPLETE"]
                fails = [n for n, v in final.items() if v != "COMPLETE"]
                if closes and fails:
                    confirmed.append({"suffix": u, "closes_on": closes, "fails_on": fails,
                                      "end_to_end": final})
                    _log(f"    *** PROOF-CARRYING WITNESS *** u={u} "
                         f"closes {closes} fails {fails}")
            except BaseException:
                pass
            finally:
                try:
                    ctx2.__exit__(None, None, None)
                except Exception:
                    pass
        rec["confirmed_witnesses"] = confirmed
        if confirmed:
            n_wit += 1
        fh_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh_out.flush()
        _log(f"[{ci}/{len(mine)}] done {time.time()-t0:.0f}s  witnesses so far: {n_wit}")

    fh_out.close()
    kill_orphan_lean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
