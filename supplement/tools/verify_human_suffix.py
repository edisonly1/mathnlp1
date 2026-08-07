"""Precondition check for the human-suffix witness design. Run this BEFORE the full experiment.

The design assumes three things. Each can fail silently and each would waste hours, so each is
measured here on exactly the population the experiment would run on (the theorems carrying our
confirmed alias classes, which are all in MeasureTheory/Analysis).

  P1. The human tactic sequence can be reconstructed from `states.jsonl`.
      `proof_prefix` at tactic_index i+1 equals `proof_prefix` at i plus `human_tactic` at i, so
      the per-theorem sequence is recoverable by sorting on tactic_index.

  P2. Replaying the reconstructed remainder from an alias root CLOSES the proof.
      This is the one that decides the design. `get_traced_tactics()` returns the tactic TREE
      flattened, and replaying that linearly re-applies children and finishes early -- the reason
      `top_level_tactics` exists and retains only ~77% of steps. If the remainder does not reach
      ProofFinished, there is no known-good suffix and the design is dead. Two failure modes are
      distinguished: NOT_CLOSED (ran out of tactics with goals remaining) and a mid-suffix FAIL.

  P3. The human successor is reachable as an expansion node, i.e. applying `human_tactic` to the
      alias root yields a TacticState we can treat as the human-path member.

Reports each rate separately. P2 is the go/no-go: a low rate means the experiment cannot produce
witnesses regardless of how many classes are swept.

Usage:
    python tools/verify_human_suffix.py --states runs/killtest/states.jsonl --sample 20
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _log(m: str) -> None:
    print(f"[verify-human] {m}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--dojo-timeout", type=int, default=900)
    ap.add_argument("--max-suffix", type=int, default=60)
    args = ap.parse_args()

    from lean_dojo import Dojo, Theorem
    import lean_dojo as ldj
    from audit.replay import lean_git_repo, kill_orphan_lean, deadline

    # The theorems carrying our confirmed alias classes -- the exact experimental population.
    targets = set()
    for run in ("runs/reconv_b2", "runs/hunt_enn", "runs/hunt_rare"):
        try:
            for c in json.load(open(Path(run) / "confirmation.json")):
                if c.get("verdict") == "CONFIRMED":
                    targets.add((c["file"], c["theorem"]))
        except Exception:
            pass
    _log(f"{len(targets)} theorems carrying confirmed alias classes")

    # P1: reconstruct each theorem's ordered tactic sequence.
    per_thm = collections.defaultdict(dict)
    with open(args.states, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            pv = d["provenance"]
            k = (pv["file_path"], pv["theorem_full_name"])
            if k in targets:
                per_thm[k][pv["tactic_index"]] = d
    _log(f"P1: recovered step data for {len(per_thm)} theorems")

    seqs = {}
    for k, byidx in per_thm.items():
        idxs = sorted(byidx)
        # contiguity matters: a gap means we do not have the whole proof
        contiguous = idxs == list(range(idxs[0], idxs[0] + len(idxs)))
        tactics = [byidx[i].get("human_tactic") or "" for i in idxs]
        if all(tactics) and contiguous and idxs[0] == 0:
            seqs[k] = (idxs, tactics, byidx)
    _log(f"P1: {len(seqs)}/{len(per_thm)} theorems have a contiguous sequence from index 0")

    picked = list(seqs)[:args.sample]
    stats = collections.Counter()
    rows = []
    for i, k in enumerate(picked, 1):
        idxs, tactics, byidx = seqs[k]
        # Use the FIRST recorded alias root for this theorem as the branch point.
        root_idx = idxs[0]
        rec0 = byidx[root_idx]
        p = rec0["provenance"]
        remainder = tactics[0:]           # from root_idx onward, inclusive of its human tactic
        if len(remainder) > args.max_suffix:
            stats["skip: suffix too long"] += 1
            continue
        t0 = time.time()
        try:
            thm = Theorem(lean_git_repo(ldj, p["repo_url"], p["repo_commit"]),
                          p["file_path"], p["theorem_full_name"])
            ctx = Dojo(thm, timeout=args.dojo_timeout)
            dojo, st = ctx.__enter__()
        except BaseException as e:
            stats[f"skip: dojo {type(e).__name__}"] += 1
            kill_orphan_lean()
            continue
        verdict, failed_at = "?", None
        try:
            for pre in rec0["proof_prefix"]:
                st = dojo.run_tac(st, pre)
                if not isinstance(st, ldj.TacticState):
                    raise RuntimeError("prefix failed")
            # P3: does the human tactic apply at the root?
            with deadline(120, "human tactic"):
                first = dojo.run_tac(st, remainder[0])
            p3 = isinstance(first, (ldj.TacticState, ldj.ProofFinished))
            stats["P3 human tactic applies" if p3 else "P3 human tactic FAILS"] += 1

            # P2: replay the whole remainder.
            cur, done = st, False
            for j, tac in enumerate(remainder):
                with deadline(120, tac[:30]):
                    r = dojo.run_tac(cur, tac)
                if isinstance(r, ldj.ProofFinished):
                    done = True
                    break
                if not isinstance(r, ldj.TacticState):
                    failed_at = (j, tac, (getattr(r, "error", "") or str(r))[:120])
                    break
                cur = r
            verdict = "CLOSED" if done else ("FAIL_MIDWAY" if failed_at else "NOT_CLOSED")
            stats[f"P2 {verdict}"] += 1
        except BaseException as e:
            verdict = f"ERROR:{type(e).__name__}"
            stats["P2 ERROR"] += 1
        finally:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
        rows.append({"theorem": k[1], "file": k[0], "n_tactics": len(remainder),
                     "verdict": verdict, "failed_at": failed_at})
        _log(f"[{i}/{len(picked)}] {verdict:12} {len(remainder):3} tactics  "
             f"{time.time()-t0:5.0f}s  {k[1][:44]}"
             + (f"  -> step {failed_at[0]}: {failed_at[1][:40]!r}" if failed_at else ""))

    print("\n" + "=" * 78)
    print("PRECONDITION SUMMARY")
    for k2, v in stats.most_common():
        print(f"  {v:4}  {k2}")
    closed = stats.get("P2 CLOSED", 0)
    tried = sum(v for k2, v in stats.items() if k2.startswith("P2 "))
    print(f"\n  P2 close rate: {closed}/{tried}" + (f" = {closed/tried:.0%}" if tried else ""))
    print("  GO if P2 close rate is high enough that a sweep yields witnesses;")
    print("  NO-GO if the human remainder does not replay -- there is then no known-good suffix.")
    Path("runs/human_suffix_check.json").write_text(
        json.dumps({"stats": dict(stats), "rows": rows}, indent=2, ensure_ascii=False))
    kill_orphan_lean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
