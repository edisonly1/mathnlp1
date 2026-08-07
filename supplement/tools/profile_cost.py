"""Measure fingerprint cost against REAL Mathlib states across a range of goal sizes.

The smoke test proves the channel works; this produces the number that decides whether a
20,000-state pilot is hours or days. It matters because cost is not flat:

  * `phi_full` (pp.all + deepTerms) expanded an 82-byte goal to 1260 bytes — ~15x — and a
    64 KB response was measured at 18s, so large states may be far more expensive than small
    ones.
  * `Dojo` startup is paid per theorem, not per state, which is why `replay.py` reuses
    sessions. If startup dominates, session reuse matters more than probe cost.

Reports goal size vs. fingerprint time, and separates Dojo startup from probe time so the
pilot estimate can be built from the right components.

Usage:
    python tools/profile_cost.py [--max-files 400] [--samples 8]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = "https://github.com/leanprover-community/mathlib4"
COMMIT = "29dcec074de168ac2bf835a77ef68bbe069194c5"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-files", type=int, default=400)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--dojo-timeout", type=int, default=900)
    args = ap.parse_args()

    from lean_dojo import Dojo, LeanGitRepo, Theorem, trace
    from audit.extract import _repo_own_files
    from audit.fingerprint import PROBES, collect_fields, fingerprint_state

    t0 = time.time()
    repo = LeanGitRepo(REPO, COMMIT)
    traced = trace(repo)
    print(f"trace (cached) load: {time.time() - t0:.0f}s", flush=True)

    own, skipped = _repo_own_files(traced)
    scan = own[:args.max_files]
    print(f"repo files: {len(own)} (skipped {skipped} dependency files); "
          f"scanning {len(scan)}", flush=True)

    # Collect (goal_size, theorem, tactic_index) across the scanned files.
    cands = []
    for tf in scan:
        try:
            for tt in tf.get_traced_theorems():
                for i, tac in enumerate(tt.get_traced_tactics()):
                    sb = getattr(tac, "state_before", None)
                    if sb:
                        cands.append((len(sb), tt, i, sb))
        except Exception:
            continue
    if not cands:
        print("no states found")
        return 1

    sizes = sorted(c[0] for c in cands)
    print(f"states found: {len(cands)}  goal bytes: "
          f"min={sizes[0]} p50={statistics.median(sizes):.0f} "
          f"p90={sizes[int(0.9 * len(sizes))]} max={sizes[-1]}", flush=True)

    # Sample across the size range rather than uniformly, so the tail is represented.
    cands.sort(key=lambda c: c[0])
    idx = [int(i * (len(cands) - 1) / max(1, args.samples - 1)) for i in range(args.samples)]
    picks = [cands[i] for i in dict.fromkeys(idx)]

    print(f"\n{'goal_B':>8} {'dojo_s':>7} {'probe_s':>8} {'per_probe':>10} "
          f"{'fields':>7} {'phi_full_B':>11}  theorem", flush=True)
    print("-" * 92, flush=True)

    rows = []
    for goal_bytes, tt, tac_idx, sb in picks:
        thm = Theorem(repo, tt.theorem.file_path, tt.theorem.full_name)
        t_open = time.time()
        try:
            ctx = Dojo(thm, timeout=args.dojo_timeout)
            dojo, state = ctx.__enter__()
        except Exception as e:
            print(f"{goal_bytes:>8} {'':>7} {'':>8} {'':>10} {'OPEN-FAIL':>7}  "
                  f"{type(e).__name__}", flush=True)
            continue
        dojo_s = time.time() - t_open
        try:
            t_p = time.time()
            fields = collect_fields(dojo, state)
            probe_s = time.time() - t_p
            full = fields.get("phi_full", "")
            rows.append((goal_bytes, dojo_s, probe_s, len(full)))
            print(f"{goal_bytes:>8} {dojo_s:>7.1f} {probe_s:>8.2f} "
                  f"{probe_s / len(PROBES):>10.3f} {len(fields):>3}/{len(PROBES)} "
                  f"{len(full):>11}  {str(tt.theorem.full_name)[:34]}", flush=True)
        except Exception as e:
            print(f"{goal_bytes:>8} {dojo_s:>7.1f}  PROBE-FAIL {type(e).__name__}", flush=True)
        finally:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass

    if rows:
        print("-" * 92)
        d = [r[1] for r in rows]
        p = [r[2] for r in rows]
        print(f"Dojo startup : median {statistics.median(d):.1f}s  max {max(d):.1f}s")
        print(f"11 probes    : median {statistics.median(p):.2f}s  max {max(p):.2f}s")
        print(f"expansion    : phi_full / goal = "
              f"{statistics.median([r[3] / max(1, r[0]) for r in rows]):.1f}x median")
        print()
        print("Pilot arithmetic (fingerprinting runs on candidate-class members only):")
        for n in (1000, 5000, 20000):
            secs = n * statistics.median(p)
            print(f"  {n:>6} members x {statistics.median(p):.2f}s = {secs / 3600:.1f}h "
                  f"probe time, EXCLUDING Dojo startup")
        print(f"  Dojo startup is per THEOREM and amortises across its states via session reuse;")
        print(f"  at {statistics.median(d):.0f}s each it dominates unless reuse works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
