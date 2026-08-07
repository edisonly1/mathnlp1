"""Do Mathlib's OWN proofs traverse two positions carrying one rendering?

The confirmed alias classes are reached by expanding a root with a tactic pool, so they are
search-reachable by construction. That leaves open a different question: does the human corpus
itself ever revisit a rendering, that is, does a written proof script pass through two positions
whose printed states are byte-identical?

The traced census answers it directly and without Lean, because every state of every proof is
recorded with its position and the tactic the author applied there. A repeat is a proof with two
positions sharing a rendering; the informative case is a repeat where the author applied the SAME
tactic at both, since only then can the two successors be compared for divergence.

Usage:
    python tools/natural_traversal.py --states runs/killtest/states.jsonl \
        --out runs/natural/summary.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="runs/killtest/states.jsonl")
    ap.add_argument("--out", default="runs/natural/summary.json")
    args = ap.parse_args()

    H = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
    # thm -> tactic_index -> (rendering hash, human tactic). Hashes only: the full renderings
    # would not fit in memory at census scale.
    by_thm: dict = collections.defaultdict(dict)
    n_states = 0
    with open(args.states, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            p = d["provenance"]
            by_thm[(p["file_path"], p["theorem_full_name"])][p["tactic_index"]] = (
                H(d["observation"]["phi_state"]), d.get("human_tactic"))
            n_states += 1

    res = {"states": n_states, "proofs": len(by_thm), "proofs_with_repeat": 0,
           "repeat_pairs": 0, "same_tactic": 0, "same_tactic_successor_differs": 0,
           "same_tactic_successor_identical": 0, "same_tactic_successor_unrecorded": 0,
           "different_tactic": 0, "pairs": []}
    for (path, thm), idxs in by_thm.items():
        groups = collections.defaultdict(list)
        for i, (h, _) in idxs.items():
            groups[h].append(i)
        reps = {h: sorted(v) for h, v in groups.items() if len(v) > 1}
        if not reps:
            continue
        res["proofs_with_repeat"] += 1
        for positions in reps.values():
            for a, b in zip(positions, positions[1:]):
                res["repeat_pairs"] += 1
                ta, tb = idxs[a][1], idxs[b][1]
                rec = {"file": path, "theorem": thm, "positions": [a, b],
                       "tactics": [ta, tb]}
                if ta != tb:
                    res["different_tactic"] += 1
                    rec["case"] = "different_tactic"
                else:
                    res["same_tactic"] += 1
                    sa, sb = idxs.get(a + 1), idxs.get(b + 1)
                    if sa is None or sb is None:
                        res["same_tactic_successor_unrecorded"] += 1
                        rec["case"] = "successor_unrecorded"
                    elif sa[0] != sb[0]:
                        res["same_tactic_successor_differs"] += 1
                        rec["case"] = "successor_differs"
                    else:
                        res["same_tactic_successor_identical"] += 1
                        rec["case"] = "successor_identical"
                res["pairs"].append(rec)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, v in res.items():
        if k != "pairs":
            print(f"{k:38} {v}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
