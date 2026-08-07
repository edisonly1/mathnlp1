"""Combined Stage A + Stage B report on within-search transposition merging.

Stage A asks whether ReProver's merge can fire at all: do two distinct tactic histories from
one root reach a byte-identical rendering? Stage B asks whether firing it is WRONG: do the
merged states behave differently under a continuation battery?

Both numerator and denominator come from one run, so the reconvergence rate and the divergence
rate describe the same population.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
from pathlib import Path


def rule_of_three(k: int, n: int) -> tuple:
    """Wilson 95% interval; for k=0 the upper bound is the usual 3/n approximation."""
    if n == 0:
        return (0.0, 1.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)

    roots = [json.loads(l) for f in glob.glob(str(run / "roots.shard*.jsonl"))
             for l in open(f) if l.strip()]
    cls = [json.loads(l) for f in glob.glob(str(run / "stageb.shard*.jsonl"))
           for l in open(f) if l.strip()]

    n_roots = len(roots)
    expandable = [r for r in roots if r["expandable"]]
    with_rec = [r for r in roots if r["n_reconv_classes"] > 0]

    print("=" * 78)
    print("STAGE A — does the merge condition ever arise?")
    print("=" * 78)
    print(f"  roots attempted                        : {n_roots}")
    print(f"  roots where any pool tactic applied    : {len(expandable)}")
    print(f"  roots with >=1 reconvergence class     : {len(with_rec)}")
    if expandable:
        lo, hi = rule_of_three(len(with_rec), len(expandable))
        print(f"  reconvergence rate (of expandable)     : "
              f"{len(with_rec)/len(expandable):.3f}  [95% CI {lo:.3f}, {hi:.3f}]")
    print(f"  reconvergence classes found            : {len(cls)}")

    print()
    print("=" * 78)
    print("STAGE B — when it arises, is merging WRONG?")
    print("=" * 78)
    # A watchdog firing mid-request leaves the Dojo pipe DESYNCHRONISED, so every later probe
    # result on that member belongs to a different probe. The signature is unmistakable: the same
    # few successor hashes reappear under shifted probe names, and the class reports 12-17
    # "divergent" probes instead of 1-2. Any class containing a CRASH is therefore unusable —
    # 7 of 11 originally-flagged classes were contaminated this way, against 9 of 150 overall.
    def crashed(c):
        return any(v == "CRASH" for o in c["outcomes"].values() for v in o.values())
    dirty = [c for c in cls if crashed(c)]
    cls = [c for c in cls if not crashed(c)]
    print(f"  classes quarantined (CRASH desync)     : {len(dirty)}")
    div = [c for c in cls if c["divergent_probes"]]
    print(f"  classes probed                         : {len(cls)}")
    print(f"  classes with a probe divergence        : {len(div)}   <== the finding")
    if cls:
        lo, hi = rule_of_three(len(div), len(cls))
        print(f"  divergence rate                        : "
              f"{len(div)/len(cls):.3f}  [95% CI {lo:.3f}, {hi:.3f}]")

    # A "no divergence" verdict is only meaningful if the battery actually did something.
    live = []
    for c in cls:
        outs = c["outcomes"]
        k0 = list(outs)[0]
        live.append(sum(1 for a, o in outs[k0].items()
                        if not o.startswith(("FAIL", "CRASH", "TIMEOUT"))))
    if live:
        print(f"  battery power: live probes per class   : "
              f"median {sorted(live)[len(live)//2]}, min {min(live)}, max {max(live)}")
        print(f"  classes where NO probe applied (vacuous): {sum(1 for x in live if x == 0)}")

    shape = collections.Counter()
    for c in cls:
        hs = [tuple(h) for h in c["histories"]]
        if len(hs) >= 2 and all(len(h) == len(hs[0]) for h in hs) and \
                all(sorted(h) == sorted(hs[0]) for h in hs) and len(hs[0]) > 1:
            shape["commutation (same steps, different order)"] += 1
        elif all(len(h) == 1 for h in hs):
            shape["sibling merge (distinct single actions)"] += 1
        else:
            shape["other"] += 1
    print("\n  reconvergence shapes:")
    for k, v in shape.most_common():
        print(f"    {v:5}  {k}")

    if div:
        print("\n  DIVERGENT CLASSES:")
        for c in div:
            print(f"    {c['theorem']}  {c['pp_hash'][:10]}")
            print(f"      histories: {[' ; '.join(h) for h in c['histories']]}")
            print(f"      probes   : {c['divergent_probes']}")
            for h, o in c["outcomes"].items():
                print(f"        {h:44} " +
                      "  ".join(f"{a}={o[a]}" for a in c["divergent_probes"]))

    out = run / "reconv_report.json"
    out.write_text(json.dumps({
        "roots_attempted": n_roots, "roots_expandable": len(expandable),
        "roots_with_reconvergence": len(with_rec), "classes": len(cls),
        "classes_divergent": len(div), "shapes": dict(shape),
        "battery_live_probes": live,
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
