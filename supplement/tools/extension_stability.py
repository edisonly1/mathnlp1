"""Kill test: is Lean's proof automation NON-MONOTONE under additive environment growth?

Lean's underlying logic is monotone — extending a theory with kernel-accepted declarations
cannot invalidate an existing theorem. Its *automation* need not be. This asks whether that
gap is real and measurable:

    E ⪯ E'   (E' adds declarations, changes none)
    same elaborated goal, same tactic, same options, same version
    Exec(E, g, a) = success   but   Exec(E', g, a) = failure

Direction is the whole question. Failure→success only restates that context helps, which
miniCTX and the premise-selection literature already establish. **Success→failure is the
finding**: valid mathematical knowledge with negative causal utility for an unchanged action.

Ordering proxy
--------------
`nconsts` counts LOCALLY declared constants at a state. Within one file it grows monotonically
as declarations accumulate, so for two members of the same file a strictly larger `nconsts`
means strictly more is in scope. This is a proxy for E ⪯ E', not a proof of it: it does not
verify that every declaration in E survives unchanged into E'. Cross-file pairs are excluded
because their environments are not comparable this way.

Tiering
-------
  tier A  same elaborated goal (F_core equal) AND same file  — the clean extension population
  tier B  same file, goal differs                            — confounded by the goal
  excluded: cross-file, degraded extraction, notation/scope mechanisms (easier explanations)

Usage:
    python tools/extension_stability.py --run runs/killtest
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

# Outcomes that count as the tactic having worked.
SUCCESS = ("COMPLETE", "SUCC")
# Excluded from divergence entirely (blueprint §7.6): never treat a timeout as logical failure.
NOT_COUNTED = ("TIMEOUT", "CRASH", "GAVE_UP")


def is_success(o: str) -> bool:
    return o.startswith(SUCCESS)


def counted(o: str) -> bool:
    return not o.startswith(NOT_COUNTED)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)

    rows = []
    for f in sorted(run.glob("classes.shard*.jsonl")) or [run / "classes.jsonl"]:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    # dedup
    seen, cls = set(), []
    for r in rows:
        if r["observation_key"] not in seen:
            seen.add(r["observation_key"])
            cls.append(r)
    print(f"classes analysed: {len(cls)}\n" + "=" * 78)

    findings = collections.Counter()
    s2f, f2s, drift = [], [], []

    for c in cls:
        reps = [r for r in c.get("reps", []) if not r.get("is_degraded")]
        if len(reps) < 2:
            findings["skip: <2 usable members"] += 1
            continue
        if len(set(c["files"])) != 1:
            findings["skip: cross-file (environments not comparable)"] += 1
            continue
        # Order by declaration count. Equal counts cannot be ordered additively.
        try:
            reps.sort(key=lambda r: int(r["nconsts"]))
        except (ValueError, TypeError):
            findings["skip: nconsts unavailable"] += 1
            continue
        lo, hi = reps[0], reps[-1]
        if int(lo["nconsts"]) >= int(hi["nconsts"]):
            findings["skip: environments not ordered by nconsts"] += 1
            continue

        tier = "A" if lo["f_core"] == hi["f_core"] else "B"
        for tac in c["divergent_tactics"]:
            a, b = lo["outcomes"].get(tac), hi["outcomes"].get(tac)
            if a is None or b is None or not counted(a) or not counted(b):
                continue
            rec = {"key": c["observation_key"][:12], "tier": tier, "tactic": tac,
                   "file": c["files"][0], "small_env": a, "large_env": b,
                   "nconsts": [lo["nconsts"], hi["nconsts"]],
                   "simpN": [lo.get("simpN"), hi.get("simpN")],
                   "mechanisms": c["mechanisms"], "severity": c["severity"]}
            if is_success(a) and not is_success(b):
                s2f.append(rec)              # THE FINDING
            elif not is_success(a) and is_success(b):
                f2s.append(rec)              # merely "context helps"
            elif is_success(a) and is_success(b) and a != b:
                drift.append(rec)            # behavioural drift

    print("\n--- DIRECTION (the whole question) ---")
    print(f"  SUCCESS -> FAILURE under added declarations : {len(s2f)}   <== the finding")
    print(f"  failure -> success (context merely helps)   : {len(f2s)}")
    print(f"  success -> different success (drift)        : {len(drift)}")

    if s2f:
        tiers = collections.Counter(r["tier"] for r in s2f)
        fams = collections.Counter(r["tactic"].split()[0] for r in s2f)
        files = {r["file"] for r in s2f}
        print(f"\n--- SUCCESS->FAILURE detail ---")
        print(f"  tier A (same elaborated goal): {tiers['A']}   tier B: {tiers['B']}")
        print(f"  distinct tactic families     : {len(fams)}  {dict(fams)}")
        print(f"  distinct files               : {len(files)}")
        simp_only = all(r["tactic"].split()[0].startswith("simp") for r in s2f)
        print(f"  ALL simp?                    : {simp_only}"
              f"{'  <-- folklore only; weak' if simp_only else '  <-- extends beyond simp'}")
        print("\n  witnesses:")
        for r in s2f[:15]:
            d = int(r["nconsts"][1]) - int(r["nconsts"][0])
            print(f"    [{r['tier']}] {r['key']} +{d:>3} decls  {r['small_env']:>12} -> "
                  f"{r['large_env']:<12} {r['tactic'][:34]!r}")
            print(f"         {r['file']}  simpN {r['simpN'][0]}->{r['simpN'][1]}")

    print("\n--- skips ---")
    for k, v in findings.most_common():
        print(f"  {v:5}  {k}")

    out = run / "extension_stability.json"
    out.write_text(json.dumps({"success_to_failure": s2f, "failure_to_success": f2s,
                               "drift": drift}, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")

    print("\n--- GO / NO-GO (thresholds set in advance) ---")
    checks = {
        ">=20 success->failure": len(s2f) >= 20,
        ">=3 tactic families": len({r["tactic"].split()[0] for r in s2f}) >= 3,
        ">=3 distinct files": len({r["file"] for r in s2f}) >= 3,
        "not all simp": not (s2f and all(r["tactic"].split()[0].startswith("simp") for r in s2f)),
        ">=1 tier-A (same goal)": any(r["tier"] == "A" for r in s2f),
    }
    for k, v in checks.items():
        print(f"  [{'x' if v else ' '}] {k}")
    print(f"\n  VERDICT: {'PURSUE THE PIVOT' if all(checks.values()) else 'PIVOT NOT SUPPORTED BY THIS DATA'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
