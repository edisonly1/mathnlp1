"""Stage 4: controlled additive-extension sweep.

The kill test found 26 success->failure cases, but 25 were tier B — the two members had
different elaborated goals, because the population came from phi_state collisions and those are
*selected* for differing goals. The goal is a confound there.

This removes the confound by intervening instead of observing. For one fixed theorem and one
fixed tactic step:

    baseline   Dojo(thm)                                E
    extended   Dojo(thm, additional_imports=[M])        E'  with E ⪯ E'

`additional_imports` only ADDS declarations: no existing declaration is modified or removed, the
theorem statement is untouched, and the tactic text is byte-identical. Verified on
`List.getElem_pmap`: importing `Mathlib.Tactic.Ring` moved the simp set from 6,436 to 10,503
lemmas while `f_core` stayed at `aae1ae4f6d9d` — the elaborated goal is unchanged.

Every trial therefore satisfies the strict controls by construction:
  * elaborated goal identity  -- checked per trial via f_core, not assumed
  * same tactic text          -- byte-identical string
  * additive-only environment -- imports cannot remove or alter declarations
  * same options / version    -- same Dojo config, same toolchain

A trial counts as a violation of OUTCOME MONOTONICITY when the baseline succeeds and the
extended environment fails. Failure->success is recorded but is not the finding: that merely
restates that context helps.

Usage:
    python tools/extension_sweep.py --states runs/killtest/states.jsonl --out runs/extsweep \\
        --trials 200 --shard 0 --n-shards 4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Extension modules. Chosen to be broadly importable and to add substantial declarations,
#: simp lemmas and instances. A module already transitively imported is a no-op and is skipped
#: by the delta check; a module that cannot be imported raises DojoInitError and is skipped.
#: HEAVY extension modules, chosen to sit high in the import DAG so they add thousands of
#: declarations rather than dozens. The first sweep used light modules and 58% of trials were
#: no-ops: mature Mathlib files already import them transitively, so the "extension" added
#: nothing and the test had no power. Where it did apply it added only +52..+311 simp lemmas,
#: against +4,067 when Mathlib.Tactic.Ring is genuinely new to a file.
EXTENSIONS = [
    "Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic",
    "Mathlib.MeasureTheory.Integral.Bochner",
    "Mathlib.NumberTheory.LucasLehmer",
    "Mathlib.Topology.MetricSpace.Basic",
]

#: Restrict to files LOW in the import DAG, which have room to grow. This is also the faithful
#: counterfactual: early-layer proofs are the ones later library growth actually reaches.
EARLY_LAYER = ("Mathlib/Logic/", "Mathlib/Order/", "Mathlib/Data/Nat/", "Mathlib/Data/List/",
               "Mathlib/Data/Set/", "Mathlib/Algebra/Group/", "Mathlib/Init/",
               "Mathlib/Data/Finset/", "Mathlib/Data/Option/", "Mathlib/Combinatorics/")


def _log(m: str) -> None:
    print(f"[extsweep] {m}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260724)
    ap.add_argument("--dojo-timeout", type=int, default=300)
    ap.add_argument("--bare-simp-only", action="store_true",
                    help="restrict to bare `simp` steps — the case the folklore is actually "
                         "about; the general sweep sampled only 3 of them")
    args = ap.parse_args()

    from lean_dojo import Dojo, Theorem
    import lean_dojo as ldj
    from audit.fingerprint import PROBES, _run, parse_probe, collect_fields, assemble
    from audit.fingerprint import build_fingerprint
    from audit.replay import lean_git_repo, kill_orphan_lean

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / f"trials.shard{args.shard}.jsonl", "a", encoding="utf-8")

    states = [json.loads(l) for l in Path(args.states).read_text().splitlines()]
    # Only steps with a human tactic are usable: we need an action known to work in E.
    usable = [s for s in states if s.get("human_tactic") and s.get("proof_prefix") is not None
              and s["provenance"]["file_path"].startswith(EARLY_LAYER)]
    if args.bare_simp_only:
        usable = [s for s in usable if s["human_tactic"].strip() == "simp"]
        _log(f"bare-simp filter: {len(usable)} steps")
    rng = random.Random(args.seed)
    rng.shuffle(usable)
    picked = usable[:args.trials]
    mine = [s for i, s in enumerate(picked) if i % args.n_shards == args.shard]
    _log(f"shard {args.shard}: {len(mine)} steps of {len(picked)} sampled "
         f"(from {len(usable)} usable)")

    def run_trial(s, imports):
        """Open a Dojo (optionally extended), restore the state, return (f_core, outcome)."""
        p = s["provenance"]
        thm = Theorem(lean_git_repo(ldj, p["repo_url"], p["repo_commit"]),
                      p["file_path"], p["theorem_full_name"])
        ctx = Dojo(thm, timeout=args.dojo_timeout, additional_imports=imports)
        dojo, st = ctx.__enter__()
        try:
            for pre in s["proof_prefix"]:
                st = dojo.run_tac(st, pre)
                if not isinstance(st, ldj.TacticState):
                    return None, "PREFIX_FAIL", None, None
            fp = build_fingerprint(assemble(collect_fields(dojo, st)), "sweep", faithful=False)
            simp = parse_probe(_run(dojo, st, PROBES["simp"]) or "", False)
            r = dojo.run_tac(st, s["human_tactic"])
            succ = None
            if isinstance(r, ldj.ProofFinished):
                o = "COMPLETE"
            elif isinstance(r, ldj.TacticState):
                o = "SUCC"
                # Transition stability (the second property): a tactic can "succeed" in both
                # environments while leaving DIFFERENT proof obligations. Recording only the
                # outcome label hides that entirely, so capture the successor goal itself.
                succ = getattr(r, "pp", "") or ""
            else:
                msg = (getattr(r, "error", "") or str(r))
                o = "TIMEOUT" if "timeout" in msg.lower() else "FAIL"
            return fp.f_core, o, simp, succ
        finally:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass

    n_viol = 0
    for i, s in enumerate(mine, 1):
        t0 = time.time()
        try:
            base_core, base_out, base_simp, base_succ = run_trial(s, [])
        except BaseException as e:
            _log(f"[{i}/{len(mine)}] baseline failed {type(e).__name__}")
            kill_orphan_lean()
            continue
        # Only steps that WORK in the base environment can violate outcome monotonicity.
        if base_out not in ("COMPLETE", "SUCC"):
            _log(f"[{i}/{len(mine)}] baseline not successful ({base_out}); skip")
            continue

        for mod in EXTENSIONS:
            try:
                ext_core, ext_out, ext_simp, ext_succ = run_trial(s, [mod])
            except BaseException as e:
                continue                      # module not importable here
            if ext_core is None or base_core is None:
                continue
            goal_same = (ext_core == base_core)
            grew = (base_simp != ext_simp)    # did the environment actually change?
            rec = {
                "file": s["provenance"]["file_path"],
                "theorem": s["provenance"]["theorem_full_name"],
                "tactic": s["human_tactic"],
                "module": mod,
                "base_outcome": base_out, "ext_outcome": ext_out,
                "goal_identical": goal_same, "env_grew": grew,
                "base_simp": base_simp, "ext_simp": ext_simp,
                # transition stability: same outcome label but different successor goal
                "successor_changed": bool(base_out == ext_out == "SUCC"
                                          and base_succ is not None and ext_succ is not None
                                          and base_succ != ext_succ),
                "base_succ_len": len(base_succ or ""), "ext_succ_len": len(ext_succ or ""),
                "violation": bool(goal_same and grew
                                  and base_out in ("COMPLETE", "SUCC")
                                  and ext_out == "FAIL"),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if rec["violation"]:
                n_viol += 1
                _log(f"  VIOLATION  {mod}  {base_out}->FAIL  "
                     f"{s['human_tactic'][:44]!r}  {s['provenance']['file_path']}")
        _log(f"[{i}/{len(mine)}] {time.time()-t0:5.1f}s  violations so far: {n_viol}")

    fh.close()
    kill_orphan_lean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
