"""Constructed retention test: does rendering-keyed retention lose a proof that
structural identity preserves, on the current Lean/Mathlib stack?

The natural corpus cannot answer this because the evaluated policy proposes no
productive step on it (repsens). This test conditions on the alias by
construction instead, using the 8 acceptance-divergent intervention pairs
already verified on Mathlib master 3b3cdbb / Lean v4.33.0-rc1: identical
renderings, distinct digests, probe accepted on variant A and rejected on B.

PROTOCOL, FIXED BEFORE EXECUTION
--------------------------------
One deterministic best-first search per (pair, arm). Action grammar, declared
in advance and identical across arms:

    at the root            : show <termA>, show <termB>   (queue order per arm)
    at any post-show state : <probe>, rfl                 (in that order)
    after an accepted probe: rfl

Node identity is the arm's key. Rendering arm: the pretty-printed state; the
two shows produce byte-identical renderings, so the second arrival MERGES into
the first and is discarded, exactly as LeanDojo's TacticState equality does.
Digest arm: rendering plus the structural digest, so both survive.

Every edge is evaluated by compiling a standalone file with `lake env lean`
(no mocks; compile results memoized by tactic prefix, justified by the 3x
determinism controls of the main study). A state is PROVED when its file
compiles with no error and no `sorry`.

Preregistered readout, all 8 pairs reported with no exclusions:
  arm R with A queued first : proof expected (A retained, probe accepted);
  arm R with B queued first : proof lost expected (B retained, probe rejected,
                              rfl not expected to close a propositional
                              coercion equation, but the compiler decides);
  arm D, both orders        : proof expected (both retained).
A pair CONTRADICTS the hypothesis if arm R proves in both orders or arm D
fails in either. Edge evaluations per arm are recorded for the efficiency
comparison.

Usage:
    python tools/stress_search.py --repo ../persist/mathlib4_current \
        --out runs/stress_search
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.intervention_gen import CAPTURE_PRE, MECHS  # noqa: E402  (pair list + probe text)

IMPORTS = MECHS.get("coercion_accept", {}).get("imports") or [
    "Mathlib.Data.ENNReal.Basic", "Mathlib.Data.ENNReal.Operations",
    "Mathlib.Tactic.NormNum"]


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _log(m: str) -> None:
    print(f"[stress] {m}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="../persist/mathlib4_current")
    ap.add_argument("--out", default="runs/stress_search")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--union-grammar", action="store_true",
                    help="control: one FIXED grammar for every pair, the union of all 8 "
                         "pairs' probes plus rfl, instead of the pair's own probe. Answers "
                         "the objection that including the pair's separating probe bakes in "
                         "the result.")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = repo / "test_pairs"
    workdir.mkdir(exist_ok=True)
    imp = "\n".join(f"import {m}" for m in IMPORTS)

    pairs = MECHS["coercion_accept"]["pairs"]
    _log(f"{len(pairs)} pairs, repo {repo.name}")

    n_compiles = 0
    memo: dict = {}

    def compile_file(tag: str, binders: str, goal: str, tactics: list,
                     capture: bool) -> dict:
        """Compile `example binders : goal := by tactics...` and classify.
        capture=True appends the pp+digest probe after the tactics (file then
        ends in sorry, so 'proved' is only meaningful when capture=False)."""
        nonlocal n_compiles
        key = (tag, tuple(tactics), capture)
        if key in memo:
            return memo[key]
        body = "\n  ".join(tactics)
        cap = f"\n  {CAPTURE_PRE}\n  all_goals sorry" if capture else ""
        fn = workdir / f"stress_{_h(tag + body + str(capture))}.lean"
        fn.write_text(f"""{imp}
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false
example {binders} : {goal} := by
  {body}{cap}
""")
        t0 = time.time()
        pr = subprocess.run(["lake", "env", "lean", str(fn.relative_to(repo))],
                            cwd=repo, capture_output=True, text=True,
                            timeout=args.timeout)
        n_compiles += 1
        out = pr.stdout + pr.stderr
        errs = re.findall(rf"{fn.name}:(\d+):\d+: error", out)
        mpp = re.search(r"(?s)PPV:(.*?)PPEND", out)
        mdig = re.search(r"DIGV:(\d+)", out)
        res = {"errors": [int(e) for e in errs],
               "pp": _h(mpp.group(1).strip()) if mpp else None,
               "dig": mdig.group(1) if mdig else None,
               "proved": (not capture) and pr.returncode == 0 and not errs,
               "secs": round(time.time() - t0, 1)}
        memo[key] = res
        _log(f"    compile {fn.name} tac={tactics} cap={capture} "
             f"errors={res['errors']} proved={res['proved']} {res['secs']}s")
        return res

    UNION_ACTIONS = None
    if args.union_grammar:
        # Fixed order: the 8 probes in MECHS declaration order, then rfl.
        UNION_ACTIONS = [pr for (_, _, _, _, pr) in MECHS["coercion_accept"]["pairs"]]
        UNION_ACTIONS = list(dict.fromkeys(UNION_ACTIONS)) + ["rfl"]

    def close_from(tag, binders, goal, show_tac, probe) -> tuple:
        """Expand a post-show state with the declared grammar. Returns
        (proved, edge_evaluations). Pair mode: probe, probe;rfl, rfl. Union
        mode: every action in the fixed order, each followed by rfl on a
        residual, first success wins."""
        edges = 0
        actions = UNION_ACTIONS if UNION_ACTIONS else [probe]
        for act in actions:
            seq = compile_file(tag, binders, goal, [show_tac, act], capture=False)
            edges += 1
            if seq["proved"]:
                return True, edges
            if act != "rfl":
                seq2 = compile_file(tag, binders, goal, [show_tac, act, "rfl"],
                                    capture=False)
                edges += 1
                if seq2["proved"]:
                    return True, edges
        if not UNION_ACTIONS:
            seq3 = compile_file(tag, binders, goal, [show_tac, "rfl"], capture=False)
            edges += 1
            return seq3["proved"], edges
        return False, edges

    results = []
    for pi, (binders, goal, tA, tB, probe) in enumerate(pairs):
        rec = {"pair": pi, "probe": probe, "goal": goal, "arms": {}}
        # Validity gates recomputed, not carried over.
        capA = compile_file(f"p{pi}A", binders, goal, [f"show {tA}"], capture=True)
        capB = compile_file(f"p{pi}B", binders, goal, [f"show {tB}"], capture=True)
        rec["pp_identical"] = (capA["pp"] is not None and capA["pp"] == capB["pp"])
        rec["dig_distinct"] = (capA["dig"] is not None and capB["dig"] is not None
                               and capA["dig"] != capB["dig"])
        rec["valid"] = rec["pp_identical"] and rec["dig_distinct"]
        provedA, eA = close_from(f"p{pi}A", binders, goal, f"show {tA}", probe)
        provedB, eB = close_from(f"p{pi}B", binders, goal, f"show {tB}", probe)
        rec["variant_closes"] = {"A": provedA, "B": provedB}

        for arm, order in (("R_Afirst", "AB"), ("R_Bfirst", "BA"),
                           ("D_Afirst", "AB"), ("D_Bfirst", "BA")):
            digest_key = arm.startswith("D")
            # root expansion: both shows evaluated (2 edges); under the
            # rendering key the second arrival merges and is discarded.
            edges = 2
            retained = list(order) if digest_key else [order[0]]
            proved = False
            for v in retained:
                ok, e = (provedA, eA) if v == "A" else (provedB, eB)
                edges += e
                if ok:
                    proved = True
                    break
            rec["arms"][arm] = {"proved": proved, "edges": edges,
                                "retained": retained}
        rec["order_dependent_R"] = (rec["arms"]["R_Afirst"]["proved"]
                                    != rec["arms"]["R_Bfirst"]["proved"])
        rec["digest_invariant"] = (rec["arms"]["D_Afirst"]["proved"]
                                   and rec["arms"]["D_Bfirst"]["proved"])
        results.append(rec)
        _log(f"[{pi}] valid={rec['valid']} closes A={provedA} B={provedB} "
             f"R(A)={rec['arms']['R_Afirst']['proved']} "
             f"R(B)={rec['arms']['R_Bfirst']['proved']} "
             f"D both={rec['digest_invariant']}")

    summary = {
        "pairs": len(results),
        "valid": sum(r["valid"] for r in results),
        "order_dependent_under_rendering_key":
            sum(r["order_dependent_R"] for r in results),
        "proof_lost_when_rejecting_variant_retained":
            sum(1 for r in results if r["arms"]["R_Afirst"]["proved"]
                and not r["arms"]["R_Bfirst"]["proved"]),
        "digest_invariant_both_orders":
            sum(r["digest_invariant"] for r in results),
        "compiles": n_compiles,
    }
    (outdir / "results.json").write_text(
        json.dumps({"summary": summary, "pairs": results}, indent=1))
    _log(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
