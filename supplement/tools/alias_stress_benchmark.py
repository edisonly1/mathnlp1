"""Alias-conditioned retention stress benchmark on current Lean and Mathlib.

The benchmark expands the rendering-preserving ENNReal intervention into 96 cases: eight
acceptance-sensitive base propositions crossed with twelve proposition contexts.  A case enters
the benchmark only when Lean verifies all three conditions:

1. the two post-``show`` states have byte-identical default renderings;
2. their structural digests differ;
3. one fixed tactic proposal closes the abbreviated member and fails on the unfolded member.

The search comparison begins after both branches reach the aliased states.  Rendering identity
retains only the first arrival.  Refined identity retains both.  Arrival order is randomized for
the empirical readout and exhaustively enumerated for the exact uniform-order estimand.

Three proposal generators and four action budgets test whether the effect depends on how the
known continuation is exposed.  Every tactic outcome is obtained by compiling Lean examples.
Search trials reuse these deterministic compiler labels; they do not mock tactic execution.

Usage:
    python tools/alias_stress_benchmark.py \
      --repo ../persist/mathlib4_current --out runs/alias_stress
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.intervention_gen import MECHS


IMPORTS = MECHS["coercion_accept"]["imports"]
BASES = MECHS["coercion_accept"]["pairs"]
WRAPPERS = (
    ("plain", "{p}"),
    ("and_true_r", "({p}) ∧ True"),
    ("and_true_l", "True ∧ ({p})"),
    ("or_false_r", "({p}) ∨ False"),
    ("or_false_l", "False ∨ ({p})"),
    ("eq_true_r", "({p}) = True"),
    ("eq_true_l", "True = ({p})"),
    ("double_neg", "¬¬({p})"),
    ("and_true_pair", "({p}) ∧ (True ∧ True)"),
    ("true_or_false_and", "(True ∨ False) ∧ ({p})"),
    ("or_false_and_true", "({p}) ∨ (False ∧ True)"),
    ("duplicate_and", "({p}) ∧ ({p})"),
)
GENERATOR_NAMES = ("targeted", "context_bank", "generic_first")
BUDGETS = (2, 4, 8, 12)


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _log(message: str) -> None:
    print(f"[alias-stress] {message}", flush=True)


def _action(probe: str, wrapper: str) -> str:
    """Lift the separating probe through a proposition context.

    The surrounding commands expose the original proposition as a subgoal and
    use only logical introduction rules.  The probe must therefore close that
    subgoal itself.  A failure on the unfolded member cannot be repaired by an
    unrestricted cleanup tactic.
    """
    templates = {
        "plain": "{probe}",
        "and_true_r": "constructor\n· {probe}\n· trivial",
        "and_true_l": "constructor\n· trivial\n· {probe}",
        "or_false_r": "left\n{probe}",
        "or_false_l": "right\n{probe}",
        "eq_true_r": (
            "apply propext\nconstructor\n· intro _\n  trivial\n"
            "· intro _\n  {probe}"),
        "eq_true_l": (
            "apply propext\nconstructor\n· intro _\n  {probe}\n"
            "· intro _\n  trivial"),
        "double_neg": "intro h\napply h\n{probe}",
        "and_true_pair": "constructor\n· {probe}\n· constructor <;> trivial",
        "true_or_false_and": "constructor\n· left\n  trivial\n· {probe}",
        "or_false_and_true": "left\n{probe}",
        "duplicate_and": "constructor\n· {probe}\n· {probe}",
    }
    return templates[wrapper].format(probe=probe)


def _cases() -> list[dict]:
    cases = []
    for base, (binders, goal, term_a, term_b, probe) in enumerate(BASES):
        for wrapper, (wrapper_name, template) in enumerate(WRAPPERS):
            wrap = lambda proposition: template.format(p=f"({proposition})")
            cases.append({
                "case_id": f"b{base:02d}_w{wrapper:02d}",
                "base": base,
                "wrapper": wrapper_name,
                "binders": binders,
                "goal": wrap(goal),
                "term_a": wrap(term_a),
                "term_b": wrap(term_b),
                "probe": probe,
                "witness_action": _action(probe, wrapper_name),
            })
    return cases


def _header() -> list[str]:
    return [*(f"import {module}" for module in IMPORTS),
            "", "set_option linter.unusedSimpArgs false",
            "set_option linter.unusedVariables false",
            # The action matrix intentionally contains failing examples.  Lean's
            # default error cap would silently leave later labels unclassified.
            "set_option maxErrors 100000", ""]


def _compile(repo: Path, file: Path, timeout: int) -> tuple[subprocess.CompletedProcess, float]:
    start = time.time()
    process = subprocess.run(
        # The matrix intentionally contains many failing declarations.  The CLI
        # override prevents Lean's default diagnostic cap from turning later
        # failures into unobserved labels.
        ["lake", "env", "lean", "-D", "maxErrors=100000",
         str(file.relative_to(repo))],
        cwd=repo, capture_output=True, text=True, timeout=timeout)
    return process, time.time() - start


def _capture_source(cases: list[dict]) -> str:
    lines = _header()
    for case in cases:
        for variant, term in (("A", case["term_a"]), ("B", case["term_b"])):
            tag = f"{case['case_id']}_{variant}"
            lines.extend([
                f"example {case['binders']} : {case['goal']} := by",
                f"  show {term}",
                "  run_tac (show Lean.Elab.Tactic.TacticM Unit from do",
                "    let g ← Lean.Elab.Tactic.getMainGoal",
                "    g.withContext do",
                "      let d ← g.getDecl",
                "      let fmt ← Lean.Meta.ppGoal g",
                "      let e ← Lean.Meta.mkForallFVars d.lctx.getFVars d.type",
                "      let e ← Lean.instantiateMVars e",
                "      let r ← Lean.Meta.abstractMVars e",
                f"      Lean.logInfo s!\"CASEV:{tag}\\nPPV:{{fmt}}\\nPPEND\\n"
                f"DIGV:{{r.expr.hash}}\\nCASEEND\")",
                "  all_goals sorry", "",
            ])
    return "\n".join(lines)


def _parse_captures(output: str) -> dict[str, dict]:
    pattern = re.compile(
        r"CASEV:([^\n]+)\nPPV:(.*?)\nPPEND\nDIGV:(\d+)\nCASEEND", re.S)
    return {tag.strip(): {"pp_hash": _h(pp.strip()), "digest": digest}
            for tag, pp, digest in pattern.findall(output)}


def _all_actions(cases: list[dict]) -> list[str]:
    witness_bank = list(dict.fromkeys(case["witness_action"] for case in cases))
    return list(dict.fromkeys(["rfl", "simp", "norm_num", *witness_bank]))


def _matrix_source(cases: list[dict], actions: list[str]) -> tuple[str, list[dict]]:
    lines = _header()
    spans = []
    for case in cases:
        for variant, term in (("A", case["term_a"]), ("B", case["term_b"])):
            for action_index, action in enumerate(actions):
                start = len(lines) + 1
                lines.extend([
                    f"-- MATRIX {case['case_id']} {variant} {action_index}",
                    f"example {case['binders']} : {case['goal']} := by",
                    f"  show {term}",
                    *(f"  {command}" for command in action.splitlines()), "",
                ])
                spans.append({"case_id": case["case_id"], "variant": variant,
                              "action": action, "start": start, "end": len(lines)})
    return "\n".join(lines), spans


def _parse_matrix(output: str, filename: str, spans: list[dict]) -> dict[tuple, bool]:
    error_lines = [int(value) for value in re.findall(
        rf"{re.escape(filename)}:(\d+):\d+: error", output)]
    matrix = {}
    for span in spans:
        failed = any(span["start"] <= line <= span["end"] for line in error_lines)
        matrix[(span["case_id"], span["variant"], span["action"])] = not failed
    return matrix


def _generator_actions(case: dict, name: str,
                       bank_by_wrapper: dict[str, list[str]]) -> list[str]:
    context_bank = bank_by_wrapper[case["wrapper"]]
    if name == "targeted":
        values = [case["witness_action"], "rfl"]
    elif name == "context_bank":
        values = [*context_bank, "rfl"]
    elif name == "generic_first":
        values = ["rfl", "simp", "norm_num", *context_bank]
    else:
        raise ValueError(name)
    return list(dict.fromkeys(values))


def _interval(rows: list[dict], cluster_key: str, iterations: int = 10000) -> list[float]:
    """Cluster bootstrap interval for the conditional loss probability."""
    rows = [row for row in rows if row["exact_loss_denominator"]]
    clusters: dict[object, list[dict]] = collections.defaultdict(list)
    for row in rows:
        clusters[row[cluster_key]].append(row)
    names = sorted(clusters)
    rng = random.Random(20260804 + (0 if cluster_key == "case_id" else 1))
    values = []
    for _ in range(iterations):
        sampled = [rng.choice(names) for _ in names]
        numerator = denominator = 0.0
        for name in sampled:
            for row in clusters[name]:
                numerator += row["exact_loss_numerator"]
                denominator += row["exact_loss_denominator"]
        values.append(numerator / denominator if denominator else 0.0)
    values.sort()
    return [values[int(.025 * iterations)], values[int(.975 * iterations)]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="../persist/mathlib4_current")
    parser.add_argument("--out", default="runs/alias_stress")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--trials-per-pair", type=int, default=200)
    parser.add_argument("--seed", type=int, default=731991)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    generated = repo / "test_pairs" / "alias_stress"
    generated.mkdir(parents=True, exist_ok=True)
    cases = _cases()
    lean_version = subprocess.run(
        ["lake", "env", "lean", "--version"], cwd=repo,
        capture_output=True, text=True, check=True).stdout.strip()
    mathlib_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo,
        capture_output=True, text=True, check=True).stdout.strip()
    _log(f"generated {len(cases)} candidate pairs")

    capture_file = generated / "Capture.lean"
    capture_file.write_text(_capture_source(cases), encoding="utf-8")
    capture_process, capture_seconds = _compile(repo, capture_file, args.timeout)
    capture_output = capture_process.stdout + capture_process.stderr
    captures = _parse_captures(capture_output)
    _log(f"capture compile {capture_seconds:.1f}s, parsed {len(captures)}/{2*len(cases)} states")

    actions = _all_actions(cases)
    matrix_file = generated / "ActionMatrix.lean"
    matrix_source, spans = _matrix_source(cases, actions)
    matrix_file.write_text(matrix_source, encoding="utf-8")
    matrix_process, matrix_seconds = _compile(repo, matrix_file, args.timeout)
    matrix_output = matrix_process.stdout + matrix_process.stderr
    matrix = _parse_matrix(matrix_output, matrix_file.name, spans)
    _log(f"action matrix {matrix_seconds:.1f}s, {len(spans)} state-action labels")

    for case in cases:
        cap_a = captures.get(f"{case['case_id']}_A")
        cap_b = captures.get(f"{case['case_id']}_B")
        case["pp_identical"] = bool(cap_a and cap_b and cap_a["pp_hash"] == cap_b["pp_hash"])
        case["digest_distinct"] = bool(
            cap_a and cap_b and cap_a["digest"] != cap_b["digest"])
        case["witness_closes_a"] = matrix.get(
            (case["case_id"], "A", case["witness_action"]), False)
        case["witness_closes_b"] = matrix.get(
            (case["case_id"], "B", case["witness_action"]), False)
        case["eligible"] = bool(case["pp_identical"] and case["digest_distinct"]
                                and case["witness_closes_a"]
                                and not case["witness_closes_b"])
    eligible = [case for case in cases if case["eligible"]]
    _log(f"eligible {len(eligible)}/{len(cases)} pairs")

    bank_by_wrapper = {
        wrapper: list(dict.fromkeys(
            case["witness_action"] for case in cases if case["wrapper"] == wrapper))
        for wrapper, _ in WRAPPERS
    }
    rng = random.Random(args.seed)
    randomized_pair_counts = []
    summary = []
    for generator in GENERATOR_NAMES:
        for budget in BUDGETS:
            pair_rows = []
            empirical_numerator = empirical_denominator = 0
            for case in eligible:
                proposed = _generator_actions(case, generator, bank_by_wrapper)[:budget]
                proves = {
                    variant: any(matrix.get((case["case_id"], variant, action), False)
                                 for action in proposed)
                    for variant in ("A", "B")
                }
                refined = proves["A"] or proves["B"]
                exact_losses = sum(int(refined and not proves[first]) for first in ("A", "B"))
                pair_row = {
                    "case_id": case["case_id"], "base": case["base"],
                    "wrapper": case["wrapper"], "generator": generator,
                    "budget": budget, "proposed": proposed, "proves": proves,
                    "refined_proves": refined,
                    "exact_loss_numerator": exact_losses,
                    "exact_loss_denominator": 2 if refined else 0,
                }
                pair_rows.append(pair_row)
                arrival_counts = {"A": 0, "B": 0}
                loss_count = 0
                for trial in range(args.trials_per_pair):
                    first = "A" if rng.random() < .5 else "B"
                    arrival_counts[first] += 1
                    rendering = proves[first]
                    loss = bool(refined and not rendering)
                    loss_count += int(loss)
                    empirical_numerator += int(loss)
                    empirical_denominator += int(refined)
                randomized_pair_counts.append({
                    "case_id": case["case_id"], "base": case["base"],
                    "generator": generator, "budget": budget,
                    "arrival_counts": arrival_counts, "losses": loss_count,
                    "refined_proves": refined,
                })
            exact_num = sum(row["exact_loss_numerator"] for row in pair_rows)
            exact_den = sum(row["exact_loss_denominator"] for row in pair_rows)
            summary.append({
                "generator": generator, "budget": budget,
                "eligible_pairs": len(eligible),
                "refined_proof_pairs": sum(row["refined_proves"] for row in pair_rows),
                "uniform_order_loss_probability": exact_num / exact_den if exact_den else None,
                "pair_cluster_bootstrap_95": _interval(pair_rows, "case_id") if exact_den else None,
                "base_cluster_bootstrap_95": _interval(pair_rows, "base") if exact_den else None,
                "randomized_trials": empirical_denominator,
                "randomized_loss_probability": (
                    empirical_numerator / empirical_denominator if empirical_denominator else None),
                "refined_identity_losses": 0,
                "pair_results": pair_rows,
            })
            _log(f"{generator:13} budget={budget:2d} refined="
                 f"{summary[-1]['refined_proof_pairs']:2d}/{len(eligible)} "
                 f"loss={summary[-1]['uniform_order_loss_probability']}")

    result = {
        "protocol": {
            "lean_repo": repo.name, "lean_version": lean_version,
            "mathlib_commit": mathlib_commit,
            "cases": "8 base propositions x 12 proposition contexts",
            "admission": "byte-identical rendering, distinct digest, member-specific proof",
            "arrival_order": "uniform random; both orders also enumerated exactly",
            "rendering_identity": "retain first arrival only",
            "refined_identity": "retain both rendering-plus-digest members",
            "trials_per_pair": args.trials_per_pair, "seed": args.seed,
            "budgets": list(BUDGETS), "generators": list(GENERATOR_NAMES),
            "uncertainty": "pair and base-template cluster bootstrap, 10000 resamples",
        },
        "compilation": {
            "capture_seconds": capture_seconds, "capture_returncode": capture_process.returncode,
            "matrix_seconds": matrix_seconds, "matrix_returncode": matrix_process.returncode,
            "state_action_labels": len(spans),
            "failed_state_action_labels": sum(not outcome for outcome in matrix.values()),
            "reported_action_errors": len(re.findall(
                rf"{re.escape(matrix_file.name)}:(\d+):\d+: error", matrix_output)),
            "capture_source_sha256": hashlib.sha256(
                capture_file.read_bytes()).hexdigest(),
            "matrix_source_sha256": hashlib.sha256(
                matrix_file.read_bytes()).hexdigest(),
            "actions": actions,
        },
        "candidate_pairs": len(cases), "eligible_pairs": len(eligible),
        "eligible_base_templates": len({case["base"] for case in eligible}),
        "eligible_proposition_contexts": len({case["wrapper"] for case in eligible}),
        "cases": cases, "summary": summary,
        "randomized_pair_counts": randomized_pair_counts,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    _log(f"wrote {out / 'results.json'}")
    return 0 if len(eligible) >= 50 else 2


if __name__ == "__main__":
    raise SystemExit(main())
