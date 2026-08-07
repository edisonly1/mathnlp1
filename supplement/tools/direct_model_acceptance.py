"""Frozen-prover acceptance test on confirmed natural observation aliases.

For each confirmed class-action contradiction, the corpus provides one hidden member on which
the tactic is accepted and one byte-identically rendered member on which it is rejected.  This
tool asks BFS-Prover-V2-7B to score that same tactic under two input channels:

* the ordinary proof-state rendering;
* the rendering followed by the member-local structural sketch.

The checkpoint, prompt template, and scoring rule are fixed.  No classifier, reranker, fold,
or calibration parameter is fitted to the alias corpus.  Paired ranking accuracy measures how
often the accepting member assigns the tactic a higher conditional log probability.  The
ordinary arm is tied by construction and supplies an exact 50 percent information control.

Results are reported separately for each hidden-delta mechanism and macro-averaged across
mechanisms.  The theorem-cluster bootstrap accounts for multiple actions and classes drawn
from one theorem.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.representation_eval import RUNS, _examples


MODEL_ID = "ByteDance-Seed/BFS-Prover-V2-7B"
MODEL_REVISION = "533d59fc6b5e04ae6e2c25f1d88cf921061169d0"
PROMPT_TEMPLATE = "{state}\n\n/- structural sketch: elaborated constants with counts\n{sketch}\n-/"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _value(margin: float, tolerance: float = 1e-9) -> float:
    if margin > tolerance:
        return 1.0
    if margin < -tolerance:
        return 0.0
    return 0.5


def _cluster_interval(values: list[float], groups: list[str],
                      iterations: int, seed: int) -> list[float]:
    import numpy as np

    if not values:
        return [float("nan"), float("nan")]
    names = sorted(set(groups))
    indices = {name: np.asarray([i for i, group in enumerate(groups) if group == name])
               for name in names}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sample = rng.choice(names, size=len(names), replace=True)
        chosen = np.concatenate([indices[name] for name in sample])
        means[iteration] = array[chosen].mean()
    return [float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975))]


def _paired_summary(rows: list[dict], score_key: str,
                    iterations: int, seed: int) -> dict:
    values = []
    margins = []
    theorem_groups = []
    by_mechanism: dict[str, list[float]] = collections.defaultdict(list)
    predictions = []
    for pair_id, pair_rows in sorted(rows_by(rows, "pair_id").items()):
        negative = next(row for row in pair_rows if row["label"] == 0)
        positive = next(row for row in pair_rows if row["label"] == 1)
        margin = positive[score_key] - negative[score_key]
        value = _value(margin)
        values.append(value)
        margins.append(margin)
        theorem_groups.append(positive["theorem"])
        by_mechanism[positive["mechanism"]].append(value)
        predictions.append({
            "pair_id": pair_id,
            "theorem": positive["theorem"],
            "mechanism": positive["mechanism"],
            "action": positive["action"],
            "margin": margin,
            "correct": value,
        })
    named = [name for name in by_mechanism if name != "residual"]
    mechanism_scores = {
        name: {"pairs": len(mechanism_values),
               "paired_accuracy": sum(mechanism_values) / len(mechanism_values)}
        for name, mechanism_values in sorted(by_mechanism.items())
    }
    return {
        "paired_ranking_accuracy": sum(values) / len(values),
        "theorem_cluster_bootstrap_95": _cluster_interval(
            values, theorem_groups, iterations, seed),
        "mean_pair_margin": sum(margins) / len(margins),
        "mechanism_macro_paired_accuracy": (
            sum(mechanism_scores[name]["paired_accuracy"] for name in named) / len(named)
            if named else float("nan")),
        "all_group_macro_paired_accuracy": (
            sum(value["paired_accuracy"] for value in mechanism_scores.values()) /
            len(mechanism_scores)),
        "by_mechanism": mechanism_scores,
        "pair_predictions": predictions,
    }


def rows_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return grouped


def _score_unique(model, items: list[tuple[str, str]], batch_size: int) -> list[dict]:
    """Score repeated state-action inputs once and restore corpus order."""
    unique = list(dict.fromkeys(items))
    scores = model.score_actions(unique, batch_size)
    lookup = dict(zip(unique, scores))
    return [lookup[item] for item in items]


def _length_control(rows: list[dict], iterations: int) -> dict:
    grouped = rows_by(rows, "pair_id")
    wanted = {
        pair_id for pair_id, pair_rows in grouped.items()
        if len({row["structural_prompt_tokens"] for row in pair_rows}) == 1
    }
    equal_rows = [row for row in rows if row["pair_id"] in wanted]
    return {
        "criterion": "equal structural prompt-token count within the pair",
        "pairs": len(wanted),
        "structural": _paired_summary(
            equal_rows, "structural_mean_logprob", iterations, 20260807),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="runs/representation")
    parser.add_argument("--runs", default=",".join(RUNS))
    parser.add_argument("--out", default="runs/direct_model_acceptance/results.json")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--reuse-results",
                        help="reuse unchanged per-example scores from an earlier result")
    parser.add_argument("--limit", type=int, default=0,
                        help="optional number of pairs for a smoke test")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()

    from audit.bfs_runner import BFSProverGenerator

    run_paths = tuple(value.strip() for value in args.runs.split(",") if value.strip())
    examples, pairs = _examples(Path(args.data), run_paths)
    if args.limit:
        wanted = {pair["pair_id"] for pair in pairs[:args.limit]}
        examples = [example for example in examples if example["pair_id"] in wanted]
        pairs = [pair for pair in pairs if pair["pair_id"] in wanted]
    if not examples:
        raise SystemExit("no confirmed class-action pairs")

    reused: dict[tuple[str, int], dict] = {}
    if args.reuse_results:
        previous = json.load(open(args.reuse_results, encoding="utf-8"))
        for row in previous.get("rows", []):
            reused[(row["pair_id"], row["label"])] = row
    new_examples = []
    rows = []
    for example in examples:
        old = reused.get((example["pair_id"], example["label"]))
        if (old is not None and old.get("action") == example["action"]
                and old.get("default_pp_hash") == _hash(example["default_pp"])
                and old.get("sketch_hash") == _hash(example["shallow_sketch"])):
            rows.append(old)
        else:
            new_examples.append(example)

    load_source = args.model_path or args.model_id
    model = BFSProverGenerator(
        load_source, device=args.device,
        revision=None if args.model_path else args.revision,
        local_files_only=not args.allow_network)
    ordinary_items = [(example["default_pp"], example["action"])
                      for example in new_examples]
    structural_items = [(
        PROMPT_TEMPLATE.format(state=example["default_pp"],
                               sketch=example["shallow_sketch"]),
        example["action"],
    ) for example in new_examples]
    print(f"[direct-model] reused {len(rows)} examples; "
          f"scoring {len(new_examples)} ordinary examples", file=sys.stderr)
    ordinary_scores = _score_unique(model, ordinary_items, args.batch_size)
    print(f"[direct-model] scoring {len(new_examples)} structural examples", file=sys.stderr)
    structural_scores = _score_unique(model, structural_items, args.batch_size)

    for example, ordinary, structural in zip(
            new_examples, ordinary_scores, structural_scores):
        rows.append({
            "pair_id": example["pair_id"], "theorem": example["theorem"],
            "pp_hash": example["pp_hash"], "mechanism": example["mechanism"],
            "action": example["action"], "label": example["label"],
            "ordinary_mean_logprob": ordinary["mean_logprob"],
            "structural_mean_logprob": structural["mean_logprob"],
            "ordinary_sum_logprob": ordinary["sum_logprob"],
            "structural_sum_logprob": structural["sum_logprob"],
            "action_tokens": structural["action_tokens"],
            "ordinary_prompt_tokens": ordinary["prompt_tokens"],
            "structural_prompt_tokens": structural["prompt_tokens"],
            "default_pp_hash": _hash(example["default_pp"]),
            "sketch_hash": _hash(example["shallow_sketch"]),
        })

    ordinary = _paired_summary(
        rows, "ordinary_mean_logprob", args.bootstrap, 20260805)
    structural = _paired_summary(
        rows, "structural_mean_logprob", args.bootstrap, 20260806)
    ordinary_ties = sum(row["correct"] == 0.5
                        for row in ordinary["pair_predictions"])
    if ordinary_ties != len(pairs):
        raise RuntimeError(
            f"ordinary rendering should tie all pairs, observed {ordinary_ties}/{len(pairs)}")
    prompt_increments = [row["structural_prompt_tokens"] - row["ordinary_prompt_tokens"]
                         for row in rows]
    result = {
        "protocol": {
            "model_id": args.model_id, "revision": args.revision,
            "device": model.device, "dtype": "float16",
            "torch_version": model.torch.__version__,
            "checkpoint_frozen": True,
            "experiment_side_fitting": "none",
            "unit": "one accepted and one rejected member per confirmed class-action pair",
            "score": "mean conditional tactic-token log probability",
            "primary_metric": "paired ranking accuracy, ties score 0.5",
            "mechanism_evaluation": (
                "prompt and scoring rule fixed globally; no corpus labels enter training; "
                "report every hidden-delta group and their unweighted macro average"),
            "uncertainty": "theorem-cluster percentile bootstrap",
            "bootstrap_iterations": args.bootstrap,
            "prompt_template_sha256": _hash(PROMPT_TEMPLATE),
        },
        "corpus": {
            "classes": len({(row["theorem"], row["pp_hash"]) for row in rows}),
            "theorems": len({row["theorem"] for row in rows}),
            "contradictory_pairs": len(pairs), "examples": len(rows),
            "mechanisms": dict(collections.Counter(pair["mechanism"] for pair in pairs)),
        },
        "prompt_tokens": {
            "mean_increment": sum(prompt_increments) / len(prompt_increments),
            "minimum_increment": min(prompt_increments),
            "maximum_increment": max(prompt_increments),
        },
        "ordinary": ordinary,
        "structural": structural,
        "equal_prompt_length_control": _length_control(rows, args.bootstrap),
        "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "pairs": len(pairs),
        "ordinary": ordinary["paired_ranking_accuracy"],
        "structural": structural["paired_ranking_accuracy"],
        "mechanism_macro": structural["mechanism_macro_paired_accuracy"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
