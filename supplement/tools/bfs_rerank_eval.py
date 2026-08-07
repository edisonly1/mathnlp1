"""End-to-end structural reranking of BFS-Prover-V2-7B candidates on aliases.

The expensive 7B representative-sensitivity run already generated one shared top-k list from
each class's byte-identical default rendering.  This tool reuses those *recorded generator
outputs* but freshly executes every candidate on every hidden member with repeat controls.
It then evaluates whether the compact member-local sketch improves tactic selection.

Two stages are deliberately separate:

``collect``
    Restore every alias member, verify its default rendering against the representation
    extraction, and execute every recorded BFS-Prover-V2 candidate three times.  This stage
    needs LeanDojo but not the 7B checkpoint.

``evaluate``
    Fit text-only and text-plus-sketch acceptance rerankers with folds grouped by theorem.
    Candidate rank is supplied to both models, so the comparison asks what the hidden member
    structure adds beyond the original generator order.  Every reported selection metric is
    out of fold.  Family holdouts are included as a generalization stress test.

This is a candidate-selection experiment on the alias-enriched natural corpus.  It does not
claim an aggregate theorem-proving gain or rerank every node in a full search.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.representation_eval import (
    RUNS,
    _examples as contradiction_examples,
    _mechanism_for_row,
    _mechanism_map,
    _fit_logistic,
    _folds,
    _hash_index,
    _tokens,
    _vector as contradiction_vector,
)


def _log(message: str) -> None:
    print(f"[bfs-rerank] {message}", file=sys.stderr, flush=True)


def _load_jsonl(pattern: str) -> list[dict]:
    return [json.loads(line) for filename in sorted(glob.glob(pattern))
            for line in open(filename, encoding="utf-8") if line.strip()]


def _representation_rows(data_dir: Path) -> dict[tuple[str, str], dict]:
    rows = _load_jsonl(str(data_dir / "members.shard*.jsonl"))
    return {(row["theorem"], row["pp_hash"]): row for row in rows
            if not row.get("error") and row.get("renderings_identical")}


def _candidate_rows(pattern: str) -> dict[tuple[str, str], dict]:
    rows = _load_jsonl(pattern)
    out = {}
    for row in rows:
        searches = row.get("searches") or {}
        lists = [tuple(search.get("first_candidates") or ()) for search in searches.values()]
        if not lists or not lists[0] or len(set(lists)) != 1:
            continue
        out[(row["theorem"], row["pp_hash"])] = {
            "candidates": list(lists[0]),
            "model_id": "ByteDance-Seed/BFS-Prover-V2-7B",
            "source_policy": row.get("policy"),
            "source_top_k": row.get("top_k"),
            "verified_identical_across_members": bool(row.get("first_candidates_identical")),
        }
    return out


def _outcome(result, lean_dojo_module) -> str:
    if isinstance(result, lean_dojo_module.ProofFinished):
        return "COMPLETE"
    if isinstance(result, lean_dojo_module.TacticState):
        return "STATE"
    return "FAIL"


def collect(args: argparse.Namespace) -> int:
    from lean_dojo import Dojo, Theorem
    import lean_dojo as ldj
    from audit.replay import deadline, kill_orphan_lean, lean_git_repo

    run_paths = tuple(value.strip() for value in args.runs.split(",") if value.strip())
    candidates = _candidate_rows(args.candidates)
    representations = _representation_rows(Path(args.representations))
    mechanisms = _mechanism_map(run_paths)

    targets = {}
    for run_s in run_paths:
        run = Path(run_s)
        stage_rows = {}
        for filename in glob.glob(str(run / "stageb.shard*.jsonl")):
            for line in open(filename, encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    stage_rows[(row["theorem"], row["pp_hash"])] = row
        for conf in json.load(open(run / "confirmation.json", encoding="utf-8")):
            if conf.get("verdict") != "CONFIRMED":
                continue
            key = (conf["theorem"], conf["pp_hash"])
            if key in candidates and key in representations and key in stage_rows:
                targets.setdefault(key, stage_rows[key])

    ordered = [item for item in targets.items()]
    mine = [item for index, item in enumerate(ordered)
            if index % args.n_shards == args.shard]
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    outfile = output / f"executions.shard{args.shard}.jsonl"
    done = set()
    if outfile.exists():
        for row in _load_jsonl(str(outfile)):
            done.add((row["theorem"], row["pp_hash"]))
    mine = [(key, row) for key, row in mine if key not in done]
    _log(f"shard {args.shard}: {len(mine)} classes to execute of {len(targets)}")

    wanted = {(row["file"], row["theorem"], row["tactic_index"])
              for _, row in mine}
    roots = {}
    with open(args.states, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            state = json.loads(line)
            prov = state["provenance"]
            key = (prov["file_path"], prov["theorem_full_name"], prov["tactic_index"])
            if key in wanted:
                roots[key] = state
    _log(f"indexed {len(roots)} of {len(wanted)} roots")
    if not roots:
        return 1
    any_prov = next(iter(roots.values()))["provenance"]
    repo = lean_git_repo(ldj, any_prov["repo_url"], any_prov["repo_commit"])

    with open(outfile, "a", encoding="utf-8") as sink:
        for index, (key, row) in enumerate(mine, 1):
            root = roots.get((row["file"], row["theorem"], row["tactic_index"]))
            rep = representations[key]
            cand = candidates[key]
            record = {
                "theorem": row["theorem"], "file": row["file"],
                "pp_hash": row["pp_hash"], "tactic_index": row["tactic_index"],
                "family": _mechanism_for_row(mechanisms, rep),
                "model_id": cand["model_id"], "candidates": cand["candidates"],
                "candidate_list_shared": cand["verified_identical_across_members"],
                "repeats": args.repeats, "members": [],
            }
            ctx = None
            try:
                if root is None:
                    raise RuntimeError("root not indexed")
                prov = root["provenance"]
                theorem = Theorem(repo, prov["file_path"], prov["theorem_full_name"])
                ctx = Dojo(theorem, timeout=args.dojo_timeout)
                dojo, state0 = ctx.__enter__()
                for tactic in root["proof_prefix"]:
                    with deadline(args.tactic_timeout, tactic[:30]):
                        state0 = dojo.run_tac(state0, tactic)

                rep_members = {member["history_label"]: member for member in rep["members"]}
                for history in row["histories"]:
                    label = " ; ".join(history)
                    member_rep = rep_members.get(label)
                    if member_rep is None:
                        continue
                    state = state0
                    for tactic in history:
                        with deadline(args.tactic_timeout, tactic[:30]):
                            state = dojo.run_tac(state, tactic)
                        if not isinstance(state, ldj.TacticState):
                            break
                    if not isinstance(state, ldj.TacticState):
                        continue
                    if state.pp != member_rep["default_pp"]:
                        raise RuntimeError(f"default rendering drift for {label}")
                    member = {
                        "history": history, "history_label": label,
                        "default_pp": state.pp,
                        "shallow_sketch": member_rep["shallow_sketch"],
                        "proof_sketch": member_rep["proof_sketch"],
                        "outcomes": {},
                    }
                    for tactic in cand["candidates"]:
                        observed = []
                        for _ in range(args.repeats):
                            with deadline(args.candidate_timeout, tactic[:30]):
                                observed.append(_outcome(dojo.run_tac(state, tactic), ldj))
                        member["outcomes"][tactic] = observed
                    record["members"].append(member)
                record["renderings_identical"] = (
                    len(record["members"]) >= 2
                    and len({member["default_pp"] for member in record["members"]}) == 1
                )
                record["complete"] = len(record["members"]) == len(row["histories"])
            except BaseException as exc:
                record["error"] = f"{type(exc).__name__}: {str(exc)[:180]}"
            finally:
                if ctx is not None:
                    try:
                        ctx.__exit__(None, None, None)
                    except Exception:
                        pass
                try:
                    kill_orphan_lean()
                except Exception:
                    pass
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            _log(f"[{index}/{len(mine)}] {row['theorem'][:46]:46} "
                 f"members={len(record['members'])} error={record.get('error')}")
    return 0


def stable_label(outcomes: list[str]) -> int | None:
    """Map repeat outcomes to compiler acceptance, excluding self-inconsistency."""
    labels = {0 if value == "FAIL" else 1 for value in outcomes}
    return next(iter(labels)) if len(labels) == 1 else None


def _examples(data_dir: Path) -> tuple[list[dict], list[dict]]:
    rows = _load_jsonl(str(data_dir / "executions.shard*.jsonl"))
    examples = []
    members = []
    for row in rows:
        if row.get("error") or not row.get("complete") or not row.get("renderings_identical"):
            continue
        candidates = row["candidates"]
        for member in row["members"]:
            member_id = f"{row['theorem']}|{row['pp_hash']}|{member['history_label']}"
            indices = []
            for raw_index, tactic in enumerate(candidates):
                label = stable_label(member["outcomes"].get(tactic, []))
                if label is None:
                    continue
                indices.append(len(examples))
                examples.append({
                    "member_id": member_id, "theorem": row["theorem"],
                    "pp_hash": row["pp_hash"], "family": row.get("family", "residual"),
                    "history_label": member["history_label"], "action": tactic,
                    "raw_rank": raw_index + 1, "label": label,
                    "default_pp": member["default_pp"],
                    "shallow_sketch": member["shallow_sketch"],
                    "proof_sketch": member["proof_sketch"],
                })
            if indices:
                members.append({"member_id": member_id, "theorem": row["theorem"],
                                "pp_hash": row["pp_hash"],
                                "family": row.get("family", "residual"),
                                "indices": indices})
    return examples, members


def _vector(example: dict, representation: str, dimension: int):
    import numpy as np

    vector = np.zeros(dimension, dtype=np.float64)
    action = _tokens(example["action"])
    visible = _tokens(example["default_pp"])
    rank = int(example["raw_rank"])
    vector[_hash_index("bias", dimension)] += 1.0
    vector[_hash_index(f"rank:{rank}", dimension)] += 1.0
    vector[_hash_index("reciprocal-rank", dimension)] += 1.0 / rank
    for token, count in action.items():
        value = math.log1p(count)
        vector[_hash_index("a:" + token, dimension)] += value
        vector[_hash_index(f"ar:{token}:{min(rank, 4)}", dimension)] += value
    for token, count in visible.items():
        value = math.log1p(count)
        vector[_hash_index("v:" + token, dimension)] += value
        for act in action:
            vector[_hash_index(f"av:{act}:{token}", dimension)] += value
    if representation != "default":
        sketch = example["shallow_sketch" if representation == "shallow"
                         else "proof_sketch"]
        for item in sketch.split():
            name, _, raw_count = item.rpartition("=")
            try:
                value = math.log1p(int(raw_count))
            except ValueError:
                name, value = item, 1.0
            name = name.lower()
            vector[_hash_index("s:" + name, dimension)] += value
            for act in action:
                vector[_hash_index(f"as:{act}:{name}", dimension)] += value
    return vector


def _first_accepted(indices: list[int], examples: list[dict], order: list[int]) -> int | None:
    labels = {index: examples[index]["label"] for index in indices}
    for position, index in enumerate(order, 1):
        if labels[index]:
            return position
    return None


def ranking_metrics(examples: list[dict], members: list[dict], scores=None) -> tuple[dict, list[dict]]:
    rows = []
    for member in members:
        indices = member["indices"]
        raw = sorted(indices, key=lambda index: examples[index]["raw_rank"])
        if scores is None:
            order = raw
        else:
            order = sorted(indices, key=lambda index: (-float(scores[index]),
                                                       examples[index]["raw_rank"]))
        first = _first_accepted(indices, examples, order)
        raw_first = _first_accepted(indices, examples, raw)
        rows.append({
            **{key: member[key] for key in ("member_id", "theorem", "pp_hash", "family")},
            "covered": first is not None,
            "first_accepted_rank": first,
            "raw_first_accepted_rank": raw_first,
            "mrr": 0.0 if first is None else 1.0 / first,
            "raw_mrr": 0.0 if raw_first is None else 1.0 / raw_first,
            "top1": bool(first == 1), "top3": bool(first is not None and first <= 3),
            "top8": bool(first is not None and first <= 8),
            "order": [examples[index]["action"] for index in order],
        })
    covered = [row for row in rows if row["covered"]]
    metrics = {
        "members": len(rows), "covered_members": len(covered),
        "top1_acceptance": sum(row["top1"] for row in rows) / len(rows),
        "top3_acceptance": sum(row["top3"] for row in rows) / len(rows),
        "top8_acceptance": sum(row["top8"] for row in rows) / len(rows),
        "mrr": sum(row["mrr"] for row in rows) / len(rows),
        "mean_first_accepted_rank_covered": (
            sum(row["first_accepted_rank"] for row in covered) / len(covered)
            if covered else None
        ),
    }
    return metrics, rows


def candidate_oracles(examples: list[dict], members: list[dict]) -> dict:
    """Compare hidden-member and best-shared-action ceilings within the fixed top-k.

    If the two ceilings coincide, this candidate population offers no top-1 selection benefit
    to *any* member-aware representation, even though a learned reranker may still improve on
    the generator's raw ordering by learning ordinary action priors.
    """
    classes: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for member in members:
        classes[(member["theorem"], member["pp_hash"])].append(member)
    per_member = 0
    best_shared = 0
    acceptance_ambiguous_classes = 0
    acceptance_ambiguous_actions = 0
    fully_covered_with_common = 0
    for class_members in classes.values():
        accepted_sets = []
        action_order = []
        for member in class_members:
            accepted = {examples[index]["action"] for index in member["indices"]
                        if examples[index]["label"] == 1}
            accepted_sets.append(accepted)
            per_member += bool(accepted)
            if not action_order:
                action_order = [examples[index]["action"] for index in member["indices"]]
        divergent = [action for action in action_order
                     if len({action in accepted for accepted in accepted_sets}) > 1]
        acceptance_ambiguous_actions += len(divergent)
        acceptance_ambiguous_classes += bool(divergent)
        if accepted_sets and all(accepted_sets) and set.intersection(*accepted_sets):
            fully_covered_with_common += 1
        best_shared += max((sum(action in accepted for accepted in accepted_sets)
                            for action in action_order), default=0)
    total = len(members)
    return {
        "member_oracle_top1": per_member / total,
        "best_shared_action_oracle_top1": best_shared / total,
        "member_specific_oracle_headroom": (per_member - best_shared) / total,
        "member_specific_oracle_extra_successes": per_member - best_shared,
        "acceptance_ambiguous_classes": acceptance_ambiguous_classes,
        "acceptance_ambiguous_class_actions": acceptance_ambiguous_actions,
        "fully_covered_classes_with_common_action": fully_covered_with_common,
        "classes": len(classes),
    }


def _bootstrap_delta(rows_a: list[dict], rows_b: list[dict], key: str,
                     iterations: int = 10000) -> list[float]:
    import numpy as np

    by_id_b = {row["member_id"]: row for row in rows_b}
    paired = [(row["theorem"], float(row[key]) - float(by_id_b[row["member_id"]][key]))
              for row in rows_a if row["member_id"] in by_id_b]
    groups = sorted({theorem for theorem, _ in paired})
    values = {group: [delta for theorem, delta in paired if theorem == group]
              for group in groups}
    rng = np.random.default_rng(20260803)
    draws = []
    for _ in range(iterations):
        chosen = rng.choice(groups, size=len(groups), replace=True)
        sample = [value for group in chosen for value in values[group]]
        draws.append(float(np.mean(sample)))
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _divergent_pair_accuracy(examples: list[dict], scores, allowed: set[int] | None = None) -> dict:
    grouped: dict[tuple[str, str, str], list[int]] = collections.defaultdict(list)
    for index, example in enumerate(examples):
        if allowed is None or index in allowed:
            grouped[(example["theorem"], example["pp_hash"], example["action"])].append(index)
    values = []
    for indices in grouped.values():
        positives = [index for index in indices if examples[index]["label"] == 1]
        negatives = [index for index in indices if examples[index]["label"] == 0]
        if not positives or not negatives:
            continue
        margin = float(scores[positives[0]] - scores[negatives[0]])
        values.append(1.0 if margin > 1e-12 else 0.0 if margin < -1e-12 else 0.5)
    return {"pairs": len(values),
            "accepted_member_ranked_higher": sum(values) / len(values) if values else None}


def _fit_scores(examples: list[dict], train_mask, test_mask, representation: str,
                args: argparse.Namespace, matrix=None):
    import numpy as np

    x = matrix if matrix is not None else np.stack(
        [_vector(example, representation, args.dim) for example in examples])
    y = np.asarray([example["label"] for example in examples], dtype=np.float64)
    weights = _fit_logistic(x[train_mask], y[train_mask], args.l2, args.steps, args.lr)
    scores = np.full(len(examples), np.nan, dtype=np.float64)
    scores[test_mask] = 1.0 / (1.0 + np.exp(-np.clip(x[test_mask] @ weights, -30.0, 30.0)))
    return scores, x


def evaluate(args: argparse.Namespace) -> int:
    import numpy as np

    examples, members = _examples(Path(args.data))
    if not examples or not members:
        raise SystemExit("no complete candidate executions found")
    pseudo_pairs = [{"theorem": member["theorem"]} for member in members]
    folds = _folds(pseudo_pairs, args.folds)
    raw_metrics, raw_rows = ranking_metrics(examples, members)
    results = {
        "protocol": {
            "generator": "ByteDance-Seed/BFS-Prover-V2-7B",
            "candidate_source": "recorded shared top-8 lists from default rendering",
            "labels": "fresh three-repeat Lean executions on every hidden member",
            "split": f"{args.folds}-fold grouped by theorem",
            "classifier": "hashed linear logistic reranker with original-rank features",
            "dimension": args.dim, "l2": args.l2, "steps": args.steps,
            "seed": 20260803,
        },
        "corpus": {
            "classes": len({(member["theorem"], member["pp_hash"]) for member in members}),
            "theorems": len({member["theorem"] for member in members}),
            "members": len(members), "candidate_executions": len(examples),
            "compiler_calls": len(examples) * 3,
            "accepted": sum(example["label"] for example in examples),
            "families": dict(collections.Counter(member["family"] for member in members)),
        },
        "raw_generator_order": raw_metrics,
        "fixed_candidate_oracles": candidate_oracles(examples, members),
        "representations": {},
    }

    all_scores = {}
    all_rows = {}
    for representation in ("default", "shallow", "proofs"):
        matrix = np.stack([_vector(example, representation, args.dim) for example in examples])
        scores = np.full(len(examples), np.nan, dtype=np.float64)
        fold_rows = []
        for fold_index, test_theorems in enumerate(folds):
            test = np.asarray([example["theorem"] in test_theorems for example in examples])
            train = ~test
            fold_scores, _ = _fit_scores(examples, train, test, representation, args, matrix)
            scores[test] = fold_scores[test]
            fold_rows.append({"fold": fold_index, "test_theorems": sorted(test_theorems),
                              "train_examples": int(train.sum()),
                              "test_examples": int(test.sum())})
        metrics, rows = ranking_metrics(examples, members, scores)
        improvements = [row["mrr"] - row["raw_mrr"] for row in rows]
        metrics.update({
            "mrr_delta_vs_raw": metrics["mrr"] - raw_metrics["mrr"],
            "mrr_delta_vs_raw_theorem_bootstrap_95": _bootstrap_delta(rows, raw_rows, "mrr"),
            "top1_delta_vs_raw": metrics["top1_acceptance"] - raw_metrics["top1_acceptance"],
            "improved_members": sum(value > 1e-12 for value in improvements),
            "worsened_members": sum(value < -1e-12 for value in improvements),
            "tied_members": sum(abs(value) <= 1e-12 for value in improvements),
            "divergent_candidate_pairs": _divergent_pair_accuracy(examples, scores),
            "folds": fold_rows,
        })
        results["representations"][representation] = metrics
        all_scores[representation] = scores
        all_rows[representation] = rows

    shallow_rows = all_rows["shallow"]
    default_rows = all_rows["default"]
    results["structural_increment"] = {
        "mrr_delta_shallow_minus_default": (
            results["representations"]["shallow"]["mrr"]
            - results["representations"]["default"]["mrr"]
        ),
        "mrr_delta_theorem_bootstrap_95": _bootstrap_delta(
            shallow_rows, default_rows, "mrr"),
        "top1_delta_shallow_minus_default": (
            results["representations"]["shallow"]["top1_acceptance"]
            - results["representations"]["default"]["top1_acceptance"]
        ),
        "improved_members_vs_default": sum(
            shallow["mrr"] > default["mrr"] + 1e-12
            for shallow, default in zip(shallow_rows, default_rows)
        ),
        "worsened_members_vs_default": sum(
            shallow["mrr"] < default["mrr"] - 1e-12
            for shallow, default in zip(shallow_rows, default_rows)
        ),
        "tied_members_vs_default": sum(
            abs(shallow["mrr"] - default["mrr"]) <= 1e-12
            for shallow, default in zip(shallow_rows, default_rows)
        ),
        "classes_with_member_specific_order": len({
            (row["theorem"], row["pp_hash"])
            for row in shallow_rows
            if len({tuple(other["order"]) for other in shallow_rows
                    if other["theorem"] == row["theorem"]
                    and other["pp_hash"] == row["pp_hash"]}) > 1
        }),
    }

    family_holdout = {}
    families = sorted(set(example["family"] for example in examples))
    for family in families:
        test = np.asarray([example["family"] == family for example in examples])
        train = ~test
        if not train.any() or not test.any():
            continue
        family_members = [member for member in members if member["family"] == family]
        entry = {"members": len(family_members)}
        for representation in ("default", "shallow"):
            scores, _ = _fit_scores(examples, train, test, representation, args)
            metrics, _ = ranking_metrics(examples, family_members, scores)
            entry[representation] = metrics
        family_holdout[family] = entry
    results["leave_one_family_out"] = family_holdout

    # Strict transfer test requested by the representation-repair hypothesis: train the
    # *original* contradiction classifier only on the repeat-confirmed probe pairs, then apply
    # it without BFS-candidate fine-tuning.  The test theorem remains absent from training.
    run_paths = tuple(value.strip() for value in args.runs.split(",") if value.strip())
    source_examples, source_pairs = contradiction_examples(
        Path(args.representations), run_paths)
    transfer = {
        "training_examples": len(source_examples),
        "training_pairs": len(source_pairs),
        "training_task": "repeat-confirmed contradictory probe acceptance",
        "target_task": "BFS-Prover-V2-7B top-8 candidate acceptance",
        "representations": {},
    }
    source_y = np.asarray([example["label"] for example in source_examples], dtype=np.float64)
    for representation in ("default", "shallow", "proofs"):
        source_x = np.stack([contradiction_vector(example, representation, args.dim)
                             for example in source_examples])
        target_x = np.stack([contradiction_vector(example, representation, args.dim)
                             for example in examples])
        scores = np.full(len(examples), np.nan, dtype=np.float64)
        fold_rows = []
        for fold_index, test_theorems in enumerate(folds):
            source_train = np.asarray(
                [example["theorem"] not in test_theorems for example in source_examples])
            target_test = np.asarray(
                [example["theorem"] in test_theorems for example in examples])
            weights = _fit_logistic(source_x[source_train], source_y[source_train],
                                    args.l2, args.steps, args.lr)
            scores[target_test] = 1.0 / (
                1.0 + np.exp(-np.clip(target_x[target_test] @ weights, -30.0, 30.0))
            )
            fold_rows.append({"fold": fold_index,
                              "source_train_examples": int(source_train.sum()),
                              "target_test_examples": int(target_test.sum()),
                              "test_theorems": sorted(test_theorems)})
        metrics, rows = ranking_metrics(examples, members, scores)
        metrics.update({
            "mrr_delta_vs_raw": metrics["mrr"] - raw_metrics["mrr"],
            "mrr_delta_vs_raw_theorem_bootstrap_95": _bootstrap_delta(rows, raw_rows, "mrr"),
            "top1_delta_vs_raw": metrics["top1_acceptance"] - raw_metrics["top1_acceptance"],
            "divergent_candidate_pairs": _divergent_pair_accuracy(examples, scores),
            "folds": fold_rows,
        })
        transfer["representations"][representation] = metrics
    transfer["structural_increment"] = {
        "mrr_delta_shallow_minus_default": (
            transfer["representations"]["shallow"]["mrr"]
            - transfer["representations"]["default"]["mrr"]
        ),
        "top1_delta_shallow_minus_default": (
            transfer["representations"]["shallow"]["top1_acceptance"]
            - transfer["representations"]["default"]["top1_acceptance"]
        ),
    }
    results["transfer_from_contradiction_classifier"] = transfer

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--states", default="runs/killtest/states.jsonl")
    collect_parser.add_argument("--runs", default=",".join(RUNS))
    collect_parser.add_argument("--representations", default="runs/representation")
    collect_parser.add_argument("--candidates", default="runs/repsens_bfs/repsens.shard*.jsonl")
    collect_parser.add_argument("--out", default="runs/bfs_rerank")
    collect_parser.add_argument("--repeats", type=int, default=3)
    collect_parser.add_argument("--dojo-timeout", type=int, default=1200)
    collect_parser.add_argument("--tactic-timeout", type=int, default=90)
    collect_parser.add_argument("--candidate-timeout", type=int, default=90)
    collect_parser.add_argument("--shard", type=int, default=0)
    collect_parser.add_argument("--n-shards", type=int, default=1)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--data", default="runs/bfs_rerank")
    evaluate_parser.add_argument("--representations", default="runs/representation")
    evaluate_parser.add_argument("--runs", default=",".join(RUNS))
    evaluate_parser.add_argument("--out", default="runs/bfs_rerank/results.json")
    evaluate_parser.add_argument("--folds", type=int, default=5)
    evaluate_parser.add_argument("--dim", type=int, default=8192)
    evaluate_parser.add_argument("--l2", type=float, default=1e-3)
    evaluate_parser.add_argument("--steps", type=int, default=500)
    evaluate_parser.add_argument("--lr", type=float, default=0.03)
    args = parser.parse_args()
    return collect(args) if args.command == "collect" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
