"""Compact structural-sketch evaluation on confirmed observation aliases.

This tool has two deliberately separate stages.

``extract`` restores every member of every repeat-confirmed alias class and records two
member-local representations:

* ``default``: LeanDojo's ordinary proof-state rendering;
* ``shallow``: a compact bag of elaborated constant paths extracted from a ``pp.all``
  rendering with proof terms suppressed.

The sketch is deployable in the important sense that it is computed from one state at a time;
it does not use the other member, the label, the history, theorem name, or the paired diff.
Repeated constants are counted and sorted, turning very large explicit renderings into a
stable textual summary.  A structural digest is recorded only as an identity control and is
never supplied to the learned evaluator.

The extractor can also record a proof-visible sketch for auxiliary diagnostics, although the
learned evaluation uses the shallow channel. ``evaluate`` joins the representations to the
three-repeat compiler labels in
``confirmation.json``.  It keeps one accepted and one rejected member for each contradictory
class--action pair, then evaluates a deterministic hashed linear classifier with folds grouped
by theorem.  Grouping by theorem prevents near-duplicate classes from the same proof leaking
across train and test.  A separate leave-one-mechanism-out analysis groups classes by their
hidden elaborated delta and tests transfer to an entirely excluded mechanism.  The primary
readout is paired ranking accuracy: how often the accepted member receives a higher score than
its byte-identical rejected partner.  A text-only model is information-theoretically tied
within every pair and therefore provides the 50% control.

This is a diagnostic representation experiment, not a claim that the sketch is universally
action-sufficient or that appending it to an existing step-prover improves theorem proving.

Examples
--------
    python tools/representation_eval.py extract --states runs/killtest/states.jsonl \
      --out runs/representation --shard 0 --n-shards 4
    python tools/representation_eval.py evaluate --data runs/representation \
      --out runs/representation/results.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


RUNS = ("runs/reconv_b2", "runs/hunt_enn", "runs/hunt_rare", "runs/hunt_rare2")
_HEAD = re.compile(r"\(@?([A-Za-z_][A-Za-z0-9_'.]*(?:\.\{[^()]*\})?)")
_ATOM = re.compile(r"[A-Za-z_][A-Za-z0-9_'.]*")


def _h(s: str, n: int = 16) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def _log(msg: str) -> None:
    print(f"[representation] {msg}", file=sys.stderr, flush=True)


def _norm_name(name: str) -> str:
    """Remove universe annotations while preserving qualified constant paths."""
    # The head regex consumes the dot before ``{u}`` as part of the identifier, so remove
    # either ``.{...}`` or the residual trailing dot.  Do not strip ordinary namespace dots.
    name = re.sub(r"\.\{[^{}]*\}$", "", name)
    return name.lstrip("@").rstrip(".")


def constant_path_counts(explicit_pp: str) -> dict[str, int]:
    """Return a deterministic bag of application heads from a ``pp.all`` rendering.

    Fully explicit Lean output parenthesizes applications.  Taking their heads retains names
    such as ``ENNReal.ofNNReal``, ``WithTop.some``, type-class methods, constructors, and
    projections while excluding most binder names and literal payload.  This intentionally
    small grammar is version-auditable and does not depend on the paired alias member.
    """
    counts: collections.Counter[str] = collections.Counter()
    for raw in _HEAD.findall(explicit_pp or ""):
        name = _norm_name(raw)
        if name:
            counts[name] += 1
    return dict(sorted(counts.items()))


def encode_sketch(counts: dict[str, int]) -> str:
    return " ".join(f"{name}={count}" for name, count in sorted(counts.items()))


def _confirmed_targets(run_paths: tuple[str, ...]) -> list[dict]:
    """Join confirmations to Stage-B roots and deduplicate on paper class identity."""
    targets: dict[tuple[str, str], dict] = {}
    for run_s in run_paths:
        run = Path(run_s)
        rows = {}
        for filename in glob.glob(str(run / "stageb.shard*.jsonl")):
            for line in open(filename, encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    rows[(row["theorem"], row["pp_hash"])] = row
        for conf in json.load(open(run / "confirmation.json", encoding="utf-8")):
            if conf.get("verdict") != "CONFIRMED":
                continue
            key = (conf["theorem"], conf["pp_hash"])
            row = rows.get(key)
            if row is None or key in targets:
                continue
            targets[key] = {"row": row, "confirmation": conf, "run": run.name}
    return list(targets.values())


def extract(args: argparse.Namespace) -> int:
    from lean_dojo import Dojo, Theorem
    import lean_dojo as ldj
    from audit.fingerprint import PROBES, _run, parse_probe
    from audit.replay import deadline, kill_orphan_lean, lean_git_repo
    from tools.digest_eval import DIGEST

    run_paths = tuple(s.strip() for s in args.runs.split(",") if s.strip())
    targets = _confirmed_targets(run_paths)
    mine = [target for i, target in enumerate(targets)
            if i % args.n_shards == args.shard]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    outfile = out / f"members.shard{args.shard}.jsonl"
    done: set[tuple[str, str]] = set()
    if outfile.exists():
        for line in open(outfile, encoding="utf-8"):
            if line.strip():
                try:
                    old = json.loads(line)
                    if not old.get("error") and old.get("members"):
                        done.add((old["theorem"], old["pp_hash"]))
                except Exception:
                    pass
    mine = [target for target in mine
            if (target["row"]["theorem"], target["row"]["pp_hash"]) not in done]
    _log(f"shard {args.shard}: {len(mine)} classes to extract of {len(targets)}")

    wanted = {(t["row"]["file"], t["row"]["theorem"], t["row"]["tactic_index"])
              for t in mine}
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
        for index, target in enumerate(mine, 1):
            row, conf = target["row"], target["confirmation"]
            root = roots.get((row["file"], row["theorem"], row["tactic_index"]))
            if root is None:
                continue
            prov = root["provenance"]
            rec = {
                "theorem": row["theorem"], "file": row["file"],
                "pp_hash": row["pp_hash"], "tactic_index": row["tactic_index"],
                "run": target["run"], "members": [],
                "confirmed_probes": conf.get("confirmed_probes", []),
                "repeat_table": conf.get("repeat_table", {}),
            }
            ctx = None
            try:
                theorem = Theorem(repo, prov["file_path"], prov["theorem_full_name"])
                ctx = Dojo(theorem, timeout=args.dojo_timeout)
                dojo, state0 = ctx.__enter__()
                for tactic in root["proof_prefix"]:
                    with deadline(args.tactic_timeout, tactic[:30]):
                        state0 = dojo.run_tac(state0, tactic)
                for history in row["histories"]:
                    state = state0
                    for tactic in history:
                        with deadline(args.tactic_timeout, tactic[:30]):
                            state = dojo.run_tac(state, tactic)
                        if not isinstance(state, ldj.TacticState):
                            break
                    if not isinstance(state, ldj.TacticState):
                        continue
                    fields = {}
                    probe_names = (("phi_all_shallow",) if args.shallow_only else
                                   ("phi_all_shallow", "phi_full_proofs"))
                    for name in probe_names:
                        with deadline(args.probe_timeout, name):
                            msg = _run(dojo, state, PROBES[name]) or ""
                        fields[name] = parse_probe(msg, multiline=True)
                    with deadline(args.probe_timeout, "digest"):
                        dig = parse_probe(_run(dojo, state, DIGEST) or "", multiline=False)
                    shallow = fields.get("phi_all_shallow") or ""
                    proofs = fields.get("phi_full_proofs") or ""
                    shallow_sketch = encode_sketch(constant_path_counts(shallow))
                    proof_sketch = encode_sketch(constant_path_counts(proofs))
                    rec["members"].append({
                        "history": history,
                        "history_label": " ; ".join(history),
                        "default_pp": state.pp,
                        "default_hash": _h(state.pp),
                        "digest": dig,
                        "shallow_hash": _h(shallow) if shallow else None,
                        "shallow_bytes": len(shallow.encode("utf-8")),
                        "shallow_sketch": shallow_sketch,
                        "shallow_sketch_bytes": len(shallow_sketch.encode("utf-8")),
                        "proof_hash": _h(proofs) if proofs else None,
                        "proof_bytes": len(proofs.encode("utf-8")),
                        "proof_sketch": proof_sketch,
                        "proof_sketch_bytes": len(proof_sketch.encode("utf-8")),
                    })
                rec["renderings_identical"] = (
                    len(rec["members"]) >= 2
                    and len({member["default_pp"] for member in rec["members"]}) == 1
                )
                rec["shallow_separates"] = len({m["shallow_sketch"] for m in rec["members"]}) > 1
                rec["proof_separates"] = len({m["proof_sketch"] for m in rec["members"]}) > 1
                rec["digest_separates"] = len({m["digest"] for m in rec["members"]}) > 1
            except BaseException as exc:
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
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
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sink.flush()
            _log(f"[{index}/{len(mine)}] {row['theorem'][:48]} "
                 f"members={len(rec['members'])} shallow={rec.get('shallow_separates')} "
                 f"proofs={rec.get('proof_separates')} error={rec.get('error')}")
    return 0


def _accepted(outcomes: list[str]) -> int | None:
    labels = {0 if str(outcome).startswith("FAIL") else 1 for outcome in outcomes}
    return next(iter(labels)) if len(labels) == 1 else None


def _history_key(histories: list[list[str]]) -> tuple[str, ...]:
    return tuple(sorted(" ; ".join(history) for history in histories))


def _mechanism_label(row: dict) -> str:
    """Classify a recorded hidden elaborated delta."""
    text = " ".join(row.get("diff_tokens") or [])
    if "ENNReal.ofNNReal" in text or "WithTop.some" in text:
        return "coercion"
    if "@Ne" in text or "+Not" in text:
        return "Ne-vs-Not"
    if "instOfNatNat" in text:
        return "Fin-eta"
    if any(token in text for token in
           ("toLE", "toPreorder", "linearOrderOfSTO", "WellOrderingRel")):
        return "instance-path"
    # These classes are produced by simplifying different local definitions before the
    # renderings reconverge.  Explicit printing shows an abbreviation on one member and its
    # unfolded body on another.  The theorem names make this classification auditable even
    # for the large Fourier goal whose full probe can exceed the watchdog under load.
    if row.get("theorem") in {
        "tendsto_integral_exp_smul_cocompact",
        "Tuple.bubble_sort_induction'",
        "max_aleph0_card_le_rank_fun_nat",
    }:
        return "local-unfolding"
    proof = row.get("rungs", {}).get("phi_deep_proofs")
    if isinstance(proof, dict) and proof.get("separates"):
        return "proof-term"
    return "residual"


def _mechanism_map(run_paths: tuple[str, ...]) -> dict[tuple[str, tuple[str, ...]], str]:
    """Map classes to mechanisms using theorem and member histories.

    The mechanism files predate ``pp_hash``.  Theorem alone is insufficient because one theorem
    contributes both a coercion class and the residual class.  The member-history set is present
    in both records and gives an exact class-level join.
    """
    out: dict[tuple[str, tuple[str, ...]], str] = {}
    for run_s in run_paths:
        path = Path(run_s) / "mechanism.json"
        if not path.exists():
            continue
        for row in json.load(open(path, encoding="utf-8")):
            histories = row.get("histories") or []
            out[(row["theorem"], _history_key(histories))] = _mechanism_label(row)
    return out


def _mechanism_for_row(mechanisms: dict, row: dict) -> str:
    histories = row.get("histories")
    if histories is None:
        histories = [member["history"] for member in row.get("members", [])]
    return mechanisms.get((row["theorem"], _history_key(histories or [])), "residual")


# Backward-compatible import used by older analysis code.
_family_map = _mechanism_map


def _examples(data_dir: Path, run_paths: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    mechanisms = _mechanism_map(run_paths)
    classes = []
    for filename in sorted(data_dir.glob("members.shard*.jsonl")):
        for line in open(filename, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                if not row.get("error") and row.get("renderings_identical"):
                    classes.append(row)
    examples = []
    pairs = []
    for row in classes:
        by_label = {m["history_label"]: m for m in row["members"]}
        for action in row.get("confirmed_probes", []):
            table = row.get("repeat_table", {}).get(action, {})
            labeled = []
            for history_label, outcomes in table.items():
                member = by_label.get(history_label)
                label = _accepted(outcomes)
                if member is not None and label is not None:
                    labeled.append((member, label))
            positives = sorted((m for m, y in labeled if y == 1),
                               key=lambda m: m["history_label"])
            negatives = sorted((m for m, y in labeled if y == 0),
                               key=lambda m: m["history_label"])
            if not positives or not negatives:
                continue
            # One balanced pair per contradictory observation/action.  Multiple members would
            # otherwise give larger classes disproportionate weight.
            pair_id = f"{row['theorem']}|{row['pp_hash']}|{action}"
            mechanism = _mechanism_for_row(mechanisms, row)
            pair_rows = []
            for member, label in ((negatives[0], 0), (positives[0], 1)):
                ex = {
                    "pair_id": pair_id, "theorem": row["theorem"],
                    "pp_hash": row["pp_hash"], "action": action, "label": label,
                    "mechanism": mechanism, "family": mechanism,
                    "default_pp": member["default_pp"],
                    "shallow_sketch": member["shallow_sketch"],
                    "proof_sketch": member["proof_sketch"],
                }
                examples.append(ex)
                pair_rows.append(len(examples) - 1)
            pairs.append({"pair_id": pair_id, "indices": pair_rows,
                          "theorem": row["theorem"], "mechanism": mechanism,
                          "family": mechanism})
    return examples, pairs


def _tokens(text: str) -> collections.Counter[str]:
    return collections.Counter(token.lower() for token in _ATOM.findall(text or ""))


def _hash_index(token: str, dim: int) -> int:
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big") % dim


def _vector(ex: dict, representation: str, dim: int) -> list[float]:
    """Hashed bag features with action--structure crosses for a linear evaluator."""
    import numpy as np

    vec = np.zeros(dim, dtype=np.float64)
    action = _tokens(ex["action"])
    visible = _tokens(ex["default_pp"])
    for token, count in action.items():
        vec[_hash_index("a:" + token, dim)] += math.log1p(count)
    for token, count in visible.items():
        vec[_hash_index("v:" + token, dim)] += math.log1p(count)
        for act in action:
            vec[_hash_index(f"av:{act}:{token}", dim)] += math.log1p(count)
    if representation != "default":
        sketch = ex["shallow_sketch" if representation == "shallow" else "proof_sketch"]
        for item in sketch.split():
            name, _, raw_count = item.rpartition("=")
            try:
                value = math.log1p(int(raw_count))
            except ValueError:
                name, value = item, 1.0
            name = name.lower()
            vec[_hash_index("s:" + name, dim)] += value
            for act in action:
                vec[_hash_index(f"as:{act}:{name}", dim)] += value
    vec[_hash_index("bias", dim)] += 1.0
    return vec


def _folds(pairs: list[dict], n_folds: int) -> list[set[str]]:
    """Greedily balance theorem groups by their number of contradictory pairs."""
    counts = collections.Counter(pair["theorem"] for pair in pairs)
    bins: list[set[str]] = [set() for _ in range(n_folds)]
    sizes = [0] * n_folds
    for theorem, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        index = min(range(n_folds), key=lambda i: (sizes[i], i))
        bins[index].add(theorem)
        sizes[index] += count
    return bins


def _fit_logistic(x, y, l2: float, steps: int, lr: float):
    """Small deterministic full-batch Adam optimizer; no external ML dependency."""
    import numpy as np

    w = np.zeros(x.shape[1], dtype=np.float64)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    for step in range(1, steps + 1):
        z = np.clip(x @ w, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = (x.T @ (p - y)) / len(y) + l2 * w
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        mh = m / (1.0 - 0.9 ** step)
        vh = v / (1.0 - 0.999 ** step)
        w -= lr * mh / (np.sqrt(vh) + 1e-8)
    return w


def _cluster_bootstrap(pair_scores: list[float], groups: list[str],
                       iterations: int = 10000) -> list[float]:
    """Percentile interval resampling theorem groups, the experimental split unit."""
    import numpy as np

    if not pair_scores:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(20260803)
    values = np.asarray(pair_scores, dtype=np.float64)
    group_names = sorted(set(groups))
    indices = {group: np.asarray([i for i, value in enumerate(groups) if value == group])
               for group in group_names}
    means = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        chosen = np.concatenate([indices[group] for group in sampled])
        means[i] = values[chosen].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def evaluate(args: argparse.Namespace) -> int:
    import numpy as np

    run_paths = tuple(s.strip() for s in args.runs.split(",") if s.strip())
    examples, pairs = _examples(Path(args.data), run_paths)
    mechanism_map = _mechanism_map(run_paths)
    if not pairs:
        raise SystemExit("no contradictory pairs found")
    fold_groups = _folds(pairs, args.folds)
    results = {
        "protocol": {
            "unit": "one accepted and one rejected member per confirmed class-action pair",
            "split": f"{args.folds}-fold grouped by theorem",
            "secondary_split": "leave one hidden-delta mechanism group out",
            "mechanism_macro": "unweighted mean across named held-out mechanisms",
            "all_group_macro": "unweighted mean including the unresolved group",
            "classifier": "hashed linear logistic evaluator with action-feature crosses",
            "dimension": args.dim, "l2": args.l2, "steps": args.steps,
            "seed": 20260803,
        },
        "corpus": {
            "classes": len({(ex["theorem"], ex["pp_hash"]) for ex in examples}),
            "theorems": len({ex["theorem"] for ex in examples}),
            "contradictory_pairs": len(pairs), "examples": len(examples),
            "mechanisms": dict(collections.Counter(
                pair["mechanism"] for pair in pairs)),
            "mechanism_classes": dict(collections.Counter(
                mechanism for _, _, mechanism in {
                    (ex["theorem"], ex["pp_hash"], ex["mechanism"])
                    for ex in examples
                })),
        },
        "representations": {},
    }
    pair_lookup = {pair["pair_id"]: pair for pair in pairs}
    for representation in ("default", "shallow"):
        x = np.stack([_vector(ex, representation, args.dim) for ex in examples])
        y = np.asarray([ex["label"] for ex in examples], dtype=np.float64)
        scores = np.zeros(len(examples), dtype=np.float64)
        fold_rows = []
        for fold, test_theorems in enumerate(fold_groups):
            test = np.asarray([ex["theorem"] in test_theorems for ex in examples])
            train = ~test
            w = _fit_logistic(x[train], y[train], args.l2, args.steps, args.lr)
            scores[test] = 1.0 / (1.0 + np.exp(-np.clip(x[test] @ w, -30.0, 30.0)))
            fold_rows.append({"fold": fold, "train_examples": int(train.sum()),
                              "test_examples": int(test.sum()),
                              "test_theorems": sorted(test_theorems)})
        pair_values = []
        margins = []
        pair_predictions = []
        by_mechanism: dict[str, list[float]] = collections.defaultdict(list)
        for pair_id, pair in pair_lookup.items():
            indices = pair["indices"]
            neg = next(i for i in indices if examples[i]["label"] == 0)
            pos = next(i for i in indices if examples[i]["label"] == 1)
            margin = float(scores[pos] - scores[neg])
            value = 1.0 if margin > 1e-12 else 0.0 if margin < -1e-12 else 0.5
            pair_values.append(value)
            margins.append(margin)
            by_mechanism[pair["mechanism"]].append(value)
            pair_predictions.append({"pair_id": pair_id, "theorem": pair["theorem"],
                                     "mechanism": pair["mechanism"], "margin": margin,
                                     "correct": value})
        predictions = (scores >= 0.5).astype(np.float64)
        accuracy = float((predictions == y).mean())
        brier = float(np.mean((scores - y) ** 2))
        mechanism_holdout = {}
        for mechanism in sorted(by_mechanism):
            test = np.asarray([ex["mechanism"] == mechanism for ex in examples])
            train = ~test
            if not train.any() or not test.any():
                continue
            w = _fit_logistic(x[train], y[train], args.l2, args.steps, args.lr)
            held_scores = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30.0, 30.0)))
            held_values = []
            for pair in pairs:
                if pair["mechanism"] != mechanism:
                    continue
                indices = pair["indices"]
                neg = next(i for i in indices if examples[i]["label"] == 0)
                pos = next(i for i in indices if examples[i]["label"] == 1)
                margin = float(held_scores[pos] - held_scores[neg])
                held_values.append(1.0 if margin > 1e-12 else
                                   0.0 if margin < -1e-12 else 0.5)
            mechanism_holdout[mechanism] = {
                "pairs": len(held_values),
                "classes": len({
                    (ex["theorem"], ex["pp_hash"])
                    for ex in examples if ex["mechanism"] == mechanism
                }),
                "paired_ranking_accuracy": float(np.mean(held_values)),
                "training_mechanisms": sorted(
                    set(ex["mechanism"] for ex in examples) - {mechanism}),
            }
        all_group_scores = [
            row["paired_ranking_accuracy"] for row in mechanism_holdout.values()
        ]
        mechanism_scores = [
            row["paired_ranking_accuracy"]
            for mechanism, row in mechanism_holdout.items()
            if mechanism != "residual"
        ]
        held_correct = sum(
            row["pairs"] * row["paired_ranking_accuracy"]
            for row in mechanism_holdout.values()
        )
        held_pairs = sum(row["pairs"] for row in mechanism_holdout.values())
        results["representations"][representation] = {
            "paired_ranking_accuracy": float(np.mean(pair_values)),
            "theorem_cluster_bootstrap_95": _cluster_bootstrap(
                pair_values, [pair["theorem"] for pair in pairs]),
            "mean_pair_margin": float(np.mean(margins)),
            "classification_accuracy": accuracy,
            "brier": brier,
            "by_mechanism_paired_accuracy": {
                mechanism: {"pairs": len(values), "accuracy": float(np.mean(values))}
                for mechanism, values in sorted(by_mechanism.items())
            },
            "leave_one_mechanism_out": mechanism_holdout,
            "mechanism_macro_paired_accuracy": float(np.mean(mechanism_scores)),
            "all_group_macro_paired_accuracy": float(np.mean(all_group_scores)),
            "all_group_pair_weighted_paired_accuracy": float(held_correct / held_pairs),
            "pair_predictions": pair_predictions,
            "folds": fold_rows,
        }

    member_files = []
    class_rows = []
    for filename in sorted(Path(args.data).glob("members.shard*.jsonl")):
        member_files.append(filename.name)
        class_rows.extend(json.loads(line) for line in open(filename, encoding="utf-8")
                          if line.strip())
    valid = [row for row in class_rows if not row.get("error") and row.get("renderings_identical")]
    results["separation"] = {
        "valid_classes": len(valid),
        "all_class_mechanisms": dict(collections.Counter(
            _mechanism_for_row(mechanism_map, row) for row in valid)),
        "shallow_sketch": sum(bool(row.get("shallow_separates")) for row in valid),
        "proof_sketch": sum(bool(row.get("proof_separates")) for row in valid),
        "digest": sum(bool(row.get("digest_separates")) for row in valid),
        "median_bytes": {
            key: float(np.median([member[key] for row in valid for member in row["members"]]))
            for key in ("shallow_bytes", "shallow_sketch_bytes", "proof_bytes",
                        "proof_sketch_bytes")
        },
    }
    results["inputs"] = member_files
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--states", required=True)
    ex.add_argument("--runs", default=",".join(RUNS))
    ex.add_argument("--out", default="runs/representation")
    ex.add_argument("--shard", type=int, default=0)
    ex.add_argument("--n-shards", type=int, default=1)
    ex.add_argument("--dojo-timeout", type=int, default=900)
    ex.add_argument("--tactic-timeout", type=int, default=90)
    ex.add_argument("--probe-timeout", type=int, default=300)
    ex.add_argument("--shallow-only", action="store_true",
                    help="omit the auxiliary proof-visible rendering")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--data", default="runs/representation")
    ev.add_argument("--runs", default=",".join(RUNS))
    ev.add_argument("--out", default="runs/representation/results.json")
    ev.add_argument("--folds", type=int, default=5)
    ev.add_argument("--dim", type=int, default=8192)
    ev.add_argument("--l2", type=float, default=1e-3)
    ev.add_argument("--steps", type=int, default=500)
    ev.add_argument("--lr", type=float, default=0.03)
    args = parser.parse_args()
    return extract(args) if args.command == "extract" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
