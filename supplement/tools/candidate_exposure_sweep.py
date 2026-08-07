"""Candidate-set exposure as a function of beam width.

The experiment has three stages. ``generate`` obtains one deterministic ranked list from
each byte-identical class rendering. ``collect`` restores every hidden member and executes
every candidate with repeat controls. ``summarize`` truncates the one ranked list at several
values of k and compares member-aware and best-shared-action oracle coverage.

Generating one maximum-width list and taking prefixes makes the candidate sets nested. The
result therefore isolates candidate-set size from changes caused by rerunning beam search at
different widths.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.representation_eval import RUNS, _mechanism_for_row, _mechanism_map


MODEL_ID = "ByteDance-Seed/BFS-Prover-V2-7B"
MODEL_REVISION = "533d59fc6b5e04ae6e2c25f1d88cf921061169d0"
MINICTX_ID = "l3lab/ntp-mathlib-st-deepseek-coder-1.3b"
MINICTX_REVISION = "7c00b1175a82ea993732e5eff727b300e7589bb5"


class _MiniCTXGenerator:
    """Deterministic miniCTX decoder using the published state-tactic prompt."""

    def __init__(self, source: str, device: str, local_files_only: bool,
                 max_new_tokens: int, revision: str | None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        load_args = {"local_files_only": local_files_only}
        if revision:
            load_args["revision"] = revision
        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(source, **load_args)
        self.model = AutoModelForCausalLM.from_pretrained(
            source, dtype=torch.float16, low_cpu_mem_usage=True, **load_args
        ).to(device).eval()
        self.max_new_tokens = max_new_tokens

    def top_k_scored(self, state_pp: str, k: int) -> list[tuple[str, float]]:
        prompt = (
            "/- You are proving a theorem in Lean 4.\n"
            "You are given the following information:\n"
            "- The current proof state, inside [STATE]...[/STATE]\n\n"
            "Your task is to generate the next tactic in the proof.\n"
            "Put the next tactic inside [TAC]...[/TAC]\n-/\n"
            f"[STATE]\n{state_pp}\n[/STATE]\n[TAC]\n"
        )
        encoded = self.tok(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **encoded, num_beams=k, num_return_sequences=k, do_sample=False,
                max_new_tokens=self.max_new_tokens, early_stopping=True,
                output_scores=True, return_dict_in_generate=True,
                pad_token_id=self.tok.eos_token_id,
            )
        prefix = encoded["input_ids"].shape[1]
        scores = output.sequences_scores
        if scores is None:
            scores = [0.0] * len(output.sequences)
        seen, ranked = set(), []
        for sequence, score in zip(output.sequences, scores):
            text = self.tok.decode(sequence[prefix:], skip_special_tokens=True)
            tactic = text.split("[/TAC]")[0].strip()
            if "[/TAC]" not in text:
                tactic = tactic.splitlines()[0].strip() if tactic else ""
            if tactic and tactic not in seen:
                seen.add(tactic)
                ranked.append((tactic, float(score)))
        return ranked


def _log(message: str) -> None:
    print(f"[candidate-sweep] {message}", file=sys.stderr, flush=True)


def _jsonl(pattern: str) -> list[dict]:
    return [json.loads(line) for filename in sorted(glob.glob(pattern))
            for line in open(filename, encoding="utf-8") if line.strip()]


def _representations(path: Path) -> dict[tuple[str, str], dict]:
    rows = _jsonl(str(path / "members.shard*.jsonl"))
    return {(row["theorem"], row["pp_hash"]): row for row in rows
            if not row.get("error") and row.get("renderings_identical")}


def generate(args: argparse.Namespace) -> int:
    from audit.bfs_runner import BFSProverGenerator

    rows = list(_representations(Path(args.representations)).values())
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    outfile = output / "candidates.jsonl"
    done = {(row["theorem"], row["pp_hash"]) for row in _jsonl(str(outfile))}
    todo = [row for row in rows if (row["theorem"], row["pp_hash"]) not in done]
    default_id = MINICTX_ID if args.generator == "minictx" else MODEL_ID
    default_revision = MINICTX_REVISION if args.generator == "minictx" else MODEL_REVISION
    model_id = args.model_id or default_id
    revision = args.revision or default_revision
    load_source = args.model_path or model_id
    load_revision = None if args.model_path else revision
    _log(f"loading {model_id} at {revision}; {len(todo)} of {len(rows)} classes remain")
    if args.generator == "minictx":
        model = _MiniCTXGenerator(
            load_source, args.device, args.local_files_only,
            args.max_new_tokens, load_revision,
        )
    else:
        model = BFSProverGenerator(
            load_source, revision=load_revision, device=args.device,
            local_files_only=args.local_files_only, max_new_tokens=args.max_new_tokens,
        )
    with open(outfile, "a", encoding="utf-8") as sink:
        for index, row in enumerate(todo, 1):
            rendering = row["members"][0]["default_pp"]
            ranked = model.top_k_scored(rendering, k=args.max_k)
            record = {
                "theorem": row["theorem"], "pp_hash": row["pp_hash"],
                "model_id": model_id, "revision": revision,
                "decoding": "deterministic beam search",
                "num_beams": args.max_k, "requested_candidates": args.max_k,
                "max_new_tokens": args.max_new_tokens,
                "returned_candidates": len(ranked),
                "candidates": [tactic for tactic, _ in ranked],
                "sequence_scores": [score for _, score in ranked],
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            _log(f"[{index}/{len(todo)}] {row['theorem'][:52]} returned={len(ranked)}")
    return 0


def _confirmed_targets(run_paths: tuple[str, ...]) -> dict[tuple[str, str], dict]:
    targets: dict[tuple[str, str], dict] = {}
    for run_string in run_paths:
        run = Path(run_string)
        stage = {}
        for filename in glob.glob(str(run / "stageb.shard*.jsonl")):
            for line in open(filename, encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    stage[(row["theorem"], row["pp_hash"])] = row
        for confirmation in json.load(open(run / "confirmation.json", encoding="utf-8")):
            key = (confirmation["theorem"], confirmation["pp_hash"])
            if confirmation.get("verdict") == "CONFIRMED" and key in stage:
                targets.setdefault(key, stage[key])
    return targets


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

    run_paths = tuple(item.strip() for item in args.runs.split(",") if item.strip())
    reps = _representations(Path(args.representations))
    mechanisms = _mechanism_map(run_paths)
    candidates = {(row["theorem"], row["pp_hash"]): row
                  for row in _jsonl(str(Path(args.candidates) / "candidates.jsonl"))}
    targets = _confirmed_targets(run_paths)
    keys = [key for key in targets if key in reps and key in candidates]
    mine = [key for index, key in enumerate(keys) if index % args.n_shards == args.shard]

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    outfile = output / f"executions.shard{args.shard}.jsonl"
    done = {(row["theorem"], row["pp_hash"]) for row in _jsonl(str(outfile))}
    mine = [key for key in mine if key not in done]
    _log(f"shard {args.shard}: {len(mine)} classes remain of {len(keys)}")

    wanted = {(targets[key]["file"], targets[key]["theorem"],
               targets[key]["tactic_index"]) for key in mine}
    roots = {}
    with open(args.states, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            state = json.loads(line)
            provenance = state["provenance"]
            root_key = (provenance["file_path"], provenance["theorem_full_name"],
                        provenance["tactic_index"])
            if root_key in wanted:
                roots[root_key] = state
    if not roots:
        raise SystemExit("no target roots indexed")
    provenance = next(iter(roots.values()))["provenance"]
    repo = lean_git_repo(ldj, provenance["repo_url"], provenance["repo_commit"])

    with open(outfile, "a", encoding="utf-8") as sink:
        for index, key in enumerate(mine, 1):
            row = targets[key]
            rep = reps[key]
            candidate_row = candidates[key]
            root = roots.get((row["file"], row["theorem"], row["tactic_index"]))
            record = {
                "theorem": row["theorem"], "file": row["file"],
                "pp_hash": row["pp_hash"], "tactic_index": row["tactic_index"],
                "mechanism": _mechanism_for_row(mechanisms, rep),
                "model_id": candidate_row["model_id"],
                "revision": candidate_row["revision"],
                "num_beams": candidate_row["num_beams"],
                "max_new_tokens": candidate_row.get("max_new_tokens"),
                "candidates": candidate_row["candidates"],
                "repeats": args.repeats, "members": [],
            }
            context = None
            try:
                if root is None:
                    raise RuntimeError("root not indexed")
                provenance = root["provenance"]
                theorem = Theorem(repo, provenance["file_path"],
                                  provenance["theorem_full_name"])
                context = Dojo(theorem, timeout=args.dojo_timeout)
                dojo, state0 = context.__enter__()
                for tactic in root["proof_prefix"]:
                    with deadline(args.tactic_timeout, tactic[:30]):
                        state0 = dojo.run_tac(state0, tactic)
                rep_members = {member["history_label"]: member for member in rep["members"]}
                for history in row["histories"]:
                    history_label = " ; ".join(history)
                    member_rep = rep_members.get(history_label)
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
                        raise RuntimeError(f"rendering drift for {history_label}")
                    outcomes = {}
                    for tactic in candidate_row["candidates"]:
                        observed = []
                        for _ in range(args.repeats):
                            with deadline(args.candidate_timeout, tactic[:30]):
                                observed.append(_outcome(dojo.run_tac(state, tactic), ldj))
                        outcomes[tactic] = observed
                    record["members"].append({
                        "history": history, "history_label": history_label,
                        "outcomes": outcomes,
                    })
                record["complete"] = len(record["members"]) == len(row["histories"])
            except BaseException as exception:
                record["error"] = f"{type(exception).__name__}: {str(exception)[:180]}"
            finally:
                if context is not None:
                    try:
                        context.__exit__(None, None, None)
                    except Exception:
                        pass
                try:
                    kill_orphan_lean()
                except Exception:
                    pass
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            _log(f"[{index}/{len(mine)}] {row['theorem'][:48]} "
                 f"members={len(record['members'])} error={record.get('error')}")
    return 0


def _stable_label(values: list[str]) -> int | None:
    labels = {0 if value == "FAIL" else 1 for value in values}
    return next(iter(labels)) if len(labels) == 1 else None


def _oracle(rows: list[dict], k: int) -> dict:
    classes = 0
    members = 0
    member_covered = 0
    best_shared = 0
    ambiguous_classes = 0
    ambiguous_actions = 0
    fully_covered_common = 0
    returned = []
    unstable = 0
    for row in rows:
        if row.get("error") or not row.get("complete"):
            continue
        candidates = row["candidates"][:k]
        returned.append(len(candidates))
        accepted_sets = []
        for member in row["members"]:
            accepted = set()
            for tactic in candidates:
                label = _stable_label(member["outcomes"].get(tactic, []))
                if label is None:
                    unstable += 1
                elif label:
                    accepted.add(tactic)
            accepted_sets.append(accepted)
        if not accepted_sets:
            continue
        classes += 1
        members += len(accepted_sets)
        member_covered += sum(bool(actions) for actions in accepted_sets)
        per_action = [sum(tactic in actions for actions in accepted_sets)
                      for tactic in candidates]
        best_shared += max(per_action, default=0)
        divergent = [tactic for tactic in candidates
                     if len({tactic in actions for actions in accepted_sets}) > 1]
        ambiguous_classes += bool(divergent)
        ambiguous_actions += len(divergent)
        fully_covered_common += bool(
            all(accepted_sets) and set.intersection(*accepted_sets))
    return {
        "k": k, "classes": classes, "members": members,
        "mean_candidates_available": sum(returned) / len(returned) if returned else None,
        "member_oracle_coverage": member_covered / members if members else None,
        "best_shared_action_coverage": best_shared / members if members else None,
        "member_specific_headroom": ((member_covered - best_shared) / members
                                     if members else None),
        "member_specific_extra_successes": member_covered - best_shared,
        "acceptance_ambiguous_classes": ambiguous_classes,
        "acceptance_ambiguous_actions": ambiguous_actions,
        "fully_covered_classes_with_common_action": fully_covered_common,
        "unstable_state_action_labels": unstable,
    }


def summarize(args: argparse.Namespace) -> int:
    rows = _jsonl(str(Path(args.data) / "executions.shard*.jsonl"))
    ks = [int(value) for value in args.k.split(",") if value]
    valid = [row for row in rows if not row.get("error") and row.get("complete")]
    results = {
        "protocol": {
            "generator": (valid[0].get("model_id") if valid else None),
            "revision": (valid[0].get("revision") if valid else None),
            "candidate_construction": (
                "one deterministic maximum-width beam list per rendering; nested prefixes"
            ),
            "max_new_tokens": (valid[0].get("max_new_tokens") if valid else None),
            "labels": "fresh repeated Lean executions on every hidden member",
            "k": ks,
        },
        "corpus": {
            "records": len(rows), "complete_classes": len(valid),
            "theorems": len({row["theorem"] for row in valid}),
            "members": sum(len(row["members"]) for row in valid),
            "mechanisms": dict(collections.Counter(row["mechanism"] for row in valid)),
        },
        "sweep": [_oracle(valid, k) for k in ks],
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generator = subparsers.add_parser("generate")
    generator.add_argument("--representations", default="runs/representation")
    generator.add_argument("--out", default="runs/candidate_sweep")
    generator.add_argument("--generator", choices=("bfs7b", "minictx"),
                           default="bfs7b")
    generator.add_argument("--model-id")
    generator.add_argument("--revision")
    generator.add_argument("--max-k", type=int, default=32)
    generator.add_argument("--max-new-tokens", type=int, default=16)
    generator.add_argument("--device", default="auto")
    generator.add_argument("--model-path",
                           help="local immutable snapshot; avoids registry metadata requests")
    generator.add_argument("--local-files-only", action="store_true")
    collector = subparsers.add_parser("collect")
    collector.add_argument("--states", default="runs/killtest/states.jsonl")
    collector.add_argument("--runs", default=",".join(RUNS))
    collector.add_argument("--representations", default="runs/representation")
    collector.add_argument("--candidates", default="runs/candidate_sweep")
    collector.add_argument("--out", default="runs/candidate_sweep")
    collector.add_argument("--repeats", type=int, default=3)
    collector.add_argument("--dojo-timeout", type=int, default=1200)
    collector.add_argument("--tactic-timeout", type=int, default=90)
    collector.add_argument("--candidate-timeout", type=int, default=90)
    collector.add_argument("--shard", type=int, default=0)
    collector.add_argument("--n-shards", type=int, default=1)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--data", default="runs/candidate_sweep")
    summary.add_argument("--out", default="runs/candidate_sweep/results.json")
    summary.add_argument("--k", default="8,16,32")
    args = parser.parse_args()
    if args.command == "generate":
        return generate(args)
    if args.command == "collect":
        return collect(args)
    return summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
