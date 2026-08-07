"""Build a blind sample of natural tactic states on a current Mathlib checkout.

This utility targets a fixed random sample of compiled Mathlib source files. Run
``prepare_current_stack.py`` first to install the matching LeanDojo-v2 extractor. The
resulting trace files are parsed with the v2 parser and converted to the audit's
version-neutral state JSONL schema.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit.extract import top_level_tactics


REPO_URL = "https://github.com/leanprover-community/mathlib4"


def _log(message: str) -> None:
    print(f"[current-blind-extract] {message}", file=sys.stderr, flush=True)


def _olean(repo: Path, source: Path) -> Path:
    relative = source.relative_to(repo).with_suffix(".olean")
    return repo / ".lake" / "build" / "lib" / "lean" / relative


def _trace_one(repo: Path, source: Path, timeout: int) -> dict:
    started = time.time()
    relative = source.relative_to(repo)
    try:
        result = subprocess.run(
            ["lake", "env", "lean", "--run", "ExtractData.lean", str(relative)],
            cwd=repo, capture_output=True, text=True, timeout=timeout,
        )
        return {
            "file": str(relative), "returncode": result.returncode,
            "seconds": time.time() - started,
            "stderr_tail": result.stderr[-500:] if result.returncode else "",
        }
    except subprocess.TimeoutExpired:
        return {"file": str(relative), "returncode": None,
                "seconds": time.time() - started, "error": "timeout"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--files", type=int, default=150)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    # These imports intentionally occur after argument parsing. The published v2 parser can be
    # supplied through PYTHONPATH without changing the primary audit environment.
    from lean_dojo_v2.lean_dojo.data_extraction.lean import LeanGitRepo
    from lean_dojo_v2.lean_dojo.data_extraction.traced_data import TracedFile

    repo_path = Path(args.repo).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"], text=True).strip()
    sources = sorted(path for path in (repo_path / "Mathlib").rglob("*.lean")
                     if _olean(repo_path, path).exists())
    rng = random.Random(args.seed)
    selected = rng.sample(sources, min(args.files, len(sources)))
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    _log(f"selected {len(selected)} of {len(sources)} compiled Mathlib files")

    trace_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_trace_one, repo_path, source, args.timeout): source
                   for source in selected}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            trace_rows.append(row)
            _log(f"[{index}/{len(selected)}] {row['file']} rc={row['returncode']} "
                 f"{row['seconds']:.1f}s")

    repo = LeanGitRepo(REPO_URL, commit)
    traced_parent = SimpleNamespace(repo=repo, dependencies={})
    records = []
    parse_failures = []
    for trace_row in trace_rows:
        if trace_row["returncode"] != 0:
            continue
        source_relative = Path(trace_row["file"])
        json_path = (repo_path / ".lake" / "build" / "ir" /
                     source_relative.with_suffix(".ast.json"))
        try:
            traced_file = TracedFile.from_traced_file(repo_path, json_path, repo)
            traced_file.traced_repo = traced_parent
            for traced_theorem in traced_file.get_traced_theorems():
                tactics, _ = top_level_tactics(traced_theorem.get_traced_tactics())
                prefix = []
                for tactic_index, traced_tactic in enumerate(tactics):
                    state_before = getattr(traced_tactic, "state_before", None)
                    tactic = traced_tactic.tactic.strip()
                    if state_before:
                        records.append({
                            "provenance": {
                                "repo_url": REPO_URL, "repo_commit": commit,
                                "file_path": str(source_relative),
                                "theorem_full_name": str(traced_theorem.theorem.full_name),
                                "tactic_index": tactic_index,
                                "lean_toolchain": (repo_path / "lean-toolchain").read_text().strip(),
                                "leandojo_version": "lean-dojo-v2-1.0.9-extractor",
                                "proof_depth": len(tactics),
                                "goal_count": state_before.count("⊢"),
                                "token_length": len(state_before),
                                "has_ellipsis": "⋯" in state_before or "…" in state_before,
                                "tactic_family": tactic.split(maxsplit=1)[0] if tactic else "",
                            },
                            "observation": {"phi_state": state_before},
                            "human_tactic": tactic, "proof_prefix": list(prefix),
                        })
                    prefix.append(tactic)
        except BaseException as exception:
            parse_failures.append({"file": str(source_relative),
                                   "error": f"{type(exception).__name__}: {exception}"})

    with open(output / "states.jsonl", "w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "protocol": "fixed-seed uniform sample of compiled Mathlib source files",
        "repo_url": REPO_URL, "commit": commit,
        "lean_toolchain": (repo_path / "lean-toolchain").read_text().strip(),
        "seed": args.seed, "eligible_files": len(sources),
        "selected_files": len(selected),
        "successfully_traced_files": sum(row["returncode"] == 0 for row in trace_rows),
        "states": len(records), "theorems": len({
            record["provenance"]["theorem_full_name"] for record in records}),
        "trace_rows": sorted(trace_rows, key=lambda row: row["file"]),
        "parse_failures": parse_failures,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({key: manifest[key] for key in (
        "commit", "lean_toolchain", "eligible_files", "selected_files",
        "successfully_traced_files", "states", "theorems")}, indent=2))
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
