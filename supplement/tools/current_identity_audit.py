"""Static and behavioral audit of node identity in LeanDojo releases.

The script deliberately reads published source without importing LeanDojo, whose package import
performs a GitHub API request. It records the exact source hashes and verifies the behavioral
consequences of the dataclass declarations where the dependencies are available.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _class_fields(source: str, class_name: str) -> dict:
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            fields = {}
            for child in node.body:
                if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    fields[child.target.id] = {
                        "annotation": ast.unparse(child.annotation),
                        "default": ast.unparse(child.value) if child.value is not None else None,
                    }
            return fields
    raise ValueError(f"class {class_name} not found")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lean-dojo-source", required=True)
    parser.add_argument("--v2-wheel", required=True)
    parser.add_argument("--pantograph", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    old_path = Path(args.lean_dojo_source)
    wheel = Path(args.v2_wheel)
    pantograph = Path(args.pantograph)
    old_source = old_path.read_text(encoding="utf-8")
    with zipfile.ZipFile(wheel) as archive:
        search_source = archive.read(
            "lean_dojo_v2/lean_agent/prover/search_tree.py").decode("utf-8")
        active_source = archive.read(
            "lean_dojo_v2/prover/base_prover.py").decode("utf-8")
        policy_sources = {
            name: archive.read(name).decode("utf-8")
            for name in (
                "lean_dojo_v2/prover/external_prover.py",
                "lean_dojo_v2/prover/hf_prover.py",
                "lean_dojo_v2/prover/retrieval_prover.py",
            )
        }
        metadata_name = next(name for name in archive.namelist()
                             if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode("utf-8")
        names = set(archive.namelist())
    goal_path = pantograph / "pantograph" / "expr.py"
    goal_source = goal_path.read_text(encoding="utf-8")
    version = re.search(r"^Version:\s*(\S+)", metadata, re.MULTILINE).group(1)
    commit = subprocess.check_output(
        ["git", "-C", str(pantograph), "rev-parse", "HEAD"], text=True).strip()

    old_fields = _class_fields(old_source, "TacticState")
    internal_fields = _class_fields(search_source, "InternalNode")
    goal_fields = _class_fields(goal_source, "GoalState")
    old_id_excluded = "compare=False" in (old_fields["id"]["default"] or "")
    old_pp_compared = old_fields["pp"]["default"] is None
    legacy_missing_state = not any(name.endswith("interaction/dojo.py") for name in names)
    active_uses_stack = "search_stack = [initial_state]" in active_source
    active_uses_state_dictionary = "self.nodes =" in active_source
    policy_uses_rendering = {
        name: "str(state)" in source for name, source in policy_sources.items()
    }

    sys.path.insert(0, str(pantograph))
    from pantograph.expr import Goal, GoalState
    sentinel = []
    goal = Goal.sentence("g", "P")
    first = GoalState(1, [goal], [], sentinel)
    second = GoalState(2, [goal], [], sentinel)
    hashable = True
    try:
        hash(first)
    except TypeError:
        hashable = False

    results = {
        "evaluated_lean_dojo": {
            "source": str(old_path), "source_sha256": _sha256(old_path),
            "tactic_state_fields": old_fields,
            "pretty_printed_state_participates_in_equality": old_pp_compared,
            "executable_id_excluded_from_equality": old_id_excluded,
        },
        "lean_dojo_v2": {
            "version": version, "wheel": wheel.name, "wheel_sha256": _sha256(wheel),
            "legacy_internal_node_fields": internal_fields,
            "legacy_interaction_state_definition_present_in_wheel": not legacy_missing_state,
            "active_prover_uses_history_stack": active_uses_stack,
            "active_prover_uses_state_dictionary": active_uses_state_dictionary,
            "active_policy_uses_str_goal_state": policy_uses_rendering,
        },
        "pantograph": {
            "commit": commit, "goal_state_source_sha256": _sha256(goal_path),
            "goal_state_fields": goal_fields,
            "same_rendering_different_handle": {
                "renderings_equal": str(first) == str(second),
                "objects_equal": first == second,
                "hashable": hashable,
            },
        },
        "conclusion": {
            "evaluated_prover_rendering_keyed": bool(old_pp_compared and old_id_excluded),
            "current_v2_active_prover_rendering_keyed": False,
            "current_v2_model_rendering_can_alias": str(first) == str(second),
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
