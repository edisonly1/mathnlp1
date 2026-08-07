"""Run the reconvergence protocol on traces produced by LeanDojo-v2.

The published v2 extractor supports current Lean syntax that the evaluated LeanDojo 2.0.2 AST
parser predates. Its traced theorem positions can nevertheless drive the 2.0.2 interactive REPL,
which was compiled against the current checkout. This adapter uses the v2 parser only to locate
and splice theorem proofs; tactic execution and every observation remain live Lean responses.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pexpect

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from github import Github

_original_github_get_repo = Github.get_repo


def _lazy_github_get_repo(self, full_name_or_id, lazy=False):
    return _original_github_get_repo(self, full_name_or_id, lazy=True)


# Both LeanDojo packages construct GitHub objects at import time. Lazy objects
# avoid unrelated API calls because all required repositories are local here.
Github.get_repo = _lazy_github_get_repo

_original_pexpect_spawn = pexpect.spawn


def _spawn_lean_without_lake_stdin_loss(command, *args, **kwargs):
    """Run Lean directly while retaining the environment computed by Lake.

    Lake 5.0 closes the child standard input after launching `lake env lean` on
    this stack.  The REPL prints its initial state and then immediately observes
    EOF.  Resolving Lake's environment and executable first gives Lean the same
    module paths while leaving its pseudoterminal open for tactic requests.
    """
    if isinstance(command, str) and command.startswith("lake env lean "):
        lake_environment = os.environ.copy()
        resolved = subprocess.run(
            ["lake", "env", "printenv"], capture_output=True, text=True,
            check=True,
        )
        for line in resolved.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                lake_environment[key] = value
        lean_executable = subprocess.run(
            ["lake", "env", "which", "lean"], capture_output=True, text=True,
            check=True,
        ).stdout.strip()
        command = command.replace("lake env lean", lean_executable, 1)
        kwargs["env"] = lake_environment
    return _original_pexpect_spawn(command, *args, **kwargs)


pexpect.spawn = _spawn_lean_without_lake_stdin_loss

from lean_dojo.data_extraction.lean import LeanFile, LeanGitRepo as OldLeanGitRepo
from lean_dojo.interaction.dojo import Dojo
from lean_dojo.interaction.dojo import DojoInitError
import lean_dojo.interaction.dojo as dojo_module

from lean_dojo_v2.lean_dojo.data_extraction.lean import (
    LeanGitRepo as V2LeanGitRepo,
    RepoType as V2RepoType,
)
from lean_dojo_v2.lean_dojo.data_extraction.traced_data import (
    TracedFile as V2TracedFile,
    get_code_without_comments as v2_code_without_comments,
)


_original_convert_pos = LeanFile.convert_pos


def _convert_pos(self, byte_idx):
    if isinstance(byte_idx, dict):
        byte_idx = byte_idx["byteIdx"]
    return _original_convert_pos(self, byte_idx)


LeanFile.convert_pos = _convert_pos

_original_traced_repo_path = dojo_module.get_traced_repo_path


def _resolved_traced_repo_path(repo):
    return _original_traced_repo_path(repo).resolve()


dojo_module.get_traced_repo_path = _resolved_traced_repo_path

_v2_repo_cache = {}


def _locate_v2_traced_file(self, traced_repo_path):
    repo_key = (self.repo.url, self.repo.commit)
    repo = _v2_repo_cache.get(repo_key)
    if repo is None:
        # The URL and immutable commit are already resolved by the outer LeanDojo
        # object. Construct the frozen v2 value without a redundant GitHub query.
        repo = object.__new__(V2LeanGitRepo)
        object.__setattr__(repo, "url", repo_key[0])
        object.__setattr__(repo, "commit", repo_key[1])
        object.__setattr__(repo, "repo", None)
        object.__setattr__(repo, "lean_version", "v4.33.0-rc1")
        object.__setattr__(repo, "repo_type", V2RepoType.GITHUB)
        _v2_repo_cache[repo_key] = repo
    root = Path(traced_repo_path).resolve()
    source = root / self.file_path
    json_path = (root / ".lake" / "build" / "ir" /
                 self.file_path.with_suffix(".ast.json"))
    if not source.exists() or not json_path.exists():
        raise DojoInitError(f"missing current trace for {self.file_path}")
    traced_file = V2TracedFile.from_traced_file(root, json_path, repo)
    traced_file.traced_repo = SimpleNamespace(repo=repo, dependencies={})
    return traced_file


def _get_v2_modified_proof(self, traced_file):
    traced_theorem = next(
        (theorem for theorem in traced_file.get_traced_theorems()
         if theorem.theorem.full_name == self.entry.full_name),
        None,
    )
    if traced_theorem is None:
        raise DojoInitError(f"failed to locate theorem {self.entry.full_name}")
    proof_start, proof_end = traced_theorem.locate_proof()
    lean_file = traced_file.lean_file
    code_before = v2_code_without_comments(
        lean_file, lean_file.start_pos, traced_theorem.start, traced_file.comments
    )
    declaration = v2_code_without_comments(
        lean_file, traced_theorem.start, proof_start, traced_file.comments
    ).strip()
    if declaration.endswith(" where"):
        raise DojoInitError("cannot interact with a declaration ending in `where`")
    if not declaration.endswith(":="):
        declaration += " := "
    if code_before.startswith("module\n"):
        code_before = code_before.replace(
            "module\n", "module\n\npublic import Lean4Repl\n", 1
        )
        imports = ""
    else:
        imports = self._get_imports()
    return str(
        imports
        + code_before
        + "\n\nset_option maxHeartbeats 0 in\n"
        + declaration
        + "by\n  lean_dojo_repl\n  sorry\n"
        + lean_file[proof_end:]
    )


Dojo._locate_traced_file = _locate_v2_traced_file
Dojo._get_modified_proof = _get_v2_modified_proof

_original_enter = Dojo.__enter__


def _enter_with_compile_diagnostic(self):
    try:
        return _original_enter(self)
    except DojoInitError:
        modified = getattr(getattr(self, "modified_file", None), "name", None)
        if modified and Path(modified).exists():
            root = _resolved_traced_repo_path(self.repo)
            relative = Path(modified).relative_to(root)
            preview = Path(modified).read_text(encoding="utf-8")[:800]
            print(f"[current-replay] modified-file preview:\n{preview}",
                  file=sys.stderr, flush=True)
            result = subprocess.run(
                ["lake", "env", "lean", str(relative)], cwd=root,
                capture_output=True, text=True, timeout=300,
            )
            diagnostic = (result.stdout + "\n" + result.stderr).strip()
            if diagnostic:
                print(f"[current-replay] compile diagnostic:\n{diagnostic[-4000:]}",
                      file=sys.stderr, flush=True)
        raise


Dojo.__enter__ = _enter_with_compile_diagnostic

import audit.replay as replay_module

_old_repo_cache = {}


def _local_old_repo(_module, url, commit):
    key = (url, commit)
    repo = _old_repo_cache.get(key)
    if repo is None:
        repo = object.__new__(OldLeanGitRepo)
        object.__setattr__(repo, "url", url)
        object.__setattr__(repo, "commit", commit)
        object.__setattr__(repo, "repo", None)
        object.__setattr__(repo, "lean_version", "v4.33.0-rc1")
        _old_repo_cache[key] = repo
    return repo


replay_module.lean_git_repo = _local_old_repo

from tools.reconv_stageb import main


if __name__ == "__main__":
    raise SystemExit(main())
