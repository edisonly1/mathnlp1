"""Install the version-adapted extraction and replay shims in a Mathlib checkout.

LeanDojo-v2 supplies the extractor required by ``current_blind_extract.py`` but no
interactive tactic API.  The replay audit therefore compiles the evaluated LeanDojo 2.0.2
REPL against the current Lean toolchain.  ``current_replay/Lean4Repl.lean`` contains the
minimal Lean 4.33 module changes and communicates through the controlling pseudoterminal,
because module elaboration buffers the ordinary output streams.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def _install(source: Path, target: Path, replace: bool) -> str:
    if target.exists():
        if target.read_bytes() == source.read_bytes():
            return "already current"
        if not replace:
            raise SystemExit(
                f"refusing to overwrite {target}; pass --replace after inspecting it"
            )
    shutil.copyfile(source, target)
    return "installed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="current Mathlib checkout")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / "lakefile.toml").exists() and not (repo / "lakefile.lean").exists():
        raise SystemExit(f"no Lake project found at {repo}")
    assets = Path(__file__).resolve().parent.parent / "current_replay"
    for name in ("ExtractData.lean", "Lean4Repl.lean"):
        status = _install(assets / name, repo / name, args.replace)
        print(f"{name}: {status}")

    # The v2 extractor resolves Lean source metadata through this conventional package path.
    lean = Path(subprocess.check_output(
        ["lake", "env", "which", "lean"], cwd=repo, text=True
    ).strip()).resolve()
    lean_package = repo / ".lake" / "packages" / "lean4"
    if not lean_package.exists():
        lean_package.symlink_to(lean.parent.parent, target_is_directory=True)
        print(f"lean4 package link: {lean_package} -> {lean.parent.parent}")

    output = repo / ".lake" / "build" / "lib" / "lean" / "Lean4Repl.olean"
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["lake", "env", "lean", "-o", str(output), "Lean4Repl.lean"],
        cwd=repo, check=True,
    )
    print(f"compiled replay module: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
