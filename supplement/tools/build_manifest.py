"""Build a deterministic SHA-256 manifest for the supplement directory."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path != MANIFEST
    )
    lines = [
        f"{file_digest(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in files
    ]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} entries to {MANIFEST.name}")


if __name__ == "__main__":
    main()
