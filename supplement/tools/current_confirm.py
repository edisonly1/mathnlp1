"""Confirm current-stack reconvergence candidates through the replay adapter."""
from __future__ import annotations

# Importing the adapter installs the v2 trace locator and current-Lean transport patches.
from tools import current_reconvergence as _current_adapter  # noqa: F401
from tools.reconv_confirm import main


if __name__ == "__main__":
    raise SystemExit(main())
