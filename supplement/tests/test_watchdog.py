"""Tests for the wall-clock watchdog (replay.py).

A 200-file probe hung for 3 hours with four Lean subprocesses alive 2+ hours each on a 20s
`Dojo(timeout=...)`, having consumed 0.2s CPU apiece — blocked in a pipe read that LeanDojo's
own timeout does not cover. The run produced no output and no indication of where it stopped.
These tests pin the guard that replaces it.
"""
import time

import pytest

from audit.replay import WatchdogTimeout, deadline, kill_orphan_lean


def test_fires_on_a_blocking_call():
    t0 = time.time()
    with pytest.raises(WatchdogTimeout):
        with deadline(1, "blocking op"):
            time.sleep(20)
    assert time.time() - t0 < 3, "must interrupt promptly, not wait for the call to return"


def test_does_not_fire_when_the_op_completes():
    with deadline(5, "fast op"):
        time.sleep(0.05)


def test_timer_is_disarmed_after_exit():
    """A leaked itimer would raise inside unrelated later code."""
    with deadline(1, "op"):
        pass
    time.sleep(1.5)          # would fire here if the timer survived


def test_timer_is_disarmed_after_firing():
    with pytest.raises(WatchdogTimeout):
        with deadline(1, "op"):
            time.sleep(5)
    time.sleep(1.5)          # no stray alarm from the fired timer


def test_repeated_timeouts_restore_the_handler():
    for i in range(3):
        with pytest.raises(WatchdogTimeout):
            with deadline(1, f"op{i}"):
                time.sleep(5)


def test_message_names_the_operation():
    """The log line has to identify WHICH call blew its budget — the hung run gave no clue."""
    with pytest.raises(WatchdogTimeout, match="Dojo open Foo.bar"):
        with deadline(1, "Dojo open Foo.bar"):
            time.sleep(5)


def test_zero_or_negative_budget_is_a_noop():
    with deadline(0, "disabled"):
        time.sleep(0.05)


def test_orphan_reaper_is_safe_with_no_children():
    assert kill_orphan_lean() == 0


def test_lean_git_repo_is_cached():
    """Constructing a LeanGitRepo calls the GitHub API. Doing it per session open meant one
    network round-trip per theorem — a transient `Connection refused` aborted a whole run that
    otherwise needed no network, since the trace is cached locally."""
    from audit import replay

    calls = []

    class _FakeLdj:
        def LeanGitRepo(self, url, commit):          # noqa: N802 - mirrors lean_dojo's API
            calls.append((url, commit))
            return f"repo:{url}@{commit}"

    replay._REPO_CACHE.clear()
    ldj = _FakeLdj()
    for _ in range(5):
        replay.lean_git_repo(ldj, "https://example/mathlib4", "abc123")
    assert len(calls) == 1, f"expected 1 API call, got {len(calls)}"

    replay.lean_git_repo(ldj, "https://example/mathlib4", "def456")
    assert len(calls) == 2, "a different commit must not reuse the cached handle"
    replay._REPO_CACHE.clear()


def test_watchdog_escapes_broad_exception_handlers():
    """The 200-file probe hung 108 minutes with the watchdog armed and zero firings: SIGALRM
    raised inside a LeanDojo call whose `except Exception` retry logic swallowed it. A guard
    that library code can catch is not a guard."""
    fired = False
    try:
        with deadline(1, "swallowed op"):
            try:
                time.sleep(5)
            except Exception:          # library-style catch-all, as in pexpect retry loops
                pass
    except WatchdogTimeout:
        fired = True
    assert fired, "WatchdogTimeout must not be catchable by `except Exception`"


def test_watchdog_is_not_an_exception_subclass():
    assert not issubclass(WatchdogTimeout, Exception)
    assert issubclass(WatchdogTimeout, BaseException)


def test_descendant_walk_finds_grandchildren():
    """`lake env lean` makes lean a GRANDchild, so a direct-children-only kill missed it and
    leaked Lean processes filled MAX_OPEN_SESSIONS."""
    import os
    import subprocess
    import time as _t

    from audit.replay import _descendants

    # bash -> sleep : the sleep is a grandchild of this process
    p = subprocess.Popen(["/bin/bash", "-c", "sleep 30"])
    _t.sleep(1.0)
    try:
        desc = _descendants(os.getpid())
        assert p.pid in desc, "direct child missing"
        kids = subprocess.run(["pgrep", "-P", str(p.pid)], capture_output=True, text=True).stdout
        grand = [int(x) for x in kids.split()]
        for g in grand:
            assert g in desc, f"grandchild {g} not found — this is the leak"
    finally:
        p.kill()
        p.wait()
