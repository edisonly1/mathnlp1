"""Component 5 — Replay worker (blueprint §7.7).

Restores a sampled state (by fast-forwarding the human proof prefix), runs a probe tactic under
fixed resources, repeats it, and canonicalizes the outcome Ψ (blueprint §4.2). Timeouts are
recorded separately and NEVER counted as logical failure (§7.6). Non-reproducible outcomes
invalidate the witness (§7.7).

## Session reuse

A naive implementation opens a fresh `Dojo` for every (state, tactic, repeat). Against Mathlib
a Dojo start costs tens of seconds and re-runs the whole proof prefix, so a single alias class
with 8 representatives and a 7-tactic panel at 2 repeats costs 100+ starts — roughly two orders
of magnitude more than a 20,000-state pilot can afford on one machine.

LeanDojo tactic states are immutable handles within a session, so every tactic in the panel can
be fanned out from one restored state. This module therefore keeps a small LRU of open sessions
keyed by theorem and caches the fast-forwarded state per (theorem, prefix).

Isolation is preserved where the blueprint actually requires it:
  * `repeats` within a pass detect intra-session nondeterminism;
  * `reset_sessions()` drops every open session, so the orchestrator's divergence-verification
    pass (§10.2 cond 5) re-runs each detected divergence in genuinely fresh processes;
  * `isolation: "process"` in config restores one-Dojo-per-tactic for the manual audit (§8.3),
    where full isolation matters more than throughput.

LeanDojo is imported lazily so the module stays importable off-box.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import signal
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

from .config import Config
from .fingerprint import fingerprint_state
from .schema import OutcomeKind, ReplayOutcome, StateRecord

MAX_OPEN_SESSIONS = 4


class WatchdogTimeout(BaseException):
    """A Dojo operation blew its wall-clock budget and was forcibly abandoned.

    Inherits BaseException, NOT Exception, on purpose. The 200-file probe hung for 108 minutes
    with the watchdog armed and ZERO firings: SIGALRM fired and raised inside a LeanDojo/pexpect
    call whose own `except Exception` retry logic swallowed it, so the read was simply re-entered
    and the alarm — already disarmed — never came back. Signal-based interruption is only useful
    if the exception can escape the library it interrupts; BaseException is what gets it out,
    the same reason KeyboardInterrupt is defined that way.

    Every `except WatchdogTimeout` in this module is explicit, so nothing catches it by accident.
    """


def _log(msg: str) -> None:
    print(f"[replay] {msg}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def deadline(seconds: float, what: str):
    """Hard wall-clock bound on a blocking Dojo call.

    LeanDojo's own `Dojo(timeout=...)` does NOT bound session startup: a 200-file probe hung
    with four Lean subprocesses alive for 2+ hours each on a 20-second timeout, having consumed
    0.2s of CPU apiece — blocked on a REPL that never answered. The Python side sat in a pipe
    read, so nothing short of an external signal could break it.

    SIGALRM interrupts the blocking read. Only valid on the main thread; elsewhere this is a
    no-op (the caller keeps LeanDojo's weaker guarantee rather than crashing).
    """
    if threading.current_thread() is not threading.main_thread() or seconds <= 0:
        yield
        return

    def _fire(_signum, _frame):
        raise WatchdogTimeout(f"{what} exceeded {seconds:.0f}s wall clock")

    old = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _descendants(pid: int) -> list:
    """Every descendant pid, not just direct children.

    `lake env lean` means `lake` is our child and `lean` is its child, so a direct-children-only
    kill leaves the actual Lean process running. That is why the census ended with 6 live Lean
    processes against a MAX_OPEN_SESSIONS cap of 4 — leaked grandchildren filled the pool.
    """
    try:
        out = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True,
                             text=True, timeout=15).stdout
    except Exception:
        return []
    children: dict = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                children.setdefault(int(parts[1]), []).append(int(parts[0]))
            except ValueError:
                continue
    out_pids, stack = [], [pid]
    while stack:
        for c in children.get(stack.pop(), []):
            out_pids.append(c)
            stack.append(c)
    return out_pids


def kill_orphan_lean(descendants_of: Optional[int] = None) -> int:
    """SIGKILL every Lean/lake descendant left behind by an abandoned session.

    A timed-out session leaks its Lean subprocess. Those fill MAX_OPEN_SESSIONS and every later
    state blocks forever, which is how two multi-hour runs produced no further output.
    """
    root = descendants_of or os.getpid()
    killed = 0
    for cpid in _descendants(root):
        try:
            cmd = subprocess.run(["ps", "-o", "command=", "-p", str(cpid)],
                                 capture_output=True, text=True, timeout=5).stdout
        except Exception:
            continue
        if "lake env lean" in cmd or "/bin/lean" in cmd or "repl" in cmd:
            try:
                os.kill(cpid, signal.SIGKILL)
                killed += 1
            except Exception:
                pass
    return killed


def _lazy_lean_dojo():
    import lean_dojo  # noqa
    return lean_dojo


_REPO_CACHE: dict = {}


def lean_git_repo(ldj, url: str, commit: str):
    """Cache `LeanGitRepo` per (url, commit).

    Constructing one calls the GitHub API. This module built a fresh repo object on every
    session open — i.e. one network round-trip per theorem, during a phase that otherwise runs
    entirely from the 18 GB local trace cache. A transient `Connection refused` to
    api.github.com killed a run outright, and at pilot scale it is also thousands of needless
    API calls against a 5,000/hr budget.
    """
    key = (url, commit)
    if key not in _REPO_CACHE:
        _REPO_CACHE[key] = ldj.LeanGitRepo(url, commit)
    return _REPO_CACHE[key]


class _Session:
    """One open Dojo plus the states fast-forwarded inside it."""

    def __init__(self, ctx, dojo, base_state):
        self.ctx = ctx
        self.dojo = dojo
        self.base_state = base_state
        self.states: dict[tuple, Any] = {}
        self.poisoned = False        # a timed-out session must never be reused

    def close(self, force: bool = False) -> None:
        # Closing a wedged session can itself block, so bound it and then kill the subprocess.
        try:
            with deadline(20, "session close"):
                self.ctx.__exit__(None, None, None)
        except Exception:
            force = True
        if force:
            kill_orphan_lean()


class DojoReplayer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.channel = cfg.fingerprint_channel
        self.isolation = getattr(cfg, "replay_isolation", "session")
        self._sessions: "OrderedDict[str, _Session]" = OrderedDict()
        # Wall-clock budgets enforced by `deadline`, independent of LeanDojo's own timeout.
        self.open_timeout = getattr(cfg, "session_open_timeout_s", 180)
        self.op_timeout = getattr(cfg, "tactic_watchdog_s", 90)
        self.stats: dict = {
            "session_opens": 0, "session_open_errors": 0, "session_open_timeouts": 0,
            "session_open_seconds": 0.0, "op_timeouts": 0, "orphans_killed": 0,
        }

    # ------------------------------------------------------------------ #
    def reset_sessions(self) -> None:
        """Close every open session. The next replay therefore runs in a fresh process —
        this is what makes the verification pass an independent rerun (§7.7 step 2)."""
        for s in self._sessions.values():
            s.close(force=s.poisoned)
        self._sessions.clear()
        n = kill_orphan_lean()
        if n:
            self.stats["orphans_killed"] += n
            _log(f"reaped {n} orphaned Lean process(es) on session reset")

    def __enter__(self) -> "DojoReplayer":
        return self

    def __exit__(self, *exc) -> None:
        self.reset_sessions()

    # ------------------------------------------------------------------ #
    def _theorem_key(self, rec: StateRecord) -> str:
        p = rec.provenance
        return f"{p.repo_url}@{p.repo_commit}::{p.file_path}::{p.theorem_full_name}"

    def _open_session(self, rec: StateRecord) -> Optional[_Session]:
        ldj = _lazy_lean_dojo()
        p = rec.provenance
        repo = lean_git_repo(ldj, p.repo_url, p.repo_commit)
        thm = ldj.Theorem(repo, p.file_path, p.theorem_full_name)
        # NB: the kwarg is `timeout`, not `hard_timeout`. `hard_timeout` does not exist in any
        # released lean-dojo (checked 2.0.2 and 4.20.0) — passing it raises TypeError on every
        # Dojo open, which the replayer would have swallowed as "session failed to open" and
        # reported as CRASH for every single state.
        ctx = ldj.Dojo(thm, timeout=self.cfg.hard_timeout_s)
        t0 = time.time()
        try:
            # Startup is the operation LeanDojo's own timeout does not cover, and deep Mathlib
            # files (Probability/Kernel/..., RingTheory/...) are where it hangs.
            with deadline(self.open_timeout, f"Dojo open {p.theorem_full_name[:40]}"):
                dojo, state = ctx.__enter__()
        except WatchdogTimeout as e:
            self.stats["session_open_timeouts"] += 1
            _log(f"WATCHDOG {e}; killing {kill_orphan_lean()} orphaned Lean process(es)")
            return None
        except Exception:
            self.stats["session_open_errors"] += 1
            return None
        self.stats["session_opens"] += 1
        self.stats["session_open_seconds"] += time.time() - t0
        return _Session(ctx, dojo, state)

    def _session_for(self, rec: StateRecord) -> Optional[_Session]:
        key = self._theorem_key(rec)
        sess = self._sessions.get(key)
        if sess is not None and not sess.poisoned:
            self._sessions.move_to_end(key)
            return sess
        if sess is not None:                 # poisoned: drop it before opening a replacement
            self._sessions.pop(key, None)
            sess.close(force=True)
        sess = self._open_session(rec)
        if sess is None:
            return None
        self._sessions[key] = sess
        while len(self._sessions) > MAX_OPEN_SESSIONS:
            _, old = self._sessions.popitem(last=False)
            old.close()
        return sess

    def _state_for(self, sess: _Session, rec: StateRecord, ldj) -> Optional[Any]:
        """Fast-forward the human proof prefix once per (theorem, prefix) and cache it."""
        key = tuple(rec.proof_prefix)
        if key in sess.states:
            return sess.states[key]
        state = sess.base_state
        try:
            with deadline(self.op_timeout, "prefix fast-forward"):
                for pre in rec.proof_prefix:
                    state = sess.dojo.run_tac(state, pre)
                    if not isinstance(state, ldj.TacticState):
                        return None
        except WatchdogTimeout as e:
            self.stats["op_timeouts"] += 1
            sess.poisoned = True
            _log(f"WATCHDOG {e}")
            return None
        sess.states[key] = state
        return state

    # ------------------------------------------------------------------ #
    def _classify(self, ldj, dojo, result: Any, wall: float) -> ReplayOutcome:
        TacticState = ldj.TacticState
        ProofFinished = ldj.ProofFinished
        LeanError = getattr(ldj, "LeanError", None)
        ProofGivenUp = getattr(ldj, "ProofGivenUp", None)

        if isinstance(result, ProofFinished):
            return ReplayOutcome(kind=OutcomeKind.COMPLETE, wall_time_s=wall)
        if ProofGivenUp is not None and isinstance(result, ProofGivenUp):
            return ReplayOutcome(kind=OutcomeKind.GAVE_UP, wall_time_s=wall)
        if isinstance(result, TacticState):
            # Nonterminal success — fingerprint the successor as its canonical signature (§4.2).
            try:
                fp = fingerprint_state(dojo, result, self.channel, getattr(result, "pp", ""))
                sig = fp.f_exec
            except Exception:
                sig = "PP:" + hashlib.sha256(getattr(result, "pp", "").encode()).hexdigest()
            return ReplayOutcome(kind=OutcomeKind.SUCC, successor_sig=[sig], wall_time_s=wall)
        # LeanError or anything else → logical failure, UNLESS message says timeout.
        if LeanError is not None and isinstance(result, LeanError):
            msg = getattr(result, "error", "") or str(result)
        else:
            msg = str(result)
        kind = OutcomeKind.TIMEOUT if "timeout" in msg.lower() else OutcomeKind.FAIL
        return ReplayOutcome(kind=kind, wall_time_s=wall,
                             message_digest=hashlib.sha256(msg.encode()).hexdigest()[:16])

    # ------------------------------------------------------------------ #
    def fingerprint_record(self, rec: StateRecord):
        """Fingerprint `rec`'s state (fills rec.fingerprint). Reuses the theorem's session."""
        ldj = _lazy_lean_dojo()
        try:
            sess = self._session_for(rec)
            if sess is None:
                return None
            state = self._state_for(sess, rec, ldj)
            if state is None:
                return None
            with deadline(self.op_timeout, "fingerprint probes"):
                fp = fingerprint_state(sess.dojo, state, self.channel,
                                       rec.observation.phi_state)
            rec.fingerprint = fp
            return fp
        except WatchdogTimeout as e:
            self.stats["op_timeouts"] += 1
            _log(f"WATCHDOG {e}")
            sess = self._sessions.pop(self._theorem_key(rec), None)
            if sess is not None:
                sess.poisoned = True
                sess.close(force=True)
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    def replay_once(self, rec: StateRecord, tactic: str) -> ReplayOutcome:
        ldj = _lazy_lean_dojo()
        t0 = time.time()
        try:
            sess = self._session_for(rec)
            if sess is None:
                return ReplayOutcome(kind=OutcomeKind.CRASH, reproducible=False,
                                     wall_time_s=time.time() - t0,
                                     message_digest="dojo_open_fail")
            state = self._state_for(sess, rec, ldj)
            if state is None:
                # prefix did not reproduce — cannot restore this state; invalidate.
                return ReplayOutcome(kind=OutcomeKind.CRASH, reproducible=False,
                                     wall_time_s=time.time() - t0,
                                     message_digest="prefix_reproduce_fail")
            with deadline(self.op_timeout, f"run_tac {tactic[:32]!r}"):
                result = sess.dojo.run_tac(state, tactic)
            return self._classify(ldj, sess.dojo, result, time.time() - t0)
        except WatchdogTimeout as e:
            # Blown budget: the session is unusable and its Lean process must go, or it fills
            # MAX_OPEN_SESSIONS and blocks every later state.
            self.stats["op_timeouts"] += 1
            _log(f"WATCHDOG {e}")
            dead = self._sessions.pop(self._theorem_key(rec), None)
            if dead is not None:
                dead.poisoned = True
                dead.close(force=True)
                self.stats["orphans_killed"] += 1
            return ReplayOutcome(kind=OutcomeKind.TIMEOUT, reproducible=False,
                                 wall_time_s=time.time() - t0,
                                 message_digest="watchdog_timeout")
        except Exception as e:  # DojoHardTimeoutError, DojoCrashError, init errors
            # A crashed session must not be reused.
            key = self._theorem_key(rec)
            dead = self._sessions.pop(key, None)
            if dead is not None:
                dead.close()
            name = type(e).__name__.lower()
            kind = (OutcomeKind.TIMEOUT
                    if "timeout" in name or "timeout" in str(e).lower()
                    else OutcomeKind.CRASH)
            return ReplayOutcome(kind=kind, reproducible=False, wall_time_s=time.time() - t0,
                                 message_digest=type(e).__name__)

    # ------------------------------------------------------------------ #
    def replay(self, rec: StateRecord, tactic: str, repeats: Optional[int] = None) -> ReplayOutcome:
        """Run the probe `repeats` times; mark non-reproducible if canonical outcomes disagree."""
        repeats = repeats or self.cfg.repeats
        goal_order = self.cfg.goal_order_sensitive
        outcomes = []
        for i in range(repeats):
            if self.isolation == "process" and i > 0:
                self.reset_sessions()   # full isolation: each repeat in a fresh process
            outcomes.append(self.replay_once(rec, tactic))
        canon = {o.canonical(goal_order) for o in outcomes}
        first = outcomes[0]
        first.reproducible = (len(canon) == 1) and all(o.reproducible for o in outcomes)
        return first
