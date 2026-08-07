"""Smoke-test the LeanDojo-dependent half of the harness on a small repo.

Everything in `audit/` except `extract.py`, `replay.py`, and `fingerprint.fingerprint_state`
is unit-tested off-box. Those three are the parts that only run against a live Dojo, and the
riskiest assumption in the whole harness lives there:

    the fingerprint metaprogram is a multi-line, indentation-sensitive `do` block submitted as
    a tactic string, and its result comes back as a deliberately-thrown ERROR MESSAGE.

Three ways that fails silently, all of which look like "no aliases found" rather than an error:

  1. `run_tac` rejects a multi-line tactic → every state degrades to the DEGRADED fallback.
  2. LeanDojo truncates long error messages → brace-matching fails on big goals, so exactly
     the large states (the ones most likely to alias) are the ones that lose their fingerprint.
  3. The payload parses but a field is missing → attribution silently loses a mechanism.

This script runs the real thing end to end on a tiny repo and reports payload sizes, so the
truncation headroom for Mathlib-scale goals can be estimated before committing to a long trace.

Usage:
    python tools/smoke_dojo.py                       # defaults to yangky11/lean4-example
    python tools/smoke_dojo.py --repo URL --commit SHA
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_REPO = "https://github.com/yangky11/lean4-example"
DEFAULT_COMMIT = "7761283d0aed994cd1c7e893786212d2a01d159e"

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
_results: list[tuple[str, str, str]] = []


def check(name: str, status: str, detail: str = "") -> None:
    _results.append((status, name, detail))
    icon = {"PASS": "  ok ", "FAIL": " FAIL", "WARN": " warn"}[status]
    print(f"{icon}  {name}" + (f"  — {detail}" if detail else ""), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--commit", default=DEFAULT_COMMIT)
    ap.add_argument("--channel", default="throwError", choices=["throwError", "frontend"])
    ap.add_argument("--dojo-timeout", type=int, default=120,
                    help="Dojo startup+tactic budget; Mathlib needs far more than the default")
    ap.add_argument("--max-files", type=int, default=0,
                    help="cap how many repo files are scanned for a theorem (0 = all)")
    ap.add_argument("--goal-rank", type=int, default=0,
                    help="0 = theorem with most tactics; higher picks progressively smaller")
    args = ap.parse_args()

    print(f"repo   : {args.repo}@{args.commit[:10]}")
    print(f"channel: {args.channel}")
    print(f"cache  : {os.environ.get('CACHE_DIR', '~/.cache/lean_dojo')}")
    print(f"token  : {'set' if os.environ.get('GITHUB_ACCESS_TOKEN') else 'NOT SET (60 req/hr)'}")
    print("-" * 72, flush=True)

    # ---------------------------------------------------------------- import
    try:
        from lean_dojo import Dojo, LeanGitRepo, Theorem, trace
        import lean_dojo
        check("import lean_dojo", PASS, getattr(lean_dojo, "__version__", "?"))
    except Exception as e:
        check("import lean_dojo", FAIL, repr(e))
        return 1

    from audit.fingerprint import (MAIN_FIELDS, MAX_TACTIC_BYTES, PROBES, _MULTILINE,
                                   _error_message, assemble, collect_fields,
                                   fingerprint_state, parse_probe)

    # ---------------------------------------------------------------- trace
    t0 = time.time()
    try:
        repo = LeanGitRepo(args.repo, args.commit)
        traced = trace(repo)
        check("trace repo", PASS, f"{time.time() - t0:.0f}s")
    except Exception as e:
        check("trace repo", FAIL, repr(e)[:300])
        return 1

    # ------------------------------------------------------- pick a theorem
    # Dependency files must be excluded: Dojo can only open theorems from the audited repo.
    from audit.extract import _repo_own_files
    own_files, skipped = _repo_own_files(traced)
    check("filter out dependency files", PASS,
          f"{len(own_files)} repo files, {skipped} dependency files skipped")

    scan = own_files if not args.max_files else own_files[:args.max_files]
    if args.max_files and len(own_files) > args.max_files:
        check("cap file scan", WARN,
              f"scanning {len(scan)} of {len(own_files)} repo files "
              f"(--max-files); theorem choice is drawn from this subset only")
    try:
        thms = []
        for tf in scan:
            for tt in tf.get_traced_theorems():
                tacs = tt.get_traced_tactics()
                if tacs:
                    thms.append((tt, tacs))
        if not thms:
            check("find traced theorem with tactics", FAIL, "none in the repo's own files")
            return 1
        thms.sort(key=lambda p: -len(p[1]))
        traced_thm, tactics = thms[min(args.goal_rank, len(thms) - 1)]
        check("find traced theorem with tactics", PASS,
              f"{traced_thm.theorem.full_name} ({len(tactics)} tactics, "
              f"{len(thms)} theorems scanned)")
    except Exception as e:
        check("find traced theorem with tactics", FAIL, repr(e)[:300])
        return 1

    # ------------------------------------------------- state_before present?
    with_state = [t for t in tactics if getattr(t, "state_before", None)]
    if with_state:
        check("traced tactics expose state_before", PASS,
              f"{len(with_state)}/{len(tactics)}")
    else:
        check("traced tactics expose state_before", FAIL,
              "extract.py yields nothing without it")

    # ---------------------------------------------------------------- Dojo
    thm = Theorem(repo, traced_thm.theorem.file_path, traced_thm.theorem.full_name)
    try:
        ctx = Dojo(thm, timeout=args.dojo_timeout)
        dojo, state = ctx.__enter__()
        check("open Dojo", PASS, type(state).__name__)
    except Exception as e:
        check("open Dojo", FAIL, repr(e)[:300])
        return 1

    exit_code = 0
    try:
        pp = getattr(state, "pp", "")
        print(f"\n    initial goal ({len(pp)} bytes):\n      " +
              pp.replace("\n", "\n      ") + "\n", flush=True)

        # ------------------------------------------- THE critical assumption
        oversize = [n for n, src in PROBES.items()
                    if len(src.encode("utf-8")) >= MAX_TACTIC_BYTES]
        check("all probes under the ~1KB run_tac limit",
              PASS if not oversize else FAIL,
              f"{len(PROBES)} probes, max "
              f"{max(len(s.encode()) for s in PROBES.values())}B"
              if not oversize else f"oversize: {oversize}")

        t_probe = time.time()
        fields = collect_fields(dojo, state)
        dt = time.time() - t_probe
        missing = [n for n in PROBES if n not in fields]
        check("collect every probe", PASS if not missing else WARN,
              f"{len(fields)}/{len(PROBES)} fields in {dt:.1f}s "
              f"({dt / max(1, len(PROBES)):.2f}s per probe)"
              + (f"; missing {missing}" if missing else ""))

        if not fields.get("phi_full"):
            check("phi_full collected", FAIL, "the fingerprint cannot be built without it")
            exit_code = 1
        else:
            data = assemble(fields)
            check("phi_full collected", PASS, f"{len(data['phi_full'])}B")
            for label in ("phi_deep", "phi_all_shallow", "phi_full"):
                v = data.get(label, "")
                first = v.splitlines()[0][:58] if v else ""
                print(f"      {label:16} {len(v):5}B  {first}")
            print(f"      {'env_behavior':16} {len(data['env_behavior']):5}B  "
                  f"{data['env_behavior'][:58]}")

            try:
                _t0 = time.time()
                fp = fingerprint_state(dojo, state, args.channel, pp)
                ok = bool(fp.phi_full) and not fp.channel.endswith("+degraded")
                check("fingerprint_state end-to-end", PASS if ok else FAIL,
                      f"{time.time()-_t0:.1f}s  f_core={fp.f_core[:10]} f_env={fp.f_env[:10]}")
                check("env_behavior excludes module identity",
                      PASS if "mod=" not in fp.env_digest else FAIL,
                      f"provenance kept separately: {fp.env_provenance!r}")
            except Exception as e:
                check("fingerprint_state end-to-end", FAIL, repr(e)[:200])
                exit_code = 1

        # ------------------------------------------------ ordinary tactics
        try:
            r = dojo.run_tac(state, "rfl")
            check("run_tac ordinary tactic", PASS, type(r).__name__)
        except Exception as e:
            check("run_tac ordinary tactic", FAIL, repr(e)[:200])

        # ------------------------------- fan-out from one state (session reuse)
        try:
            kinds = []
            for tac in ["rfl", "simp", "assumption", "constructor"]:
                kinds.append(type(dojo.run_tac(state, tac)).__name__)
            check("fan out panel from one restored state", PASS, ", ".join(kinds))
        except Exception as e:
            check("fan out panel from one restored state", FAIL, repr(e)[:200])
            exit_code = 1

    finally:
        try:
            ctx.__exit__(None, None, None)
        except Exception:
            pass

    # ---------------------------------------------------------------- summary
    print("-" * 72)
    n_fail = sum(1 for s, _, _ in _results if s == FAIL)
    n_warn = sum(1 for s, _, _ in _results if s == WARN)
    print(f"{len(_results)} checks: {len(_results) - n_fail - n_warn} pass, "
          f"{n_warn} warn, {n_fail} fail")
    if n_fail:
        print("\nThe throwError channel is not usable as-is. Fallbacks, in order:")
        print("  1. channel='frontend' — put lean_fingerprint/Fingerprint.lean on the import")
        print("     path and inject the single tactic `fingerprint_throw`.")
        print("  2. Return hashes + short digests from Lean instead of full renderings, and")
        print("     fetch full renderings only for candidate-class members.")
    return exit_code or (1 if n_fail else 0)


if __name__ == "__main__":
    raise SystemExit(main())
