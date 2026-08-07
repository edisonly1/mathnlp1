"""Isolate WHY the injected fingerprint metaprogram fails under `Dojo.run_tac`.

The smoke test showed the metaprogram raising and then wedging the session (subsequent `rfl`
timed out). That confounds several hypotheses, so each probe here runs in a FRESH Dojo and they
escalate one variable at a time:

  1. plain tactic                      — is the Dojo usable at all?
  2. one-line `run_tac`                — does run_tac work at all?
  3. multi-line `run_tac`              — does the protocol survive embedded newlines?
  4. multi-line + `throwError`         — does the error channel carry a payload back?
  5. ppGoal under options              — is the rendering ladder itself affordable?
  6. getSimpTheorems                   — is the simp-set digest the expensive part?
  7. the full production metaprogram   — the real thing

Each probe reports the exception TYPE, which is what distinguishes a parse failure from a
timeout from a protocol violation.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = "https://github.com/yangky11/lean4-example"
COMMIT = "7b6ecb9ad4829e4e73600a3329baeb3b5df8d23f"
TIMEOUT = 600

PROBES: list[tuple[str, str]] = [
    ("1. plain tactic", "rfl"),

    ("2. one-line run_tac", "run_tac (do pure ())"),

    ("3. one-line run_tac + println", 'run_tac (do IO.println "hi")'),

    ("4. multi-line run_tac", """run_tac (do
  let _g ← Lean.Elab.Tactic.getMainGoal
  pure ())"""),

    ("5. multi-line + throwError", """run_tac (show Lean.Elab.Tactic.TacticM Unit from do
  let _g ← Lean.Elab.Tactic.getMainGoal
  throwError "FPRINT_JSON:{\\"probe\\":1}")"""),

    ("6. ppGoal under options", """run_tac (show Lean.Elab.Tactic.TacticM Unit from do
  let g ← Lean.Elab.Tactic.getMainGoal
  let s ← Lean.withOptions (fun o => (o.setBool `pp.all true).setBool `pp.deepTerms true) (Lean.Meta.ppGoal g)
  throwError s!"FPRINT_JSON:{toString s}")"""),

    ("7. getSimpTheorems digest", """run_tac (show Lean.Elab.Tactic.TacticM Unit from do
  let s ← Lean.Meta.getSimpTheorems
  let names := ((s.lemmaNames.toList.map (fun o => toString o.key)).toArray.qsort (· < ·))
  throwError s!"FPRINT_JSON:simpN={names.size}")"""),
]


def main() -> int:
    from lean_dojo import Dojo, LeanGitRepo, Theorem, trace
    from audit.extract import _repo_own_files
    from audit.fingerprint import TACTIC_SRC_INLINE, _error_message, _extract_json

    repo = LeanGitRepo(REPO, COMMIT)
    traced = trace(repo)
    own, _ = _repo_own_files(traced)
    tt = None
    for tf in own:
        for cand in tf.get_traced_theorems():
            if cand.get_traced_tactics():
                tt = cand
                break
        if tt:
            break
    if tt is None:
        print("no theorem found")
        return 1
    thm = Theorem(repo, tt.theorem.file_path, tt.theorem.full_name)
    print(f"theorem: {tt.theorem.full_name}\n" + "=" * 74, flush=True)

    probes = PROBES + [("8. FULL production metaprogram", TACTIC_SRC_INLINE)]

    for label, src in probes:
        t0 = time.time()
        # Fresh Dojo per probe: a wedged session poisons every later probe.
        try:
            ctx = Dojo(thm, timeout=TIMEOUT)
            dojo, state = ctx.__enter__()
        except Exception as e:
            print(f"{label:34} DOJO-OPEN-FAIL {type(e).__name__}", flush=True)
            continue
        try:
            res = dojo.run_tac(state, src)
            kind = type(res).__name__
            msg = _error_message(res) or ""
            payload = _extract_json(msg) if "FPRINT_JSON:" in msg else None
            extra = ""
            if payload is not None:
                extra = f" payload_ok({len(str(payload))}B)"
            elif "FPRINT_JSON:" in msg:
                extra = f" marker_present_but_unparsed({len(msg)}B)"
            elif msg:
                extra = f" msg[{len(msg)}B]={msg[:70]!r}"
            print(f"{label:34} {time.time()-t0:6.1f}s  {kind}{extra}", flush=True)
        except Exception as e:
            print(f"{label:34} {time.time()-t0:6.1f}s  RAISED {type(e).__name__}: "
                  f"{str(e)[:90]!r}", flush=True)
        finally:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
