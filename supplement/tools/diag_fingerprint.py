"""Narrow down which part of the production fingerprint metaprogram is unusable.

`diag_runtac.py` established that `run_tac` accepts multi-line programs and that `throwError`
carries a payload back, but that `Lean.Meta.getSimpTheorems` raises `internal exception #3` and
the full program does not return in reasonable time.

Two things to separate:
  * does the simp digest merely FAIL (caught by its try/catch, field degrades to NA), or does it
    HANG (in which case the try/catch cannot help and the field must be dropped or made lazy)?
  * is the rest of the ladder affordable on its own?

`internal exception #3` is Lean's `Exception.internal`, which is used for control flow and is
NOT reliably caught by a `try ... catch _` in `TacticM` — so a try/catch around it is not
sufficient protection, which is exactly what the production program assumes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = "https://github.com/yangky11/lean4-example"
COMMIT = "7b6ecb9ad4829e4e73600a3329baeb3b5df8d23f"
PER_PROBE_TIMEOUT = 60

RENDER_PRELUDE = """run_tac (show Lean.Elab.Tactic.TacticM Unit from do
  let g ← Lean.Elab.Tactic.getMainGoal
  let render := fun (f : Lean.Options → Lean.Options) => do
    return toString (← Lean.withOptions f (Lean.Meta.ppGoal g))
  let phiDeep       ← render (fun o => (o.setBool `pp.deepTerms true).setBool `pp.proofs false)
  let phiDeepProofs ← render (fun o => (o.setBool `pp.deepTerms true).setBool `pp.proofs true)
  let phiAllShallow ← render (fun o => ((o.setBool `pp.all true).setBool `pp.notation false).setBool `pp.proofs false)
  let phiFull       ← render (fun o => (((o.setBool `pp.all true).setBool `pp.notation false).setBool `pp.deepTerms true).setBool `pp.proofs false)
  let phiFullProofs ← render (fun o => (((o.setBool `pp.all true).setBool `pp.notation false).setBool `pp.deepTerms true).setBool `pp.proofs true)"""

def _direct(n: int) -> str:
    """n renderings written out directly — no higher-order lambda."""
    opts = [
        "(fun o => (o.setBool `pp.deepTerms true).setBool `pp.proofs false)",
        "(fun o => (o.setBool `pp.deepTerms true).setBool `pp.proofs true)",
        "(fun o => ((o.setBool `pp.all true).setBool `pp.notation false).setBool `pp.proofs false)",
        "(fun o => (((o.setBool `pp.all true).setBool `pp.notation false).setBool `pp.deepTerms true).setBool `pp.proofs false)",
        "(fun o => (((o.setBool `pp.all true).setBool `pp.notation false).setBool `pp.deepTerms true).setBool `pp.proofs true)",
    ][:n]
    lets = "\n".join(
        f"  let r{i} := toString (← Lean.withOptions {o} (Lean.Meta.ppGoal g))"
        for i, o in enumerate(opts))
    fields = ",\n    ".join(f'("r{i}", Lean.Json.str r{i})' for i in range(n))
    return f"""run_tac (show Lean.Elab.Tactic.TacticM Unit from do
  let g ← Lean.Elab.Tactic.getMainGoal
{lets}
  let j := Lean.Json.mkObj [
    {fields}]
  throwError s!"FPRINT_JSON:{{j.compress}}")"""


VARIANTS = {
    "F. 1 rendering, direct (control)": _direct(1),
    "G. 2 renderings, direct": _direct(2),
    "H. 3 renderings, direct": _direct(3),
    "I. 5 renderings, direct (no lambda)": _direct(5),

    "A. renderings only (no env)": RENDER_PRELUDE + """
  let j := Lean.Json.mkObj [
    ("phi_deep", Lean.Json.str phiDeep),
    ("phi_deep_proofs", Lean.Json.str phiDeepProofs),
    ("phi_all_shallow", Lean.Json.str phiAllShallow),
    ("phi_full", Lean.Json.str phiFull),
    ("phi_full_proofs", Lean.Json.str phiFullProofs)]
  throwError s!"FPRINT_JSON:{j.compress}")""",

    "B. renderings + ns/open/linst": RENDER_PRELUDE + """
  let nsPart ← (try do pure s!"ns={← Lean.getCurrNamespace}" catch _ => pure "ns=NA")
  let openPart ← (try do
     let ds ← Lean.getOpenDecls
     pure s!"open={((ds.map toString).toArray.qsort (· < ·)).toList}"
   catch _ => pure "open=NA")
  let linstPart ← (try do
     let insts ← Lean.Meta.getLocalInstances
     let strs ← insts.toList.mapM (fun li => do
       let t ← Lean.instantiateMVars (← Lean.Meta.inferType li.fvar)
       return s!"{li.className}:{toString (← Lean.Meta.ppExpr t)}")
     pure s!"linst={((strs.toArray.qsort (· < ·)).toList)}"
   catch _ => pure "linst=NA")
  let j := Lean.Json.mkObj [
    ("phi_full", Lean.Json.str phiFull),
    ("env_behavior", Lean.Json.str ("|".intercalate [nsPart, openPart, linstPart]))]
  throwError s!"FPRINT_JSON:{j.compress}")""",

    "C. simp digest INSIDE try/catch": """run_tac (show Lean.Elab.Tactic.TacticM Unit from do
  let simpPart ← (try do
     let s ← Lean.Meta.getSimpTheorems
     let names := ((s.lemmaNames.toList.map (fun o => toString o.key)).toArray.qsort (· < ·))
     pure s!"simpN={names.size}"
   catch _ => pure "simpN=NA")
  throwError s!"FPRINT_JSON:CAUGHT[{simpPart}]")""",

    "D. simp via getSimpExtension?": """run_tac (show Lean.Elab.Tactic.TacticM Unit from do
  let part ← (try do
     let env ← Lean.getEnv
     pure s!"envConsts={env.constants.size}"
   catch _ => pure "env=NA")
  throwError s!"FPRINT_JSON:ENV[{part}]")""",
}


def main() -> int:
    from lean_dojo import Dojo, LeanGitRepo, Theorem, trace
    from audit.extract import _repo_own_files
    from audit.fingerprint import TACTIC_SRC_INLINE, _error_message, _extract_json

    repo = LeanGitRepo(REPO, COMMIT)
    traced = trace(repo)
    own, _ = _repo_own_files(traced)
    tt = next(c for tf in own for c in tf.get_traced_theorems() if c.get_traced_tactics())
    thm = Theorem(repo, tt.theorem.file_path, tt.theorem.full_name)
    print(f"theorem: {tt.theorem.full_name}\n" + "=" * 78, flush=True)

    probes = [(k, v) for k, v in VARIANTS.items() if k[0] in "FGHIA"]

    for label, src in probes:
        t0 = time.time()
        try:
            ctx = Dojo(thm, timeout=PER_PROBE_TIMEOUT)
            dojo, state = ctx.__enter__()
        except Exception as e:
            print(f"{label:34} DOJO-OPEN-FAIL {type(e).__name__}", flush=True)
            continue
        try:
            res = dojo.run_tac(state, src)
            msg = _error_message(res) or ""
            data = _extract_json(msg) if "FPRINT_JSON:" in msg else None
            dt = time.time() - t0
            if data is not None:
                keys = ",".join(sorted(data))
                print(f"{label:34} {dt:6.1f}s  OK json({len(str(data))}B) keys=[{keys}]", flush=True)
            elif "FPRINT_JSON:" in msg:
                i = msg.find("FPRINT_JSON:")
                print(f"{label:34} {dt:6.1f}s  MARKER {msg[i:i+110]!r}", flush=True)
            else:
                print(f"{label:34} {dt:6.1f}s  {type(res).__name__} {msg[:100]!r}", flush=True)
        except Exception as e:
            print(f"{label:34} {time.time()-t0:6.1f}s  RAISED {type(e).__name__}", flush=True)
        finally:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
