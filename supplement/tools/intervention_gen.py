"""Matched hidden-structure interventions.

For each mechanism, pairs of compilations differ ONLY in a `show` step that replaces the goal
with a definitionally equal term carrying different hidden syntax. Everything else is held
fixed: theorem-free standalone example, environment (same imports), binders, default rendering,
and probe text. A pair is a VALID intervention when the two variants' pre-probe renderings are
byte-identical (trace comparison) while their structural digests differ; among valid pairs, the
measurement is whether the probe's outcome (failure vs acceptance vs successor state) diverges.

Outcome parsing per variant: FAIL if a compile error occurs at the probe line; otherwise the
post-probe trace hash (or NOGOALS).
"""
from __future__ import annotations
import hashlib, json, re, subprocess, sys
from pathlib import Path

CAPTURE_PRE = '''run_tac (show Lean.Elab.Tactic.TacticM Unit from do
    let g ← Lean.Elab.Tactic.getMainGoal
    g.withContext do
      let d ← g.getDecl
      let fmt ← Lean.Meta.ppGoal g
      let e ← Lean.Meta.mkForallFVars d.lctx.getFVars d.type
      let e ← Lean.instantiateMVars e
      let r ← Lean.Meta.abstractMVars e
      Lean.logInfo s!"PPV:{fmt}\\nPPEND"
      Lean.logInfo s!"DIGV:{r.expr.hash}")'''
CAPTURE_POST = '''first
  | run_tac (show Lean.Elab.Tactic.TacticM Unit from do
      let g ← Lean.Elab.Tactic.getMainGoal
      g.withContext do
        let fmt ← Lean.Meta.ppGoal g
        Lean.logInfo s!"POSTV:{fmt}\\nPOSTEND")
  | run_tac (Lean.logInfo "POSTV:NOGOALS\\nPOSTEND")'''

MECHS = {
 "coercion": {
   "imports": ["Mathlib.Data.ENNReal.Basic", "Mathlib.Data.ENNReal.Operations",
               "Mathlib.Tactic.NormNum", "Mathlib.Tactic.Ring"],
   "pairs": [
     # (binders, goal-as-written, termA, termB, probe)
     ("(x : NNReal) (y : ENNReal)", "(x : ENNReal) ≤ x + y",
      "ENNReal.ofNNReal x ≤ ENNReal.ofNNReal x + y",
      "WithTop.some x ≤ WithTop.some x + y", "simp"),
     ("(x y : NNReal)", "(x : ENNReal) + y = ((x + y : NNReal) : ENNReal)",
      "ENNReal.ofNNReal x + ENNReal.ofNNReal y = ENNReal.ofNNReal (x + y)",
      "WithTop.some x + WithTop.some y = WithTop.some (x + y)", "simp"),
     ("(x : NNReal)", "(x : ENNReal) ≠ ⊤",
      "ENNReal.ofNNReal x ≠ ⊤", "WithTop.some x ≠ ⊤", "simp"),
     ("(x : NNReal)", "0 ≤ (x : ENNReal)",
      "0 ≤ ENNReal.ofNNReal x", "0 ≤ WithTop.some x", "simp"),
     ("(x y : NNReal) (h : x ≤ y)", "(x : ENNReal) ≤ y",
      "ENNReal.ofNNReal x ≤ ENNReal.ofNNReal y",
      "WithTop.some x ≤ WithTop.some y", "simp [h]"),
     ("(x : NNReal)", "(x : ENNReal) * 1 = x",
      "ENNReal.ofNNReal x * 1 = ENNReal.ofNNReal x",
      "WithTop.some x * 1 = WithTop.some x", "simp"),
     ("(x y : NNReal)", "(x : ENNReal) * y = ((x * y : NNReal) : ENNReal)",
      "ENNReal.ofNNReal x * ENNReal.ofNNReal y = ENNReal.ofNNReal (x * y)",
      "WithTop.some x * WithTop.some y = WithTop.some (x * y)", "simp"),
     ("(x : NNReal) (h : x ≠ 0)", "(x : ENNReal) ≠ 0",
      "ENNReal.ofNNReal x ≠ 0", "WithTop.some x ≠ 0", "simp [h]"),
     ("(x : NNReal)", "(x : ENNReal) < ⊤",
      "ENNReal.ofNNReal x < ⊤", "WithTop.some x < ⊤", "simp"),
     ("(x y : NNReal) (h : x < y)", "(x : ENNReal) < y",
      "ENNReal.ofNNReal x < ENNReal.ofNNReal y",
      "WithTop.some x < WithTop.some y", "simp [h]"),
     ("(x : NNReal)", "(x : ENNReal) ⊔ x = x",
      "ENNReal.ofNNReal x ⊔ ENNReal.ofNNReal x = ENNReal.ofNNReal x",
      "WithTop.some x ⊔ WithTop.some x = WithTop.some x", "simp"),
     ("(x : NNReal)", "(x : ENNReal) + 0 = x",
      "ENNReal.ofNNReal x + 0 = ENNReal.ofNNReal x",
      "WithTop.some x + 0 = WithTop.some x", "simp"),
     ("(x y : NNReal)", "(x : ENNReal) ≤ x + y",
      "ENNReal.ofNNReal x ≤ ENNReal.ofNNReal x + ENNReal.ofNNReal y",
      "WithTop.some x ≤ WithTop.some x + WithTop.some y", "simp"),
     ("(x : NNReal) (y : ENNReal) (h : y ≠ 0)", "(x : ENNReal) ≤ x + y",
      "ENNReal.ofNNReal x ≤ ENNReal.ofNNReal x + y",
      "WithTop.some x ≤ WithTop.some x + y", "simp"),
     ("(x : NNReal)", "(x : ENNReal) = x",
      "ENNReal.ofNNReal x = ENNReal.ofNNReal x",
      "WithTop.some x = WithTop.some x", "simp"),
   ]},
 "coercion_accept": {
   "imports": ["Mathlib.Data.ENNReal.Basic", "Mathlib.Data.ENNReal.Operations",
               "Mathlib.Tactic.NormNum"],
   # Probes chosen so that ACCEPTANCE (not just the successor) can differ: a
   # `simp only` keyed on a coercion lemma matches the abbreviated spelling and
   # reports "no progress" (i.e. fails) on the unfolded one, which is the natural
   # mechanism reproduced under intervention.
   "pairs": [
     ("(x y : NNReal)", "(x : ENNReal) + y = ((x + y : NNReal) : ENNReal)",
      "ENNReal.ofNNReal x + ENNReal.ofNNReal y = ENNReal.ofNNReal (x + y)",
      "WithTop.some x + WithTop.some y = WithTop.some (x + y)",
      "simp only [ENNReal.coe_add]"),
     ("(x y : NNReal)", "(x : ENNReal) * y = ((x * y : NNReal) : ENNReal)",
      "ENNReal.ofNNReal x * ENNReal.ofNNReal y = ENNReal.ofNNReal (x * y)",
      "WithTop.some x * WithTop.some y = WithTop.some (x * y)",
      "simp only [ENNReal.coe_mul]"),
     ("(x y : NNReal)", "((x : ENNReal) ≤ y) = (x ≤ y)",
      "(ENNReal.ofNNReal x ≤ ENNReal.ofNNReal y) = (x ≤ y)",
      "(WithTop.some x ≤ WithTop.some y) = (x ≤ y)",
      "simp only [ENNReal.coe_le_coe]"),
     ("(x y : NNReal)", "((x : ENNReal) = y) = (x = y)",
      "(ENNReal.ofNNReal x = ENNReal.ofNNReal y) = (x = y)",
      "(WithTop.some x = WithTop.some y) = (x = y)",
      "simp only [ENNReal.coe_inj]"),
     ("(x y : NNReal)", "((x : ENNReal) < y) = (x < y)",
      "(ENNReal.ofNNReal x < ENNReal.ofNNReal y) = (x < y)",
      "(WithTop.some x < WithTop.some y) = (x < y)",
      "simp only [ENNReal.coe_lt_coe]"),
     ("(x : NNReal)", "((x : ENNReal) = 0) = (x = 0)",
      "(ENNReal.ofNNReal x = 0) = (x = 0)",
      "(WithTop.some x = 0) = (x = 0)",
      "simp only [ENNReal.coe_eq_zero]"),
     ("(x y : NNReal)", "(x : ENNReal) + y = ((x + y : NNReal) : ENNReal)",
      "ENNReal.ofNNReal x + ENNReal.ofNNReal y = ENNReal.ofNNReal (x + y)",
      "WithTop.some x + WithTop.some y = WithTop.some (x + y)",
      "rw [ENNReal.coe_add]"),
     ("(x y : NNReal)", "((x : ENNReal) ≤ y) = (x ≤ y)",
      "(ENNReal.ofNNReal x ≤ ENNReal.ofNNReal y) = (x ≤ y)",
      "(WithTop.some x ≤ WithTop.some y) = (x ≤ y)",
      "rw [ENNReal.coe_le_coe]"),
   ]},
 "NeNot": {
   "imports": ["Mathlib.Tactic.Push", "Mathlib.Tactic.NormNum",
               "Mathlib.Order.Basic"],
   "pairs": [
     ("(n m : Nat) (h : n < m)", "n ≠ m",
      "n ≠ m", "¬(n = m)", "push_neg"),
     ("(n : Nat)", "n + 1 ≠ n",
      "n + 1 ≠ n", "¬(n + 1 = n)", "simp"),
     ("(a b : Nat) (h : a ≠ b)", "b ≠ a",
      "b ≠ a", "¬(b = a)", "push_neg"),
     ("(n : Nat)", "(0 : Nat) ≠ n + 1",
      "(0 : Nat) ≠ n + 1", "¬((0 : Nat) = n + 1)", "simp"),
     ("(p : Prop) (a b : Nat) (h : a ≠ b)", "a ≠ b ∨ p",
      "a ≠ b ∨ p", "¬(a = b) ∨ p", "push_neg"),
     ("(n : Nat)", "n ≠ n + 1",
      "n ≠ n + 1", "¬(n = n + 1)", "omega"),
     ("(a b : Nat)", "a + b ≠ a + b + 1",
      "a + b ≠ a + b + 1", "¬(a + b = a + b + 1)", "simp"),
     ("(n m : Nat) (h : n ≠ m)", "n ≠ m",
      "n ≠ m", "¬(n = m)", "simp only [ne_eq]"),
     ("(x : Int)", "x ≠ x + 1",
      "x ≠ x + 1", "¬(x = x + 1)", "push_neg"),
     ("(n : Nat)", "2 * n + 1 ≠ 2 * n",
      "2 * n + 1 ≠ 2 * n", "¬(2 * n + 1 = 2 * n)", "simp"),
   ]},
 "FinEta": {
   "imports": ["Mathlib.Logic.Basic", "Mathlib.Tactic.NormNum",
               "Mathlib.Data.Fin.Basic"],
   "pairs": [
     ("(f : Nat → Nat)", "f 0 = f 0",
      "f 0 = f 0", "(fun i => f i) 0 = f 0", "simp"),
     ("(f : Nat → Nat) (h : f 1 = 2)", "f 1 = 2",
      "f 1 = 2", "(fun i => f i) 1 = 2", "simp [h]"),
     ("(f : Nat → Nat)", "f 0 + 0 = f 0",
      "f 0 + 0 = f 0", "(fun i => f i) 0 + 0 = f 0", "simp"),
     ("(f g : Nat → Nat) (h : ∀ n, f n = g n)", "f 3 = g 3",
      "f 3 = g 3", "(fun i => f i) 3 = g 3", "simp [h]"),
     ("(f : Fin 1 → Nat)", "f 0 = f 0",
      "f 0 = f 0", "(fun i => f i) 0 = f 0", "simp"),
     ("(f : Nat → Nat)", "f (0 + 0) = f 0",
      "f (0 + 0) = f 0", "(fun i => f i) (0 + 0) = f 0", "simp"),
     ("(f : Nat → Nat) (h : f 0 = 0)", "f 0 ≤ 0",
      "f 0 ≤ 0", "(fun i => f i) 0 ≤ 0", "simp [h]"),
     ("(f : Nat → Nat → Nat)", "f 0 1 = f 0 1",
      "f 0 1 = f 0 1", "(fun i => f i) 0 1 = f 0 1", "simp"),
     ("(f : Nat → Nat)", "f 2 = f (1 + 1)",
      "f 2 = f (1 + 1)", "(fun i => f i) 2 = f (1 + 1)", "norm_num"),
     ("(f : Nat → Nat) (h : ∀ n, f n = n)", "f 5 = 5",
      "f 5 = 5", "(fun i => f i) 5 = 5", "simp [h]"),
   ]},
}

def main():
    repo = Path(sys.argv[1]).resolve()
    only = sys.argv[2] if len(sys.argv) > 2 else None
    if only:
        for k in list(MECHS):
            if k != only: del MECHS[k]
    outdir = repo / "test_pairs"
    outdir.mkdir(exist_ok=True)
    results = []
    for mech, spec in MECHS.items():
        imp = "\n".join(f"import {m}" for m in spec["imports"])
        for pi, (binders, goal, tA, tB, probe) in enumerate(spec["pairs"]):
            rec = {"mechanism": mech, "pair": pi, "probe": probe, "variants": {}}
            for tag, term in (("A", tA), ("B", tB)):
                fn = outdir / f"{mech}_{pi}_{tag}.lean"
                fn.write_text(f"""{imp}
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false
example {binders} : {goal} := by
  show {term}
  {CAPTURE_PRE}
  {probe}
  {CAPTURE_POST}
  all_goals sorry
""")
                pr = subprocess.run(["lake", "env", "lean", str(fn.relative_to(repo))],
                                    cwd=repo, capture_output=True, text=True, timeout=600)
                out = pr.stdout + pr.stderr
                errs = re.findall(rf"{fn.name}:(\d+):\d+: error", out)
                mpp = re.search(r"(?s)PPV:(.*?)PPEND", out)
                pp = mpp.group(1).strip() if mpp else None
                mdig = re.search(r"DIGV:(\d+)", out)
                dig = mdig.group(1) if mdig else None
                mpost = re.search(r"(?s)POSTV:(.*?)POSTEND", out)
                post = mpost.group(1).strip() if mpost else None
                # pre-capture present => show and captures elaborated; any error then
                # belongs to the probe or later
                probe_failed = bool(errs) and pp is not None
                pre_failed = bool(errs) and pp is None
                rec["variants"][tag] = {
                    "pre_error": pre_failed, "probe_error": probe_failed,
                    "pp": hashlib.sha256((pp or "").encode()).hexdigest()[:12] if pp else None,
                    "dig": dig,
                    "post": ("FAIL" if probe_failed else
                             hashlib.sha256((post or "NOPOST").encode()).hexdigest()[:12]),
                }
            A, B = rec["variants"]["A"], rec["variants"]["B"]
            rec["constructible"] = bool(not (A["pre_error"] or B["pre_error"])
                                        and A["pp"] and B["pp"])
            rec["pp_identical"] = rec["constructible"] and A["pp"] == B["pp"]
            rec["dig_distinct"] = bool(A["dig"] and B["dig"] and A["dig"] != B["dig"])
            rec["valid"] = rec["pp_identical"] and rec["dig_distinct"]
            rec["outcome_diverges"] = rec["valid"] and A["post"] != B["post"]
            results.append(rec)
            print(f"[{mech} {pi}] constructible={rec['constructible']} "
                  f"pp_id={rec['pp_identical']} dig_diff={rec['dig_distinct']} "
                  f"DIVERGES={rec['outcome_diverges']} "
                  f"(A:{A['post']} B:{B['post']})", flush=True)
    outj = Path("runs/interventions.json")
    old = json.loads(outj.read_text()) if outj.exists() else []
    old = [r for r in old if r["mechanism"] not in MECHS]
    outj.write_text(json.dumps(old + results, indent=1))
    import collections
    for mech in MECHS:
        rs = [r for r in results if r["mechanism"] == mech]
        v = [r for r in rs if r["valid"]]
        d = [r for r in v if r["outcome_diverges"]]
        print(f"{mech}: pairs {len(rs)}, valid interventions {len(v)}, "
              f"outcome-divergent {len(d)}")

if __name__ == "__main__":
    main()
