import Lean
/-!
# Execution-fingerprint exporter (blueprint §7.4 two-level fingerprint)

`φ_state` (default `ppGoal`) is what the neural prover sees. It hides instances, implicit
arguments, universes, deep subterms, proof terms, active simp lemmas, and the ambient
environment. To decide whether two states sharing `φ_state` are *internally distinct* (a latent
alias, §4.3) — and, crucially, to attribute *which* hidden component differs — we export a
ladder of renderings rather than a single "full" one.

## Renderings

| field             | options                                   | reveals                          |
|-------------------|-------------------------------------------|----------------------------------|
| `phi_deep`        | default + `pp.deepTerms`                  | deep non-proof subterms (R2)     |
| `phi_deep_proofs` | `phi_deep` + `pp.proofs`                  | proof terms (R3, negative ctrl)  |
| `phi_all_shallow` | `pp.all`, no notation, proofs OFF         | instances/implicits/universes    |
| `phi_full`        | `pp.all` + `pp.deepTerms`, proofs OFF     | the above AND deep subterms (R4) |
| `phi_full_proofs` | `phi_full` + `pp.proofs`                  | everything renderable            |

Two invariants that are easy to get wrong and were both wrong in the first version of this
file:

* **`pp.all` does NOT imply `pp.deepTerms`.** Without setting it explicitly the fingerprint
  inherits the ambient elision, so any alias whose hidden difference is an elided deep subterm
  renders identically at *every* level and is invisible — precisely blueprint §5.3 mechanism 2.
* **`pp.all` switches `pp.proofs` ON.** It must be switched back off explicitly, or
  shallow-vs-full differs in both elision *and* proof visibility and the overflow discriminator
  cannot separate a printer artifact from a proof-irrelevant difference.

The pair (`phi_all_shallow`, `phi_full`) differs only in deep-term elision, which is what makes
the attribution in `audit/analysis.py` sound:

    shallow differs                      ⇒ instances / implicits / universes
    shallow same,  full differs          ⇒ printer overflow (difference lies in elided region)
    full same,     full_proofs differs   ⇒ proof term only (inert under proof irrelevance)

## Environment digest

`env_behavior` holds only ambient state that can change *tactic behavior*: current namespace,
open declarations, local instances (with their types — the class name alone cannot distinguish
two `Add Box` instances), and the default simp set.

`env_provenance` (the module name) is exported **separately and is never hashed into
`F_exec`**. Folding module identity into the execution fingerprint makes every cross-file class
trivially "latent" and manufactures a spurious `namespace_scope` mechanism on every such class,
which in turn disables the overflow-only guard that Gate 1 depends on.

`F_core = hash(α-normalized phi_full)`, `F_exec = hash(F_core ⊕ canonical(env_behavior))` are
computed in Python (`audit/fingerprint.py`) so the canonicalization policy is versioned in one
place and is unit-testable off-box.
-/
open Lean Elab Tactic Meta

namespace StateAliasing

/-- Raw option set. `Lean.Options` has a private backing map in v4.32, so we emit its
    deterministic `toString` (backed by a `NameMap`, hence sorted and stable) and let the Python
    canonicalizer strip rendering-only options (`pp.*`, `format.*`, `trace.*`). -/
def rawOptions : TacticM String := do
  try
    let opts ← getOptions
    return toString opts
  catch _ => return "NA"

/-- Open declarations — the ambient state behind mechanism C (same tactic text, different
    `open`, different resolved lemma). `getCurrNamespace` alone does NOT capture this. -/
def openDeclsDigest : TacticM String := do
  try
    let ds ← getOpenDecls
    let strs := (ds.map toString).toArray.qsort (· < ·)
    return s!"open={strs.toList}"
  catch _ => return "open=NA"

/-- Default simp set: size plus an order-independent digest. Sorted before hashing, because
    `lemmaNames` iteration order is insertion-dependent and would otherwise manufacture
    spurious differences between states that share a simp set. -/
def simpDigest : TacticM String := do
  try
    let s ← Meta.getSimpTheorems
    let names := (s.lemmaNames.toList.map (fun o => toString o.key)).toArray.qsort (· < ·)
    return s!"simpN={names.size}|simpH={hash names.toList}"
  catch _ => return "simpN=NA|simpH=NA"

/-- Local instances, keyed by class name *and* instance type. The class name alone is too
    coarse: two `Add Box` instances with different values share a class name. -/
def localInstDigest : TacticM String := do
  try
    let insts ← getLocalInstances
    let strs ← insts.toList.mapM (fun li => do
      let t ← instantiateMVars (← inferType li.fvar)
      return s!"{li.className}:{toString (← Meta.ppExpr t)}")
    return s!"linst={(strs.toArray.qsort (· < ·)).toList}"
  catch _ => return "linst=NA"

/-- Number of LOCALLY declared constants (`map₂` is the file's own extension of the
    environment; `SMap.size` does not exist in every toolchain, and the total count is
    dominated by imports anyway). Every other ambient field is unchanged by adding a lemma,
    so without this two states at different points in one file compare as the same environment —
    which silently turned a declaration-order divergence into a "same-environment" witness. -/
def constCountDigest : TacticM String := do
  try
    let e ← getEnv
    return s!"nconsts={e.constants.map₂.foldl (fun n _ _ => n + 1) 0}"
  catch _ => return "nconsts=NA"

/-- Behavior-relevant ambient state. Deliberately EXCLUDES the module name (see header). -/
def envBehavior : TacticM String := do
  let mut parts : Array String := #[]
  try parts := parts.push s!"ns={← getCurrNamespace}" catch _ => parts := parts.push "ns=NA"
  parts := parts.push (← openDeclsDigest)
  parts := parts.push (← localInstDigest)
  parts := parts.push (← simpDigest)
  parts := parts.push (← constCountDigest)
  return "|".intercalate parts.toList

/-- Pure provenance: reported for attribution, never hashed into `F_exec`. -/
def envProvenance : TacticM String := do
  try
    let env ← getEnv
    return s!"mod={env.mainModule}"
  catch _ => return "mod=NA"

/-- Render the main goal under an explicit option override. -/
def render (f : Options → Options) : TacticM String := do
  let g ← getMainGoal
  return toString (← withOptions f (Meta.ppGoal g))

def optDeep          (o : Options) : Options := (o.setBool `pp.deepTerms true).setBool `pp.proofs false
def optDeepProof     (o : Options) : Options := (o.setBool `pp.deepTerms true).setBool `pp.proofs true
def optAllShallow    (o : Options) : Options :=
  ((o.setBool `pp.all true).setBool `pp.notation false).setBool `pp.proofs false
def optAllDeep       (o : Options) : Options :=
  (((o.setBool `pp.all true).setBool `pp.notation false).setBool `pp.deepTerms true).setBool `pp.proofs false
def optAllDeepProof  (o : Options) : Options :=
  (((o.setBool `pp.all true).setBool `pp.notation false).setBool `pp.deepTerms true).setBool `pp.proofs true

/-- Simp-set digest. Kept separate from `fingerprintPayload` because `getSimpTheorems` can
    raise Lean's `Exception.internal` (e.g. in a file whose environment never initialised the
    simp extension), and a `try ... catch _` in `TacticM` does not reliably intercept that. -/
def simpDigestOrNA : TacticM String := do
  try
    let s ← Meta.getSimpTheorems
    let names := (s.lemmaNames.toList.map (fun o => toString o.key)).toArray.qsort (· < ·)
    return s!"simpN={names.size}|simpH={hash names.toList}"
  catch _ => return "simpN=NA|simpH=NA"

/-- Build the fingerprint payload for the current proof state.

    Fields are UNIT-SEPARATED (U+001F), not JSON, and in the order `audit/fingerprint.py`
    declares as `MAIN_FIELDS`. The separator cannot occur in pretty-printer output, so nothing
    needs escaping in either direction. -/
def fingerprintPayload : TacticM String := do
  let fields := [
    (← render optDeep),
    (← render optDeepProof),
    (← render optAllShallow),
    (← render optAllDeep),
    (← render optAllDeepProof),
    (← envBehavior) ++ "|" ++ (← simpDigestOrNA),
    (← envProvenance),
    (← rawOptions)
  ]
  return "\u001F".intercalate fields

/-- Injected tactic: emit the fingerprint by throwing it, so LeanDojo returns it as the error
    message. Marker `FPRINT_V2:` lets `fingerprint.py` locate the payload.

    Unlike the inline `throwError` channel, this is a single compiled tactic, so it is not
    subject to the ~1 KB limit `Dojo.run_tac` imposes on submitted tactic text — which is the
    reason this channel exists and why it collects every field in one call. -/
elab "fingerprint_throw" : tactic => do
  throwError s!"FPRINT_V2:{← fingerprintPayload}"

/-- Frontend form: dump the fingerprint of the current goal to stdout (channel=frontend).
    Use as: `example : P := by <tactics>; fingerprint_print` -/
elab "fingerprint_print" : tactic => do
  IO.println s!"FPRINT_V2:{← fingerprintPayload}"

end StateAliasing
