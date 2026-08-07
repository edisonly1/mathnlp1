"""Component 7 — Measurements and statistical analysis (blueprint §8, §4.6, §7.10, §10.2).

Pure Python (no LeanDojo / no GPU) so it is fully unit-testable off-box. Consumes plain
summaries the orchestrator assembles from replay outcomes.

Implements:
  * latent-alias prevalence at BOTH fingerprint levels                (§8.1.1)
  * action-disagreement rate D_A and severity decomposition           (§8.1.2-3)
  * alias regret R = V_full - V_φ, the observation-only ceiling       (§4.6, §8.1.5)
  * the representation-repair ladder and its Pareto curve             (§7.10, §8.1.7)
  * cluster-robust bootstrap CIs (resample files/repos)               (§8.2)
  * mechanism source attribution                                      (§8.1.6)
  * Gate-1 GO / TERMINATE evaluation                                  (§10.2)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from .canon import env_fields

_NOTCOUNTED_PREFIXES = ("TIMEOUT", "CRASH", "GAVE_UP")   # excluded from divergence (§7.6)

# Mechanisms that make a class dismissible as a renderer/tokenizer artifact (§10.2 cond 4).
# `proof_term_only` is deliberately NOT here: proof irrelevance predicts such a class is inert,
# so an action-divergent proof-only class is a finding, not something to discount.
DISMISSIBLE_MECHANISMS = {"printer_overflow", "token_truncation"}


@dataclass
class RepSummary:
    """One distinct executable state (representative of an F_exec equivalence subclass).

    The rendering ladder is carried through from the fingerprint because mechanism attribution
    and the repair curve both need to know *where* two members differ, not merely that they do.
    """
    f_exec: str
    weight: int                              # multiplicity within the alias class
    outcomes: dict[str, str]                 # tactic -> canonical Ψ (reproducible outcomes only)
    phi_full: str = ""                       # pp.all + deepTerms, proofs off
    env_digest: str = ""                     # behavior-relevant ambient digest (no module id)
    phi_state: str = ""                      # the shared default rendering (repair baseline R1)
    phi_deep: str = ""                       # default pp + deepTerms                      (R2)
    phi_deep_proofs: str = ""                # R2 + proof terms                            (R3)
    phi_all_shallow: str = ""                # pp.all, proofs off, deep elision intact
    phi_full_proofs: str = ""                # pp.all + deepTerms + proofs
    env_provenance: str = ""                 # module identity — reported, never a mechanism
    f_core: str = ""
    f_proof: str = ""
    f_env: str = ""

    @property
    def has_ladder(self) -> bool:
        """True when the fingerprint carried enough renderings to attribute a mechanism."""
        return bool(self.phi_all_shallow and self.phi_full)

    @property
    def is_degraded(self) -> bool:
        """True when fingerprint extraction failed for this member, so its fields are
        placeholders rather than observations and must not be compared against real ones."""
        return self.phi_full.startswith("DEGRADED:") or not self.phi_full


@dataclass
class ClassSummary:
    observation_key: str
    layer: str
    is_natural: bool
    reps: list[RepSummary]
    files: list[str] = field(default_factory=list)
    repos: list[str] = field(default_factory=list)
    has_ellipsis: bool = False
    any_truncated: bool = False
    # filled by analysis
    action_divergent: bool = False
    divergent_tactics: list[str] = field(default_factory=list)
    severity: str = ""
    mechanisms: list[str] = field(default_factory=list)
    overflow_only: bool = True
    #: The action set divergence was DETECTED on (core panel + human + structural tactics).
    #: Repair and regret must be evaluated on this same set: evaluating them on the core panel
    #: alone silently omits the very tactics the divergence was found on, which made R1 — whose
    #: key is constant and therefore cannot partition anything — report classes as "repaired".
    action_set: list[str] = field(default_factory=list)

    @property
    def is_latent(self) -> bool:
        """Latent at the *executable* level: ≥2 distinct F_exec (blueprint §4.3)."""
        return len({r.f_exec for r in self.reps}) > 1

    @property
    def is_core_latent(self) -> bool:
        """Latent at the *goal-structure* level: ≥2 distinct F_core.

        This is the meaningful prevalence level. `F_exec` folds in the ambient environment,
        and two Mathlib files essentially never share a simp set, so `is_latent` saturates
        across files and cannot carry an RQ1 prevalence claim on its own.
        """
        cores = {r.f_core for r in self.reps if r.f_core}
        return len(cores) > 1

    @property
    def same_environment(self) -> bool:
        """All members share the ambient environment — so any divergence is attributable to
        the state itself, not to context the model could recover from elsewhere."""
        return len({r.env_digest for r in self.reps}) == 1


# --------------------------------------------------------------------------- #
# Divergence & severity (§8.1.2-3)
# --------------------------------------------------------------------------- #
def _counted(outcome: str) -> bool:
    return not any(outcome.startswith(p) for p in _NOTCOUNTED_PREFIXES)


def _divergent_outcomes(reps: list[RepSummary], a: str) -> set[str]:
    return {r.outcomes[a] for r in reps if a in r.outcomes and _counted(r.outcomes[a])}


def class_divergence(cls: ClassSummary, action_set: Iterable[str]) -> tuple[bool, list[str]]:
    """A class is action-divergent if some tactic yields ≥2 distinct *counted* outcomes across
    reps (blueprint §4.3). Timeouts/crashes are excluded so a flaky timeout is never divergence."""
    divergent = [a for a in action_set if len(_divergent_outcomes(cls.reps, a)) > 1]
    return (len(divergent) > 0), divergent


def class_severity(cls: ClassSummary, divergent_tactics: list[str]) -> str:
    """Rank the strongest disagreement kind present (blueprint §8.1.3)."""
    sev = ""
    for a in divergent_tactics:
        vals = _divergent_outcomes(cls.reps, a)
        has_complete = any(v == "COMPLETE" for v in vals)
        has_fail = any(v == "FAIL" for v in vals)
        has_succ = any(v.startswith("SUCC") for v in vals)
        if has_complete and has_fail:
            return "completion_vs_failure"          # strongest
        if has_succ and has_fail:
            sev = sev or "success_vs_failure"
        elif has_succ and len({v for v in vals if v.startswith("SUCC")}) > 1:
            sev = sev or "successor_disagreement"
    return sev


# --------------------------------------------------------------------------- #
# Alias regret (§4.6, §8.1.5)
# --------------------------------------------------------------------------- #
def alias_regret(cls: ClassSummary, action_set: Iterable[str],
                 success_is: Optional[set[str]] = None) -> float:
    """R = V_full - V_φ over the audited action set.

    V_φ  = max_a Σ_s w_s · 1[a ∈ A⁺(s)]      (best single observation-only action)
    V_full = Σ_s w_s · max_a 1[a ∈ A⁺(s)]    (per-state oracle)
    Weights w_s are the (normalized) empirical multiplicities. Success target defaults to
    'nonfailure' (a counted, non-FAIL outcome).

    A tactic absent from a rep's `outcomes` (dropped as non-reproducible) counts as unsuccessful
    for that rep, which biases R downward — conservative for a claim that R > 0.
    """
    action_set = list(action_set)
    reps = cls.reps
    total_w = sum(r.weight for r in reps) or 1

    def is_success(outcome: str) -> bool:
        if not _counted(outcome):
            return False
        if success_is is None:                      # 'nonfailure'
            return outcome != "FAIL"
        return outcome in success_is or (
            "SUCC" in success_is and outcome.startswith("SUCC"))

    # V_phi: best single action across the whole class.
    best_v_phi = 0.0
    for a in action_set:
        v = sum(r.weight for r in reps if a in r.outcomes and is_success(r.outcomes[a]))
        best_v_phi = max(best_v_phi, v)
    v_phi = best_v_phi / total_w

    # V_full: each state uses its own best action.
    v_full_num = sum(
        r.weight for r in reps
        if any(a in r.outcomes and is_success(r.outcomes[a]) for a in action_set))
    v_full = v_full_num / total_w
    return max(0.0, v_full - v_phi)


def repair_efficiency(divergent_removed: int, median_added_tokens: float, eps: float = 1.0) -> float:
    """E(r) = action-divergent classes removed / (median added tokens + ε)  (blueprint §8.1.7)."""
    return divergent_removed / (median_added_tokens + eps)


# --------------------------------------------------------------------------- #
# Mechanism attribution (§8.1.6)
# --------------------------------------------------------------------------- #
def attribute_mechanisms(cls: ClassSummary) -> list[str]:
    """Attribute a class's hidden difference by contrasting the reps' rendering ladder.

    The in-state discriminator relies on `phi_all_shallow` and `phi_full` differing *only* in
    deep-term elision (both render at pp.all with proofs off — see
    `lean_fingerprint/Fingerprint.lean`):

        shallow differs                    ⇒ instances / implicits / universes
        shallow same, full differs         ⇒ printer overflow (difference is in the elided region)
        full same, full_proofs differs     ⇒ proof term only (inert under proof irrelevance)

    Module identity is never consulted: it lives in `env_provenance`, not `env_digest`. Folding
    it in labelled every cross-file class `namespace_scope`, which made `overflow_only` — and
    therefore the §10.2 guard against a pure pretty-printer result — impossible to trigger.
    """
    # A member whose extraction failed differs from a successful one in EVERY dimension, so
    # including it invents `active_environment_*`, `namespace_scope` and `options_transparency`
    # out of nothing. Compare only members that actually produced a fingerprint.
    reps = [r for r in cls.reps if not r.is_degraded]
    if len(reps) < 2:
        return ["degraded_extraction"] if len(cls.reps) >= 2 else []
    mechs: set[str] = set()
    base = reps[0]
    bd = env_fields(base.env_digest)
    for r in reps[1:]:
        rd = env_fields(r.env_digest)
        # --- ambient mechanisms ---
        if bd.get("simpN") != rd.get("simpN") or bd.get("simpH") != rd.get("simpH"):
            mechs.add("active_environment_simp")
        if bd.get("linst") != rd.get("linst"):
            mechs.add("active_environment_instance")
        if bd.get("open") != rd.get("open") or bd.get("ns") != rd.get("ns"):
            mechs.add("namespace_scope")
        if bd.get("optsH") != rd.get("optsH"):
            mechs.add("options_transparency")
        if bd.get("nconsts") != rd.get("nconsts"):
            # Different numbers of locally-declared constants: the two states sit at different
            # points in the file, so a lemma available to one is not available to the other.
            # This is what actually drove the §8.3-audited divergence (`simp [getElem_pmap]`
            # fails inside `getElem_pmap`'s own proof and completes inside `get_pmap`'s).
            # Without this branch such classes get NO mechanism, and an empty mechanism list
            # is not `overflow_only`, so they silently count toward §10.2 conditions 1 and 4.
            mechs.add("declaration_context")
        # --- in-state mechanisms, most-visible first ---
        if not (base.has_ladder and r.has_ladder):
            if base.phi_full != r.phi_full:
                mechs.add("unattributed_incomplete_fingerprint")
        elif base.phi_all_shallow != r.phi_all_shallow:
            # `pp.proofs false` elides only proofs above `pp.proofs.threshold`; an ATOMIC proof
            # (a bare hypothesis name) always prints, and setting the threshold to 0 does not
            # change that. So a proof-term difference CAN reach the shallow rendering, and
            # "shallow differs" alone does not establish an instance/implicit difference.
            #
            # Asymmetric `⋯` is the tell: one side shows a term where the other shows elided
            # content, so the difference lives in something the renderer hid. The §8.3 audit of
            # class 8c33dbf195 was mislabelled `hidden_ingoal_instances_or_implicits` on exactly
            # this pattern (`n hn)` vs `n ⋯)`), when the real difference was a proof term.
            if base.phi_all_shallow.count("⋯") != r.phi_all_shallow.count("⋯"):
                mechs.add("elided_content_proof_or_deepterm")
            else:
                mechs.add("hidden_ingoal_instances_or_implicits")
        elif base.phi_full != r.phi_full:
            mechs.add("printer_overflow")
        elif base.phi_full_proofs and r.phi_full_proofs and base.phi_full_proofs != r.phi_full_proofs:
            mechs.add("proof_term_only")
    if cls.any_truncated and cls.layer == "phi_tok":
        mechs.add("token_truncation")
    return sorted(mechs)


def is_overflow_only(mechanisms: list[str]) -> bool:
    """True if every attributed mechanism is a pure print/token-length artifact (§10.2 cond 4)."""
    if not mechanisms:
        return False
    return set(mechanisms).issubset(DISMISSIBLE_MECHANISMS)


# --------------------------------------------------------------------------- #
# Representation-repair ladder (§7.10)
# --------------------------------------------------------------------------- #
@dataclass
class RepairRung:
    """One rung of the repair ladder.

    `key` maps a representative to the observation it would receive under this repair; two
    members sharing a key remain indistinguishable. `cost` is the observation growth in bytes
    relative to φ_state, measured rather than assumed (blueprint §7.10).
    """
    name: str
    describe: str
    key: Callable[[RepSummary], Optional[str]]
    cost: Callable[[RepSummary], int]

    def available_for(self, rep: RepSummary) -> bool:
        return self.key(rep) is not None


def _added(rep: RepSummary, rendering: str) -> int:
    return max(0, len(rendering) - len(rep.phi_state))


def _or_none(s: str) -> Optional[str]:
    return s if s else None


#: The implementable rungs. R5 (metavariable-context digest) and R7 (exact retrieved premises)
#: are declared in the blueprint but need, respectively, `MetavarContext` access that LeanDojo's
#: pp-only interface does not expose and the wired historical retriever. They are reported as
#: unavailable rather than silently omitted.
REPAIR_LADDER: list[RepairRung] = [
    RepairRung("R1_default", "default printed state",
               key=lambda r: "", cost=lambda r: 0),
    RepairRung("R2_deep", "full non-proof terms, raised print budget",
               key=lambda r: _or_none(r.phi_deep), cost=lambda r: _added(r, r.phi_deep)),
    RepairRung("R3_proofs_negctrl", "proof terms visible (negative control)",
               key=lambda r: _or_none(r.phi_deep_proofs),
               cost=lambda r: _added(r, r.phi_deep_proofs)),
    RepairRung("R4_struct", "structural: instances, implicits, universes, deep terms",
               key=lambda r: _or_none(r.phi_full), cost=lambda r: _added(r, r.phi_full)),
    RepairRung("R6_env", "R4 + compact active-environment digest",
               key=lambda r: (f"{r.phi_full}\x00{r.f_env}" if r.phi_full and r.f_env else None),
               # the digest is added as a fixed-width hash, not the raw environment
               cost=lambda r: _added(r, r.phi_full) + 16),
    RepairRung("R8_full", "R6 + proof terms",
               key=lambda r: (f"{r.phi_full_proofs}\x00{r.f_env}"
                              if r.phi_full_proofs and r.f_env else None),
               cost=lambda r: _added(r, r.phi_full_proofs) + 16),
]

UNAVAILABLE_RUNGS = {
    "R5_mvar_digest": "needs MetavarContext access; LeanDojo exposes pretty-printed state only",
    "R7_retrieved_premise": "needs the historical ReProver retriever wired to the pinned corpus",
}


def class_repaired_by(cls: ClassSummary, rung: RepairRung,
                      action_set: Iterable[str]) -> Optional[bool]:
    """Does `rung` remove this class's action divergence?

    The repair partitions the class by the repaired observation; the class is repaired iff no
    partition block is still action-divergent. Returns None if the rung is unavailable for any
    member (so it is neither counted as a success nor as a failure).
    """
    action_set = list(action_set)
    if not all(rung.available_for(r) for r in cls.reps):
        return None
    blocks: dict[str, list[RepSummary]] = {}
    for r in cls.reps:
        blocks.setdefault(rung.key(r) or "", []).append(r)
    for members in blocks.values():
        if any(len(_divergent_outcomes(members, a)) > 1 for a in action_set):
            return False
    return True


def repair_curve(classes: list[ClassSummary], action_set: Iterable[str],
                 ladder: Optional[list[RepairRung]] = None) -> dict:
    """Repair Pareto curve over the action-divergent classes (blueprint §7.10, §8.1.7, fig 6).

    Reports, per rung: how many divergent classes it removes, the median observation growth in
    bytes, the efficiency ratio, and how many classes it could not be evaluated on. Nothing is
    silently dropped — `n_unevaluable` is part of the output.
    """
    ladder = ladder if ladder is not None else REPAIR_LADDER
    default_actions = list(action_set)
    divergent = [c for c in classes if c.action_divergent]

    def actions_for(c: ClassSummary) -> list:
        return c.action_set or default_actions
    rows = []
    for rung in ladder:
        removed = 0
        unevaluable = 0
        costs: list[int] = []
        for c in divergent:
            verdict = class_repaired_by(c, rung, actions_for(c))
            if verdict is None:
                unevaluable += 1
                continue
            costs.extend(rung.cost(r) for r in c.reps)
            if verdict:
                removed += 1
        evaluable = len(divergent) - unevaluable
        median_cost = float(np.median(costs)) if costs else 0.0
        rows.append({
            "rung": rung.name,
            "describes": rung.describe,
            "divergent_classes": len(divergent),
            "evaluable": evaluable,
            "n_unevaluable": unevaluable,
            "removed": removed,
            "residual": max(0, evaluable - removed),
            "median_added_bytes": median_cost,
            "efficiency": repair_efficiency(removed, median_cost),
        })
    return {
        "rungs": rows,
        "unavailable_rungs": UNAVAILABLE_RUNGS,
        "n_action_divergent": len(divergent),
    }


def residual_after_repair(classes: list[ClassSummary], action_set: Iterable[str],
                          rung_name: str = "R6_env") -> list[ClassSummary]:
    """Classes whose divergence survives a given rung — the crux for a main-conference claim
    (README limitation 2: whether anything survives the environment repair)."""
    rung = next((r for r in REPAIR_LADDER if r.name == rung_name), None)
    if rung is None:
        raise ValueError(f"unknown rung {rung_name!r}")
    out = []
    for c in classes:
        if not c.action_divergent:
            continue
        if class_repaired_by(c, rung, c.action_set or list(action_set)) is False:
            out.append(c)
    return out


# --------------------------------------------------------------------------- #
# Prevalence + CIs (§8.1.1, §8.2)
# --------------------------------------------------------------------------- #
def prevalence(classes: list[ClassSummary], n_sampled_states: int) -> dict:
    """Latent-alias prevalence at both fingerprint levels (§8.1.1).

    `core_*` is the headline: states whose *goal structure* differs behind an identical
    observation. `exec_*` additionally counts ambient-environment differences and saturates
    across files, so it is reported for completeness but must not be quoted as RQ1 prevalence.
    """
    latent = [c for c in classes if c.is_latent]
    core_latent = [c for c in classes if c.is_core_latent]
    same_env = [c for c in core_latent if c.same_environment]

    def mass(cs: list[ClassSummary]) -> int:
        return sum(sum(r.weight for r in c.reps) for c in cs)

    denom = n_sampled_states or 1
    return {
        "n_sampled_states": n_sampled_states,
        "n_candidate_classes": len(classes),
        "n_latent_classes_exec": len(latent),
        "n_latent_classes_core": len(core_latent),
        "n_latent_classes_core_same_env": len(same_env),
        "state_weighted_prevalence_core": mass(core_latent) / denom,
        "state_weighted_prevalence_exec": mass(latent) / denom,
        "class_weighted_prevalence_core": (len(core_latent) / len(classes)) if classes else 0.0,
        "class_weighted_prevalence_exec": (len(latent) / len(classes)) if classes else 0.0,
        "note": ("exec-level latency folds in the ambient environment and saturates across "
                 "files; quote the core-level figure for RQ1."),
    }


def cluster_bootstrap_ci(values: list[float], clusters: list[str], seed: int = 0,
                         n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    """Cluster-robust percentile CI for a mean, resampling whole clusters (files/repos) (§8.2)."""
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    by_cluster: dict[str, list[float]] = {}
    for v, c in zip(values, clusters):
        by_cluster.setdefault(c, []).append(v)
    cluster_names = list(by_cluster.keys())
    means = np.empty(n_boot)
    for b in range(n_boot):
        picks = rng.choice(len(cluster_names), size=len(cluster_names), replace=True)
        pooled = [x for i in picks for x in by_cluster[cluster_names[i]]]
        means[b] = np.mean(pooled) if pooled else 0.0
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


# --------------------------------------------------------------------------- #
# Gate-1 GO / TERMINATE (§10.2)
# --------------------------------------------------------------------------- #
def evaluate_gate1(classes: list[ClassSummary], *, reproduce_rate: float,
                   regression_passed: bool,
                   model_persists_or_repairs: Optional[bool],
                   sampling_mode: str = "stratified_7.2",
                   primary_layer: str = "phi_state") -> dict:
    """Evaluate the seven GO conditions and the TERMINATE conditions (blueprint §10.2).

    `model_persists_or_repairs` is tri-state: None means the model pass was not run, which is
    reported as *unevaluated* rather than silently failing condition 7. GO still requires it,
    but the verdict distinguishes "not established" from "established false".
    """
    # Count the PRIMARY layer only. `φ_tok` classes are the same underlying states re-keyed:
    # ReProver's ByT5 tokenizer is byte-level, so identical φ_state gives identical token input
    # and the two layers produce duplicate classes under different hashes. Counting both doubled
    # every figure and pushed condition 1 (≥5) over the line on 4 real classes. §7.5 makes
    # φ_state the primary object, so it is the one the gate counts.
    layered = [c for c in classes if c.layer == primary_layer] or classes
    divergent = [c for c in layered if c.action_divergent and c.is_natural]
    non_overflow = [c for c in divergent if not c.overflow_only]
    severe = [c for c in divergent if c.severity == "completion_vs_failure"]
    files = {f for c in non_overflow for f in c.files}
    repos = {r for c in non_overflow for r in c.repos}
    # A dense/within-file sample is enriched by construction; §10.2's thresholds are calibrated
    # for the §7.2 stratified pilot. Evaluating them on an enriched sample is not a GO.
    #
    # An EXHAUSTIVE census is different in kind: it audits every auditable state, so there is no
    # selection at all and prevalence is exact rather than estimated. It is therefore gate-
    # applicable — and strictly more informative than the stratified sample §10.2 assumed, which
    # was specified because a census was presumed infeasible.
    UNENRICHED = ("stratified_7.2", "exhaustive_census")
    enriched = sampling_mode not in UNENRICHED

    go_conditions = {
        "1_ge5_verified_divergent": len(non_overflow) >= 5,
        "2_ge1_completion_vs_failure": len(severe) >= 1,
        "3_ge2_independent_files_or_repos": (len(files) >= 2 or len(repos) >= 2),
        "4_ge1_nonoverflow_mechanism": len(non_overflow) >= 1,
        "5_ge90pct_reproduce": reproduce_rate >= 0.90,
        "6_pipelines_pass_regression": regression_passed,
        "7_model_persists_or_retrieval_repairs": bool(model_persists_or_repairs),
    }
    go = all(go_conditions.values()) and not enriched

    terminate_conditions = {
        "no_reproducible_divergence": len(divergent) == 0,
        "all_divergence_overflow_only": (len(divergent) > 0 and len(non_overflow) == 0),
        "regression_failed": not regression_passed,
    }
    terminate = any(terminate_conditions.values())

    unevaluated = [] if model_persists_or_repairs is not None else \
        ["7_model_persists_or_retrieval_repairs"]

    # Amendment A2: a class whose members differ in ambient environment (declaration context,
    # simp set, scope) is repairable by supplying context the model could in principle have, so
    # it cannot carry the central claim. The §10.2 thresholds are evaluated a second time over
    # non-dismissible classes only. BOTH verdicts are reported: `GO` is §10.2 as literally
    # written, `GO_non_dismissible` is the qualified reading. They disagree exactly when every
    # witness is ambient — which is the case the distinction exists to expose.
    non_dismissible = [c for c in non_overflow if c.same_environment]
    nd_severe = [c for c in non_dismissible if c.severity == "completion_vs_failure"]
    nd_files = {f for c in non_dismissible for f in c.files}
    nd_repos = {r for c in non_dismissible for r in c.repos}
    nd_conditions = dict(go_conditions)
    nd_conditions["1_ge5_verified_divergent"] = len(non_dismissible) >= 5
    nd_conditions["2_ge1_completion_vs_failure"] = len(nd_severe) >= 1
    nd_conditions["3_ge2_independent_files_or_repos"] = (len(nd_files) >= 2 or len(nd_repos) >= 2)
    nd_conditions["4_ge1_nonoverflow_mechanism"] = len(non_dismissible) >= 1
    go_nd = all(nd_conditions.values()) and not enriched

    return {
        "GO": go and not terminate,
        "GO_non_dismissible": go_nd and not terminate,
        "go_conditions_non_dismissible": nd_conditions,
        "sampling_mode": sampling_mode,
        "gate_applicable": not enriched,
        "gate_note": ("" if not enriched else
                      f"sampling_mode={sampling_mode!r} is enriched by construction; §10.2 "
                      "thresholds are calibrated for the §7.2 stratified pilot, so GO is "
                      "withheld regardless of the condition flags below."),
        "counted_layer": primary_layer,
        "go_conditions": go_conditions,
        "unevaluated_conditions": unevaluated,
        "terminate": terminate,
        "terminate_conditions": terminate_conditions,
        "counts": {
            "verified_natural_divergent": len(divergent),
            "non_overflow_divergent": len(non_overflow),
            "completion_vs_failure": len(severe),
            "distinct_files": len(files),
            "distinct_repos": len(repos),
            "same_environment_divergent": len(non_dismissible),
            "ambient_divergent": len(non_overflow) - len(non_dismissible),
        },
    }
