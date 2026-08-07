"""Unit tests for the analysis math (blueprint §8). Pure Python — runs off-box (no LeanDojo)."""
from audit import analysis as A


def _cls(reps, **kw):
    return A.ClassSummary(observation_key="o", layer="phi_state", is_natural=True, reps=reps, **kw)


def test_divergence_detected():
    reps = [
        A.RepSummary(f_exec="e1", weight=1, outcomes={"simp": "COMPLETE"}),
        A.RepSummary(f_exec="e2", weight=1, outcomes={"simp": "FAIL"}),
    ]
    cls = _cls(reps)
    div, tacs = A.class_divergence(cls, ["simp"])
    assert div and tacs == ["simp"]
    assert A.class_severity(cls, tacs) == "completion_vs_failure"


def test_no_divergence_when_equal():
    reps = [
        A.RepSummary(f_exec="e1", weight=1, outcomes={"simp": "FAIL"}),
        A.RepSummary(f_exec="e2", weight=1, outcomes={"simp": "FAIL"}),
    ]
    div, tacs = A.class_divergence(_cls(reps), ["simp"])
    assert not div and tacs == []


def test_timeout_not_counted_as_divergence():
    reps = [
        A.RepSummary(f_exec="e1", weight=1, outcomes={"simp": "COMPLETE"}),
        A.RepSummary(f_exec="e2", weight=1, outcomes={"simp": "TIMEOUT"}),
    ]
    div, _ = A.class_divergence(_cls(reps), ["simp"])
    assert not div, "a timeout must never create a divergence (blueprint §7.6)"


def test_alias_regret_positive_when_actions_split():
    # rep1 solved only by simp; rep2 solved only by rfl -> a single action can't cover both.
    reps = [
        A.RepSummary(f_exec="e1", weight=1, outcomes={"simp": "COMPLETE", "rfl": "FAIL"}),
        A.RepSummary(f_exec="e2", weight=1, outcomes={"simp": "FAIL", "rfl": "COMPLETE"}),
    ]
    r = A.alias_regret(_cls(reps), ["simp", "rfl"])
    assert abs(r - 0.5) < 1e-9, f"expected 0.5 regret, got {r}"


def test_alias_regret_zero_when_single_action_covers():
    reps = [
        A.RepSummary(f_exec="e1", weight=1, outcomes={"simp": "COMPLETE"}),
        A.RepSummary(f_exec="e2", weight=1, outcomes={"simp": "COMPLETE"}),
    ]
    assert A.alias_regret(_cls(reps), ["simp"]) == 0.0


def test_mechanism_attribution_and_overflow():
    """Synthetic counterpart to `test_fingerprint_attribution.py`, which runs the same rules
    against real Lean renderings. The in-state discriminator needs the shallow/full pair:
    equal `phi_all_shallow` with differing `phi_full` means the difference sits inside the
    deep-elided region, i.e. a printer artifact."""
    reps = [
        A.RepSummary(f_exec="e1", weight=1, outcomes={}, phi_full="G",
                     phi_all_shallow="G", env_digest="ns=N|simpN=10|simpH=1"),
        A.RepSummary(f_exec="e2", weight=1, outcomes={}, phi_full="G",
                     phi_all_shallow="G", env_digest="ns=N|simpN=11|simpH=2"),
    ]
    mechs = A.attribute_mechanisms(_cls(reps))
    assert "active_environment_simp" in mechs
    assert not A.is_overflow_only(mechs)

    over = A.attribute_mechanisms(_cls(
        [A.RepSummary("e1", 1, {}, phi_full="G_deep_a", phi_all_shallow="G⋯", env_digest="ns=N"),
         A.RepSummary("e2", 1, {}, phi_full="G_deep_b", phi_all_shallow="G⋯", env_digest="ns=N")],
        has_ellipsis=True))
    assert over == ["printer_overflow"]
    assert A.is_overflow_only(over)


def test_incomplete_fingerprint_is_flagged_not_guessed():
    """Without the shallow rendering the mechanism cannot be determined; say so rather than
    attributing it to whichever branch happens to fire."""
    mechs = A.attribute_mechanisms(_cls(
        [A.RepSummary("e1", 1, {}, phi_full="A"), A.RepSummary("e2", 1, {}, phi_full="B")]))
    assert mechs == ["unattributed_incomplete_fingerprint"]
    assert not A.is_overflow_only(mechs)


def test_proof_only_difference_is_not_dismissible():
    """Proof irrelevance predicts such a class is inert; if it IS action-divergent that is a
    finding, so it must not be lumped in with renderer artifacts."""
    assert not A.is_overflow_only(["proof_term_only"])


def test_gate1_go_and_terminate():
    def divergent_cls(i, sev="completion_vs_failure", overflow=False, fpair=("a", "b")):
        c = _cls([A.RepSummary("e1", 1, {"simp": "COMPLETE"}), A.RepSummary("e2", 1, {"simp": "FAIL"})],
                 files=[f"F{i}.lean"])
        c.action_divergent = True
        c.severity = sev
        c.overflow_only = overflow
        c.reps = c.reps  # keep latent (2 distinct f_exec)
        return c

    classes = [divergent_cls(i) for i in range(5)]
    v = A.evaluate_gate1(classes, reproduce_rate=1.0, regression_passed=True,
                         model_persists_or_repairs=True)
    assert v["GO"] is True
    assert v["counts"]["non_overflow_divergent"] == 5

    # overflow-only divergence must NOT pass and should TERMINATE (blueprint §10.2).
    over = [divergent_cls(i, overflow=True) for i in range(5)]
    v2 = A.evaluate_gate1(over, reproduce_rate=1.0, regression_passed=True,
                          model_persists_or_repairs=True)
    assert v2["GO"] is False
    assert v2["terminate"] is True


def test_prevalence_counts_latent_only():
    latent = _cls([A.RepSummary("e1", 3, {}, f_core="c1"),
                   A.RepSummary("e2", 2, {}, f_core="c2")])   # 2 distinct -> latent
    trivial = _cls([A.RepSummary("e9", 4, {}, f_core="c9"),
                    A.RepSummary("e9", 1, {}, f_core="c9")])  # 1 distinct -> not latent
    p = A.prevalence([latent, trivial], n_sampled_states=100)
    assert p["n_latent_classes_exec"] == 1
    assert p["n_latent_classes_core"] == 1
    assert abs(p["state_weighted_prevalence_exec"] - 5 / 100) < 1e-9


def test_prevalence_separates_core_from_exec_latency():
    """`F_exec` folds in the ambient environment, which differs between essentially any two
    Mathlib files, so exec-latency saturates and cannot carry the RQ1 prevalence claim. A class
    whose members share a goal but differ only in environment is exec-latent, NOT core-latent."""
    env_only = _cls([A.RepSummary("e1", 1, {}, f_core="same", env_digest="simpH=1"),
                     A.RepSummary("e2", 1, {}, f_core="same", env_digest="simpH=2")])
    p = A.prevalence([env_only], n_sampled_states=10)
    assert p["n_latent_classes_exec"] == 1
    assert p["n_latent_classes_core"] == 0
    assert not env_only.same_environment


def test_same_environment_classes_are_counted():
    """Divergence within one environment is the non-dismissible case: nothing outside the
    state could have disambiguated it."""
    same = _cls([A.RepSummary("e1", 1, {"simp": "COMPLETE"}, f_core="a", env_digest="E"),
                 A.RepSummary("e2", 1, {"simp": "FAIL"}, f_core="b", env_digest="E")],
                files=["F.lean"])
    same.action_divergent, _ = A.class_divergence(same, ["simp"])
    same.overflow_only = False
    v = A.evaluate_gate1([same], reproduce_rate=1.0, regression_passed=True,
                         model_persists_or_repairs=True)
    assert v["counts"]["same_environment_divergent"] == 1


def test_condition7_unevaluated_is_distinct_from_false():
    """`--with-model` was not run: condition 7 must report as unevaluated, not as a measured
    failure. It still blocks GO, but the verdict says which."""
    classes = []
    v = A.evaluate_gate1(classes, reproduce_rate=1.0, regression_passed=True,
                         model_persists_or_repairs=None)
    assert v["unevaluated_conditions"] == ["7_model_persists_or_retrieval_repairs"]
    assert v["go_conditions"]["7_model_persists_or_retrieval_repairs"] is False

    v2 = A.evaluate_gate1(classes, reproduce_rate=1.0, regression_passed=True,
                          model_persists_or_repairs=False)
    assert v2["unevaluated_conditions"] == []


def test_low_reproduce_rate_blocks_go():
    """Condition 5 was previously hardcoded to 1.0 whenever any divergence existed, making it
    impossible to fail."""
    def cls(i):
        c = _cls([A.RepSummary("e1", 1, {"simp": "COMPLETE"}),
                  A.RepSummary("e2", 1, {"simp": "FAIL"})], files=[f"F{i}.lean"])
        c.action_divergent, c.severity, c.overflow_only = True, "completion_vs_failure", False
        return c
    classes = [cls(i) for i in range(5)]
    v = A.evaluate_gate1(classes, reproduce_rate=0.5, regression_passed=True,
                         model_persists_or_repairs=True)
    assert v["go_conditions"]["5_ge90pct_reproduce"] is False
    assert v["GO"] is False


def test_declaration_context_is_attributed():
    """Two states at different points in one file have different locally-declared constants, so
    a lemma available to one is not available to the other. Without this branch such a class
    gets NO mechanism — and an empty list is not `overflow_only`, so it would silently count
    toward §10.2 conditions 1 and 4 as a non-dismissible witness."""
    reps = [
        A.RepSummary("e1", 1, {}, phi_full="G", phi_all_shallow="G",
                     env_digest="ns=N|simpN=10|simpH=a|nconsts=120"),
        A.RepSummary("e2", 1, {}, phi_full="G", phi_all_shallow="G",
                     env_digest="ns=N|simpN=10|simpH=a|nconsts=121"),
    ]
    mechs = A.attribute_mechanisms(_cls(reps))
    assert mechs == ["declaration_context"]
    assert not A.is_overflow_only(mechs)


def _div(i, layer="phi_state"):
    c = A.ClassSummary(observation_key=f"k{i}", layer=layer, is_natural=True,
                       reps=[A.RepSummary("e1", 1, {"simp": "COMPLETE"}),
                             A.RepSummary("e2", 1, {"simp": "FAIL"})],
                       files=[f"F{i}.lean"])
    c.action_divergent, c.severity, c.overflow_only = True, "completion_vs_failure", False
    return c


def test_phi_tok_classes_do_not_double_count():
    """ByT5 is byte-level, so φ_tok classes are the SAME states re-keyed under a different hash.
    Counting both layers doubled every figure and pushed condition 1 (≥5) over the line on 4
    real classes — a false GO."""
    classes = [_div(i) for i in range(4)] + [_div(i, layer="phi_tok") for i in range(4)]
    v = A.evaluate_gate1(classes, reproduce_rate=1.0, regression_passed=True,
                         model_persists_or_repairs=True)
    assert v["counts"]["verified_natural_divergent"] == 4, "must count φ_state only"
    assert v["go_conditions"]["1_ge5_verified_divergent"] is False
    assert v["GO"] is False


def test_enriched_sample_never_yields_go():
    """§10.2 thresholds are calibrated for the §7.2 stratified pilot. A dense within-file sample
    is enriched by construction, so it can answer existence but never GO."""
    classes = [_div(i) for i in range(6)]
    ok = A.evaluate_gate1(classes, reproduce_rate=1.0, regression_passed=True,
                          model_persists_or_repairs=True)
    assert ok["GO"] is True and ok["gate_applicable"] is True

    enr = A.evaluate_gate1(classes, reproduce_rate=1.0, regression_passed=True,
                           model_persists_or_repairs=True,
                           sampling_mode="dense_within_file")
    assert enr["GO"] is False
    assert enr["gate_applicable"] is False
    assert "enriched" in enr["gate_note"]


def test_ambient_only_witnesses_fail_the_qualified_gate():
    """Amendment A2. §10.2 as literally written counts any non-overflow divergent class, so a
    run whose witnesses are ALL ambient (declaration context, simp set, scope) can pass — the
    20k pilot did exactly that with 7/7 ambient. The qualified reading applies the same
    thresholds to non-dismissible classes only, and must disagree in that case."""
    def amb(i):
        c = _cls([A.RepSummary("e1", 1, {"simp": "COMPLETE"}, env_digest=f"E{i}a"),
                  A.RepSummary("e2", 1, {"simp": "FAIL"}, env_digest=f"E{i}b")],
                 files=[f"F{i}.lean"])
        c.action_divergent, c.severity, c.overflow_only = True, "completion_vs_failure", False
        return c
    v = A.evaluate_gate1([amb(i) for i in range(7)], reproduce_rate=1.0,
                         regression_passed=True, model_persists_or_repairs=True)
    assert v["GO"] is True, "literal §10.2 counts ambient classes"
    assert v["GO_non_dismissible"] is False, "qualified reading must reject an all-ambient run"
    assert v["counts"]["same_environment_divergent"] == 0
    assert v["counts"]["ambient_divergent"] == 7


def test_qualified_gate_passes_when_witnesses_are_non_dismissible():
    def nd(i):
        c = _cls([A.RepSummary("e1", 1, {"simp": "COMPLETE"}, env_digest="SAME"),
                  A.RepSummary("e2", 1, {"simp": "FAIL"}, env_digest="SAME")],
                 files=[f"F{i}.lean"])
        c.action_divergent, c.severity, c.overflow_only = True, "completion_vs_failure", False
        return c
    v = A.evaluate_gate1([nd(i) for i in range(5)], reproduce_rate=1.0,
                         regression_passed=True, model_persists_or_repairs=True)
    assert v["GO"] is True and v["GO_non_dismissible"] is True


def test_census_is_gate_applicable_but_dense_sample_is_not():
    """An exhaustive census audits every auditable state, so there is no selection and
    prevalence is exact — it is gate-applicable. A dense within-file sample selects the densest
    files and is enriched, so it never is."""
    def d(i):
        c = _cls([A.RepSummary("e1", 1, {"simp": "COMPLETE"}, env_digest="S"),
                  A.RepSummary("e2", 1, {"simp": "FAIL"}, env_digest="S")], files=[f"F{i}.lean"])
        c.action_divergent, c.severity, c.overflow_only = True, "completion_vs_failure", False
        return c
    cs = [d(i) for i in range(5)]
    kw = dict(reproduce_rate=1.0, regression_passed=True, model_persists_or_repairs=True)
    assert A.evaluate_gate1(cs, sampling_mode="exhaustive_census", **kw)["gate_applicable"] is True
    assert A.evaluate_gate1(cs, sampling_mode="dense_within_file", **kw)["gate_applicable"] is False
    assert A.evaluate_gate1(cs, sampling_mode="stratified_7.2", **kw)["gate_applicable"] is True
