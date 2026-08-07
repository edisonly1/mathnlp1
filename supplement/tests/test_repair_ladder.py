"""Tests for the representation-repair ladder (blueprint §7.10, §8.1.7).

The ladder is contribution-package item 4 (§2.3): the curve that says *which* omitted
information matters and at what observation cost. It was previously unimplemented — only the
`E(r)` division existed — so these tests cover the actual repair semantics: a rung repairs a
class iff, after partitioning the class by the repaired observation, no block is still
action-divergent.
"""
import json
from pathlib import Path

from audit import analysis as A
from audit.fingerprint import build_fingerprint

FIXTURE = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "gate0_fingerprints.json")
    .read_text(encoding="utf-8"))


def _rep(label: str, outcomes: dict) -> A.RepSummary:
    fp = build_fingerprint(FIXTURE[label], channel="test", faithful=True)
    return A.RepSummary(
        f_exec=fp.f_exec, weight=1, outcomes=outcomes,
        phi_full=fp.phi_full, env_digest=fp.env_digest,
        phi_state="⊢ shared", phi_deep=fp.phi_state_deep,
        phi_deep_proofs=fp.phi_deep_proofs, phi_all_shallow=fp.phi_all_shallow,
        phi_full_proofs=fp.phi_full_proofs, f_core=fp.f_core, f_proof=fp.f_proof,
        f_env=fp.f_env)


def _divergent_class(a: str, b: str) -> A.ClassSummary:
    cls = A.ClassSummary(observation_key="o", layer="phi_state", is_natural=True,
                         reps=[_rep(a, {"decide": "COMPLETE"}), _rep(b, {"decide": "FAIL"})])
    cls.action_divergent, cls.divergent_tactics = A.class_divergence(cls, ["decide"])
    return cls


def _rung(name: str) -> A.RepairRung:
    return next(r for r in A.REPAIR_LADDER if r.name == name)


# --------------------------------------------------------------------------- #
def test_baseline_rung_never_repairs():
    """R1 is the default observation — by construction it cannot separate class members."""
    cls = _divergent_class("D_over_true", "D_over_false")
    assert cls.action_divergent
    assert A.class_repaired_by(cls, _rung("R1_default"), ["decide"]) is False


def test_deep_terms_rung_repairs_print_overflow():
    """R2 raises the print budget, which is exactly what an overflow alias needs."""
    cls = _divergent_class("D_over_true", "D_over_false")
    assert A.class_repaired_by(cls, _rung("R2_deep"), ["decide"]) is True


def test_deep_terms_rung_does_not_repair_instance_aliasing():
    """R2 keeps default notation, so instance-implicit arguments stay hidden — an in-goal
    instance alias needs the structural rung."""
    cls = _divergent_class("E_ingoal_2", "E_ingoal_0")
    assert A.class_repaired_by(cls, _rung("R2_deep"), ["decide"]) is False
    assert A.class_repaired_by(cls, _rung("R4_struct"), ["decide"]) is True


def test_structural_rung_does_not_repair_ambient_aliasing():
    """The scope pair has an identical goal; no amount of goal rendering separates it. This is
    the distinction that decides whether a class is closable by R4 or needs R6."""
    cls = _divergent_class("C_scope_N1", "C_scope_N2")
    assert A.class_repaired_by(cls, _rung("R4_struct"), ["decide"]) is False
    assert A.class_repaired_by(cls, _rung("R6_env"), ["decide"]) is True


def test_env_rung_repairs_simp_aliasing():
    cls = _divergent_class("A_with_simp", "A_without_simp")
    assert A.class_repaired_by(cls, _rung("R4_struct"), ["decide"]) is False
    assert A.class_repaired_by(cls, _rung("R6_env"), ["decide"]) is True


def test_unavailable_rung_returns_none_not_false():
    """A rung that cannot be evaluated must not be scored as a failed repair."""
    reps = [A.RepSummary("e1", 1, {"simp": "COMPLETE"}),   # no renderings at all
            A.RepSummary("e2", 1, {"simp": "FAIL"})]
    cls = A.ClassSummary(observation_key="o", layer="phi_state", is_natural=True, reps=reps)
    cls.action_divergent = True
    assert A.class_repaired_by(cls, _rung("R4_struct"), ["simp"]) is None


# --------------------------------------------------------------------------- #
def test_repair_curve_counts_and_reports_unevaluable():
    classes = [
        _divergent_class("D_over_true", "D_over_false"),   # repaired by R2
        _divergent_class("E_ingoal_2", "E_ingoal_0"),      # needs R4
        _divergent_class("C_scope_N1", "C_scope_N2"),      # needs R6
    ]
    curve = A.repair_curve(classes, ["decide"])
    by_name = {r["rung"]: r for r in curve["rungs"]}

    assert curve["n_action_divergent"] == 3
    assert by_name["R1_default"]["removed"] == 0
    assert by_name["R2_deep"]["removed"] == 1
    assert by_name["R4_struct"]["removed"] == 2
    assert by_name["R6_env"]["removed"] == 3
    assert by_name["R6_env"]["residual"] == 0
    # every rung must account for all divergent classes
    for row in curve["rungs"]:
        assert row["removed"] + row["residual"] + row["n_unevaluable"] == 3


def test_repair_curve_declares_unavailable_rungs():
    """R5 and R7 are in the blueprint but cannot be built on a pp-only interface. They must be
    declared, not silently omitted."""
    curve = A.repair_curve([], ["decide"])
    assert "R5_mvar_digest" in curve["unavailable_rungs"]
    assert "R7_retrieved_premise" in curve["unavailable_rungs"]


def test_repair_costs_are_measured_and_monotone():
    """Cost is observation growth in bytes vs φ_state — measured, not assumed (§7.10)."""
    classes = [_divergent_class("E_ingoal_2", "E_ingoal_0")]
    curve = A.repair_curve(classes, ["decide"])
    by_name = {r["rung"]: r for r in curve["rungs"]}
    assert by_name["R1_default"]["median_added_bytes"] == 0
    assert by_name["R4_struct"]["median_added_bytes"] > 0
    assert by_name["R6_env"]["median_added_bytes"] >= by_name["R4_struct"]["median_added_bytes"]


def test_residual_after_env_repair_is_the_crux_metric():
    """Whether anything survives R6 decides interface-note vs paper (README limitation 2)."""
    classes = [_divergent_class("C_scope_N1", "C_scope_N2")]
    assert A.residual_after_repair(classes, ["decide"], "R6_env") == []


def test_repair_efficiency_ratio():
    assert A.repair_efficiency(10, 99.0) == 0.1
    assert A.repair_efficiency(0, 0.0) == 0.0
