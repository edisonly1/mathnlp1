"""End-to-end attribution tests against REAL Lean output (blueprint §8.1.6).

The fixture in `fixtures/gate0_fingerprints.json` is produced by running the exact metaprogram
the harness injects (`audit.fingerprint.TACTIC_SRC_INLINE`) over the certified Gate-0 witnesses
under the pinned toolchain — see `gen_fingerprint_fixture.py`. These tests therefore check the
whole chain (Lean rendering → JSON → Fingerprint → mechanism attribution) rather than hand-
written strings that could agree with a wrong implementation.
"""
import json
from pathlib import Path

import pytest

from audit import analysis as A
from audit.fingerprint import (MAX_TACTIC_BYTES, PROBES, build_fingerprint,
                               degraded_fingerprint)

FIXTURE = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "gate0_fingerprints.json")
    .read_text(encoding="utf-8"))


def _rep(label: str, weight: int = 1, outcomes=None) -> A.RepSummary:
    fp = build_fingerprint(FIXTURE[label], channel="test", faithful=True)
    return A.RepSummary(
        f_exec=fp.f_exec, weight=weight, outcomes=outcomes or {},
        phi_full=fp.phi_full, env_digest=fp.env_digest,
        phi_deep=fp.phi_state_deep, phi_deep_proofs=fp.phi_deep_proofs,
        phi_all_shallow=fp.phi_all_shallow, phi_full_proofs=fp.phi_full_proofs,
        env_provenance=fp.env_provenance,
        f_core=fp.f_core, f_proof=fp.f_proof, f_env=fp.f_env)


def _cls(a: str, b: str, **kw) -> A.ClassSummary:
    return A.ClassSummary(observation_key="o", layer="phi_state", is_natural=True,
                          reps=[_rep(a), _rep(b)], **kw)


# --------------------------------------------------------------------------- #
# Mechanism attribution, one case per certified mechanism
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a,b,expected", [
    ("A_with_simp", "A_without_simp", "active_environment_simp"),
    ("B_normal", "B_degenerate", "hidden_ingoal_instances_or_implicits"),
    ("C_scope_N1", "C_scope_N2", "namespace_scope"),
    ("D_over_true", "D_over_false", "printer_overflow"),
    ("E_ingoal_2", "E_ingoal_0", "hidden_ingoal_instances_or_implicits"),
    ("CTRL_proof_aa", "CTRL_proof_ab", "proof_term_only"),
])
def test_mechanism_attributed(a, b, expected):
    mechs = A.attribute_mechanisms(_cls(a, b, has_ellipsis=("over" in a)))
    assert expected in mechs, f"{a}/{b}: expected {expected}, got {mechs}"


def test_only_print_overflow_is_dismissible():
    """§10.2 cond 4 turns on this distinction: the overflow pair must be dismissible and every
    other certified mechanism must not be."""
    overflow = A.attribute_mechanisms(_cls("D_over_true", "D_over_false", has_ellipsis=True))
    assert A.is_overflow_only(overflow), overflow
    for a, b in [("A_with_simp", "A_without_simp"), ("B_normal", "B_degenerate"),
                 ("C_scope_N1", "C_scope_N2"), ("E_ingoal_2", "E_ingoal_0")]:
        mechs = A.attribute_mechanisms(_cls(a, b))
        assert not A.is_overflow_only(mechs), f"{a}/{b} wrongly dismissed as overflow: {mechs}"


def test_overflow_pair_is_core_latent():
    """Regression for the blindness this suite was written to catch: `pp.all` does not imply
    `pp.deepTerms`, so without setting it the overflow pair rendered identically at every level,
    F_core collapsed, and the class was silently dropped as a trivial duplicate."""
    cls = _cls("D_over_true", "D_over_false", has_ellipsis=True)
    assert cls.is_core_latent
    assert cls.reps[0].phi_all_shallow == cls.reps[1].phi_all_shallow, \
        "shallow renderings must agree — the difference lives only in the elided region"
    assert cls.reps[0].phi_full != cls.reps[1].phi_full


def test_scope_pair_is_detected_at_all():
    """Mechanism C differs only in `open` declarations. The v1 digest recorded `getCurrNamespace`
    (identical here) and the module name (identical here), so F_exec collapsed and the pair was
    invisible to the pipeline."""
    cls = _cls("C_scope_N1", "C_scope_N2")
    assert cls.is_latent, "scope pair must be latent at the executable level"
    assert cls.reps[0].f_core == cls.reps[1].f_core, "the goal itself is identical"
    assert cls.reps[0].f_env != cls.reps[1].f_env, "the ambient environment differs"


def test_proof_control_is_inert_at_core_level():
    """Proof irrelevance: the pair must NOT be core-latent, since only the proof term differs."""
    cls = _cls("CTRL_proof_aa", "CTRL_proof_ab")
    assert not cls.is_core_latent
    assert cls.reps[0].f_proof != cls.reps[1].f_proof
    assert not A.is_overflow_only(A.attribute_mechanisms(cls)), \
        "a proof-only difference is inert, not a dismissible renderer artifact"


def test_module_identity_is_not_a_mechanism():
    """Module identity is provenance. Folding it into the environment digest labelled every
    cross-file class `namespace_scope`, which made `overflow_only` unreachable and disabled the
    §10.2 guard against a pure pretty-printer result."""
    r1, r2 = _rep("D_over_true"), _rep("D_over_false")
    r1.env_provenance, r2.env_provenance = "mod=Mathlib.Alpha", "mod=Mathlib.Beta"
    cls = A.ClassSummary(observation_key="o", layer="phi_state", is_natural=True,
                         reps=[r1, r2], has_ellipsis=True,
                         files=["Alpha.lean", "Beta.lean"])
    mechs = A.attribute_mechanisms(cls)
    assert "namespace_scope" not in mechs
    assert A.is_overflow_only(mechs), mechs


def test_same_environment_flag():
    assert _cls("D_over_true", "D_over_false").same_environment
    assert not _cls("A_with_simp", "A_without_simp").same_environment


# --------------------------------------------------------------------------- #
# Guards on the injected tactic itself
# --------------------------------------------------------------------------- #
def test_probes_set_the_two_critical_pp_options():
    """`pp.all` implies neither of these, and getting either wrong silently breaks a whole
    mechanism class. Guard the literal source rather than trusting review."""
    assert "`pp.deepTerms true" in PROBES["phi_full"]
    assert "`pp.proofs false" in PROBES["phi_full"]
    # the shallow rendering must also pin proofs off, or the overflow discriminator is unsound
    assert "`pp.proofs false" in PROBES["phi_all_shallow"]
    # ...and must NOT force deep terms, or it stops being the shallow half of the pair
    assert "`pp.deepTerms" not in PROBES["phi_all_shallow"]


def test_every_probe_is_under_the_run_tac_size_limit():
    """`Dojo.run_tac` deadlocks at ~1 KB of tactic text, wedging the session for every later
    state. Measured cliff: 950B ok, 1000B hangs. This is the guard that keeps a future edit
    from silently reintroducing a whole-run failure."""
    for name, src in PROBES.items():
        n = len(src.encode("utf-8"))
        assert n < MAX_TACTIC_BYTES, f"probe {name!r} is {n}B, at/over the {MAX_TACTIC_BYTES}B limit"


def test_module_identity_is_a_separate_probe_from_env_behavior():
    """`mod=` must be collected as provenance only, never folded into `env_behavior`."""
    from audit.fingerprint import ENV_FIELDS
    assert "env_provenance" not in ENV_FIELDS
    assert "mod=" in PROBES["env_provenance"]
    for f in ENV_FIELDS:
        assert "mainModule" not in PROBES[f]


def test_degraded_fingerprint_cannot_manufacture_latency():
    """Two members that both failed extraction must collapse to one f_exec, so a fingerprinting
    outage can never look like a discovered alias."""
    a = degraded_fingerprint("⊢ True", "throwError")
    b = degraded_fingerprint("⊢ True", "throwError")
    assert a.f_exec == b.f_exec
    assert not a.faithful


def test_fixture_covers_every_certified_mechanism():
    expected = {"A_with_simp", "A_without_simp", "B_normal", "B_degenerate",
                "C_scope_N1", "C_scope_N2", "D_over_true", "D_over_false",
                "E_ingoal_2", "E_ingoal_0", "CTRL_proof_aa", "CTRL_proof_ab"}
    assert expected == set(FIXTURE)


def test_asymmetric_ellipsis_is_not_called_an_instance_difference():
    """`pp.proofs false` still prints ATOMIC proofs (a bare hypothesis name), and
    `pp.proofs.threshold 0` does not change that — verified against Lean v4.32. So a proof-term
    difference reaches `phi_all_shallow`, and "shallow differs" alone cannot mean instances.

    Regression for the §8.3 audit of class 8c33dbf195, whose real difference was `n hn)` vs
    `n ⋯)` — a proof term — but which was reported as an in-goal instance difference.
    """
    reps = [
        A.RepSummary("e1", 1, {}, phi_full="F(n hn)", phi_all_shallow="S(n hn)", env_digest="E"),
        A.RepSummary("e2", 1, {}, phi_full="F(n ⋯)", phi_all_shallow="S(n ⋯)", env_digest="E"),
    ]
    mechs = A.attribute_mechanisms(
        A.ClassSummary(observation_key="o", layer="phi_state", is_natural=True, reps=reps))
    assert "elided_content_proof_or_deepterm" in mechs
    assert "hidden_ingoal_instances_or_implicits" not in mechs


def test_symmetric_ellipsis_still_reads_as_instance_difference():
    """When neither side hides anything the shallow difference really is structural."""
    reps = [
        A.RepSummary("e1", 1, {}, phi_full="F(instA)", phi_all_shallow="S(instA)", env_digest="E"),
        A.RepSummary("e2", 1, {}, phi_full="F(instB)", phi_all_shallow="S(instB)", env_digest="E"),
    ]
    mechs = A.attribute_mechanisms(
        A.ClassSummary(observation_key="o", layer="phi_state", is_natural=True, reps=reps))
    assert "hidden_ingoal_instances_or_implicits" in mechs
