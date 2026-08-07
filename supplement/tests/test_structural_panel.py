"""Tests for the structural tactic panel (blueprint §7.6).

The blueprint requires "structurally instantiated `rw`, `apply`, and `exact` candidates when
the required identifier appears in the model-visible input". `structural_panel` was present in
config.yaml but nothing read it, so the core panel was the only action set actually replayed.
"""
from audit.run_gate1 import structural_actions

PREFIXES = ["rw", "apply", "exact"]

STATE = """h : a = b
hb : b = c
⊢ a = c"""


def test_instantiates_on_visible_hypotheses():
    acts = structural_actions(STATE, PREFIXES)
    assert "rw [h]" in acts
    assert "apply h" in acts
    assert "exact hb" in acts


def test_only_uses_identifiers_the_model_can_see():
    """Every instantiated identifier must actually occur in φ_state — the point of the rule is
    that the model could have named it."""
    acts = structural_actions(STATE, PREFIXES)
    assert acts
    for act in acts:
        ident = act[4:-1] if act.startswith("rw [") else act.split(maxsplit=1)[1]
        assert ident in STATE, f"{ident!r} is not visible in the state"


def test_goal_only_state_yields_nothing():
    assert structural_actions("⊢ f 0 = 0", PREFIXES) == []


def test_stops_at_the_turnstile():
    """Text after ⊢ is the goal, not a hypothesis binder."""
    acts = structural_actions("h : P\n⊢ q : Nat", PREFIXES)
    assert "rw [h]" in acts
    assert not any("q" in a for a in acts)


def test_respects_cap_and_is_deterministic():
    state = "\n".join(f"h{i} : P{i}" for i in range(50)) + "\n⊢ Q"
    a = structural_actions(state, PREFIXES, cap=12)
    b = structural_actions(state, PREFIXES, cap=12)
    assert len(a) == 12 and a == b


def test_handles_unicode_hypothesis_names():
    acts = structural_actions("h₀ : a = b\n⊢ a = b", PREFIXES)
    assert "rw [h₀]" in acts


def test_empty_state_is_safe():
    assert structural_actions("", PREFIXES) == []
