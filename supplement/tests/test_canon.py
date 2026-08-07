"""Unit tests for the canonicalization policy (blueprint §7.4, §9). Pure Python."""
from audit.canon import (alpha_normalize, behavioral_options, canonical_env, env_fields,
                         options_digest, parse_options)


# --------------------------------------------------------------------------- #
# Alpha-renaming (blueprint §9)
# --------------------------------------------------------------------------- #
def test_inaccessible_names_are_positional():
    a = "x✝ : Nat\ny✝ : Nat\n⊢ x✝ = y✝"
    b = "a✝ : Nat\nb✝ : Nat\n⊢ a✝ = b✝"
    assert alpha_normalize(a) == alpha_normalize(b)


def test_alpha_normalization_preserves_distinctness():
    """Renaming must not collapse states that genuinely differ in structure."""
    a = alpha_normalize("x✝ : Nat\ny✝ : Nat\n⊢ x✝ = y✝")
    b = alpha_normalize("x✝ : Nat\ny✝ : Nat\n⊢ y✝ = x✝")
    assert a != b


def test_superscripted_shadow_names():
    a = "n✝¹ : Nat\nn✝ : Nat\n⊢ n✝¹ = n✝"
    b = "m✝¹ : Nat\nm✝ : Nat\n⊢ m✝¹ = m✝"
    assert alpha_normalize(a) == alpha_normalize(b)


def test_metavariable_ids_are_renumbered():
    assert alpha_normalize("⊢ ?m.4711 = ?m.4712") == alpha_normalize("⊢ ?m.99 = ?m.100")
    # but distinct mvars stay distinct
    assert alpha_normalize("⊢ ?m.1 = ?m.2") != alpha_normalize("⊢ ?m.1 = ?m.1")


def test_uniq_names_renumbered():
    assert alpha_normalize("_uniq.55 _uniq.56") == alpha_normalize("_uniq.1 _uniq.2")


def test_plain_goals_untouched():
    assert alpha_normalize("⊢ f 0 = 0") == "⊢ f 0 = 0"


def test_empty_is_safe():
    assert alpha_normalize("") == ""


# --------------------------------------------------------------------------- #
# Option filtering
# --------------------------------------------------------------------------- #
RAW = "[(Elab.async, true), (pp.deepTerms.threshold, 2), (internal.cmdlineSnapshots, true), (pp.deepTerms, false)]"


def test_parse_options():
    opts = parse_options(RAW)
    assert opts["Elab.async"] == "true"
    assert opts["pp.deepTerms"] == "false"
    assert opts["pp.deepTerms.threshold"] == "2"
    assert len(opts) == 4


def test_rendering_options_are_stripped():
    """A `set_option pp.*` cannot change tactic behavior; leaving it in the digest makes the
    print-overflow witness look like an `options_transparency` mechanism."""
    kept = behavioral_options(RAW)
    assert set(kept) == {"Elab.async", "internal.cmdlineSnapshots"}


def test_digest_ignores_rendering_only_differences():
    with_pp = "[(Elab.async, true), (pp.deepTerms, false)]"
    without = "[(Elab.async, true)]"
    assert options_digest(with_pp) == options_digest(without)


def test_digest_detects_behavioral_differences():
    a = "[(maxHeartbeats, 200000)]"
    b = "[(maxHeartbeats, 400000)]"
    assert options_digest(a) != options_digest(b)


def test_digest_is_order_independent():
    a = "[(alpha, 1), (beta, 2)]"
    b = "[(beta, 2), (alpha, 1)]"
    assert options_digest(a) == options_digest(b)


def test_empty_options():
    assert parse_options("") == {}
    assert parse_options("NA") == {}


# --------------------------------------------------------------------------- #
# Env digest plumbing
# --------------------------------------------------------------------------- #
def test_canonical_env_appends_options_digest():
    env = "ns=[anonymous]|open=[]|linst=[]|simpN=10|simpH=abc"
    out = canonical_env(env, RAW)
    assert out.startswith(env)
    assert "optsH=" in out


def test_env_fields_roundtrip():
    f = env_fields("ns=Foo|open=[N1]|simpN=10|simpH=abc")
    assert f["ns"] == "Foo"
    assert f["open"] == "[N1]"
    assert f["simpN"] == "10"
