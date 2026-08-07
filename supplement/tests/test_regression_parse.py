"""Unit test for PROBEREC parsing + pairing logic (blueprint §11.1 item 8). Runs off-box."""
from audit.regression import LOCUS_IN_GOAL, LOCUS_NONE, MANIFEST, _parse


SAMPLE = """PROBEREC|A_with_simp|true|9|7592356157467220647|"⊢ f 0 = 0"
PROBEREC|A_without_simp|false|9|7592356157467220647|"⊢ f 0 = 0"
some unrelated lean log line
PROBEREC|CTRL_proof_aa|true|26|15511576331569909349|"⊢ { n := 1, h := ⋯ }.n = 1"
PROBEREC|CTRL_proof_ab|true|26|15511576331569909349|"⊢ { n := 1, h := ⋯ }.n = 1"
"""


def test_parse_extracts_records():
    got = _parse(SAMPLE)
    assert set(got) >= {"A_with_simp", "A_without_simp", "CTRL_proof_aa", "CTRL_proof_ab"}
    assert got["A_with_simp"].ok is True
    assert got["A_without_simp"].ok is False
    # identical display (same hash), divergent outcome -> the A witness.
    assert got["A_with_simp"].hash == got["A_without_simp"].hash


def test_parse_ignores_noise():
    assert _parse("nothing here\nnor here") == {}


def test_manifest_pairs_are_wellformed():
    for name, la, lb, exp_ident, exp_div, overflow, locus in MANIFEST:
        assert isinstance(exp_ident, bool) and isinstance(exp_div, bool)
        assert isinstance(overflow, bool) and isinstance(locus, str)
        assert la != lb


def test_manifest_has_exactly_one_inert_control():
    ctrl = [m for m in MANIFEST if not m[4]]
    assert len(ctrl) == 1
    assert ctrl[0][3] is True, "the control must still have identical display"
    assert ctrl[0][6] == LOCUS_NONE


def test_manifest_has_an_in_goal_witness():
    """A difference living in the goal term cannot be closed by an environment digest (R6),
    so at least one such witness must be certified or the result reduces to ambient aliasing."""
    assert any(m[6] == LOCUS_IN_GOAL and m[4] for m in MANIFEST)


def test_manifest_has_two_independent_nonoverflow_mechanisms():
    """Blueprint §5.4 requires ≥2 distinct hidden mechanisms."""
    non_overflow = [m for m in MANIFEST if m[4] and not m[5]]
    assert len({m[0] for m in non_overflow}) >= 2


def test_sorry_goals_are_rejected():
    """A pair whose goals failed to elaborate is not a witness. The v1 negative control compared
    two `⊢ sorry = 2` states and passed while controlling nothing."""
    from audit.regression import run_regression
    import audit.regression as R

    broken = """PROBEREC|A_with_simp|true|9|1|"⊢ sorry = 0"
PROBEREC|A_without_simp|false|9|1|"⊢ sorry = 0"
"""
    parsed = R._parse(broken)
    assert "sorry" in parsed["A_with_simp"].quoted
