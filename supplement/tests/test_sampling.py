"""Tests for the stratified sampling allocation (blueprint §7.2).

`allocate_quotas` is the pure core of `sample_states`, split out so the sampling policy is
testable without LeanDojo. It replaces an allocation that gave every stratum
`max(1, round(share*n))` and then truncated a shuffled concatenation to `n` — with six
stratification keys the stratum count exceeds `n`, so the floor alone overshot the budget and
the truncation degenerated into a uniform sample over strata, discarding the stratification.
"""
from audit.extract import _tactic_family, _has_ellipsis, allocate_quotas


def _avail(seen: dict, cap: int = 512) -> dict:
    return {k: min(v, cap) for k, v in seen.items()}


def test_allocation_sums_to_budget():
    seen = {"a": 500, "b": 300, "c": 200}
    q = allocate_quotas(seen, _avail(seen), 100)
    assert sum(q.values()) == 100


def test_allocation_is_proportional():
    seen = {"a": 500, "b": 300, "c": 200}
    q = allocate_quotas(seen, _avail(seen), 100)
    assert q["a"] == 50 and q["b"] == 30 and q["c"] == 20


def test_more_strata_than_budget_does_not_overshoot():
    """The failure mode of the previous implementation: 5,000 strata and a budget of 100 gave
    every stratum a floor of 1, allocating 5,000 records before truncating."""
    seen = {f"s{i}": 10 for i in range(5000)}
    q = allocate_quotas(seen, _avail(seen), 100)
    assert sum(q.values()) == 100
    assert all(v >= 0 for v in q.values())


def test_allocation_respects_availability():
    """A stratum cannot contribute more records than its reservoir actually holds."""
    seen = {"big": 10_000, "small": 10}
    avail = {"big": 512, "small": 10}
    q = allocate_quotas(seen, avail, 1000)
    assert q["small"] <= 10
    assert q["big"] <= 512
    assert sum(q.values()) == min(1000, 522)


def test_allocation_is_deterministic():
    seen = {f"s{i}": i + 1 for i in range(50)}
    a = allocate_quotas(seen, _avail(seen), 97)
    b = allocate_quotas(seen, _avail(seen), 97)
    assert a == b


def test_rare_strata_are_retained():
    """Oversampling rare strata is the point of stratifying (§7.2); a rare stratum must not be
    allocated zero purely because its proportional share rounds down."""
    seen = {"common": 100_000, "rare": 3}
    avail = _avail(seen)
    q = allocate_quotas(seen, avail, 1000)
    assert q["rare"] >= 1


def test_empty_input_is_safe():
    assert allocate_quotas({}, {}, 100) == {}
    assert allocate_quotas({"a": 5}, {"a": 5}, 0) == {}


def test_budget_larger_than_corpus():
    seen = {"a": 3, "b": 2}
    q = allocate_quotas(seen, _avail(seen), 1000)
    assert sum(q.values()) == 5


# --------------------------------------------------------------------------- #
# Stratification helpers
# --------------------------------------------------------------------------- #
def test_tactic_family_is_the_head_token():
    assert _tactic_family("  simp [foo, bar]  ") == "simp"
    assert _tactic_family("rw [h]") == "rw"
    assert _tactic_family("") == ""


def test_ellipsis_detection_covers_lean_renderings():
    assert _has_ellipsis("⊢ id (id ⋯) = 5")
    assert _has_ellipsis("a … b")
    assert _has_ellipsis("a ... b")
    assert not _has_ellipsis("⊢ f 0 = 0")
