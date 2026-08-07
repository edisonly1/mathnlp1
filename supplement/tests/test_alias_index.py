"""Unit tests for the alias index (blueprint §4.3, §9). Pure Python — runs off-box."""
from audit.alias_index import candidate_classes, finalize_class, latent_classes
from audit.schema import Fingerprint, Observation, Provenance, StateRecord


def _rec(phi_state: str, phi_full: str, env: str, i: int, fp: bool = True) -> StateRecord:
    prov = Provenance(
        repo_url="r", repo_commit="c", file_path=f"F{i}.lean",
        theorem_full_name=f"thm{i}", tactic_index=0,
        lean_toolchain="t", leandojo_version="v")
    rec = StateRecord(provenance=prov, observation=Observation(phi_state=phi_state))
    if fp:
        rec.fingerprint = Fingerprint.from_parts(phi_full, env, "test", True)
    return rec


def test_candidate_requires_two_members():
    recs = [_rec("G", "G", "e", 0), _rec("H", "H", "e", 1)]  # distinct φ_state -> no class
    assert candidate_classes(recs, "phi_state") == []


def test_latent_class_detected_when_exec_differs():
    # same φ_state "G", different env digest -> different F_exec -> latent alias.
    recs = [_rec("G", "G", "envA", 0), _rec("G", "G", "envB", 1)]
    cands = candidate_classes(recs, "phi_state")
    assert len(cands) == 1
    obs_key, members = cands[0]
    cls, reps = finalize_class(obs_key, "phi_state", members)
    assert cls.is_latent
    assert len(reps) == 2  # two distinct executable states


def test_trivial_duplicate_not_latent():
    # identical φ_state AND identical fingerprint -> just a duplicate, not an alias.
    recs = [_rec("G", "G", "envA", 0), _rec("G", "G", "envA", 1)]
    obs_key, members = candidate_classes(recs, "phi_state")[0]
    cls, reps = finalize_class(obs_key, "phi_state", members)
    assert not cls.is_latent
    assert len(reps) == 1


def test_latent_filter():
    recs = [_rec("G", "G", "envA", 0), _rec("G", "G", "envB", 1),
            _rec("K", "K", "envA", 2), _rec("K", "K", "envA", 3)]
    finalized = [finalize_class(k, "phi_state", m) for k, m in candidate_classes(recs, "phi_state")]
    lat = latent_classes(finalized)
    assert len(lat) == 1
    assert lat[0][0].observation_key == recs[0].observation.state_hash()
