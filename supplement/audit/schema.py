"""Typed records shared across the pipeline (blueprint §11.2 data schema).

Every record is JSON-serializable so a run is fully reconstructable from disk
(blueprint §11.3: "do not reconstruct the model input later from partially logged fields").
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Identity / provenance (blueprint §11.2 "Identity", §7.3 observation recorder)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Provenance:
    repo_url: str
    repo_commit: str
    file_path: str
    theorem_full_name: str
    tactic_index: int          # position within the human proof (0-based)
    lean_toolchain: str
    leandojo_version: str
    # stratification metadata (blueprint §7.2)
    proof_depth: int = 0
    goal_count: int = 0
    token_length: int = 0
    has_ellipsis: bool = False
    tactic_family: str = ""

    @property
    def key(self) -> str:
        return f"{self.file_path}::{self.theorem_full_name}::{self.tactic_index}"


# --------------------------------------------------------------------------- #
# Observation layers (blueprint §4.1: φ_state, φ_RAG, φ_tok)
# --------------------------------------------------------------------------- #
@dataclass
class Observation:
    phi_state: str                     # ppGoal, default options — what a text prover sees
    phi_rag: Optional[str] = None      # retrieved premises ⊕ φ_state
    phi_tok_ids: Optional[list[int]] = None    # tokenized+truncated model input (the real input)
    truncated: bool = False            # did φ_tok drop content?

    def state_hash(self) -> str:
        return _sha(self.phi_state)

    def tok_hash(self) -> Optional[str]:
        if self.phi_tok_ids is None:
            return None
        return _sha(",".join(map(str, self.phi_tok_ids)))


# --------------------------------------------------------------------------- #
# Execution fingerprint (blueprint §7.4: two-level F_core / F_exec)
# --------------------------------------------------------------------------- #
@dataclass
class Fingerprint:
    """The rendering ladder plus the derived hashes (blueprint §7.4).

    Four hashes rather than two, because attribution needs to distinguish *where* a hidden
    difference lives, and prevalence needs a level that is not saturated by provenance:

      * `f_core`  — α-normalized `phi_full`: the goal's non-proof structure, deep, at pp.all.
                    **This is the prevalence headline.** Two states sharing φ_state with
                    different `f_core` differ in the mathematical content of the goal itself.
      * `f_proof` — same including proof terms. Differs-only-here ⇒ inert under proof
                    irrelevance (blueprint §5.3 negative control).
      * `f_env`   — behavior-relevant ambient state (namespace, opens, local instances, simp
                    set, non-rendering options). Module identity is NOT included.
      * `f_exec`  — hash(f_core ⊕ f_env): the executable state.

    `f_exec` saturates across files (two Mathlib files essentially always have different simp
    sets), so `f_exec`-latency is close to vacuous at corpus scale and must not be reported as
    the RQ1 prevalence number. See `analysis.prevalence`, which reports both.
    """
    phi_full: str                      # pp.all + deepTerms, proofs OFF — basis of F_core
    env_digest: str                    # canonical behavior-relevant ambient digest
    f_core: str
    f_exec: str
    # richer ladder (optional so degraded/legacy records still construct)
    phi_state_deep: str = ""           # default pp + deepTerms                  (repair rung R2)
    phi_deep_proofs: str = ""          # R2 + proof terms                        (repair rung R3)
    phi_all_shallow: str = ""          # pp.all, proofs off, deep elision intact (overflow probe)
    phi_full_proofs: str = ""          # pp.all + deepTerms + proofs
    env_provenance: str = ""           # module identity — reported, never hashed into f_exec
    f_proof: str = ""
    f_env: str = ""
    schema_version: str = ""
    # provenance of how it was obtained, for auditing
    channel: str = "throwError"
    faithful: bool = False             # True only if produced by the frontend re-elaboration exporter

    @staticmethod
    def from_parts(phi_full: str, env_digest: str, channel: str, faithful: bool,
                   *, phi_state_deep: str = "", phi_deep_proofs: str = "",
                   phi_all_shallow: str = "", phi_full_proofs: str = "",
                   env_provenance: str = "", raw_options: str = "") -> "Fingerprint":
        from .canon import SCHEMA_VERSION, alpha_normalize, canonical_env

        canon_env = canonical_env(env_digest, raw_options) if raw_options else env_digest
        f_core = _sha(alpha_normalize(phi_full))
        f_proof = _sha(alpha_normalize(phi_full_proofs or phi_full))
        f_env = _sha(canon_env)
        f_exec = _sha(f_core + "|" + f_env)
        return Fingerprint(
            phi_full=phi_full, env_digest=canon_env,
            f_core=f_core, f_exec=f_exec, f_proof=f_proof, f_env=f_env,
            phi_state_deep=phi_state_deep, phi_deep_proofs=phi_deep_proofs,
            phi_all_shallow=phi_all_shallow, phi_full_proofs=phi_full_proofs,
            env_provenance=env_provenance, schema_version=SCHEMA_VERSION,
            channel=channel, faithful=faithful)


@dataclass
class StateRecord:
    """One sampled executable tactic state (blueprint §11.2 full record)."""
    provenance: Provenance
    observation: Observation
    fingerprint: Optional[Fingerprint] = None   # filled lazily, only for candidate-class members
    human_tactic: str = ""                       # the tactic the human actually ran here
    human_outcome: Optional["ReplayOutcome"] = None
    # Tactic strings 0..idx-1, needed to fast-forward a Dojo to this state for replay (§7.7).
    proof_prefix: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Canonicalized tactic outcomes (blueprint §4.2 Ψ)
# --------------------------------------------------------------------------- #
class OutcomeKind(str, Enum):
    FAIL = "FAIL"           # logical failure (tactic error, unsolved after)
    TIMEOUT = "TIMEOUT"     # resource/hard timeout — NEVER merged with FAIL (§7.6)
    COMPLETE = "COMPLETE"   # proof finished (no goals remain)
    SUCC = "SUCC"           # succeeded, nonterminal — carries successor fingerprints
    GAVE_UP = "GAVE_UP"
    CRASH = "CRASH"         # Dojo/toolchain crash — audited separately, not a logical outcome


@dataclass
class ReplayOutcome:
    kind: OutcomeKind
    # For SUCC: canonical successor signatures (multiset- or order-normalized per §4.2).
    successor_sig: list[str] = field(default_factory=list)
    wall_time_s: float = 0.0
    message_digest: str = ""     # hash of tactic message / stderr for audit
    reproducible: bool = True    # set False if repeats disagree (§7.7)

    def canonical(self, goal_order_sensitive: bool) -> str:
        """Ψ(T(s,a)) — the value compared across alias-class members (blueprint §4.2)."""
        if self.kind != OutcomeKind.SUCC:
            return self.kind.value
        sig = self.successor_sig if goal_order_sensitive else sorted(self.successor_sig)
        return "SUCC(" + ";".join(sig) + ")"


# --------------------------------------------------------------------------- #
# Alias classes (blueprint §4.3)
# --------------------------------------------------------------------------- #
@dataclass
class AliasClass:
    observation_key: str               # the shared φ (state or tok) hash
    layer: str                         # 'phi_state' | 'phi_tok'
    member_keys: list[str]             # provenance keys of members
    exec_fingerprints: list[str]       # distinct F_exec present
    is_natural: bool = True            # natural vs certified (blueprint §7.8) — never pooled
    # filled by replay/analysis
    action_divergent: bool = False
    divergent_tactics: list[str] = field(default_factory=list)
    severity: str = ""                 # 'completion_vs_failure' | 'success_vs_failure' | 'succ_count' | ''
    mechanisms: list[str] = field(default_factory=list)   # source attribution (§8.1.6)
    overflow_only: bool = True         # True if every divergence is pure print-step overflow

    @property
    def is_latent(self) -> bool:
        """Latent alias: same observation, ≥2 distinct execution fingerprints (blueprint §4.3)."""
        return len(set(self.exec_fingerprints)) > 1

    @property
    def multiplicity(self) -> int:
        return len(self.member_keys)
