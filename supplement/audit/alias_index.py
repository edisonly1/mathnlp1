"""Component 4 — Alias index (blueprint §4.3, §7.5, §9).

Groups sampled states by their exact model-visible observation and identifies **latent alias
classes**: groups sharing an observation `o` that contain ≥2 distinct execution fingerprints
`F_exec` (blueprint §4.3, `|{F_exec(s)} : s ∈ C_o| > 1`).

Two grouping layers are produced (blueprint §7.5):
  * `phi_state` — identical ppGoal (the theorem-prover interface claim);
  * `phi_tok`   — identical tokenized+truncated model input (the strongest model-level claim).

Discipline (blueprint §9): do NOT dedup by the visible string before fingerprinting — duplication
is the object of study. DO dedup exact repeated executable states (identical `F_exec`) *after*
fingerprinting, preserving multiplicity separately.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .schema import AliasClass, StateRecord


def observation_key(rec: StateRecord, layer: str) -> Optional[str]:
    if layer == "phi_state":
        return rec.observation.state_hash()
    if layer == "phi_tok":
        return rec.observation.tok_hash()  # None if φ_tok not built yet
    raise ValueError(f"unknown layer {layer!r}")


def group_by_observation(records: list[StateRecord], layer: str) -> dict[str, list[StateRecord]]:
    groups: dict[str, list[StateRecord]] = defaultdict(list)
    for rec in records:
        k = observation_key(rec, layer)
        if k is None:
            continue
        groups[k].append(rec)
    return groups


def candidate_classes(records: list[StateRecord], layer: str,
                      min_members: int = 2) -> list[tuple[str, list[StateRecord]]]:
    """Groups with ≥2 members — the candidates worth the expensive fingerprint/replay pass.

    A class with a single member cannot be an alias. Note a group may still be a *trivial*
    duplicate (the same executable state traced twice); fingerprinting resolves that.
    """
    groups = group_by_observation(records, layer)
    return [(k, members) for k, members in groups.items() if len(members) >= min_members]


def _exec_of(rec: StateRecord) -> str:
    """Key a member by its executable fingerprint, or by φ_state if it has none.

    A DEGRADED fingerprint counts as "none". It carries `phi_full="DEGRADED:<φ_state>"` and
    `env_digest="NA"`, so comparing it against a member that extracted successfully differs in
    every dimension — which manufactures latency out of an extraction failure and, downstream,
    manufactures `active_environment_*` / `namespace_scope` mechanisms. Both false-latent
    classes in the first shakedown run came from exactly this.
    """
    from .fingerprint import is_degraded

    if rec.fingerprint is not None and not is_degraded(rec.fingerprint):
        return rec.fingerprint.f_exec
    # Unfingerprinted or degraded: key on φ_state so it never spuriously creates latency.
    return "UNFP:" + rec.observation.state_hash()


def finalize_class(observation_key_hash: str, layer: str, members: list[StateRecord],
                   is_natural: bool = True) -> tuple[AliasClass, dict[str, StateRecord]]:
    """Build an AliasClass and a map {F_exec -> representative member}.

    Replay only needs one representative per distinct F_exec (identical F_exec ⇒ identical
    executable state ⇒ identical deterministic replay), which bounds replay cost by the number
    of *distinct* internal states, not raw multiplicity.
    """
    reps: dict[str, StateRecord] = {}
    per_member_exec: list[str] = []
    for m in members:
        fe = _exec_of(m)
        per_member_exec.append(fe)
        reps.setdefault(fe, m)

    cls = AliasClass(
        observation_key=observation_key_hash,
        layer=layer,
        member_keys=[m.provenance.key for m in members],
        exec_fingerprints=per_member_exec,
        is_natural=is_natural,
    )
    return cls, reps


def latent_classes(finalized: list[tuple[AliasClass, dict[str, StateRecord]]]
                   ) -> list[tuple[AliasClass, dict[str, StateRecord]]]:
    """Filter to genuinely latent classes: ≥2 distinct F_exec (blueprint §4.3)."""
    return [(c, reps) for (c, reps) in finalized if c.is_latent]


def summarize(finalized: list[tuple[AliasClass, dict[str, StateRecord]]]) -> dict:
    """Quick counts for logging / the report header."""
    total = len(finalized)
    latent = sum(1 for c, _ in finalized if c.is_latent)
    unfingerprinted = sum(
        1 for c, reps in finalized
        if any(fe.startswith("UNFP:") for fe in c.exec_fingerprints))
    return {
        "candidate_classes": total,
        "latent_classes": latent,
        "classes_with_unfingerprinted_members": unfingerprinted,
    }
