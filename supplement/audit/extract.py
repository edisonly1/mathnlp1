"""Component 1 — Lean exporter (blueprint §7.3, §11.1).

Traces the pinned corpus with LeanDojo and emits one StateRecord per sampled tactic
state: the *default* pretty-printed state (φ_state, exactly what a text-conditioned prover
consumes) plus full provenance. Fingerprints (φ_full / F_exec) are NOT computed here — that
is deferred to `fingerprint.py` and run only on candidate-class members, because it requires
opening a Dojo per state and is expensive.

Requires Linux/macOS + LeanDojo + a GITHUB_ACCESS_TOKEN. Import is guarded so the rest of
the package remains importable (and unit-testable) off-box.
"""
from __future__ import annotations

import random
import sys as _sys
import time as _time
from collections import defaultdict
from typing import Iterator, Optional

from .config import Config
from .schema import Observation, Provenance, StateRecord

try:
    import lean_dojo  # noqa: F401
    from lean_dojo import LeanGitRepo, trace
    _HAVE_LEANDOJO = True
except Exception:  # pragma: no cover - platform dependent
    _HAVE_LEANDOJO = False


def is_dependency_path(path: str) -> bool:
    """True for a traced file that belongs to a dependency rather than the audited repo.

    LeanDojo traces the whole dependency closure: for `lean4-example`, 906 of 907 traced files
    are Lean core / Lake / Std under `.lake/`, and only one is the repo's own. Theorems in
    dependency files cannot be opened in a `Dojo` (it raises `DojoInitError: Cannot find the
    *.ast.json file`), so sampling them yields states that can never be fingerprinted or
    replayed — the pilot budget would be spent almost entirely on unusable states.
    """
    p = str(path)
    return p.startswith(".lake/") or p.startswith("src/") or "/.lake/" in p


def _repo_own_files(traced_repo) -> tuple[list, int]:
    """Split traced files into the audited repo's own files and a count of skipped deps."""
    own, skipped = [], 0
    root_repo = getattr(traced_repo, "repo", None)
    for tf in traced_repo.traced_files:
        tf_repo = getattr(tf, "repo", None)
        # Prefer the structural check; fall back to the path prefix when `repo` is unavailable
        # or compares equal across the closure.
        is_dep = is_dependency_path(tf.path)
        if root_repo is not None and tf_repo is not None and tf_repo != root_repo:
            is_dep = True
        if is_dep:
            skipped += 1
        else:
            own.append(tf)
    return own, skipped


def _span(tt) -> Optional[tuple]:
    """Source span of a traced tactic, as a comparable (start, end) pair."""
    s, e = getattr(tt, "start", None), getattr(tt, "end", None)
    if s is None or e is None:
        return None
    try:
        return ((s.line_nb, s.column_nb), (e.line_nb, e.column_nb))
    except AttributeError:
        return (tuple(s), tuple(e)) if isinstance(s, (list, tuple)) else None


def top_level_tactics(tactics: list) -> tuple[list, int]:
    """Keep only tactics not nested inside another, and count what was dropped.

    `get_traced_tactics()` returns the proof's tactic TREE flattened in traversal order: a
    focused block and each of its children all appear as separate entries. Replaying such a
    list linearly re-applies the children and finishes the proof early — which is exactly what
    broke the shakedown run:

        [2] `· exact congrArg (·.toAdd.add x y) h`   <- container
        [3] `· exact congrArg (·.toMul.mul x y) h`   <- container, proof completes here
        [4]   `exact congrArg (·.toAdd.add x y) h`   <- child of [2], replayed again

    Fast-forwarding through that yields `ProofFinished` instead of a `TacticState`, so the state
    cannot be restored: the fingerprint degrades and every replay on it reports CRASH.

    Restricting to top-level tactics makes sequential replay valid. It is a real sampling
    restriction — states inside focused blocks are not audited — and it biases the sample toward
    shallower proof positions, so the count is returned and must be reported (blueprint §7.2
    stratifies by proof depth; this truncates that stratum).
    """
    spans = [(_span(t), t) for t in tactics]
    if any(s is None for s, _ in spans):
        return list(tactics), 0          # no span info: cannot tell, keep everything
    keep = []
    nested = 0
    for i, (si, ti) in enumerate(spans):
        contained = any(
            j != i and sj[0] <= si[0] and si[1] <= sj[1] and sj != si
            for j, (sj, _) in enumerate(spans))
        if contained:
            nested += 1
        else:
            keep.append(ti)
    return keep, nested


def _leandojo_version() -> str:
    try:
        return getattr(lean_dojo, "__version__", "unknown")
    except Exception:
        return "unavailable"


def _tactic_family(tactic: str) -> str:
    """Coarse family for stratification (blueprint §7.2)."""
    head = tactic.strip().split(maxsplit=1)
    return head[0] if head else ""


def _has_ellipsis(pp: str) -> bool:
    return "⋯" in pp or "…" in pp or "..." in pp


def iter_state_records(cfg: Config) -> Iterator[StateRecord]:
    """Yield a StateRecord for every traced tactic state in the pinned corpus.

    Note: this is the *full* stream (millions of states for Mathlib). Callers sample it via
    `sample_states`. We stream rather than materialize to keep memory bounded.
    """
    if not _HAVE_LEANDOJO:
        raise RuntimeError(
            "LeanDojo is unavailable. Extraction requires Linux/macOS with lean-dojo installed. "
            "Run `python -m audit.run_gate1 --stage regression` off-box; run extraction on the "
            "target machine.")

    repo = LeanGitRepo(cfg.repo_url, cfg.repo_commit)
    # Usually a fast download of a pre-traced archive from LeanDojo's remote cache; falls back
    # to a local build+trace (hours, tens of GB) when no published trace exists for the commit.
    traced_repo = trace(repo)
    ldj_ver = _leandojo_version()

    own_files, skipped_dep_files = _repo_own_files(traced_repo)
    if skipped_dep_files:
        print(f"[extract] {len(own_files)} repo files; skipped {skipped_dep_files} dependency "
              f"files (Lean core / .lake packages) — their theorems cannot be opened in a Dojo.",
              file=_sys.stderr, flush=True)

    n_nested_skipped = 0
    for traced_file in own_files:
        rel_path = str(traced_file.path)
        for traced_thm in traced_file.get_traced_theorems():
            thm = traced_thm.theorem
            all_tactics = traced_thm.get_traced_tactics()
            tactics, nested = top_level_tactics(all_tactics)
            n_nested_skipped += nested
            prefix: list[str] = []
            for idx, tt in enumerate(tactics):
                state_before = getattr(tt, "state_before", None)
                if not state_before:  # some traced tactics have no explicit pre-state
                    prefix.append(tt.tactic.strip())
                    continue
                prov = Provenance(
                    repo_url=cfg.repo_url,
                    repo_commit=cfg.repo_commit,
                    file_path=rel_path,
                    theorem_full_name=str(thm.full_name),
                    tactic_index=idx,
                    lean_toolchain=cfg.lean_toolchain,
                    leandojo_version=ldj_ver,
                    proof_depth=len(tactics),
                    goal_count=state_before.count("⊢"),
                    token_length=len(state_before),
                    has_ellipsis=_has_ellipsis(state_before),
                    tactic_family=_tactic_family(tt.tactic),
                )
                obs = Observation(phi_state=state_before)
                yield StateRecord(
                    provenance=prov,
                    observation=obs,
                    human_tactic=tt.tactic.strip(),
                    proof_prefix=list(prefix),
                )
                prefix.append(tt.tactic.strip())


def sample_dense(cfg: Config, n_files: int = 20, n: Optional[int] = None,
                 file_substr: Optional[str] = None,
                 stats: Optional[dict] = None) -> list[StateRecord]:
    """Exhaustive within-file sample: every state from the `n_files` densest files.

    This is deliberately NOT the §7.2 stratified sample, and it is not for prevalence.

    The stratified sample spreads n states across 4,448 files, so two states from the same file
    are almost never both drawn — which suppresses exactly what the audit is looking for, since
    byte-identical `φ_state` is far likelier within a file than across the corpus. §7.2's
    stratification is right for an unbiased prevalence estimate (RQ1) and wrong as an instrument
    for *finding* witnesses.

    Sampling every state from a few files instead gives two things at once:
      * maximum collision opportunity per state examined;
      * a constant ambient environment within each file, so any divergence found is reported as
        `same_environment` — the non-dismissible case that no amount of added context could have
        disambiguated.

    Prevalence figures must NOT be quoted from this sample: it is enriched by construction.
    """
    if not _HAVE_LEANDOJO:
        raise RuntimeError("sample_dense requires lean-dojo (Linux/macOS).")

    repo = LeanGitRepo(cfg.repo_url, cfg.repo_commit)
    traced_repo = trace(repo)
    own_files, skipped = _repo_own_files(traced_repo)
    ldj_ver = _leandojo_version()

    if file_substr:
        own_files = [f for f in own_files if file_substr in str(f.path)]
    print(f"[extract] dense sampling over {len(own_files)} repo files "
          f"({skipped} dependency files skipped)", file=_sys.stderr, flush=True)

    per_file: dict[str, list[StateRecord]] = {}
    nested_total = 0
    _t0 = _time.time()
    for _i, traced_file in enumerate(own_files, 1):
        if _i % 500 == 0:
            print(f"[extract]   scanned {_i}/{len(own_files)} files, "
                  f"{sum(len(v) for v in per_file.values())} states, "
                  f"{_time.time()-_t0:.0f}s", file=_sys.stderr, flush=True)
        rel_path = str(traced_file.path)
        recs: list[StateRecord] = []
        try:
            for traced_thm in traced_file.get_traced_theorems():
                thm = traced_thm.theorem
                tactics, nested = top_level_tactics(traced_thm.get_traced_tactics())
                nested_total += nested
                prefix: list[str] = []
                for idx, tt in enumerate(tactics):
                    state_before = getattr(tt, "state_before", None)
                    if not state_before:
                        prefix.append(tt.tactic.strip())
                        continue
                    recs.append(StateRecord(
                        provenance=Provenance(
                            repo_url=cfg.repo_url, repo_commit=cfg.repo_commit,
                            file_path=rel_path, theorem_full_name=str(thm.full_name),
                            tactic_index=idx, lean_toolchain=cfg.lean_toolchain,
                            leandojo_version=ldj_ver, proof_depth=len(tactics),
                            goal_count=state_before.count("⊢"),
                            token_length=len(state_before),
                            has_ellipsis=_has_ellipsis(state_before),
                            tactic_family=_tactic_family(tt.tactic)),
                        observation=Observation(phi_state=state_before),
                        human_tactic=tt.tactic.strip(),
                        proof_prefix=list(prefix)))
                    prefix.append(tt.tactic.strip())
        except Exception:
            continue
        if recs:
            per_file[rel_path] = recs

    ranked = sorted(per_file.items(), key=lambda kv: -len(kv[1]))[:n_files]
    out: list[StateRecord] = []
    for path, recs in ranked:
        out.extend(recs)
    if n:
        out = out[:n]

    if stats is not None:
        stats.update({
            "files_available": len(per_file),
            "files_used": len(ranked),
            "states_per_file": {p: len(r) for p, r in ranked},
            "nested_tactics_skipped": nested_total,
            "states_collected": len(out),
        })
    print(f"[extract] dense sample: {len(out)} states from {len(ranked)} files "
          f"(largest: {ranked[0][0]} with {len(ranked[0][1])})" if ranked else
          "[extract] dense sample: no states", file=_sys.stderr, flush=True)
    return out


def _stratum_key(rec: StateRecord, keys: list[str]) -> tuple:
    p = rec.provenance
    buckets = []
    for k in keys:
        if k == "repo_file":
            buckets.append(p.file_path)
        elif k == "proof_depth":
            buckets.append(min(p.proof_depth, 20))                # cap tail
        elif k == "goal_count":
            buckets.append(min(p.goal_count, 5))
        elif k == "token_length":
            buckets.append(p.token_length // 200)                 # 200-char bins
        elif k == "has_ellipsis":
            buckets.append(p.has_ellipsis)
        elif k == "tactic_family":
            buckets.append(p.tactic_family)
        else:
            buckets.append(None)
    return tuple(buckets)


def allocate_quotas(seen_per_stratum: dict, available: dict, n: int) -> dict:
    """Largest-remainder allocation of a budget of `n` across strata (blueprint §7.2).

    Each stratum's exact share is `seen/total · n`; every stratum takes the floor of its share,
    and the leftover budget goes to the largest fractional remainders. A stratum can never be
    allocated more than the `available` records its reservoir actually holds, and the result
    sums to `min(n, total available)` — no floor that overshoots, no truncation afterwards.

    Pure and deterministic, so the sampling policy is unit-testable without LeanDojo.
    """
    total_seen = sum(seen_per_stratum.values())
    if total_seen <= 0 or n <= 0:
        return {}
    quotas: dict = {}
    remainders: list[tuple[float, str, object]] = []
    allocated = 0
    for sk, seen in seen_per_stratum.items():
        exact = seen / total_seen * n
        base = min(int(exact), available.get(sk, 0))
        quotas[sk] = base
        allocated += base
        remainders.append((exact - int(exact), str(sk), sk))

    # Deterministic tie-break on the stringified key so a fixed seed gives a fixed sample.
    remainders.sort(key=lambda t: (-t[0], t[1]))
    changed = True
    while allocated < n and changed:
        changed = False
        for _, _, sk in remainders:
            if allocated >= n:
                break
            if quotas[sk] < available.get(sk, 0):
                quotas[sk] += 1
                allocated += 1
                changed = True
    return quotas


def sample_states(cfg: Config, n: Optional[int] = None,
                  stratum_sizes: Optional[dict] = None) -> list[StateRecord]:
    """Stratified sample of `n` states (blueprint §7.2).

    Single streaming pass, bounded memory: each stratum keeps a reservoir of at most
    `per_stratum_cap` records, so peak memory is O(n_strata · cap) rather than O(corpus).

    Two bugs this replaces, both of which defeated the stratification it claimed to implement:

    * `per_stratum_cap` was `n`, so every stratum could retain the full 20,000 records and the
      "we never hold the whole corpus in memory" docstring was false for any stratum smaller
      than n — i.e. almost all of them.
    * The final rebalance gave every stratum `max(1, round(share·n))` and then truncated a
      shuffled concatenation to `n`. With six stratification keys the stratum count exceeds n,
      so the `max(1, …)` floor alone overshot the budget and the truncation degenerated to a
      uniform sample over strata — discarding the stratification entirely.

    Allocation is now largest-remainder over observed stratum shares, which sums to exactly n
    without a floor, and rare strata are retained by construction rather than by luck. Observed
    stratum sizes are written into `stratum_sizes` (if provided) so the caller can form the
    inverse-probability weights §7.2 requires for prevalence estimates.

    IMPORTANT: we deliberately do NOT dedup by visible string here — duplication is the object
    of study (blueprint §7.2, §9).
    """
    n = n or cfg.n_states
    rng = random.Random(cfg.seed)
    keys = cfg.stratify_by
    # Bounded per-stratum retention. A stratum can never contribute more than this many
    # records, which caps memory; it is generous relative to any plausible per-stratum quota.
    per_stratum_cap = max(8, min(n, 512))

    strata: dict[tuple, list[StateRecord]] = defaultdict(list)
    seen_per_stratum: dict[tuple, int] = defaultdict(int)

    for rec in iter_state_records(cfg):
        sk = _stratum_key(rec, keys)
        seen_per_stratum[sk] += 1
        reservoir = strata[sk]
        if len(reservoir) < per_stratum_cap:
            reservoir.append(rec)
        else:
            # Classic reservoir sampling: uniform over everything seen in this stratum.
            j = rng.randint(0, seen_per_stratum[sk] - 1)
            if j < per_stratum_cap:
                reservoir[j] = rec

    if stratum_sizes is not None:
        stratum_sizes.update({str(k): v for k, v in seen_per_stratum.items()})

    quotas = allocate_quotas(seen_per_stratum,
                             {sk: len(v) for sk, v in strata.items()}, n)

    out: list[StateRecord] = []
    for sk, take in quotas.items():
        if take <= 0:
            continue
        reservoir = strata[sk]
        rng.shuffle(reservoir)
        out.extend(reservoir[:take])

    rng.shuffle(out)
    return out
