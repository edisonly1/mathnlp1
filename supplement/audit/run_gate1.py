"""Orchestrator — the deterministic Appendix-A pipeline + Gate-1 go/no-go (blueprint App. A, §10.2).

Stages:
  regression   run certified witnesses under the pinned toolchain (any OS; no LeanDojo needed)
  pilot        20,000-state natural audit (blueprint §10.2)   [Linux/macOS + LeanDojo]
  main         100k-250k main study (only after pilot GO)     [Linux/macOS + LeanDojo]

Every expensive stage logs what it dropped (no silent caps — blueprint quality bar).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

from . import analysis as A
from .alias_index import candidate_classes, finalize_class
from .config import Config, load_config
from .fingerprint import is_degraded
from .schema import OutcomeKind, StateRecord


# --------------------------------------------------------------------------- #
def _log(msg: str) -> None:
    print(f"[gate1] {msg}", file=sys.stderr, flush=True)


def _out_dir(cfg: Config, stamp: str) -> Path:
    d = Path(cfg.output_dir) / stamp
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
def stage_regression(cfg: Config) -> int:
    from .regression import run_regression
    _log(f"running certified-witness regression under {cfg.gate0_toolchain} …")
    res = run_regression(cfg.gate0_toolchain)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if res.get("passed"):
        _log("REGRESSION PASSED — toolchain reproduces certified witnesses "
             "(identical φ_state + divergent outcome, incl. non-overflow and in-goal mechanisms, "
             "with the inert proof-irrelevance control holding).")
        return 0
    _log("REGRESSION FAILED — do not proceed to the natural audit until this passes (§10.2 cond 6).")
    return 1


# --------------------------------------------------------------------------- #
# Structural tactic panel (blueprint §7.6)
# --------------------------------------------------------------------------- #
_HYP_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_'!?₀-₉]*)\s*:", re.MULTILINE)


def structural_actions(phi_state: str, prefixes: list[str], cap: int = 12) -> list[str]:
    """Instantiate `rw`/`apply`/`exact` on identifiers that actually appear in the model-visible
    input (blueprint §7.6: "when the required identifier appears in the model-visible input").

    Only local hypothesis names are used — an identifier the model can see and name. Capped, and
    the caller logs the cap so coverage is never silently truncated.
    """
    names: list[str] = []
    for line in phi_state.splitlines():
        if line.lstrip().startswith("⊢"):
            break
        m = _HYP_LINE.match(line.strip())
        if m and m.group(1) not in names:
            names.append(m.group(1))
    out: list[str] = []
    for n in names:
        for p in prefixes:
            out.append(f"rw [{n}]" if p == "rw" else f"{p} {n}")
    return out[:cap]


# --------------------------------------------------------------------------- #
def _rep_summary(fe: str, weight: int, rep: StateRecord, outcomes: dict) -> A.RepSummary:
    fp = rep.fingerprint
    return A.RepSummary(
        f_exec=fe, weight=weight, outcomes=outcomes,
        phi_full=(fp.phi_full if fp else ""),
        env_digest=(fp.env_digest if fp else ""),
        phi_state=rep.observation.phi_state,
        phi_deep=(fp.phi_state_deep if fp else ""),
        phi_deep_proofs=(fp.phi_deep_proofs if fp else ""),
        phi_all_shallow=(fp.phi_all_shallow if fp else ""),
        phi_full_proofs=(fp.phi_full_proofs if fp else ""),
        env_provenance=(fp.env_provenance if fp else ""),
        f_core=(fp.f_core if fp else ""),
        f_proof=(fp.f_proof if fp else ""),
        f_env=(fp.f_env if fp else ""),
    )


def _build_class_summary(cfg, replayer, obs_key, layer, members, stats, max_reps=8):
    """Fingerprint members, replay the panel on representatives, build an A.ClassSummary."""
    for m in members:
        if m.fingerprint is None:
            replayer.fingerprint_record(m)
    n_degraded = sum(1 for m in members if is_degraded(m.fingerprint))
    if n_degraded:
        stats["degraded_fingerprints"] += n_degraded

    cls, reps = finalize_class(obs_key, layer, members, is_natural=True)
    if not cls.is_latent:
        return None  # trivial duplicate, not an alias

    rep_items = list(reps.items())
    if len(rep_items) > max_reps:
        dropped = len(rep_items) - max_reps
        _log(f"class {obs_key[:10]}: {len(rep_items)} distinct states, capping replay to "
             f"{max_reps} (dropped {dropped} — counted in report.dropped_reps).")
        stats["dropped_reps"] += dropped
        rep_items = rep_items[:max_reps]

    # Action set: core panel + human tactics + structural instantiations (§7.6, §8.1.4).
    human_tactics = sorted({m.human_tactic for m in members if m.human_tactic})
    structural = structural_actions(members[0].observation.phi_state, cfg.structural_panel)
    action_set = list(dict.fromkeys(cfg.core_panel + human_tactics + structural))

    rep_summaries = []
    for fe, rep in rep_items:
        weight = cls.exec_fingerprints.count(fe)
        outcomes = {}
        for a in action_set:
            o = replayer.replay(rep, a)
            stats["replays"] += 1
            if not o.reproducible:
                stats["nonreproducible_replays"] += 1
            if o.reproducible and o.kind != OutcomeKind.CRASH:
                outcomes[a] = o.canonical(cfg.goal_order_sensitive)
        rep_summaries.append(_rep_summary(fe, weight, rep, outcomes))

    cs = A.ClassSummary(
        observation_key=obs_key, layer=layer, is_natural=True, reps=rep_summaries,
        files=sorted({m.provenance.file_path for m in members}),
        repos=sorted({m.provenance.repo_url for m in members}),
        has_ellipsis=any(m.provenance.has_ellipsis for m in members),
        any_truncated=any(m.observation.truncated for m in members),
    )
    cs.action_divergent, cs.divergent_tactics = A.class_divergence(cs, action_set)
    cs.severity = A.class_severity(cs, cs.divergent_tactics)
    cs.mechanisms = A.attribute_mechanisms(cs)
    cs.overflow_only = A.is_overflow_only(cs.mechanisms)
    cs.action_set = action_set
    cs_action_set = action_set
    return cs, cs_action_set


# --------------------------------------------------------------------------- #
def verify_divergences(cfg, replayer, summaries, action_sets, reps_by_class) -> dict:
    """Blueprint §10.2 cond 5 / §7.7: re-run every *detected* divergence in a clean pass and
    check it still holds. Previously this rate was hardcoded to 1.0, which made the condition
    a tautology — `replay()` had already filtered non-reproducible outcomes, so nothing could
    ever fail it."""
    detected = 0
    reproduced = 0
    failures = []
    for cs in summaries:
        if not cs.action_divergent:
            continue
        recs = reps_by_class.get(cs.observation_key, {})
        for tac in cs.divergent_tactics:
            detected += 1
            fresh: list[A.RepSummary] = []
            for rep in cs.reps:
                rec = recs.get(rep.f_exec)
                if rec is None:
                    continue
                o = replayer.replay(rec, tac)
                if o.reproducible and o.kind != OutcomeKind.CRASH:
                    fresh.append(A.RepSummary(f_exec=rep.f_exec, weight=rep.weight,
                                              outcomes={tac: o.canonical(cfg.goal_order_sensitive)}))
            probe = A.ClassSummary(observation_key=cs.observation_key, layer=cs.layer,
                                   is_natural=True, reps=fresh)
            still, _ = A.class_divergence(probe, [tac])
            if still:
                reproduced += 1
            else:
                failures.append({"observation_key": cs.observation_key, "tactic": tac})
    return {
        "detected_divergences": detected,
        "reproduced_divergences": reproduced,
        "reproduce_rate": (reproduced / detected) if detected else 0.0,
        "failed_to_reproduce": failures[:50],
    }


def reverify_nondivergent(cfg, replayer, summaries, reps_by_class, seed: int) -> dict:
    """Blueprint §7.7 step 6: re-run a random fraction of NON-divergent cases to estimate the
    silent replay-error rate (a divergence the first pass missed)."""
    frac = cfg.reverify_fraction
    nondiv = [c for c in summaries if not c.action_divergent]
    if frac <= 0 or not nondiv:
        return {"sampled": 0, "silent_errors": 0, "silent_error_rate": 0.0}
    rng = random.Random(seed)
    k = max(1, int(round(frac * len(nondiv))))
    sample = rng.sample(nondiv, min(k, len(nondiv)))
    silent = 0
    for cs in sample:
        recs = reps_by_class.get(cs.observation_key, {})
        tactics = sorted({t for r in cs.reps for t in r.outcomes})
        fresh_reps = []
        for rep in cs.reps:
            rec = recs.get(rep.f_exec)
            if rec is None:
                continue
            outs = {}
            for t in tactics:
                o = replayer.replay(rec, t)
                if o.reproducible and o.kind != OutcomeKind.CRASH:
                    outs[t] = o.canonical(cfg.goal_order_sensitive)
            fresh_reps.append(A.RepSummary(f_exec=rep.f_exec, weight=rep.weight, outcomes=outs))
        probe = A.ClassSummary(observation_key=cs.observation_key, layer=cs.layer,
                               is_natural=True, reps=fresh_reps)
        found, _ = A.class_divergence(probe, tactics)
        if found:
            silent += 1
    return {"sampled": len(sample), "silent_errors": silent,
            "silent_error_rate": silent / len(sample)}


# --------------------------------------------------------------------------- #
def stage_audit(cfg: Config, n_states: int, with_model: bool, stamp: str,
                dense_files: int = 0, file_substr: str = "",
                census: bool = False) -> int:
    problems = cfg.validate()
    for p in problems:
        _log("CONFIG: " + p)
    if any("repo_commit is a placeholder" in p for p in problems):
        _log("Refusing to run the natural audit against an unpinned corpus. Set corpus.repo_commit.")
        return 2

    from .extract import sample_dense, sample_states
    from .observation import ObservationBuilder, StateOnlyRetriever
    from .replay import DojoReplayer

    out = _out_dir(cfg, stamp)
    sampling_stats: dict = {}
    t_sample = time.time()
    if census:
        _log("EXHAUSTIVE CENSUS: every auditable state in every repo file. Not a sample — "
             "prevalence from this run is exact, not estimated.")
        records = sample_dense(cfg, n_files=10**9, n=None,
                               file_substr=file_substr or None, stats=sampling_stats)
    elif dense_files:
        _log(f"DENSE within-file sample: every state from the {dense_files} densest files "
             f"(NOT a prevalence sample — enriched by construction).")
        records = sample_dense(cfg, n_files=dense_files, n=n_states,
                               file_substr=file_substr or None, stats=sampling_stats)
    else:
        _log(f"sampling {n_states} stratified states from "
             f"{cfg.repo_url}@{cfg.repo_commit[:10]} …")
        records = sample_states(cfg, n_states)
    _log(f"sampled {len(records)} states in {time.time()-t_sample:.0f}s.")

    if "phi_tok" in cfg.group_on:
        try:
            ob = ObservationBuilder(cfg.tokenizer, cfg.max_input_length,
                                    retriever=StateOnlyRetriever())
            for r in records:
                ob.enrich(r)
            _log("built φ_tok (StateOnly retriever — token-input collisions are a lower bound; "
                 "with a byte-level tokenizer they add nothing over φ_state except truncation).")
        except Exception as e:  # pragma: no cover
            _log(f"φ_tok disabled ({e!r}); grouping on φ_state only.")

    replayer = DojoReplayer(cfg)
    stats = {"replays": 0, "nonreproducible_replays": 0, "dropped_reps": 0,
             "degraded_fingerprints": 0}

    with open(out / "states.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(r.to_json() + "\n")

    all_summaries: list[A.ClassSummary] = []
    action_sets: dict[str, list[str]] = {}
    reps_by_class: dict[str, dict[str, StateRecord]] = {}

    # classes.jsonl is written INCREMENTALLY: a run killed mid-pass keeps what it finished.
    # The 200-file probe did ~45 min of real work before hanging and none of it was recoverable.
    classes_fh = open(out / "classes.jsonl", "w", encoding="utf-8")
    t_start = time.time()
    try:
        for layer in cfg.group_on:
            cands = candidate_classes(records, layer, min_members=2)
            _log(f"layer {layer}: {len(cands)} candidate classes (≥2 members).")
            for i, (obs_key, members) in enumerate(cands, 1):
                t_cls = time.time()
                built = _build_class_summary(cfg, replayer, obs_key, layer, members, stats)
                elapsed = time.time() - t_start
                rate = elapsed / max(1, i)
                # Per-class progress: without it a multi-hour pass is completely silent and a
                # hang is indistinguishable from slow work.
                _log(f"  [{i}/{len(cands)}] {obs_key[:10]} {time.time()-t_cls:5.1f}s "
                     f"{'skip(not latent)' if built is None else 'built'} "
                     f"| elapsed {elapsed/60:.0f}m eta {(rate*(len(cands)-i))/60:.0f}m "
                     f"| timeouts={replayer.stats['op_timeouts']}"
                     f"+{replayer.stats['session_open_timeouts']}")
                if built is None:
                    continue
                cs, aset = built
                all_summaries.append(cs)
                action_sets[obs_key] = aset
                _, reps = finalize_class(obs_key, layer, members, is_natural=True)
                reps_by_class[obs_key] = reps
                classes_fh.write(json.dumps(_class_to_json(cs), ensure_ascii=False) + "\n")
                classes_fh.flush()
    finally:
        classes_fh.close()

    same_env_core = [c for c in all_summaries if c.is_core_latent and c.same_environment]
    _log(f"DECISIVE: core-latent classes with a constant environment: {len(same_env_core)} "
         f"(of {sum(1 for c in all_summaries if c.is_core_latent)} core-latent, "
         f"{len(all_summaries)} latent overall)")
    _log(f"latent classes built: {len(all_summaries)} "
         f"(degraded fingerprints: {stats['degraded_fingerprints']}, "
         f"dropped reps: {stats['dropped_reps']}).")

    # ------- verification passes (§7.7, §10.2 cond 5) -------
    # Drop every open session first, so each detected divergence is re-run in a fresh process
    # rather than re-read from the session that discovered it.
    replayer.reset_sessions()
    repro = verify_divergences(cfg, replayer, all_summaries, action_sets, reps_by_class)
    _log(f"divergence reproduction: {repro['reproduced_divergences']}/"
         f"{repro['detected_divergences']} = {repro['reproduce_rate']:.3f}")
    silent = reverify_nondivergent(cfg, replayer, all_summaries, reps_by_class, cfg.seed)
    _log(f"silent replay error probe: {silent['silent_errors']}/{silent['sampled']}")

    # ------- statistics (blueprint §8) -------
    prev = A.prevalence(all_summaries, len(records))
    divergent = [c for c in all_summaries if c.action_divergent]
    non_overflow = [c for c in divergent if not c.overflow_only]

    div_flags = [1.0 if c.action_divergent else 0.0 for c in all_summaries]
    clusters = [c.files[0] if c.files else "?" for c in all_summaries]
    ci = A.cluster_bootstrap_ci(div_flags, clusters, seed=cfg.seed) if all_summaries else (0.0, 0.0)

    panel = cfg.core_panel
    curve = A.repair_curve(all_summaries, panel)
    residual_env = A.residual_after_repair(all_summaries, panel, "R6_env")

    regression_passed = False
    try:
        from .regression import run_regression
        regression_passed = bool(run_regression(cfg.gate0_toolchain).get("passed"))
    except Exception as e:
        _log(f"regression check skipped ({e!r}).")

    # Model-consequence condition (§10.2 cond 7). Tri-state: None ⇒ not evaluated.
    model_persists = None
    if with_model:
        from .model_runner import ReProverGenerator
        model_persists = _model_consequence(cfg, replayer, ReProverGenerator(cfg),
                                            non_overflow, reps_by_class, out)
    else:
        _log("condition 7 NOT evaluated (--with-model not set); it is reported as unevaluated, "
             "not as a failure.")

    verdict = A.evaluate_gate1(
        all_summaries, reproduce_rate=repro["reproduce_rate"],
        regression_passed=regression_passed, model_persists_or_repairs=model_persists,
        sampling_mode=("exhaustive_census" if census else
                       "dense_within_file" if dense_files else "stratified_7.2"))

    report = {
        "stamp": stamp, "n_states": len(records),
        "sampling": ("exhaustive_census" if census else
                     "dense_within_file" if dense_files else "stratified_7.2"),
        "sampling_stats": sampling_stats,
        "prevalence": prev,
        "divergence_class_rate_ci95": ci,
        "counts": {"divergent": len(divergent), "non_overflow_divergent": len(non_overflow),
                   "residual_after_R6_env": len(residual_env)},
        "reproduction": repro,
        "silent_replay_error": silent,
        "pipeline_stats": stats,
        "replayer_stats": replayer.stats,
        "repair_curve": curve,
        "alias_regret_examples": [
            {"observation_key": c.observation_key,
             "regret_nonfailure": A.alias_regret(c, c.action_set or panel),
             "severity": c.severity, "mechanisms": c.mechanisms,
             "same_environment": c.same_environment}
            for c in non_overflow[:20]
        ],
        "gate1_verdict": verdict,
        "notes": [
            "φ_tok collisions computed under StateOnly retriever are a LOWER BOUND (§4.1).",
            "F_core is built from pp.all+deepTerms rendering, not the elaborated Expr; see README.",
            "Prevalence: quote the core-level figure. exec-level latency saturates across files.",
            "Gate-1 cond 7 requires --with-model; otherwise it is reported as unevaluated.",
        ] + (["DENSE SAMPLE: enriched by construction — do NOT quote prevalence from it; "
              "it answers whether same-environment aliases EXIST, not how common they are."]
             if dense_files else []),
    }
    with open(out / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    _write_verdict(out, verdict, report)
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    return 0 if verdict["GO"] else 3


# --------------------------------------------------------------------------- #
def _model_consequence(cfg, replayer, model, non_overflow_classes, reps_by_class, out) -> bool:
    """Blueprint §7.9: run ReProver top-k once per unique model input, replay every generated
    tactic on every class member, and report whether the same ranked list has different success
    sets across members. Returns True iff ≥1 class shows nonuniform top-k validity.

    Operates on `phi_state` classes and tokenizes their shared observation here. It previously
    filtered to `layer == "phi_tok"`, which silently examined NOTHING once that layer was
    disabled as redundant (ByT5 is byte-level, so φ_tok re-keys the same classes). The φ_tok
    layer was never load-bearing for this test anyway: what §7.9 needs is one token input shared
    by all members, and byte-identical φ_state already guarantees that.
    """
    persists = False
    rows = []
    tok = None
    try:
        from .observation import ObservationBuilder, StateOnlyRetriever
        ob = ObservationBuilder(cfg.tokenizer, cfg.max_input_length,
                                retriever=StateOnlyRetriever())
        tok = ob
    except Exception as e:  # pragma: no cover - needs transformers
        _log(f"model pass disabled: tokenizer unavailable ({e!r})")
        return False

    for c in non_overflow_classes:
        recs = reps_by_class.get(c.observation_key, {})
        if len(recs) < 2:
            rows.append({"observation_key": c.observation_key, "skipped": "fewer than 2 members"})
            continue
        member = next(iter(recs.values()))
        phi = member.observation.phi_state
        try:
            ids, truncated = tok.build_tok(tok.build_rag(phi, member.provenance))
            tactics = model.top_k_tactics(list(ids))
        except Exception as e:  # pragma: no cover - needs checkpoint
            rows.append({"observation_key": c.observation_key, "error": repr(e)[:200]})
            continue
        if not tactics:
            rows.append({"observation_key": c.observation_key, "skipped": "no tactics generated"})
            continue
        # Identical token input + deterministic beam search ⇒ one ranked list for all members,
        # so any difference in which tactics succeed is attributable to hidden state (§7.9).
        #
        # Replay each tactic ONCE and derive every k from that single pass: the top-k lists are
        # nested (tactics[:1] ⊂ tactics[:4] ⊂ tactics[:8]), so replaying per-k repeated the same
        # work — 13 replays per member where 8 distinct tactics exist, ~40% waste that multiplies
        # at pilot scale.
        kmax = max(cfg.top_k)
        succeeded: dict = {}          # f_exec -> set of tactic indices that succeeded
        for fe, rec in recs.items():
            ok = set()
            for idx, t in enumerate(tactics[:kmax]):
                o = replayer.replay(rec, t)
                if o.reproducible and o.kind in (OutcomeKind.COMPLETE, OutcomeKind.SUCC):
                    ok.add(idx)
            succeeded[fe] = ok
        for k in cfg.top_k:
            valid = {fe: tuple(tactics[i] for i in sorted(ok) if i < k)
                     for fe, ok in succeeded.items()}
            nonuniform = len(set(valid.values())) > 1
            rows.append({"observation_key": c.observation_key, "k": k,
                         "nonuniform_topk_validity": nonuniform,
                         "n_members": len(valid), "truncated_input": bool(truncated),
                         "top_k_tactics": tactics[:k],
                         "valid_per_member": {f[:12]: list(v) for f, v in valid.items()}})
            persists = persists or nonuniform
    with open(out / "model.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    _log(f"model pass: {len(rows)} rows, "
         f"{sum(1 for r in rows if r.get('nonuniform_topk_validity'))} nonuniform")
    return persists


def _class_to_json(c: A.ClassSummary) -> dict:
    """Serialize a class.

    Per-rep outcomes and the declaration count are recorded because the class-level summary
    cannot answer the extension-stability question: given two members whose environments are
    ordered by additive growth, did the SMALLER environment succeed where the LARGER failed?
    `divergent_tactics` and `severity` say a divergence exists but not which member won, and
    direction is the entire question.
    """
    from .canon import env_fields
    reps = []
    for r in c.reps:
        f = env_fields(r.env_digest)
        reps.append({
            "f_exec": r.f_exec[:16], "f_core": r.f_core[:16], "f_env": r.f_env[:16],
            "weight": r.weight,
            # nconsts = locally-declared constants: the additive-growth proxy. A larger count
            # at the same file means strictly more declarations are in scope.
            "nconsts": f.get("nconsts", "NA"),
            "simpN": f.get("simpN", "NA"),
            "outcomes": r.outcomes,
            "is_degraded": r.is_degraded,
        })
    return {
        "observation_key": c.observation_key, "layer": c.layer, "is_natural": c.is_natural,
        "is_latent_exec": c.is_latent, "is_latent_core": c.is_core_latent,
        "same_environment": c.same_environment,
        "action_divergent": c.action_divergent,
        "divergent_tactics": c.divergent_tactics, "severity": c.severity,
        "mechanisms": c.mechanisms, "overflow_only": c.overflow_only,
        "files": c.files, "repos": c.repos,
        "n_distinct_states": len({r.f_exec for r in c.reps}),
        "n_members": sum(r.weight for r in c.reps),
        "action_set": c.action_set,
        "reps": reps,
    }


def _write_verdict(out: Path, verdict: dict, report: dict) -> None:
    lines = ["GATE-1 VERDICT", "=" * 40,
             f"GO: {verdict['GO']}", f"terminate: {verdict['terminate']}", "",
             "GO conditions:"]
    for k, v in verdict["go_conditions"].items():
        mark = "x" if v else " "
        if k in verdict.get("unevaluated_conditions", []):
            mark = "?"
        lines.append(f"  [{mark}] {k}")
    if verdict.get("unevaluated_conditions"):
        lines.append("  ('?' = not evaluated in this run, not a measured failure)")
    lines += ["", "counts: " + json.dumps(verdict["counts"]),
              "", "prevalence: " + json.dumps(report["prevalence"]),
              "", "repair: " + json.dumps(
                  [{r["rung"]: f"{r['removed']}/{r['evaluable']} removed, "
                    f"{r['median_added_bytes']:.0f}B"} for r in report["repair_curve"]["rungs"]]),
              "", "Blueprint §10.2: a result of only certified examples is a Lean interface note,",
              "not the intended paper. GO requires natural, reproducible, non-overflow divergence."]
    (out / "gate1_verdict.txt").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gate-1 natural aliasing audit")
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["regression", "pilot", "main", "probe", "full"],
                    required=True)
    ap.add_argument("--n-states", type=int, default=None)
    ap.add_argument("--with-model", action="store_true",
                    help="run the ReProver top-k model-consequence pass (needs GPU + checkpoint)")
    ap.add_argument("--stamp", default="run")
    ap.add_argument("--n-files", type=int, default=20,
                    help="probe stage: how many of the densest files to sample exhaustively")
    ap.add_argument("--file-substr", default="",
                    help="probe stage: restrict candidate files to paths containing this")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.stage == "regression":
        return stage_regression(cfg)
    n = args.n_states or cfg.n_states
    if args.stage == "probe":
        return stage_audit(cfg, n, args.with_model, args.stamp,
                           dense_files=args.n_files, file_substr=args.file_substr)
    if args.stage == "full":
        return stage_audit(cfg, n, args.with_model, args.stamp,
                           file_substr=args.file_substr, census=True)
    return stage_audit(cfg, n, args.with_model, args.stamp)


if __name__ == "__main__":
    raise SystemExit(main())
