"""Move 2: does the unsound transposition key cost a real prover anything?

Two arms of the SAME best-first search over the same theorems, differing only in the key used to
decide that two search nodes are the same node:

    arm A  key = pp                      -- stock ReProver / LeanDojo `TacticState.__eq__`
    arm B  key = (pp, phi_all_shallow)   -- the R4 repair: implicits and instances made explicit

`phi_all_shallow` is the rung that separated all three confirmed within-search witnesses, where
`pp` and `pp.deepTerms` both collided. It costs one extra probe per node.

Two measurements, deliberately, because they have very different statistical power:

1. **Pass rate A vs B.** What was asked for, and the thing a reader wants. It is also badly
   underpowered on a CPU budget: confirmed divergence runs at ~2% of reconvergence classes and
   every witness so far was a self-loop, so the true effect may be far below what a few hundred
   theorems can resolve. A null here bounds the effect; it does not establish absence.

2. **Merge firing and unsoundness rate.** Arm A records, at every merge, whether the discarded
   state's `phi_all_shallow` differs from the one already stored. That is a direct count of
   unsound merges in real search, needs no pass-rate delta to be informative, and is well powered
   because merges are common even when proofs are not. When a discarded state differs, we also
   record whether it was the deeper or shallower node, since discarding a *distinct* state is
   the precondition for losing a branch.

Arms share one process and one model cache per theorem, so identical states generate once.

Usage:
    python tools/search_ab.py --out runs/search_ab --theorems 40 --shard 0 --n-shards 4
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _log(m: str) -> None:
    print(f"[searchAB] {m}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="runs/killtest/states.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--theorems", type=int, default=40)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=91003)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-expansions", type=int, default=25)
    ap.add_argument("--budget-s", type=float, default=600.0,
                    help="wall-clock budget per (theorem, arm)")
    ap.add_argument("--dojo-timeout", type=int, default=600)
    ap.add_argument("--goal-contains", default="",
                    help="restrict to goals containing one of these comma-separated terms")
    ap.add_argument("--only-theorems", default="",
                    help="comma-separated substrings; keep only matching theorems")
    ap.add_argument("--arms", default="AB",
                    help="which arms to run per theorem: any subset of ABC")
    ap.add_argument("--probe-unsound", action="store_true",
                    help="when a merge collapses phi_shallow-distinct states, probe BOTH the "
                         "retained and discarded state with a small battery (3x repeats) in the "
                         "live session. Fingerprint disagreement alone shows the key collapsed "
                         "refined-distinguishable states; only operational divergence under a "
                         "shared tactic shows the merge was operationally unsound.")
    args = ap.parse_args()

    from lean_dojo import Dojo, Theorem
    import lean_dojo as ldj
    from audit.config import load_config
    from audit.fingerprint import PROBES, _run, parse_probe
    from audit.model_runner import ReProverGenerator
    from audit.observation import ObservationBuilder, StateOnlyRetriever
    from audit.replay import lean_git_repo, kill_orphan_lean, deadline

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / f"search.shard{args.shard}.jsonl", "a", encoding="utf-8")

    model = ReProverGenerator(cfg)
    ob = ObservationBuilder(cfg.tokenizer, cfg.max_input_length,
                            retriever=StateOnlyRetriever())

    states = [json.loads(l) for l in Path(args.states).read_text().splitlines()]
    # One search per THEOREM, from its initial state: tactic_index 0 with an empty prefix.
    roots = [s for s in states
             if s.get("proof_prefix") == [] and s["provenance"]["tactic_index"] == 0]
    if args.goal_contains:
        terms = [t for t in args.goal_contains.split(",") if t]
        roots = [s for s in roots
                 if any(t in s["observation"]["phi_state"] for t in terms)]
    seen_thm, uniq = set(), []
    for s in roots:
        t = (s["provenance"]["file_path"], s["provenance"]["theorem_full_name"])
        if t not in seen_thm:
            seen_thm.add(t)
            uniq.append(s)
    if args.only_theorems:
        pats = [t for t in args.only_theorems.split(",") if t]
        uniq = [s for s in uniq
                if any(t in s["provenance"]["theorem_full_name"] for t in pats)]
        _log(f"theorem filter: {len(uniq)} roots match")
    rng = random.Random(args.seed)
    rng.shuffle(uniq)
    picked = uniq[:args.theorems]
    mine = [s for i, s in enumerate(picked) if i % args.n_shards == args.shard]
    _log(f"shard {args.shard}: {len(mine)} theorems of {len(picked)} "
         f"(from {len(uniq)} distinct roots)")

    DIGEST_PROBE = (
        "run_tac (show Lean.Elab.Tactic.TacticM Unit from do\n"
        "  let g \u2190 Lean.Elab.Tactic.getMainGoal\n"
        "  g.withContext do\n"
        "    let d \u2190 g.getDecl\n"
        "    let e \u2190 Lean.Meta.mkForallFVars d.lctx.getFVars d.type\n"
        "    let e \u2190 Lean.instantiateMVars e\n"
        "    let r \u2190 Lean.Meta.abstractMVars e\n"
        "    throwError s!\"FPRINT_V2:dig={r.expr.hash}\")"
    )

    def digest(dojo, st) -> str:
        """64-bit structural digest of the elaborated goal closed over its context.
        Falls back to pp on probe failure, degrading that node to arm-A behaviour."""
        try:
            with deadline(45, "probe digest"):
                v = parse_probe(_run(dojo, st, DIGEST_PROBE) or "", False)
            return v if v else st.pp
        except BaseException:
            return st.pp

    def all_shallow(dojo, st) -> str:
        """The R4 rung. Falls back to pp so a probe failure degrades arm B to arm A rather
        than silently splitting every node."""
        try:
            with deadline(45, "probe all_shallow"):
                v = parse_probe(_run(dojo, st, PROBES["phi_all_shallow"]) or "", True)
            return v if v else st.pp
        except BaseException:
            return st.pp

    def search(dojo, root, arm: str, prov) -> dict:
        """Best-first search. `arm` selects the node-identity key."""
        t0 = time.time()
        stats = {"arm": arm, "expansions": 0, "nodes": 0, "merges": 0,
                 "unsound_merges": 0, "gen_calls": 0, "proved": False,
                 "proof": None, "stopped": None, "probe_fallbacks": 0}
        keyfn = (lambda st_: st_.pp) if arm == "A" else \
                (lambda st_: st_.pp + "\x01" + digest(dojo, st_)) if arm == "C" else \
                (lambda st_: st_.pp + "\x00" + all_shallow(dojo, st_))
        r_sh = all_shallow(dojo, root) if arm != "C" else digest(dojo, root)
        stats["probe_fallbacks"] += int(r_sh == root.pp)
        key0 = keyfn(root)
        # stored[key] = (phi_all_shallow, live state, path) of the node kept under that key
        stored = {key0: (r_sh, root, [])}
        stats["unsound_events"] = []
        # contrapose!/push_neg included deliberately: the escape measurement showed they are
        # the tactic class whose output depends on the hidden syntax. A battery without them is
        # exactly the probe-composition hazard this project documents.
        PROBE_BATTERY = ["simp", "simp_all", "norm_num", "constructor", "rfl", "omega",
                         "assumption", "contrapose!", "push_neg", "norm_cast"]

        def _oc(x):
            if isinstance(x, ldj.ProofFinished):
                return "COMPLETE"
            if isinstance(x, ldj.TacticState):
                return "S:" + _h(x.pp)
            return "FAIL"

        def probe_pair(st_a, st_b):
            """Repeat-controlled battery on retained vs discarded; returns divergent probes."""
            res = {}
            for tac in PROBE_BATTERY:
                pair = []
                bad = False
                for st_ in (st_a, st_b):
                    vals = []
                    for _ in range(3):
                        try:
                            with deadline(45, tac):
                                vals.append(_oc(dojo.run_tac(st_, tac)))
                        except BaseException:
                            bad = True
                            break
                    if bad or len(set(vals)) > 1:
                        bad = True
                        break
                    pair.append(vals[0])
                if not bad:
                    res[tac] = pair
            div = sorted(t for t, (a, b) in res.items() if a != b)
            return res, div
        counter = 0
        pq = [(0.0, counter, root, [])]
        while pq and stats["expansions"] < args.max_expansions:
            if time.time() - t0 > args.budget_s:
                stats["stopped"] = "budget"
                break
            negscore, _, st, path = heapq.heappop(pq)
            stats["expansions"] += 1
            try:
                ids, _ = ob.build_tok(ob.build_rag(st.pp, prov))
                cands = model.top_k_scored(list(ids), k=args.top_k)
                stats["gen_calls"] += 1
            except BaseException:
                continue
            for tac, sc in cands:
                if time.time() - t0 > args.budget_s:
                    stats["stopped"] = "budget"
                    break
                try:
                    with deadline(60, f"run {tac[:24]}"):
                        r = dojo.run_tac(st, tac)
                except BaseException:
                    # Watchdog interrupted a request: this session's pipe is now desynced and
                    # every later result would be shifted. Abort the arm rather than record
                    # corrupted search statistics.
                    stats["stopped"] = "desync"
                    return stats
                if isinstance(r, ldj.ProofFinished):
                    stats["proved"] = True
                    stats["proof"] = path + [tac]
                    stats["elapsed"] = time.time() - t0
                    return stats
                if not isinstance(r, ldj.TacticState):
                    continue
                sh = all_shallow(dojo, r) if arm != "C" else digest(dojo, r)
                stats["probe_fallbacks"] += int(sh == r.pp)
                key = keyfn(r)
                if key in stored:
                    stats["merges"] += 1
                    kept_sh, kept_st, kept_path = stored[key]
                    # Arm A merges by rendering alone; a merge whose discarded state carries a
                    # different separating rung has collapsed phi_shallow-DISTINCT states.
                    # Whether that is operationally unsound is decided by probing, not assumed.
                    if kept_sh != sh:
                        stats["unsound_merges"] += 1
                        if args.probe_unsound:
                            outs, div = probe_pair(kept_st, r)
                            stats["unsound_events"].append({
                                "retained_path": kept_path, "discarded_path": path + [tac],
                                "pp_bytes": len(r.pp), "probes": outs,
                                "divergent_probes": div})
                            _log(f"    merge probe: divergent={div}")
                    continue
                stored[key] = (sh, r, path + [tac])
                stats["nodes"] += 1
                counter += 1
                heapq.heappush(pq, (negscore - sc, counter, r, path + [tac]))
        stats["stopped"] = stats["stopped"] or (
            "exhausted" if not pq else "max_expansions")
        stats["elapsed"] = time.time() - t0
        return stats

    _repo_cache = {}      # ONE GitHub API resolution per repo; anonymous quota is 60/hour
    for ti, s in enumerate(mine, 1):
        p = s["provenance"]
        rec = {"file": p["file_path"], "theorem": p["theorem_full_name"], "arms": {}}
        for arm in args.arms:
            try:
                rk = (p["repo_url"], p["repo_commit"])
                if rk not in _repo_cache:
                    _repo_cache[rk] = lean_git_repo(ldj, *rk)
                thm = Theorem(_repo_cache[rk], p["file_path"], p["theorem_full_name"])
                ctx = Dojo(thm, timeout=args.dojo_timeout)
                dojo, root = ctx.__enter__()
            except BaseException:
                kill_orphan_lean()
                continue
            try:
                rec["arms"][arm] = search(dojo, root, arm, p)
            except BaseException as e:
                rec["arms"][arm] = {"arm": arm, "error": type(e).__name__}
            finally:
                try:
                    ctx.__exit__(None, None, None)
                except Exception:
                    pass
        a, b = rec["arms"].get("A", {}), rec["arms"].get("B", {})
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        _log(f"[{ti}/{len(mine)}] {p['theorem_full_name'][:44]:44} "
             f"A:{'PROVED' if a.get('proved') else '------'} "
             f"m={a.get('merges', 0)}/u={a.get('unsound_merges', 0)} "
             f"| B:{'PROVED' if b.get('proved') else '------'} "
             f"({a.get('elapsed', 0):.0f}+{b.get('elapsed', 0):.0f}s)")

    fh.close()
    kill_orphan_lean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
