"""End-to-end cost accounting for the three node-identity keys.

`search_ab.py` answers whether refining node identity changes search OUTCOMES. It cannot answer
what the refinement COSTS, for two reasons that are properties of that harness rather than of the
keys:

  1. it computes `phi_all_shallow` on every state in EVERY arm, because arm A needs it to report
     whether a rendering merge collapsed fingerprint-distinct states. A deployed arm-A search pays
     no probe at all, so arm A's recorded wall clock overstates the baseline;
  2. arms run in the fixed order A, B, C for every theorem, so any warm-cache drift is confounded
     with the arm.

This tool removes both. Arm A runs with no probe of any kind, so it is the true stock baseline;
the arm order rotates with the theorem index; and the wall clock is decomposed into generator
time, tactic-execution time, and identity-probe time, so the probe's share of a real search is
reported rather than inferred from per-probe microbenchmarks.

It also records the queue statistics that "no node-count overhead" was previously asserted over:
expansions are capped by `--max-expansions` and are therefore equal across arms largely by
construction, so the informative counts are distinct nodes inserted, successors generated, merges,
and peak queue size.

    arm A  key = pp                      -- stock LeanDojo/ReProver node identity, no probe
    arm B  key = (pp, phi_all_shallow)   -- printer-based refinement, one pp.all probe per state
    arm C  key = (pp, digest)            -- 64-bit structural digest, one run_tac probe per state

Usage:
    python tools/search_cost.py --out runs/search_cost --theorems 24 \
        --max-expansions 12 --shard 0 --n-shards 2
"""
from __future__ import annotations

import argparse
import heapq
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _log(m: str) -> None:
    print(f"[searchCost] {m}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="runs/killtest/states.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--theorems", type=int, default=24)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=91003)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-expansions", type=int, default=12)
    ap.add_argument("--budget-s", type=float, default=600.0)
    ap.add_argument("--dojo-timeout", type=int, default=600)
    ap.add_argument("--arms", default="ABC")
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
    fh = open(out / f"cost.shard{args.shard}.jsonl", "a", encoding="utf-8")

    model = ReProverGenerator(cfg)
    ob = ObservationBuilder(cfg.tokenizer, cfg.max_input_length,
                            retriever=StateOnlyRetriever())

    states = [json.loads(l) for l in Path(args.states).read_text().splitlines()]
    roots = [s for s in states
             if s.get("proof_prefix") == [] and s["provenance"]["tactic_index"] == 0]
    seen_thm, uniq = set(), []
    for s in roots:
        t = (s["provenance"]["file_path"], s["provenance"]["theorem_full_name"])
        if t not in seen_thm:
            seen_thm.add(t)
            uniq.append(s)
    rng = random.Random(args.seed)
    rng.shuffle(uniq)
    picked = uniq[:args.theorems]
    mine = [(i, s) for i, s in enumerate(picked) if i % args.n_shards == args.shard]
    _log(f"shard {args.shard}: {len(mine)} theorems of {len(picked)}")

    DIGEST_PROBE = (
        "run_tac (show Lean.Elab.Tactic.TacticM Unit from do\n"
        "  let g ← Lean.Elab.Tactic.getMainGoal\n"
        "  g.withContext do\n"
        "    let d ← g.getDecl\n"
        "    let e ← Lean.Meta.mkForallFVars d.lctx.getFVars d.type\n"
        "    let e ← Lean.instantiateMVars e\n"
        "    let r ← Lean.Meta.abstractMVars e\n"
        "    throwError s!\"FPRINT_V2:dig={r.expr.hash}\")"
    )

    def search(dojo, root, arm: str, prov) -> dict:
        """Best-first search with per-component timing. `arm` selects the node-identity key."""
        t0 = time.time()
        stats = {"arm": arm, "expansions": 0, "nodes": 0, "successors": 0, "merges": 0,
                 "gen_calls": 0, "key_probes": 0, "probe_fallbacks": 0,
                 "peak_queue": 0, "final_queue": 0, "proved": False, "proof": None,
                 "stopped": None, "t_gen": 0.0, "t_tac": 0.0, "t_key": 0.0}

        def key_of(st_) -> str:
            """The arm's node-identity key. Arm A probes nothing, which is the whole point:
            it is the deployed baseline, not an instrumented one."""
            if arm == "A":
                return st_.pp
            tk = time.time()
            try:
                if arm == "C":
                    with deadline(45, "probe digest"):
                        v = parse_probe(_run(dojo, st_, DIGEST_PROBE) or "", False)
                else:
                    with deadline(45, "probe all_shallow"):
                        v = parse_probe(_run(dojo, st_, PROBES["phi_all_shallow"]) or "", True)
            except BaseException:
                v = None
            stats["t_key"] += time.time() - tk
            stats["key_probes"] += 1
            # A probe failure degrades this node to arm-A behaviour rather than splitting it.
            stats["probe_fallbacks"] += int(not v)
            return st_.pp + "\x00" + (v or "")

        stored = {key_of(root)}
        counter = 0
        pq = [(0.0, counter, root, [])]
        while pq and stats["expansions"] < args.max_expansions:
            if time.time() - t0 > args.budget_s:
                stats["stopped"] = "budget"
                break
            negscore, _, st, path = heapq.heappop(pq)
            stats["expansions"] += 1
            try:
                tg = time.time()
                ids, _ = ob.build_tok(ob.build_rag(st.pp, prov))
                cands = model.top_k_scored(list(ids), k=args.top_k)
                stats["t_gen"] += time.time() - tg
                stats["gen_calls"] += 1
            except BaseException:
                continue
            for tac, sc in cands:
                if time.time() - t0 > args.budget_s:
                    stats["stopped"] = "budget"
                    break
                try:
                    tt = time.time()
                    with deadline(60, f"run {tac[:24]}"):
                        r = dojo.run_tac(st, tac)
                    stats["t_tac"] += time.time() - tt
                except BaseException:
                    # Watchdog interrupt desyncs the pipe; every later result would be shifted.
                    stats["stopped"] = "desync"
                    stats["elapsed"] = time.time() - t0
                    return stats
                if isinstance(r, ldj.ProofFinished):
                    stats["proved"] = True
                    stats["proof"] = path + [tac]
                    stats["final_queue"] = len(pq)
                    stats["elapsed"] = time.time() - t0
                    return stats
                if not isinstance(r, ldj.TacticState):
                    continue
                stats["successors"] += 1
                key = key_of(r)
                if key in stored:
                    stats["merges"] += 1
                    continue
                stored.add(key)
                stats["nodes"] += 1
                counter += 1
                heapq.heappush(pq, (negscore - sc, counter, r, path + [tac]))
                stats["peak_queue"] = max(stats["peak_queue"], len(pq))
        stats["stopped"] = stats["stopped"] or ("exhausted" if not pq else "max_expansions")
        stats["final_queue"] = len(pq)
        stats["elapsed"] = time.time() - t0
        return stats

    # Warm the generator once so the first arm of the first theorem does not absorb model
    # lazy-init; the arm rotation below handles the remaining order effects.
    try:
        w = picked[0]
        ids, _ = ob.build_tok(ob.build_rag(w["observation"]["phi_state"], w["provenance"]))
        model.top_k_scored(list(ids), k=args.top_k)
        _log("generator warm-up done")
    except BaseException as e:
        _log(f"warm-up skipped: {type(e).__name__}")

    _repo_cache = {}
    for n, (gi, s) in enumerate(mine, 1):
        p = s["provenance"]
        # Rotate arm order with the GLOBAL theorem index so warm-cache drift is not confounded
        # with the arm under test.
        k = gi % len(args.arms)
        order = args.arms[k:] + args.arms[:k]
        rec = {"file": p["file_path"], "theorem": p["theorem_full_name"],
               "arm_order": order, "arms": {}}
        for arm in order:
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
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        parts = " | ".join(
            f"{a}:{rec['arms'].get(a, {}).get('elapsed', 0):.0f}s"
            f"/n{rec['arms'].get(a, {}).get('nodes', 0)}"
            f"/q{rec['arms'].get(a, {}).get('peak_queue', 0)}"
            for a in args.arms)
        _log(f"[{n}/{len(mine)}] {p['theorem_full_name'][:38]:38} order={order} {parts}")

    fh.close()
    kill_orphan_lean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
