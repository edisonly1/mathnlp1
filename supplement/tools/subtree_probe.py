"""Did the discarded branch contain the proof?

The branch-losing witnesses establish that two rendering-identical states, merged by a pp-keyed
transposition table, have a divergent action reaching DIFFERENT successors. A pp-keyed search
stores whichever member arrives first and therefore reaches only one of those successors. What
that costs depends on something the witnesses do not say: whether the unreachable successor leads
to a proof the reachable one does not.

This runs a real ReProver best-first search from each successor independently and compares.

    both prove              -> the merge cost nothing here; the branches are redundant
    exactly one proves      -> ORDER-DEPENDENT PROOF LOSS. Whether the search finds the proof
                               depends on which member happened to be inserted first, which is
                               an artifact of expansion order, not of the mathematics.
    neither proves          -> inconclusive at this budget; report as such, not as "no loss"

Two witness shapes are handled. When the divergent probe succeeds on both members there are two
successors to compare. When it succeeds on only one (e.g. `aesop` FAILs on the other), there is a
single successor: if its subtree proves the goal, then a search holding the *other* member has no
edge there at all, and the merge can eliminate the only path.

Budgets are deliberately much larger than the A/B sweep's: this is 2-4 searches total, and the
question is what is reachable in principle, not what a 12-expansion budget finds.

Usage:
    python tools/subtree_probe.py --run runs/hunt_enn --states runs/killtest/states.jsonl \\
        --max-expansions 40 --budget-s 900
"""
from __future__ import annotations

import argparse
import glob
import heapq
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _log(m: str) -> None:
    print(f"[subtree] {m}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max-expansions", type=int, default=40)
    ap.add_argument("--budget-s", type=float, default=900.0)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--dojo-timeout", type=int, default=900)
    args = ap.parse_args()

    from lean_dojo import Dojo, Theorem
    import lean_dojo as ldj
    from audit.config import load_config
    from audit.model_runner import ReProverGenerator
    from audit.observation import ObservationBuilder, StateOnlyRetriever
    from audit.replay import lean_git_repo, kill_orphan_lean, deadline

    cfg = load_config(args.config)
    run = Path(args.run)
    model = ReProverGenerator(cfg)
    ob = ObservationBuilder(cfg.tokenizer, cfg.max_input_length,
                            retriever=StateOnlyRetriever())

    conf = {(c["theorem"], c["pp_hash"]): c
            for c in json.load(open(run / "confirmation.json"))}
    rows = {}
    for f in glob.glob(str(run / "stageb.shard*.jsonl")):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                rows[(r["theorem"], r["pp_hash"])] = r

    # Only classes that are BOTH confirmed and branch-losing, and only the probes that survived
    # the confirmation controls.
    targets = []
    for k, r in rows.items():
        c = conf.get(k, {})
        if c.get("verdict") != "CONFIRMED":
            continue
        surviving = [a for a in (r.get("branch_losing_probes") or [])
                     if a in (c.get("confirmed_probes") or [])]
        if surviving:
            targets.append((k, r, surviving))
    _log(f"{len(targets)} confirmed branch-losing classes")

    states = [json.loads(l) for l in Path(args.states).read_text().splitlines()]
    prov = {(s["provenance"]["file_path"], s["provenance"]["theorem_full_name"],
             s["provenance"]["tactic_index"]): s for s in states}

    def search(dojo, start, p, tag) -> dict:
        """Best-first ReProver search from `start`. pp-keyed dedup, as a real prover does."""
        t0 = time.time()
        st_ = {"tag": tag, "proved": False, "proof": None, "expansions": 0,
               "nodes": 0, "stopped": None}
        seen = {start.pp}
        counter = 0
        pq = [(0.0, counter, start, [])]
        while pq and st_["expansions"] < args.max_expansions:
            if time.time() - t0 > args.budget_s:
                st_["stopped"] = "budget"
                break
            neg, _, s_, path = heapq.heappop(pq)
            st_["expansions"] += 1
            try:
                ids, _ = ob.build_tok(ob.build_rag(s_.pp, p))
                cands = model.top_k_scored(list(ids), k=args.top_k)
            except BaseException:
                continue
            for tac, sc in cands:
                if time.time() - t0 > args.budget_s:
                    st_["stopped"] = "budget"
                    break
                try:
                    with deadline(90, f"run {tac[:24]}"):
                        r = dojo.run_tac(s_, tac)
                except BaseException:
                    st_["stopped"] = "desync"
                    st_["elapsed"] = time.time() - t0
                    return st_
                if isinstance(r, ldj.ProofFinished):
                    st_["proved"] = True
                    st_["proof"] = path + [tac]
                    st_["elapsed"] = time.time() - t0
                    return st_
                if not isinstance(r, ldj.TacticState) or r.pp in seen:
                    continue
                seen.add(r.pp)
                st_["nodes"] += 1
                counter += 1
                heapq.heappush(pq, (neg - sc, counter, r, path + [tac]))
        st_["stopped"] = st_["stopped"] or ("exhausted" if not pq else "max_expansions")
        st_["elapsed"] = time.time() - t0
        return st_

    results = []
    for (k, r, probes) in targets:
        s = prov.get((r["file"], r["theorem"], r["tactic_index"]))
        if s is None:
            _log(f"{r['theorem']}: root not found")
            continue
        p = s["provenance"]
        probe = probes[0]
        _log(f"=== {r['theorem']}  probe={probe!r} ===")
        rec = {"theorem": r["theorem"], "file": r["file"], "probe": probe,
               "histories": r["histories"], "subtrees": {}}
        try:
            thm = Theorem(lean_git_repo(ldj, p["repo_url"], p["repo_commit"]),
                          p["file_path"], p["theorem_full_name"])
            ctx = Dojo(thm, timeout=args.dojo_timeout)
            dojo, st0 = ctx.__enter__()
        except BaseException:
            kill_orphan_lean()
            continue
        try:
            for pre in s["proof_prefix"]:
                st0 = dojo.run_tac(st0, pre)
            succ = {}
            for hist in r["histories"]:
                st = st0
                ok = True
                for tac in hist:
                    with deadline(90, tac):
                        st = dojo.run_tac(st, tac)
                    if not isinstance(st, ldj.TacticState):
                        ok = False
                        break
                if not ok:
                    continue
                name = " ; ".join(hist)
                with deadline(120, f"probe {probe}"):
                    rr = dojo.run_tac(st, probe)
                if isinstance(rr, ldj.ProofFinished):
                    succ[name] = "COMPLETE"
                elif isinstance(rr, ldj.TacticState):
                    succ[name] = rr
                else:
                    succ[name] = "FAIL"
            rec["successor_kind"] = {n: (v if isinstance(v, str) else "STATE")
                                     for n, v in succ.items()}
            live = {n: v for n, v in succ.items() if not isinstance(v, str)}
            distinct = {v.pp for v in live.values()}
            rec["distinct_successor_renderings"] = len(distinct)
            _log(f"  successors: {rec['successor_kind']}  distinct={len(distinct)}")
            for n, stt in live.items():
                _log(f"  searching subtree of [{n}] …")
                rec["subtrees"][n] = search(dojo, stt, p, n)
                _log(f"    -> proved={rec['subtrees'][n]['proved']} "
                     f"({rec['subtrees'][n]['expansions']} exp, "
                     f"{rec['subtrees'][n]['elapsed']:.0f}s) "
                     f"{rec['subtrees'][n].get('proof')}")
            proved = {n: v["proved"] for n, v in rec["subtrees"].items()}
            n_true = sum(proved.values())
            if any(v == "COMPLETE" for v in succ.values()):
                rec["verdict"] = "PROBE_CLOSED_ON_ONE_MEMBER"
            elif len(live) < len(succ):
                rec["verdict"] = ("ONE_SIDED_PROOF_LOSS" if n_true
                                  else "ONE_SIDED_NO_PROOF_FOUND")
            elif n_true == 1:
                rec["verdict"] = "ORDER_DEPENDENT_PROOF_LOSS"
            elif n_true == len(proved) and n_true > 0:
                rec["verdict"] = "BOTH_PROVE_REDUNDANT"
            else:
                rec["verdict"] = "NEITHER_PROVES_INCONCLUSIVE"
            _log(f"  VERDICT: {rec['verdict']}")
        except BaseException as e:
            rec["verdict"] = f"ERROR:{type(e).__name__}"
            _log(f"  ERROR {type(e).__name__}")
        finally:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
        results.append(rec)

    (run / "subtree_probe.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("\n" + "=" * 78)
    for r in results:
        print(f"{r.get('verdict','?'):32} {r['theorem'][:44]}  probe={r['probe']!r}")
    print(f"\nwrote {run / 'subtree_probe.json'}")
    kill_orphan_lean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
