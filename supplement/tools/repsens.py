"""Representative sensitivity: does which alias member the search keeps decide the outcome?

A rendering-keyed search that reaches two members of an alias class merges them and keeps
whichever arrived first. If the members have different bounded proof languages under the
policy's action distribution, that arrival order decides whether the theorem is proved. This
measures the effect directly, without needing the merge to fire by chance in a random search:

    for each confirmed class, restore every member, then run an INDEPENDENT bounded
    best-first search from each one under an identical budget, model, and seed.

    Delta_rep = Pr(proof | member i retained) - Pr(proof | member j retained)

The control that makes this tight: the members render identically, so the generator's first
candidate list is byte-identical across members and the searches can diverge only through
tactic EXECUTION, never through the policy seeing something different.

Note on what a positive result would and would not mean. The members are definitionally equal,
so a single `show` step converts either into the other and no member can be unprovable in
principle. Asymmetry here is therefore asymmetry of policy reachability within a budget, which
is the quantity that matters for a real prover, not a proof-theoretic fact about Lean.

Usage:
    python tools/repsens.py --out runs/repsens --runs runs/reconv_b2,runs/hunt_enn,runs/hunt_rare \
        --max-expansions 64 --top-k 16 --shard 0 --n-shards 4
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import heapq
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _log(m: str) -> None:
    print(f"[repsens] {m}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="runs/killtest/states.jsonl")
    ap.add_argument("--runs", default="runs/reconv_b2,runs/hunt_enn,runs/hunt_rare")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--policy", default="reprover", choices=["reprover", "bfs"],
                    help="tactic generator: ReProver ByT5 (pinned protocol) or "
                         "BFS-Prover-V2-7B via MPS (state+':::' interface)")
    ap.add_argument("--model-id", default="ByteDance-Seed/BFS-Prover-V2-7B")
    ap.add_argument("--max-expansions", type=int, default=64)
    ap.add_argument("--only-theorems", default="",
                    help="comma-separated substrings; restrict to matching classes")
    ap.add_argument("--budget-s", type=float, default=900.0,
                    help="wall-clock budget per (class, member) search")
    ap.add_argument("--dojo-timeout", type=int, default=1800)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    from lean_dojo import Dojo, Theorem
    import lean_dojo as ldj
    from audit.config import load_config
    from audit.model_runner import ReProverGenerator
    from audit.observation import ObservationBuilder, StateOnlyRetriever
    from audit.replay import lean_git_repo, kill_orphan_lean, deadline

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / f"repsens.shard{args.shard}.jsonl", "a", encoding="utf-8")

    if args.policy == "bfs":
        from audit.bfs_runner import BFSProverGenerator
        bfs = BFSProverGenerator(args.model_id)

        def propose(pp: str, prov) -> list:
            return bfs.top_k_scored(pp, k=args.top_k)
    else:
        model = ReProverGenerator(cfg)
        ob = ObservationBuilder(cfg.tokenizer, cfg.max_input_length,
                                retriever=StateOnlyRetriever())

        def propose(pp: str, prov) -> list:
            ids, _ = ob.build_tok(ob.build_rag(pp, prov))
            return model.top_k_scored(list(ids), k=args.top_k)

    # Confirmed classes joined to their Stage B rows for histories + tactic_index, deduplicated
    # on (theorem, pp_hash) exactly as the paper's 59-class corpus is.
    targets, seen = [], set()
    for run in args.runs.split(","):
        run = run.strip()
        try:
            conf = json.load(open(Path(run) / "confirmation.json"))
        except Exception:
            continue
        rows = {}
        for f in glob.glob(str(Path(run) / "stageb.shard*.jsonl")):
            for l in open(f):
                if l.strip():
                    r = json.loads(l)
                    rows[(r["theorem"], r["pp_hash"])] = r
        for c in conf:
            if c.get("verdict") != "CONFIRMED":
                continue
            k = (c["theorem"], c["pp_hash"])
            if k in seen:
                continue
            r = rows.get(k)
            if r:
                seen.add(k)
                targets.append(r)
    mine = [t for i, t in enumerate(targets) if i % args.n_shards == args.shard]

    done = set()
    for f in out.glob("repsens.shard*.jsonl"):
        for l in open(f):
            if l.strip():
                try:
                    d = json.loads(l)
                    done.add((d["theorem"], d["pp_hash"]))
                except Exception:
                    pass
    mine = [r for r in mine if (r["theorem"], r["pp_hash"]) not in done]
    if args.only_theorems:
        pats = [t for t in args.only_theorems.split(",") if t]
        mine = [r for r in mine if any(t in r["theorem"] for t in pats)]
    _log(f"shard {args.shard}: {len(mine)} classes to do of {len(targets)} unique")

    wanted = {(r["file"], r["theorem"], r["tactic_index"]) for r in mine}
    prov = {}
    with open(args.states, encoding="utf-8") as sfh:   # streamed: 168k records will not fit
        for line in sfh:
            if not line.strip():
                continue
            d = json.loads(line)
            pv = d["provenance"]
            k = (pv["file_path"], pv["theorem_full_name"], pv["tactic_index"])
            if k in wanted:
                prov[k] = d
    _log(f"{len(prov)} roots indexed")

    def search(dojo, start, provenance) -> dict:
        """Bounded best-first search from one member. Node identity is the rendering, which is
        what a stock prover uses; the point here is the starting state, not the key."""
        t0 = time.time()
        st = {"proved": False, "proof": None, "expansions": 0, "nodes": 0, "merges": 0,
              "gen_calls": 0, "stopped": None, "first_candidates": None,
              "accepted_first_step": [], "reached": [], "elapsed": 0.0}
        seen_pp = {start.pp}
        counter = 0
        pq = [(0.0, counter, start, [])]
        while pq and st["expansions"] < args.max_expansions:
            if time.time() - t0 > args.budget_s:
                st["stopped"] = "budget"
                break
            negscore, _, cur, path = heapq.heappop(pq)
            st["expansions"] += 1
            try:
                cands = propose(cur.pp, provenance)
                st["gen_calls"] += 1
            except BaseException:
                continue
            if not path:
                # Identical across members by construction; recorded to verify that control.
                st["first_candidates"] = [t for t, _ in cands]
            for tac, sc in cands:
                if time.time() - t0 > args.budget_s:
                    st["stopped"] = "budget"
                    break
                try:
                    with deadline(60, f"run {tac[:24]}"):
                        r = dojo.run_tac(cur, tac)
                except BaseException:
                    st["stopped"] = "desync"      # pipe is unusable; abandon this class
                    st["elapsed"] = time.time() - t0
                    return st
                if isinstance(r, ldj.ProofFinished):
                    st["proved"] = True
                    st["proof"] = path + [tac]
                    if not path:
                        st["accepted_first_step"].append(tac)
                    st["elapsed"] = time.time() - t0
                    return st
                if not isinstance(r, ldj.TacticState):
                    continue
                if not path:
                    st["accepted_first_step"].append(tac)
                if r.pp in seen_pp:
                    st["merges"] += 1
                    continue
                seen_pp.add(r.pp)
                st["reached"].append(_h(r.pp))   # to compare the trees the members explore
                st["nodes"] += 1
                counter += 1
                heapq.heappush(pq, (negscore - sc, counter, r, path + [tac]))
        st["stopped"] = st["stopped"] or ("exhausted" if not pq else "max_expansions")
        st["elapsed"] = time.time() - t0
        return st

    for ci, r in enumerate(mine, 1):
        s = prov.get((r["file"], r["theorem"], r["tactic_index"]))
        if s is None:
            continue
        p = s["provenance"]
        rec = {"theorem": r["theorem"], "file": r["file"], "pp_hash": r["pp_hash"],
               "tactic_index": r["tactic_index"], "histories": r["histories"],
               "policy": args.policy, "top_k": args.top_k,
               "max_expansions": args.max_expansions}
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
            members = {}
            for hist in r["histories"]:
                cur, ok = st0, True
                for tac in hist:
                    with deadline(90, tac[:30]):
                        cur = dojo.run_tac(cur, tac)
                    if not isinstance(cur, ldj.TacticState):
                        ok = False
                        break
                if ok:
                    members[" ; ".join(hist)] = cur
            pps = {m.pp for m in members.values()}
            rec["members"] = len(members)
            rec["identical"] = (len(pps) == 1) if len(members) >= 2 else None
            if len(members) < 2 or not rec["identical"]:
                rec["skipped"] = "not restorable / renderings not identical"
            else:
                rec["searches"] = {}
                for label, mst in members.items():
                    rec["searches"][label] = search(dojo, mst, p)
                    if rec["searches"][label]["stopped"] == "desync":
                        rec["skipped"] = "channel desync during search"
                        break
                if "skipped" not in rec:
                    proved = {l: v["proved"] for l, v in rec["searches"].items()}
                    rec["proved_any"] = any(proved.values())
                    rec["proved_all"] = all(proved.values())
                    # The headline: does the outcome depend on which member is kept?
                    rec["representative_sensitive"] = (rec["proved_any"]
                                                       and not rec["proved_all"])
                    firsts = [tuple(v["first_candidates"] or ()) for v in rec["searches"].values()]
                    rec["first_candidates_identical"] = len(set(firsts)) == 1
                    acc = [frozenset(v["accepted_first_step"]) for v in rec["searches"].values()]
                    rec["first_step_acceptance_identical"] = len(set(acc)) == 1
                    # How much of the explored state space is shared? Identical observations
                    # with identical candidate lists should give identical trees if the
                    # rendering were action-sufficient.
                    trees = [set(v["reached"]) for v in rec["searches"].values()]
                    inter = set.intersection(*trees) if trees else set()
                    union = set.union(*trees) if trees else set()
                    rec["tree_jaccard"] = (len(inter) / len(union)) if union else None
                    rec["tree_sizes"] = [len(t) for t in trees]
        except BaseException as e:
            rec["error"] = type(e).__name__
        finally:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
            kill_orphan_lean()
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        tag = ("SENSITIVE" if rec.get("representative_sensitive") else
               "all-proved" if rec.get("proved_all") else
               "none-proved" if rec.get("proved_any") is False else
               rec.get("skipped") or rec.get("error") or "?")
        _log(f"[{ci}/{len(mine)}] {r['theorem'][:44]:44} m={rec.get('members')} {tag}")

    fh.close()
    kill_orphan_lean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
