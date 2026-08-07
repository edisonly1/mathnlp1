"""Observation-induced action regret: does aliasing intersect the policy's own top-k?

Battery-level action ambiguity is already established: in 58 of 60 confirmed alias classes some
tactic is accepted on one member and rejected on another (3x repeats, fresh sessions). What that
does not settle is whether the ambiguity touches the actions the deployed policy actually ranks.
This measures it directly.

For each confirmed class C with shared observation o (members byte-identical):

    A_k(o) = the policy's top-k tactics from o        -- ONE generation per class, since the
                                                         observation is identical and decoding
                                                         is deterministic beam search
    r(s,a) = 1 iff Lean accepts a at s                -- 3 repeats; self-inconsistent or
                                                         timeout/interrupted actions EXCLUDED
    R_k(C) = mean_i max_a r(s_i,a)  -  max_a mean_i r(s_i,a)

Reported per class:
  * top1_divergent      -- the highest-ranked action's outcome kind differs across members
  * viable sets V_i, their Jaccard, and whether the intersection is empty while every V_i is
    nonempty (strict conflict: each member solvable within top-k, no shared solution)
  * R_k               -- positive iff strict conflict
  * policy mass (softmax over beam scores) on actions whose outcome kind differs across members

A note on interpretation: the oracle in R_k is restricted to A_k(o), so without assumptions
about actions outside A_k it bounds unrestricted regret in neither direction. An earlier
version of this quantity measured 0.0 on census-derived
classes, which are ~99% ambient declaration-context pairs; the within-search population here is
the one where a transposition merge actually fires and has not been measured before.

Usage:
    tools/with_github_token.sh .venv/bin/python tools/action_regret.py \\
        --states runs/killtest/states.jsonl --out runs/regret --top-k 8 --shard 0 --n-shards 2
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _log(m: str) -> None:
    print(f"[regret] {m}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", default="runs/regret")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--runs", default="runs/reconv_b2,runs/hunt_enn,runs/hunt_rare")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dojo-timeout", type=int, default=900)
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
    model = ReProverGenerator(cfg)
    ob = ObservationBuilder(cfg.tokenizer, cfg.max_input_length,
                            retriever=StateOnlyRetriever())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / f"regret.shard{args.shard}.jsonl", "a", encoding="utf-8")

    # Confirmed classes joined to their Stage B rows (histories + tactic_index). Joining on
    # (theorem, pp_hash); (file, theorem) alone picks the wrong step of the theorem.
    targets = []
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
            if c.get("verdict") == "CONFIRMED":
                r = rows.get((c["theorem"], c["pp_hash"]))
                if r:
                    targets.append(r)
    mine = [t for i, t in enumerate(targets) if i % args.n_shards == args.shard]
    _log(f"shard {args.shard}: {len(mine)} confirmed classes of {len(targets)}")

    # Resume + streamed root index (materialising all 168k records gets OOM-killed).
    done = set()
    for f in out.glob("regret.shard*.jsonl"):
        for l in open(f):
            if l.strip():
                try:
                    d = json.loads(l)
                    done.add((d["theorem"], d["pp_hash"]))
                except Exception:
                    pass
    mine = [r for r in mine if (r["theorem"], r["pp_hash"]) not in done]
    wanted = {(r["file"], r["theorem"], r["tactic_index"]) for r in mine}
    prov = {}
    with open(args.states, encoding="utf-8") as sfh:
        for line in sfh:
            if not line.strip():
                continue
            d = json.loads(line)
            pv = d["provenance"]
            k = (pv["file_path"], pv["theorem_full_name"], pv["tactic_index"])
            if k in wanted:
                prov[k] = d
    _log(f"{len(mine)} to do, {len(prov)} roots indexed")

    def kind(o) -> str:
        if isinstance(o, ldj.ProofFinished):
            return "COMPLETE"
        if isinstance(o, ldj.TacticState):
            return "S:" + _h(o.pp)
        msg = (getattr(o, "error", "") or str(o) or "")
        return "TIMEOUT" if "timeout" in msg.lower() else "FAIL"

    for ci, r in enumerate(mine, 1):
        s = prov.get((r["file"], r["theorem"], r["tactic_index"]))
        if s is None:
            continue
        p = s["provenance"]
        t0 = time.time()
        rec = {"theorem": r["theorem"], "file": r["file"], "pp_hash": r["pp_hash"],
               "tactic_index": r["tactic_index"], "histories": r["histories"]}
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
                st = st0
                ok = True
                for tac in hist:
                    with deadline(90, tac[:30]):
                        st = dojo.run_tac(st, tac)
                    if not isinstance(st, ldj.TacticState):
                        ok = False
                        break
                if ok:
                    members[" ; ".join(hist)] = st
            pps = {m.pp for m in members.values()}
            rec["members"] = len(members)
            rec["identical"] = (len(pps) == 1 if len(members) >= 2 else None)
            if len(members) < 2 or not rec["identical"]:
                rec["skipped"] = "not restorable / not identical"
                raise StopIteration

            # ONE generation for the whole class: identical observation, deterministic decoding.
            shared_pp = next(iter(pps))
            ids, trunc = ob.build_tok(ob.build_rag(shared_pp, p))
            cands = model.top_k_scored(list(ids), k=args.top_k)
            rec["truncated"] = bool(trunc)
            rec["topk"] = [{"tactic": t, "score": sc} for t, sc in cands]

            # Execute every candidate on every member, with the nondeterminism control.
            table, excluded = {}, []
            for tac, _sc in cands:
                obs = {}
                bad = False
                for n, st in members.items():
                    vals = []
                    for _ in range(args.repeats):
                        try:
                            with deadline(60, tac[:30]):
                                vals.append(kind(dojo.run_tac(st, tac)))
                        except BaseException:
                            vals.append("CRASH")
                            break
                    obs[n] = vals
                    if len(set(vals)) > 1 or any(v in ("CRASH", "TIMEOUT") for v in vals):
                        bad = True
                if bad:
                    excluded.append(tac)      # nondeterministic or resource-failed: not counted
                else:
                    table[tac] = {n: v[0] for n, v in obs.items()}
            rec["excluded_actions"] = excluded
            rec["outcomes"] = table
            if not table:
                rec["skipped"] = "no usable actions"
                raise StopIteration

            names = list(members)
            accept = {a: {n: (0 if o[n] == "FAIL" else 1) for n in names}
                      for a, o in table.items()}
            # top-1 = highest-ranked usable action
            usable_order = [t for t, _ in cands if t in table]
            t1 = usable_order[0]
            rec["top1"] = t1
            rec["top1_outcomes"] = table[t1]
            rec["top1_divergent"] = len(set(table[t1].values())) > 1
            rec["top1_viability_divergent"] = len(set(accept[t1].values())) > 1

            V = {n: {a for a in table if accept[a][n]} for n in names}
            inter = set.intersection(*V.values()) if V else set()
            union = set.union(*V.values()) if V else set()
            rec["viable_sets"] = {n: sorted(v) for n, v in V.items()}
            rec["jaccard"] = (len(inter) / len(union)) if union else None
            rec["strict_conflict"] = bool(all(V.values()) and not inter)
            oracle = sum(1 for n in names if V[n]) / len(names)
            best_shared = max((sum(accept[a][n] for n in names) / len(names)
                               for a in table), default=0.0)
            rec["regret"] = oracle - best_shared

            # policy mass on outcome-divergent actions (softmax over beam scores of usable set)
            zs = [math.exp(sc) for t, sc in cands if t in table]
            z = sum(zs) or 1.0
            mass = 0.0
            for (t, sc) in cands:
                if t in table and len(set(table[t].values())) > 1:
                    mass += math.exp(sc) / z
            rec["divergent_policy_mass"] = mass
            _log(f"[{ci}/{len(mine)}] top1_div={rec['top1_divergent']} "
                 f"viab_div={rec['top1_viability_divergent']} "
                 f"strict={rec['strict_conflict']} R={rec['regret']:.2f} "
                 f"mass={mass:.2f}  {r['theorem'][:36]}")
        except StopIteration:
            pass
        except BaseException as e:
            rec["error"] = type(e).__name__
        finally:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
            try:
                kill_orphan_lean()       # per-class reap: Dojo.__exit__ leaks lake/lean children
            except Exception:
                pass
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()

    fh.close()
    kill_orphan_lean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
