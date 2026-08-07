"""Phase 3 of the second-policy evaluation: execute BFS-Prover candidates on every member.

Identical protocol to the ReProver top-k measurement: every candidate executed on every hidden
member with three repetitions; self-inconsistent or resource-failed candidates excluded; report
outcome divergence, acceptance divergence, top-1 divergence, best-shared-action regret, and
normalized score share on divergent candidates.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def _h(s): return hashlib.sha256(s.encode()).hexdigest()[:12]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="runs/policy2/candidates.jsonl")
    ap.add_argument("--states", default="runs/killtest/states.jsonl")
    ap.add_argument("--out", default="runs/policy2/results.jsonl")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dojo-timeout", type=int, default=900)
    args = ap.parse_args()

    from lean_dojo import Dojo, Theorem
    import lean_dojo as ldj
    from audit.replay import lean_git_repo, kill_orphan_lean, deadline

    cands = [json.loads(l) for l in open(args.candidates) if l.strip()]
    outp = Path(args.out); done = set()
    if outp.exists():
        for l in open(outp):
            if l.strip():
                d = json.loads(l); done.add((d["theorem"], d["pp_hash"]))
    todo = [c for c in cands if (c["theorem"], c["pp_hash"]) not in done]
    wanted = {(c["file"], c["theorem"], c["tactic_index"]) for c in todo}
    prov = {}
    with open(args.states) as fh:
        for line in fh:
            if not line.strip(): continue
            d = json.loads(line); pv = d["provenance"]
            k = (pv["file_path"], pv["theorem_full_name"], pv["tactic_index"])
            if k in wanted: prov[k] = d
    any_p = next(iter(prov.values()))["provenance"]
    repo = lean_git_repo(ldj, any_p["repo_url"], any_p["repo_commit"])
    fh_out = open(outp, "a")

    def kind(x):
        if isinstance(x, ldj.ProofFinished): return "COMPLETE"
        if isinstance(x, ldj.TacticState): return "S:" + _h(x.pp)
        m = (getattr(x, "error", "") or str(x) or "")
        return "TIMEOUT" if "timeout" in m.lower() else "FAIL"

    for ci, c in enumerate(todo, 1):
        s = prov.get((c["file"], c["theorem"], c["tactic_index"]))
        if s is None: continue
        p = s["provenance"]; t0 = time.time()
        rec = {"theorem": c["theorem"], "pp_hash": c["pp_hash"]}
        try:
            thm = Theorem(repo, p["file_path"], p["theorem_full_name"])
            ctx = Dojo(thm, timeout=args.dojo_timeout)
            dojo, st0 = ctx.__enter__()
        except BaseException:
            kill_orphan_lean(); continue
        try:
            for pre in s["proof_prefix"]: st0 = dojo.run_tac(st0, pre)
            members = {}
            for hist in c["histories"]:
                st = st0; ok = True
                for tac in hist:
                    with deadline(90, tac[:30]): st = dojo.run_tac(st, tac)
                    if not isinstance(st, ldj.TacticState): ok = False; break
                if ok: members[" ; ".join(hist)] = st
            if len(members) < 2 or len({m.pp for m in members.values()}) != 1:
                rec["skipped"] = "not restorable"; raise StopIteration
            table, excluded = {}, []
            for cd in c["candidates"]:
                tac = cd["tactic"]; obs = {}; bad = False
                for n, st in members.items():
                    vals = []
                    for _ in range(args.repeats):
                        try:
                            with deadline(60, tac[:30]):
                                vals.append(kind(dojo.run_tac(st, tac)))
                        except BaseException:
                            vals.append("CRASH"); break
                    obs[n] = vals
                    if len(set(vals)) > 1 or any(v in ("CRASH","TIMEOUT") for v in vals):
                        bad = True
                if bad: excluded.append(tac)
                else: table[tac] = {n: v[0] for n, v in obs.items()}
            rec["excluded"] = excluded; rec["outcomes"] = table
            rec["scores"] = {cd["tactic"]: cd["score"] for cd in c["candidates"]}
            if not table: rec["skipped"] = "no usable candidates"; raise StopIteration
            names = list(members)
            acc = {a: {n: 0 if o[n]=="FAIL" else 1 for n in names} for a,o in table.items()}
            order = [cd["tactic"] for cd in c["candidates"] if cd["tactic"] in table]
            t1 = order[0]
            rec["top1"] = t1
            rec["top1_outcome_div"] = len(set(table[t1].values())) > 1
            rec["top1_accept_div"] = len(set(acc[t1].values())) > 1
            V = {n: {a for a in table if acc[a][n]} for n in names}
            inter = set.intersection(*V.values()); union = set.union(*V.values())
            oracle = sum(1 for n in names if V[n]) / len(names)
            best = max((sum(acc[a][n] for n in names)/len(names) for a in table), default=0.0)
            rec["regret"] = oracle - best
            rec["nonvacuous"] = all(bool(v) for v in V.values())
            rec["outcome_div_any"] = any(len(set(o.values()))>1 for o in table.values())
            rec["accept_div_any"] = any(len(set(acc[a][n] for n in names))>1 for a in acc)
            zs = {a: math.exp(rec["scores"].get(a,0.0)) for a in table}
            z = sum(zs.values()) or 1.0
            rec["div_score_share"] = sum(zs[a]/z for a in table
                                          if len(set(table[a].values()))>1)
        except StopIteration: pass
        except BaseException as e: rec["error"] = type(e).__name__
        finally:
            try: ctx.__exit__(None, None, None)
            except Exception: pass
            try: kill_orphan_lean()
            except Exception: pass
        fh_out.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh_out.flush()
        print(f"[exec] [{ci}/{len(todo)}] {time.time()-t0:4.0f}s "
              f"o={rec.get('outcome_div_any')} a={rec.get('accept_div_any')} "
              f"R={rec.get('regret')} {c['theorem'][:36]}", flush=True)
    fh_out.close(); kill_orphan_lean()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
