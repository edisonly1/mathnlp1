"""Compact structural digest: separation, stability, and cost over the 59 confirmed classes.

The refinement of the paper keys search identity on phi_all_shallow, a large printed string
costing ~0.3s per node. The production form is a fixed-width native digest of the elaborated
goal. This implements one in a single 322-byte probe:

    mkForallFVars (lctx.getFVars) type     -- close the goal over its local context
    instantiateMVars                        -- resolve assigned metavariables
    abstractMVars                           -- Lean core's canonical renumbering of remaining
                                            -- expr AND level metavariables (built for
                                            -- discrimination-tree keying)
    Expr.hash                               -- 64-bit structural hash

Closing over fvars replaces free variables with de Bruijn indices, so gensym'd FVarIds cannot
split alpha-equivalent contexts; abstractMVars does the same for metavariables. The hash is
structural over constructor tags, constant names, universe levels, and literals, so it is
deterministic within a Lean version.

Per class: restore members, probe the digest three times per member (self-consistency control),
time it against the phi_all_shallow probe, and record separation. The class the printer cannot
separate at any rung is of particular interest: its members differ operationally, so a direct
expression hash may separate what no rendering can.

Usage:
    python tools/digest_eval.py --states runs/killtest/states.jsonl --out runs/digest_eval
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DIGEST = (
    "run_tac (show Lean.Elab.Tactic.TacticM Unit from do\n"
    "  let g ← Lean.Elab.Tactic.getMainGoal\n"
    "  g.withContext do\n"
    "    let d ← g.getDecl\n"
    "    let e ← Lean.Meta.mkForallFVars d.lctx.getFVars d.type\n"
    "    let e ← Lean.instantiateMVars e\n"
    "    let r ← Lean.Meta.abstractMVars e\n"
    "    throwError s!\"FPRINT_V2:dig={r.expr.hash}\")"
)


def _log(m: str) -> None:
    print(f"[digest] {m}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", default="runs/digest_eval")
    ap.add_argument("--dojo-timeout", type=int, default=900)
    args = ap.parse_args()

    from lean_dojo import Dojo, Theorem
    import lean_dojo as ldj
    from audit.fingerprint import PROBES, _run, parse_probe
    from audit.replay import lean_git_repo, kill_orphan_lean, deadline

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / "digest.jsonl", "a", encoding="utf-8")

    # Deduplicated confirmed classes joined to stageb rows.
    uniq = {}
    for run in ("runs/reconv_b2", "runs/hunt_enn", "runs/hunt_rare"):
        rows = {}
        for f in glob.glob(f"{run}/stageb.shard*.jsonl"):
            for l in open(f):
                if l.strip():
                    r = json.loads(l)
                    rows[(r["theorem"], r["pp_hash"])] = r
        for c in json.load(open(f"{run}/confirmation.json")):
            if c.get("verdict") == "CONFIRMED":
                k = (c["theorem"], c["pp_hash"])
                if k in rows:
                    uniq.setdefault(k, rows[k])
    done = set()
    for l in open(out / "digest.jsonl") if (out / "digest.jsonl").exists() else []:
        if l.strip():
            try:
                d = json.loads(l)
                done.add((d["theorem"], d["pp_hash"]))
            except Exception:
                pass
    todo = [(k, r) for k, r in uniq.items() if k not in done]
    _log(f"{len(todo)} classes to evaluate of {len(uniq)}")

    wanted = {(r["file"], r["theorem"], r["tactic_index"]) for _, r in todo}
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

    # ONE LeanGitRepo for the whole run: repository metadata resolution is the GitHub API call,
    # and the anonymous quota is 60/hour.
    any_p = next(iter(prov.values()))["provenance"]
    repo = lean_git_repo(ldj, any_p["repo_url"], any_p["repo_commit"])

    n_sep = n_col = 0
    for ci, ((thm_name, ph), r) in enumerate(todo, 1):
        s = prov.get((r["file"], r["theorem"], r["tactic_index"]))
        if s is None:
            continue
        p = s["provenance"]
        t0 = time.time()
        rec = {"theorem": thm_name, "pp_hash": ph, "file": r["file"]}
        try:
            thm = Theorem(repo, p["file_path"], p["theorem_full_name"])
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
            if len(members) < 2 or len({m.pp for m in members.values()}) != 1:
                rec["skipped"] = "not restorable / not identical"
                raise StopIteration
            digs, dt, sht = {}, [], []
            stable = True
            for name, st in members.items():
                vals = []
                for _ in range(3):
                    t1 = time.time()
                    with deadline(60, "digest"):
                        vals.append(parse_probe(_run(dojo, st, DIGEST) or "", False))
                    dt.append(time.time() - t1)
                if len(set(vals)) != 1 or vals[0] is None:
                    stable = False
                digs[name] = vals[0]
                t1 = time.time()
                with deadline(120, "shallow"):
                    _run(dojo, st, PROBES["phi_all_shallow"])
                sht.append(time.time() - t1)
            rec["digests"] = digs
            rec["stable"] = stable
            rec["separates"] = stable and len(set(digs.values())) > 1
            rec["digest_s"] = sum(dt) / len(dt)
            rec["shallow_s"] = sum(sht) / len(sht)
            if rec["separates"]:
                n_sep += 1
            else:
                n_col += 1
                _log(f"  COLLIDES: {thm_name[:50]} stable={stable} {set(digs.values())}")
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
                kill_orphan_lean()
            except Exception:
                pass
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        _log(f"[{ci}/{len(todo)}] {time.time()-t0:4.0f}s sep={n_sep} col={n_col} "
             f"{thm_name[:40]}")

    fh.close()
    kill_orphan_lean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
