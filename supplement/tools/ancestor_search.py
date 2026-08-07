"""Ancestor-rooted retention test for the natural sensitive class.

The representative-sensitivity experiment runs independent searches from each restored alias
member, which establishes that retention changes the bounded continuation outcome FROM THE
MERGED STATE, not that a full search from the common ancestor fails whenever the wrong member
is retained: an ancestor-rooted search might recover the theorem through another path. This
tool runs that full search directly.

PROTOCOL, FIXED BEFORE EXECUTION
--------------------------------
One class, MeasureTheory.snormEssSup_add_le, the repeat-verified sensitive class. From its
common ancestor (the class root), the two canonical histories are injected as the search's
first expansions, in an order fixed by the arm; every subsequent step is ordinary best-first
search under BFS-Prover-V2-7B, k=8, deterministic beams, 32 expansions, 900 s.

    arm R-prod   : rendering key, productive history (intro h; simp) arrives first
    arm R-unprod : rendering key, unproductive history (simp; intro h) arrives first
    arm D-unprod : rendering + digest key, unproductive first (the harder order)

Under the rendering key the second history's endpoint merges into the first's and is
discarded, exactly as the evaluated stack's state equality does; under the digest key both
survive. Injected nodes carry neutral priority 0, so they are explored before policy
successors (whose scores are negative logprobs), with insertion order breaking ties.

Readout, declared in advance: proved per arm. If R-unprod fails while R-prod and D-unprod
prove, the full-search claim holds for this class. If R-unprod proves through another path,
the correct claim is the weaker one, that retention discards the only member proved under
the tested continuation search, and the paper must say so.

Usage:
    python tools/ancestor_search.py --out runs/ancestor
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

THEOREM = "MeasureTheory.snormEssSup_add_le"
PROD = ["intro h", "simp"]
UNPROD = ["simp", "intro h"]

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


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _log(m: str) -> None:
    print(f"[ancestor] {m}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="runs/killtest/states.jsonl")
    ap.add_argument("--runs", default="runs/reconv_b2,runs/hunt_enn,runs/hunt_rare")
    ap.add_argument("--out", default="runs/ancestor")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-expansions", type=int, default=32)
    ap.add_argument("--budget-s", type=float, default=900.0)
    ap.add_argument("--dojo-timeout", type=int, default=1800)
    args = ap.parse_args()

    from lean_dojo import Dojo, Theorem
    import lean_dojo as ldj
    from audit.bfs_runner import BFSProverGenerator
    from audit.fingerprint import _run, parse_probe
    from audit.replay import lean_git_repo, kill_orphan_lean, deadline

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Locate the class row (histories + tactic_index) and the root provenance.
    row = None
    for run in args.runs.split(","):
        for f in glob.glob(f"{run.strip()}/stageb.shard*.jsonl"):
            for l in open(f):
                if l.strip():
                    r = json.loads(l)
                    if r["theorem"] == THEOREM:
                        row = r
    assert row, "class row not found"
    prov = None
    with open(args.states, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            p = d["provenance"]
            if (p["theorem_full_name"] == THEOREM
                    and p["tactic_index"] == row["tactic_index"]):
                prov = d
                break
    assert prov, "root provenance not found"
    _log(f"class at tactic_index {row['tactic_index']}, histories {row['histories']}")

    gen = BFSProverGenerator()

    def digest(dojo, st) -> str:
        try:
            with deadline(45, "digest"):
                v = parse_probe(_run(dojo, st, DIGEST_PROBE) or "", False)
            return v or st.pp
        except BaseException:
            return st.pp

    def run_arm(arm: str, first: list, second: list, use_digest: bool) -> dict:
        p = prov["provenance"]
        thm = Theorem(lean_git_repo(ldj, p["repo_url"], p["repo_commit"]),
                      p["file_path"], p["theorem_full_name"])
        ctx = Dojo(thm, timeout=args.dojo_timeout)
        dojo, root = ctx.__enter__()
        stats = {"arm": arm, "proved": False, "proof": None, "expansions": 0,
                 "nodes": 0, "merges": 0, "merged_member_discarded": None,
                 "stopped": None, "elapsed": 0.0}
        try:
            for pre in prov["proof_prefix"]:
                root = dojo.run_tac(root, pre)
            keyfn = (lambda st: st.pp + "\x01" + digest(dojo, st)) if use_digest \
                else (lambda st: st.pp)
            t0 = time.time()
            counter = 0
            seen = {keyfn(root)}
            pq = [(0.0, counter, root, [])]
            # Inject the two histories as the first expansions, in arm order.
            for hist in (first, second):
                cur, path = root, []
                for tac in hist:
                    with deadline(90, tac[:24]):
                        nxt = dojo.run_tac(cur, tac)
                    assert isinstance(nxt, ldj.TacticState), f"history step failed: {tac}"
                    cur, path = nxt, path + [tac]
                    key = keyfn(cur)
                    if key in seen:
                        stats["merges"] += 1
                        if len(path) == len(hist):
                            stats["merged_member_discarded"] = " ; ".join(hist)
                        continue
                    seen.add(key)
                    stats["nodes"] += 1
                    counter += 1
                    heapq.heappush(pq, (0.0, counter, cur, path))
            # Ordinary best-first search from here on.
            while pq and stats["expansions"] < args.max_expansions:
                if time.time() - t0 > args.budget_s:
                    stats["stopped"] = "budget"
                    break
                negscore, _, cur, path = heapq.heappop(pq)
                stats["expansions"] += 1
                try:
                    cands = gen.top_k_scored(cur.pp, k=args.top_k)
                except BaseException:
                    continue
                for tac, sc in cands:
                    if time.time() - t0 > args.budget_s:
                        stats["stopped"] = "budget"
                        break
                    try:
                        with deadline(60, tac[:24]):
                            r = dojo.run_tac(cur, tac)
                    except BaseException:
                        stats["stopped"] = "desync"
                        stats["elapsed"] = time.time() - t0
                        return stats
                    if isinstance(r, ldj.ProofFinished):
                        stats["proved"] = True
                        stats["proof"] = path + [tac]
                        stats["elapsed"] = time.time() - t0
                        return stats
                    if not isinstance(r, ldj.TacticState):
                        continue
                    key = keyfn(r)
                    if key in seen:
                        stats["merges"] += 1
                        continue
                    seen.add(key)
                    stats["nodes"] += 1
                    counter += 1
                    heapq.heappush(pq, (negscore - sc, counter, r, path + [tac]))
            stats["stopped"] = stats["stopped"] or (
                "exhausted" if not pq else "max_expansions")
            stats["elapsed"] = time.time() - t0
            return stats
        finally:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
            kill_orphan_lean()

    results = {}
    for arm, first, second, dig in (
            ("R_prod_first", PROD, UNPROD, False),
            ("R_unprod_first", UNPROD, PROD, False),
            ("D_unprod_first", UNPROD, PROD, True)):
        _log(f"=== arm {arm} ===")
        try:
            results[arm] = run_arm(arm, first, second, dig)
        except BaseException as e:
            results[arm] = {"arm": arm, "error": type(e).__name__, "detail": str(e)[:200]}
        _log(f"    {json.dumps(results[arm])[:220]}")
        (out / "results.json").write_text(json.dumps(
            {"theorem": THEOREM, "histories": row["histories"], "arms": results},
            indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
