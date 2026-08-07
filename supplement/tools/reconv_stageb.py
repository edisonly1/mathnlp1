"""Stage B: are within-search reconverged states OPERATIONALLY equivalent?

Stage A established that distinct tactic histories from one root do reach a byte-identical
rendering (commuting pairs like `intro h; simp only [fs]` vs `simp only [fs]; intro h`, and
sibling merges like `rw [this]` vs `simp only [this]`). That is only the PRECONDITION for
ReProver's transposition merge to fire, not yet a defect: `TacticState.__eq__` compares `pp`
and excludes `id`, so the merge is SOUND exactly when pp-equal implies future-equal.

This tests that implication directly. For each reconvergence class we hold one live `TacticState`
per canonical history inside a single Dojo session and put every member through the same
continuation battery:

    member i, member j   with   O(s_i) = O(s_j)   byte-identical
    for each probe a:    Exec(s_i, a)  vs  Exec(s_j, a)

A divergence on any probe is a counterexample to action sufficiency that fires WITHIN one search
tree — which is where ReProver's `self.nodes[response]` dict actually collapses the two nodes.

Behaviour, not fingerprints, is the primary measure. Two states may carry different metavariable
numbering and still be operationally identical; that is a rendering artifact, not a defect. The
probe outcomes are what the search would actually experience, so they decide. Fingerprints are
recorded alongside as a diagnostic only.

Usage:
    python tools/reconv_stageb.py --states runs/killtest/states.jsonl --out runs/reconv_b \\
        --roots 80 --depth 2 --shard 0 --n-shards 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.reconvergence import GENERIC_POOL, goal_pool, _h  # noqa: E402

#: Continuation battery. Deliberately broad and cheap: closers (`rfl`, `omega`, `assumption`),
#: normalisers (`simp`, `norm_num`), and structural moves. Goal-derived rewrites are appended
#: per member so the battery reaches the hypotheses actually in scope.
BATTERY = ["rfl", "assumption", "omega", "simp", "norm_num", "constructor",
           "exact?", "trivial", "decide", "simp_all", "tauto", "aesop"]


def _log(m: str) -> None:
    print(f"[stageB] {m}", file=sys.stderr, flush=True)


def _outcome(ldj, r) -> str:
    if isinstance(r, ldj.ProofFinished):
        return "COMPLETE"
    if isinstance(r, ldj.TacticState):
        return "SUCC:" + _h(r.pp)[:12]      # successor rendering matters, not just success
    msg = (getattr(r, "error", "") or str(r) or "")
    return "TIMEOUT" if "timeout" in msg.lower() else "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", type=int, default=80)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260724)
    ap.add_argument("--dojo-timeout", type=int, default=300)
    ap.add_argument("--goal-contains", default="",
                    help="comma-separated OR terms; keep only roots whose goal contains one. "
                         "The confirmed mechanism is a coercion the printer collapses, so "
                         "targeting coercion-bearing goals raises the hit rate over blind "
                         "sampling without narrowing to the single ENNReal instance.")
    args = ap.parse_args()

    from lean_dojo import Dojo, Theorem
    import lean_dojo as ldj
    from audit.replay import lean_git_repo, kill_orphan_lean, deadline

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / f"stageb.shard{args.shard}.jsonl", "a", encoding="utf-8")
    stat = open(out / f"roots.shard{args.shard}.jsonl", "a", encoding="utf-8")

    states = [json.loads(l) for l in Path(args.states).read_text().splitlines()]
    usable = [s for s in states if s.get("proof_prefix") is not None]
    if args.goal_contains:
        terms = [t for t in args.goal_contains.split(",") if t]
        usable = [s for s in usable
                  if any(t in s["observation"]["phi_state"] for t in terms)]
        _log(f"goal filter {terms}: {len(usable)} roots match")
    rng = random.Random(args.seed)          # same seed/order as Stage A => same roots
    rng.shuffle(usable)
    picked = usable[:args.roots]
    mine = [s for i, s in enumerate(picked) if i % args.n_shards == args.shard]
    _log(f"shard {args.shard}: {len(mine)} roots")

    n_classes = n_diverg = n_branch = 0
    for ri, s in enumerate(mine, 1):
        p = s["provenance"]
        t0 = time.time()
        try:
            thm = Theorem(lean_git_repo(ldj, p["repo_url"], p["repo_commit"]),
                          p["file_path"], p["theorem_full_name"])
            ctx = Dojo(thm, timeout=args.dojo_timeout)
            dojo, root = ctx.__enter__()
        except BaseException as exception:
            _log(f"[{ri}/{len(mine)}] startup failed for "
                 f"{p['theorem_full_name'][:48]}: "
                 f"{type(exception).__name__}: {str(exception)[:180]}")
            _log(traceback.format_exc(limit=12).strip())
            kill_orphan_lean()
            continue
        try:
            with deadline(240, "restore root"):
                for pre in s["proof_prefix"]:
                    root = dojo.run_tac(root, pre)
                    if not isinstance(root, ldj.TacticState):
                        raise RuntimeError("prefix failed")
        except BaseException:
            ctx.__exit__(None, None, None)
            continue

        # Same expansion as Stage A, but KEEP one live state per canonical history.
        root_hash = _h(root.pp)
        seen: dict = {root_hash: {(): root}}
        frontier = [((), root)]
        try:
            for _ in range(args.depth):
                nxt = []
                for hist, st in frontier:
                    for tac in GENERIC_POOL + goal_pool(st.pp):
                        try:
                            with deadline(30, f"run {tac}"):
                                r = dojo.run_tac(st, tac)
                        except BaseException:
                            continue
                        if not isinstance(r, ldj.TacticState):
                            continue
                        k = _h(r.pp)
                        changed = (k != _h(st.pp))
                        if not changed:
                            continue
                        hist2 = hist + (tac,)
                        seen.setdefault(k, {}).setdefault(hist2, r)
                        nxt.append((hist2, r))
                frontier = nxt
                if not frontier:
                    break

            # Probe every class that >=2 prefix-free canonical histories reach.
            for k, byhist in seen.items():
                hs = [h for h in byhist if h]
                if k == root_hash or len(hs) < 2:
                    continue
                if not any(not (len(a) <= len(b) and b[:len(a)] == a)
                           and not (len(b) <= len(a) and a[:len(b)] == b)
                           for a in hs for b in hs if a != b):
                    continue
                members = [(h, byhist[h]) for h in sorted(hs)]
                pp = members[0][1].pp
                assert all(m.pp == pp for _, m in members)     # byte-identical by construction
                battery = BATTERY + goal_pool(pp)
                rows, poisoned = {}, False
                for h, st in members:
                    res = {}
                    for a in battery:
                        try:
                            with deadline(60, f"probe {a}"):
                                res[a] = _outcome(ldj, dojo.run_tac(st, a))
                        except BaseException:
                            # The watchdog interrupted a request mid-flight, so `run_tac` now
                            # returns the PREVIOUS request's answer and every later result on
                            # this session is shifted. Nothing measured from here is usable:
                            # abandon the whole root rather than record a shifted divergence.
                            res[a] = "CRASH"
                            poisoned = True
                            break
                    rows[" ; ".join(h)] = res
                    if poisoned:
                        break
                if poisoned:
                    _log(f"  session desync on {p['theorem_full_name'][:40]}; abandoning root")
                    break
                # A probe divergence only counts if neither side timed out or crashed
                # (blueprint §7.6: resource failures are never logical failures).
                keys = list(rows)
                diverg = sorted({a for a in battery
                                 for x in keys for y in keys
                                 if rows[x][a] != rows[y][a]
                                 and not rows[x][a].startswith(("TIMEOUT", "CRASH"))
                                 and not rows[y][a].startswith(("TIMEOUT", "CRASH"))})
                # SELF-LOOP vs BRANCH-LOSING. A divergence only costs the search something if
                # the action reaches a rendering the search does not already have. In the three
                # confirmed witnesses so far, `simp` succeeded but its successor rendered
                # IDENTICALLY to the parent, so the edge a pp-keyed merge discards is a
                # self-loop and no node becomes unreachable. A probe is branch-losing when, on
                # some member, it closes the proof or reaches a DIFFERENT rendering.
                def _reaches_new(o: str) -> bool:
                    return o == "COMPLETE" or (o.startswith("SUCC:") and o[5:17] != k[:12])
                branch = sorted({a for a in diverg
                                 if any(_reaches_new(rows[x][a]) for x in keys)})
                n_classes += 1
                if diverg:
                    n_diverg += 1
                if branch:
                    n_branch += 1
                fh.write(json.dumps({
                    "file": p["file_path"], "theorem": p["theorem_full_name"],
                    "tactic_index": p["tactic_index"], "pp_hash": k[:16],
                    "pp_bytes": len(pp), "n_members": len(members),
                    "histories": [list(h) for h, _ in members],
                    "divergent_probes": diverg,
                    "branch_losing_probes": branch,
                    "outcomes": rows,
                }, ensure_ascii=False) + "\n")
                fh.flush()
                if branch:
                    _log(f"  *** BRANCH-LOSING *** {p['theorem_full_name']} {k[:10]} "
                         f"on {branch}")
                elif diverg:
                    _log(f"  divergence (self-loop) {p['theorem_full_name']} {k[:10]} "
                         f"on {diverg}")
        except BaseException:
            pass
        finally:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
        stat.write(json.dumps({
            "theorem": p["theorem_full_name"], "file": p["file_path"],
            "distinct_renderings": len(seen),
            "expandable": len(seen) > 1,
            "n_reconv_classes": sum(
                1 for k, bh in seen.items()
                if k != root_hash and len([h for h in bh if h]) >= 2),
        }) + "\n")
        stat.flush()
        _log(f"[{ri}/{len(mine)}] {time.time()-t0:5.1f}s  classes={n_classes} "
             f"divergent={n_diverg} BRANCH-LOSING={n_branch}")

    fh.close()
    stat.close()
    kill_orphan_lean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
