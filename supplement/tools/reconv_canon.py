"""Paired check: does the reconvergence rate depend on how histories are canonicalized?

The Stage A rate (17.4%) strips a step from a history when it does not change the RENDERING.
That definition is circular under this paper's own thesis: a rendering-inert step can change
hidden executable state (our dominant witness class is exactly such a step), so rendering-based
stripping merges histories that are executably distinct and prunes exec-changed successors from
the frontier. Both effects UNDER-count reconvergence.

This measures the gap directly. For each root, the same expansion is run twice in one session:

    render-canon : a step joins a history iff it changes the rendering        (the paper's 17.4%)
    exec-canon   : a step joins a history iff it changes phi_all_shallow      (the separating rung,
                                                                               59/60 exec-faithful)

Classes are keyed by rendering in both modes (that is what a pp-keyed table merges); only the
history/frontier definition changes. Prefix-free filtering is unchanged, so parent-child
self-aliases do not count in either mode. Reported per root: reconvergence class count under each
canon, plus probe-fallback counts (a failed shallow probe degrades that node to rendering-inertness,
which is conservative).

Usage:
    tools/with_github_token.sh .venv/bin/python tools/reconv_canon.py \\
        --states runs/killtest/states.jsonl --out runs/canon --roots 400 --shard 0 --n-shards 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.reconvergence import GENERIC_POOL, goal_pool  # noqa: E402


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _log(m: str) -> None:
    print(f"[canon] {m}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", type=int, default=400)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=515254)
    ap.add_argument("--dojo-timeout", type=int, default=600)
    args = ap.parse_args()

    from lean_dojo import Dojo, Theorem
    import lean_dojo as ldj
    from audit.fingerprint import PROBES, _run, parse_probe
    from audit.replay import lean_git_repo, kill_orphan_lean, deadline

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / f"canon.shard{args.shard}.jsonl", "a", encoding="utf-8")

    # Two-pass root sampling: pass 1 records byte offsets of usable lines only (materialising all
    # 168k records costs GBs per worker and gets the process SIGKILLed under a live Dojo).
    offsets = []
    with open(args.states, "rb") as sfh:
        pos = sfh.tell()
        for line in sfh:
            if b'"proof_prefix"' in line and b'"proof_prefix": null' not in line:
                offsets.append(pos)
            pos = sfh.tell()
    rng = random.Random(args.seed)
    rng.shuffle(offsets)
    picked = offsets[:args.roots]
    mine_off = [o for i, o in enumerate(picked) if i % args.n_shards == args.shard]
    mine = []
    with open(args.states, encoding="utf-8") as sfh:
        for o in mine_off:
            sfh.seek(o)
            d = json.loads(sfh.readline())
            if d.get("proof_prefix") is not None:
                mine.append(d)
    # resume
    done = set()
    for f in out.glob("canon.shard*.jsonl"):
        for l in open(f):
            if l.strip():
                try:
                    d = json.loads(l)
                    done.add((d["file"], d["theorem"], d["tactic_index"]))
                except Exception:
                    pass
    before = len(mine)
    mine = [d for d in mine
            if (d["provenance"]["file_path"], d["provenance"]["theorem_full_name"],
                d["provenance"]["tactic_index"]) not in done]
    _log(f"shard {args.shard}: {before} roots, {before-len(mine)} done, {len(mine)} to do")

    def expand(dojo, root, canon: str) -> dict:
        """One expansion; returns class count and diagnostics. `canon` picks the inertness key."""
        fallbacks = 0

        def key(st) -> str:
            nonlocal fallbacks
            if canon == "render":
                return "R" + _h(st.pp)
            try:
                with deadline(45, "probe shallow"):
                    v = parse_probe(_run(dojo, st, PROBES["phi_all_shallow"]) or "", True)
                if v:
                    return "X" + _h(v)
            except BaseException:
                pass
            fallbacks += 1
            return "R" + _h(st.pp)      # degrade to rendering-inertness: conservative

        root_render = _h(root.pp)
        seen: dict = {root_render: {(): True}}
        frontier = [((), root, key(root))]
        for _ in range(args.depth):
            nxt = []
            for hist, st, stk in frontier:
                for tac in GENERIC_POOL + goal_pool(st.pp):
                    try:
                        with deadline(30, tac[:24]):
                            r = dojo.run_tac(st, tac)
                    except BaseException:
                        continue
                    if not isinstance(r, ldj.TacticState):
                        continue
                    rk = key(r)
                    if rk == stk:
                        continue                      # inert under this canon
                    hist2 = hist + (tac,)
                    seen.setdefault(_h(r.pp), {})[hist2] = True
                    nxt.append((hist2, r, rk))
            frontier = nxt
            if not frontier:
                break
        n_cls = 0
        for k, hists in seen.items():
            hs = [h for h in hists if h]
            if k == root_render or len(hs) < 2:
                continue
            if any(not (len(a) <= len(b) and b[:len(a)] == a)
                   and not (len(b) <= len(a) and a[:len(b)] == b)
                   for a in hs for b in hs if a != b):
                n_cls += 1
        return {"classes": n_cls, "renderings": len(seen), "fallbacks": fallbacks}

    for ri, s in enumerate(mine, 1):
        p = s["provenance"]
        t0 = time.time()
        row = {"file": p["file_path"], "theorem": p["theorem_full_name"],
               "tactic_index": p["tactic_index"]}
        try:
            thm = Theorem(lean_git_repo(ldj, p["repo_url"], p["repo_commit"]),
                          p["file_path"], p["theorem_full_name"])
            ctx = Dojo(thm, timeout=args.dojo_timeout)
            dojo, root = ctx.__enter__()
        except BaseException:
            kill_orphan_lean()
            continue
        try:
            with deadline(240, "prefix"):
                for pre in s["proof_prefix"]:
                    root = dojo.run_tac(root, pre)
                    if not isinstance(root, ldj.TacticState):
                        raise RuntimeError("prefix failed")
            row["render"] = expand(dojo, root, "render")
            row["exec"] = expand(dojo, root, "exec")
        except BaseException as e:
            row["error"] = type(e).__name__
        finally:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
            try:
                kill_orphan_lean()   # per-root reap: leaked lean children otherwise accumulate
            except Exception:
                pass
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        r_, x_ = row.get("render", {}), row.get("exec", {})
        _log(f"[{ri}/{len(mine)}] {time.time()-t0:5.0f}s render={r_.get('classes','?')} "
             f"exec={x_.get('classes','?')} fb={x_.get('fallbacks','?')} "
             f"{p['theorem_full_name'][:36]}")

    fh.close()
    kill_orphan_lean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
