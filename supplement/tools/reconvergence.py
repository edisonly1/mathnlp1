"""Stage A: do distinct tactic histories reconverge to a byte-identical visible state?

Why this specific question. LeanDojo's `TacticState` is `@dataclass(frozen=True)` with
`pp: str` compared and `id: int = field(compare=False)` — so equality and hashing are by the
RENDERED STATE ALONE, and the executable state id is deliberately excluded. ReProver's
best-first search uses it as a transposition-table key (`self.nodes[response]`), so any two
search nodes whose renderings match are merged into one, later silently overwriting earlier.

Our cross-theorem alias classes prove pp-identical states can have different executable futures,
but two different theorems never co-occur in one search tree, so that merge never fires on them.
The merge fires on WITHIN-SEARCH reconvergence: one root, two tactic histories, same rendering.
This measures whether that happens at all.

    δ(s₀,h₁) = s₁ ,  δ(s₀,h₂) = s₂ ,  h₁ ≠ h₂ ,  O(s₁) = O(s₂) byte-identical

No model needed: a fixed deterministic tactic pool is enough to expose reconvergence, and it
keeps the measurement independent of any particular prover's tactic distribution.

Usage:
    python tools/reconvergence.py --states runs/killtest/states.jsonl --out runs/reconv \\
        --roots 60 --depth 2 --shard 0 --n-shards 4
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

#: Deterministic, general-purpose pool. Deliberately NOT adversarial and NOT hand-picked per
#: goal: reconvergence found with ordinary tactics is the relevant phenomenon. Commuting or
#: idempotent operations (`simp` variants, `norm_num`, `constructor`) are the plausible
#: reconvergence sources.
#: The first pool was almost all normalisers (norm_cast, push_cast, dsimp only, ring_nf).
#: On most goals those are INERT or idempotent, so "reconvergence" was two tactics that both
#: did nothing — 38 of 76 classes were literally the root state. Genuine reconvergence needs
#: steps that transform the goal and commute, i.e. rewrites by DIFFERENT hypotheses.
GENERIC_POOL = ["simp", "constructor", "intro h", "omega"]


def goal_pool(pp: str, cap: int = 8) -> list:
    """Rewrites/specialisations derived from hypotheses visible in the goal.

    `rw [h₁]` then `rw [h₂]` versus the reverse order is the canonical way two distinct
    histories reach one state, and it requires both steps to actually change the goal.
    """
    import re
    names = []
    for line in pp.splitlines():
        if line.lstrip().startswith("⊢"):
            break
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_'!?₀-₉]*)\s*:", line.strip())
        if m and m.group(1) not in names:
            names.append(m.group(1))
    out = []
    for n in names[:cap]:
        out += [f"rw [{n}]", f"simp only [{n}]"]
    return out[:cap * 2]


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _log(m: str) -> None:
    print(f"[reconv] {m}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", type=int, default=60)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260724)
    ap.add_argument("--dojo-timeout", type=int, default=300)
    args = ap.parse_args()

    from lean_dojo import Dojo, Theorem
    import lean_dojo as ldj
    from audit.replay import lean_git_repo, kill_orphan_lean, deadline, WatchdogTimeout

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / f"reconv.shard{args.shard}.jsonl", "a", encoding="utf-8")

    states = [json.loads(l) for l in Path(args.states).read_text().splitlines()]
    usable = [s for s in states if s.get("proof_prefix") is not None]
    rng = random.Random(args.seed)
    rng.shuffle(usable)
    picked = usable[:args.roots]
    mine = [s for i, s in enumerate(picked) if i % args.n_shards == args.shard]
    _log(f"shard {args.shard}: {len(mine)} roots, depth {args.depth}, pool {len(GENERIC_POOL)}+goal-derived")

    n_reconv = 0
    for ri, s in enumerate(mine, 1):
        p = s["provenance"]
        t0 = time.time()
        try:
            thm = Theorem(lean_git_repo(ldj, p["repo_url"], p["repo_commit"]),
                          p["file_path"], p["theorem_full_name"])
            ctx = Dojo(thm, timeout=args.dojo_timeout)
            dojo, root = ctx.__enter__()
        except BaseException:
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

        # Breadth-first expansion. `seen` maps rendered-state hash -> list of histories that
        # reached it. States are NOT merged here — that merging is the thing under test.
        # Canonical history = the tactic steps that ACTUALLY changed the rendered state.
        # Without this, an inert tactic anywhere in a history fakes a distinct path.
        seen: dict = {_h(root.pp): [()]}
        frontier = [((), root)]
        root_hash = _h(root.pp)
        try:
            for d in range(args.depth):
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
                        # Only extend the canonical history when the state actually moved.
                        changed = (k != _h(st.pp))
                        hist2 = hist + (tac,) if changed else hist
                        seen.setdefault(k, []).append(hist2)
                        if changed:
                            nxt.append((hist2, r))
                frontier = nxt
                if not frontier:
                    break
        except BaseException:
            pass
        finally:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass

        # A reconvergence: one rendering, reached by >=2 DISTINCT histories.
        recs = []
        for k, hists in seen.items():
            uniq = {h for h in hists if h}          # drop the empty (root) history
            # Genuine: two CANONICAL histories (no-ops already stripped), neither a prefix of
            # the other, reaching a state that is not the root.
            if k != root_hash and len(uniq) >= 2 and any(
                    not (len(a) <= len(b) and b[:len(a)] == a) and
                    not (len(b) <= len(a) and a[:len(b)] == b)
                    for a in uniq for b in uniq if a != b):
                recs.append({"pp_hash": k[:16], "n_histories": len(uniq),
                             "histories": [list(h) for h in sorted(uniq)][:6]})
        if recs:
            n_reconv += 1
        fh.write(json.dumps({
            "file": p["file_path"], "theorem": p["theorem_full_name"],
            "tactic_index": p["tactic_index"],
            "states_explored": sum(len(v) for v in seen.values()),
            "distinct_renderings": len(seen),
            "reconvergences": recs,
        }, ensure_ascii=False) + "\n")
        fh.flush()
        _log(f"[{ri}/{len(mine)}] {time.time()-t0:5.1f}s  renderings={len(seen):4} "
             f"reconv={len(recs):3}  roots_with_reconv={n_reconv}")

    fh.close()
    kill_orphan_lean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
