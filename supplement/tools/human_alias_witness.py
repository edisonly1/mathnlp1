"""Proof-carrying witnesses against the HUMAN Mathlib proof.

Both earlier attempts failed on power, not on science: they required *finding* a proof from a
state, and all 60 confirmed alias classes sit in MeasureTheory/Analysis where no automation closes
anything (attempt 1: 0 suffixes in 59 classes; attempt 2: 0 closes in ~400 battery invocations).

This removes the dependency entirely. Every root is a step of a real Mathlib proof, so the human
remainder closes it *by construction* — measured at 94% replay success on exactly this population
by `tools/verify_human_suffix.py`. The test becomes:

    s_H = T(root, human_tactic)                     the human-path successor
    s_A reached by a DIFFERENT history, φ(s_A) = φ(s_H) byte-identical
    u   = the human remainder, which closes from s_H by construction

    u fails on s_A   ⟹   u ∈ L(s_H) \\ L(s_A)

A pp-keyed transposition table merging s_H and s_A keeps one representative. If it keeps s_A, the
actual library proof is erased from the search space. That is a stronger and more interpretable
claim than erasing a synthetic closer.

Every witness carries its own control: the same remainder is replayed on s_H in the same session
and MUST close there. If it does not, the suffix was never known-good and the class is discarded
rather than counted — so a "witness" can never be an artifact of an unreplayable human proof.

Histories are canonical (steps that do not change the rendering are stripped), so an inert tactic
cannot manufacture a spurious "different history".

Usage:
    python tools/human_alias_witness.py --states runs/killtest/states.jsonl --out runs/humanwit \\
        --roots 400 --depth 2 --shard 0 --n-shards 4
"""
from __future__ import annotations

import argparse
import collections
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
    print(f"[humanwit] {m}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", type=int, default=400)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=606061)
    ap.add_argument("--max-suffix", type=int, default=40)
    ap.add_argument("--min-remainder", type=int, default=1,
                    help="minimum number of tactics AFTER the human tactic. A 1-tactic remainder "
                         "only re-tests depth-1 closure invariance (already established: 95/95 "
                         "closers closed on every member), so it cannot exhibit proof-language "
                         "separation. 56%% of the first sweep was uninformative for this reason; "
                         "sample for length instead of taking what the population offers.")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dojo-timeout", type=int, default=900)
    ap.add_argument("--goal-contains", default="")
    args = ap.parse_args()

    from lean_dojo import Dojo, Theorem
    import lean_dojo as ldj
    from audit.replay import lean_git_repo, kill_orphan_lean, deadline

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / f"humanwit.shard{args.shard}.jsonl", "a", encoding="utf-8")

    # TWO PASSES, because holding all 168k full records costs several GB and five workers
    # doing it simultaneously get SIGKILLed by the OS with no traceback. Pass 1 keeps only
    # strings (the per-theorem human tactic sequence); pass 2 re-reads and materialises only the
    # few hundred root records this shard actually needs.
    seq = collections.defaultdict(dict)          # (file, thm) -> {idx: human_tactic}
    with open(args.states, encoding="utf-8") as sfh:
        for line in sfh:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("proof_prefix") is None or not d.get("human_tactic"):
                continue
            pv = d["provenance"]
            seq[(pv["file_path"], pv["theorem_full_name"])][pv["tactic_index"]] = \
                d["human_tactic"]
    _log(f"pass 1: {len(seq)} theorems with human tactics")

    # A usable root needs the whole remainder after it: contiguous indices to the end.
    cand = []                                    # (file, thm, idx, remainder)
    for (f, t), byidx in seq.items():
        idxs = sorted(byidx)
        if idxs != list(range(idxs[0], idxs[0] + len(idxs))):
            continue
        for pos, i in enumerate(idxs[:-1]):
            remainder = [byidx[j] for j in idxs[pos:]]
            # remainder[0] is the human tactic itself; the suffix under test is remainder[1:].
            if args.min_remainder + 1 <= len(remainder) <= args.max_suffix:
                cand.append((f, t, i, remainder))
    del seq
    rng = random.Random(args.seed)
    rng.shuffle(cand)
    cand = cand[:args.roots]
    mine_keys = {(f, t, i): rem for n, (f, t, i, rem) in enumerate(cand)
                 if n % args.n_shards == args.shard}
    _log(f"pass 1: {len(mine_keys)} roots assigned to shard {args.shard} "
         f"of {len(cand)} sampled")

    # Pass 2: materialise only this shard's root records.
    mine = []
    with open(args.states, encoding="utf-8") as sfh:
        for line in sfh:
            if not line.strip():
                continue
            d = json.loads(line)
            pv = d["provenance"]
            k = (pv["file_path"], pv["theorem_full_name"], pv["tactic_index"])
            if k in mine_keys:
                if args.goal_contains:
                    terms = [x for x in args.goal_contains.split(",") if x]
                    if not any(x in d["observation"]["phi_state"] for x in terms):
                        continue
                mine.append((d, mine_keys[k]))
    # Resume: skip roots any previous run already recorded. Five concurrent Lean processes
    # exhaust 24GB (free RAM 0.3GB, swap 4.8/6GB) and macOS SIGKILLs workers silently, so runs
    # get restarted at a narrower width and must not redo completed work.
    done = set()
    for f_ in out.glob("humanwit.shard*.jsonl"):
        for line in open(f_, encoding="utf-8"):
            if line.strip():
                try:
                    d_ = json.loads(line)
                    done.add((d_["file"], d_["theorem"], d_["tactic_index"]))
                except Exception:
                    pass
    before = len(mine)
    mine = [(d, r) for d, r in mine
            if (d["provenance"]["file_path"], d["provenance"]["theorem_full_name"],
                d["provenance"]["tactic_index"]) not in done]
    _log(f"pass 2: materialised {before} root records, {before-len(mine)} already done, "
         f"{len(mine)} to do")

    n_alias = n_wit = n_ctrl_fail = 0
    for ri, (rec, remainder) in enumerate(mine, 1):
        p = rec["provenance"]
        human = remainder[0]
        rest = remainder[1:]                        # what must run from the human successor
        t0 = time.time()
        row = {"file": p["file_path"], "theorem": p["theorem_full_name"],
               "tactic_index": p["tactic_index"], "human_tactic": human,
               "remainder_len": len(rest)}
        try:
            thm = Theorem(lean_git_repo(ldj, p["repo_url"], p["repo_commit"]),
                          p["file_path"], p["theorem_full_name"])
            ctx = Dojo(thm, timeout=args.dojo_timeout)
            dojo, root = ctx.__enter__()
        except BaseException:
            kill_orphan_lean()
            continue
        try:
            with deadline(300, "prefix"):
                for pre in rec["proof_prefix"]:
                    root = dojo.run_tac(root, pre)
                    if not isinstance(root, ldj.TacticState):
                        raise RuntimeError("prefix failed")
            with deadline(120, "human tactic"):
                s_h = dojo.run_tac(root, human)
            if not isinstance(s_h, ldj.TacticState):
                row["skipped"] = "human tactic did not yield a state"
                raise StopIteration
            human_pp = s_h.pp

            # Expand with the standard pool, canonical histories, looking for a DIFFERENT history
            # that renders exactly like the human successor.
            root_h = _h(root.pp)
            aliases, frontier = {}, [((), root)]
            for _ in range(args.depth):
                nxt = []
                for hist, st in frontier:
                    for tac in GENERIC_POOL + goal_pool(st.pp):
                        if not hist and tac == human:
                            continue                # that is the human path itself
                        try:
                            with deadline(45, tac[:24]):
                                r = dojo.run_tac(st, tac)
                        except BaseException:
                            continue
                        if not isinstance(r, ldj.TacticState):
                            continue
                        if _h(r.pp) == _h(st.pp):
                            continue                # inert: canonical history unchanged
                        hist2 = hist + (tac,)
                        if r.pp == human_pp and hist2 != (human,):
                            aliases.setdefault(" ; ".join(hist2), r)
                        nxt.append((hist2, r))
                frontier = nxt
                if not frontier:
                    break
            row["n_aliases"] = len(aliases)
            if not aliases:
                raise StopIteration
            n_alias += 1

            # CONTROL: the remainder must close from the human successor in THIS session.
            def run_suffix(st):
                cur = st
                for j, tac in enumerate(rest):
                    with deadline(120, tac[:30]):
                        r = dojo.run_tac(cur, tac)
                    if isinstance(r, ldj.ProofFinished):
                        return "COMPLETE", j
                    if not isinstance(r, ldj.TacticState):
                        return "FAIL@%d" % j, j
                    cur = r
                return "NOT_CLOSED", len(rest)

            ctrl, _ = run_suffix(s_h)
            row["control_on_human"] = ctrl
            if ctrl != "COMPLETE":
                n_ctrl_fail += 1
                row["skipped"] = "human remainder does not close from the human successor"
                raise StopIteration

            # The test: same suffix, verbatim, from each alias.
            res = {}
            for name, st in aliases.items():
                vals = [run_suffix(st)[0] for _ in range(args.repeats)]
                res[name] = vals
            row["alias_results"] = res
            sep = {n: v for n, v in res.items()
                   if len(set(v)) == 1 and v[0] != "COMPLETE"}
            row["separating_aliases"] = sep
            if sep:
                n_wit += 1
                _log(f"  *** WITNESS *** {p['theorem_full_name']} "
                     f"human={human!r} aliases={list(sep)} outcomes={sep}")
        except StopIteration:
            pass
        except BaseException as e:
            row["error"] = type(e).__name__
        finally:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
            # Dojo.__exit__ does not reliably reap its `lake env lean` grandchild. Over a few
            # dozen roots the leaked Lean processes (~1.5GB each) exhaust the machine and macOS
            # SIGKILLs the worker mid-sweep with no traceback -- observed repeatedly at 3 and 5
            # workers alike, always after 40-90 roots. Reap per root, scoped to THIS process's
            # own descendants so parallel shards cannot kill each other's live sessions.
            try:
                kill_orphan_lean()
            except Exception:
                pass
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        _log(f"[{ri}/{len(mine)}] {time.time()-t0:5.0f}s aliased={n_alias} "
             f"ctrl_fail={n_ctrl_fail} WITNESSES={n_wit}")

    fh.close()
    kill_orphan_lean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
