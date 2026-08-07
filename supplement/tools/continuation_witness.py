"""Proof-carrying continuation witnesses and the visibility-escape survival curve.

Two things this fixes about the earlier subtree experiment.

**The logic was backwards.** `tools/subtree_probe.py` asked "can ReProver find a proof from the
discarded successor?" and got NEITHER_PROVES_INCONCLUSIVE. Bounded search failure cannot establish
that no proof exists, so that design could never produce evidence. Here the logic is reversed:
harvest a continuation that is KNOWN to close the proof on one member, then replay it verbatim on
the other. Lean adjudicates. A suffix `u` with

    u ∈ L(s₁)  and  u ∉ L(s₂)      where  φ(s₁) = φ(s₂) byte-identically

is an executable witness that two identically-rendered states have different verified continuation
languages — not a search that came up empty.

**One step is not enough.** A scan of all 862 recorded classes found 95 where some probe closed the
proof, and in every one of those it closed on *every* member. Proof closure is invariant under
rendering-aliasing at depth 1, so a witness must escape later. That motivates the second output.

**Visibility-escape depth.** For a common continuation u = (a₁ … a_k),

    τ_φ(s₁,s₂;u) = min { j : φ(T_{a₁:j}(s₁)) ≠ φ(T_{a₁:j}(s₂))  or terminal outcomes differ }

A defect is *confined* while the executions differ but stay inside one observed equivalence class,
and *escapes* when that difference becomes visible. Running one shared continuation in lockstep
across all members and recording per-step renderings gives τ_φ directly, and the survival curve
S(k) = Pr(τ_φ > k) measures how fast the defect attenuates. This is produced whether or not any
proof-carrying witness turns up, so the run is informative either way.

Continuations come from the model policy (greedy ReProver rollout from the shared observation),
which is what a real prover would actually do from the merged node.

Usage:
    python tools/continuation_witness.py --runs runs/reconv_b2,runs/hunt_enn,runs/hunt_rare \\
        --states runs/killtest/states.jsonl --rollout-depth 8 --search-expansions 25
"""
from __future__ import annotations

import argparse
import glob as _glob
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
    print(f"[contwit] {m}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated run dirs")
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", default="runs/contwit")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--rollout-depth", type=int, default=8)
    ap.add_argument("--search-expansions", type=int, default=25)
    ap.add_argument("--search-budget-s", type=float, default=420.0)
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
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / f"witness.shard{args.shard}.jsonl", "a", encoding="utf-8")
    model = ReProverGenerator(cfg)
    ob = ObservationBuilder(cfg.tokenizer, cfg.max_input_length,
                            retriever=StateOnlyRetriever())

    # Confirmed classes across every run, joined to their Stage B rows for histories.
    targets, wanted = [], set()
    for run in args.runs.split(","):
        run = run.strip()
        try:
            conf = json.load(open(Path(run) / "confirmation.json"))
        except Exception:
            continue
        rows = {}
        for f in _glob.glob(str(Path(run) / "stageb.shard*.jsonl")):
            for l in open(f):
                if l.strip():
                    r = json.loads(l)
                    rows[(r["theorem"], r["pp_hash"])] = r
        for c in conf:
            if c.get("verdict") != "CONFIRMED":
                continue
            r = rows.get((c["theorem"], c["pp_hash"]))
            if r:
                targets.append((run, r, c))
                wanted.add((r["file"], r["theorem"], r["tactic_index"]))
    mine = [t for i, t in enumerate(targets) if i % args.n_shards == args.shard]
    _log(f"shard {args.shard}: {len(mine)} confirmed classes of {len(targets)}")

    # Stream the state file; materialising all 168k records costs several GB and gets the
    # process OOM-killed once a Dojo session is also live.
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
    _log(f"indexed {len(prov)} roots")

    def outcome(r):
        if isinstance(r, ldj.ProofFinished):
            return "COMPLETE"
        if isinstance(r, ldj.TacticState):
            return "S:" + _h(r.pp)
        msg = (getattr(r, "error", "") or str(r) or "")
        return "TIMEOUT" if "timeout" in msg.lower() else "FAIL"

    def find_proof(dojo, start, p):
        """Best-first ReProver search for a suffix that CLOSES the proof. Returns the tactic
        list or None. This is the only place a bounded search is used, and its failure is never
        interpreted as evidence -- only its successes are used, as known-good suffixes."""
        t0 = time.time()
        seen, counter = {start.pp}, 0
        pq = [(0.0, counter, start, [])]
        exp = 0
        while pq and exp < args.search_expansions:
            if time.time() - t0 > args.search_budget_s:
                return None
            neg, _, s_, path = heapq.heappop(pq)
            exp += 1
            try:
                ids, _ = ob.build_tok(ob.build_rag(s_.pp, p))
                cands = model.top_k_scored(list(ids), k=args.top_k)
            except BaseException:
                continue
            for tac, sc in cands:
                try:
                    with deadline(60, tac[:24]):
                        r = dojo.run_tac(s_, tac)
                except BaseException:
                    return None                       # pipe desync: abandon
                if isinstance(r, ldj.ProofFinished):
                    return path + [tac]
                if not isinstance(r, ldj.TacticState) or r.pp in seen:
                    continue
                seen.add(r.pp)
                counter += 1
                heapq.heappush(pq, (neg - sc, counter, r, path + [tac]))
        return None

    def replay(dojo, st, suffix):
        """Apply a suffix verbatim, recording the rendering after every step."""
        trace, cur = [], st
        for tac in suffix:
            try:
                with deadline(60, tac[:24]):
                    r = dojo.run_tac(cur, tac)
            except BaseException:
                trace.append("CRASH")
                return trace, False
            o = outcome(r)
            trace.append(o)
            if o == "COMPLETE":
                return trace, True
            if not isinstance(r, ldj.TacticState):
                return trace, False
            cur = r
        return trace, False

    n_wit = 0
    for ti, (run, r, c) in enumerate(mine, 1):
        s = prov.get((r["file"], r["theorem"], r["tactic_index"]))
        if s is None:
            continue
        p = s["provenance"]
        t0 = time.time()
        rec = {"run": run, "theorem": r["theorem"], "file": r["file"],
               "pp_hash": r["pp_hash"], "tactic_index": r["tactic_index"],
               "histories": r["histories"]}
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
                    with deadline(90, tac):
                        st = dojo.run_tac(st, tac)
                    if not isinstance(st, ldj.TacticState):
                        ok = False
                        break
                if ok:
                    members[" ; ".join(hist)] = st
            if len(members) < 2 or len({m.pp for m in members.values()}) != 1:
                rec["skipped"] = "members not restorable or renderings differ"
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
                continue
            names = list(members)

            # ---- (1) visibility-escape depth via a lockstep policy rollout ----------------
            # One shared continuation, generated from the observation the members share, applied
            # to every member. The first step at which their renderings or outcomes diverge is
            # tau_phi. This is what a prover merging the two nodes would actually walk into.
            cur = dict(members)
            escape, rollout = None, []
            # The greedy rollout can cycle (observed: `swap` then `swap` returning to an
            # already-seen rendering), which burns depth without probing anything new. Refuse a
            # tactic whose result revisits a rendering this rollout has already occupied.
            visited = {cur[names[0]].pp}
            for step in range(args.rollout_depth):
                ref = cur[names[0]]
                try:
                    ids, _ = ob.build_tok(ob.build_rag(ref.pp, p))
                    cands = model.top_k_scored(list(ids), k=args.top_k)
                except BaseException:
                    break
                chosen, res0 = None, None
                for tac, _sc in cands:
                    try:
                        with deadline(60, tac[:24]):
                            rr = dojo.run_tac(ref, tac)
                    except BaseException:
                        chosen = None
                        break
                    if isinstance(rr, ldj.ProofFinished):
                        chosen, res0 = tac, rr
                        break
                    if isinstance(rr, ldj.TacticState) and rr.pp not in visited:
                        chosen, res0 = tac, rr
                        visited.add(rr.pp)
                        break
                if chosen is None:
                    break
                step_out = {}
                nxt = {}
                for n, stt in cur.items():
                    if n == names[0]:
                        rr = res0
                    else:
                        try:
                            with deadline(60, chosen[:24]):
                                rr = dojo.run_tac(stt, chosen)
                        except BaseException:
                            rr = None
                    step_out[n] = outcome(rr) if rr is not None else "CRASH"
                    if isinstance(rr, ldj.TacticState):
                        nxt[n] = rr
                rollout.append({"step": step, "tactic": chosen, "outcomes": step_out})
                if len({step_out[n] for n in step_out}) > 1 and escape is None:
                    escape = step + 1                   # 1-indexed depth of first visible split
                    break
                if len(nxt) < len(cur):                 # someone terminated; all agreed, so stop
                    break
                cur = nxt
            rec["rollout"] = rollout
            rec["escape_depth"] = escape
            rec["rollout_steps"] = len(rollout)

            # ---- (2) proof-carrying continuation witnesses --------------------------------
            suffixes = {}
            for n, stt in members.items():
                u = find_proof(dojo, stt, p)
                if u:
                    suffixes[n] = u
                    _log(f"    proof found from [{n}]: {u}")
            rec["suffixes_found"] = {n: u for n, u in suffixes.items()}
            wits = []
            for src, u in suffixes.items():
                cross = {}
                for n, stt in members.items():
                    trace, closed = replay(dojo, stt, u)
                    cross[n] = {"closed": closed, "trace": trace}
                closers = [n for n, v in cross.items() if v["closed"]]
                failers = [n for n, v in cross.items()
                           if not v["closed"] and "CRASH" not in v["trace"]]
                if closers and failers:
                    # depth at which the suffix first behaves differently across members
                    d = None
                    L = max(len(v["trace"]) for v in cross.values())
                    for j in range(L):
                        vals = {tuple(v["trace"][j:j+1]) for v in cross.values()}
                        if len(vals) > 1:
                            d = j + 1
                            break
                    wits.append({"suffix": u, "source": src, "closes_on": closers,
                                 "fails_on": failers, "separation_depth": d, "cross": cross})
            rec["witnesses"] = wits
            if wits:
                n_wit += 1
                _log(f"  *** PROOF-CARRYING WITNESS *** {r['theorem']}  "
                     f"{wits[0]['closes_on']} vs {wits[0]['fails_on']} "
                     f"depth={wits[0]['separation_depth']}")
        except BaseException as e:
            rec["error"] = type(e).__name__
        finally:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        _log(f"[{ti}/{len(mine)}] {time.time()-t0:5.0f}s escape={rec.get('escape_depth')} "
             f"suffixes={len(rec.get('suffixes_found') or {})} witnesses={n_wit}  "
             f"{r['theorem'][:40]}")

    fh.close()
    kill_orphan_lean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
