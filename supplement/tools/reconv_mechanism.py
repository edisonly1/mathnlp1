"""Stage C: what hides the difference, and does the proposed repair expose it?

Stage B confirmed three classes where two histories reach a byte-identical rendering yet Lean
disagrees on whether `simp` makes progress. `simp`'s progress test compares the resulting `Expr`
to the original structurally, so the states must differ in the term — invisibly to the printer.

This is the paper's own claim under test on a real within-search witness. The fingerprint ladder
climbs from what a prover actually sees to what Lean can actually be asked to render:

    phi_state        default pretty-printed goal      -- what ReProver keys on today
    phi_all_shallow  pp.all, notation off             -- implicits and instances made explicit
    phi_deep         pp.deepTerms                     -- no `⋯` elision at depth
    phi_full         pp.all + deepTerms + proofs      -- the R4 structural repair

If `phi_state` collides while a higher rung separates, the repair works and we can say which rung
is needed. If every rung collides, the difference is not renderable at all through the pp
interface, and no observation-side repair can reach it -- that would be the stronger and more
uncomfortable result, and it is the one worth knowing.

Usage:
    python tools/reconv_mechanism.py --run runs/reconv_b2 --states runs/killtest/states.jsonl
"""
from __future__ import annotations

import argparse
import glob as _glob
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--dojo-timeout", type=int, default=600)
    args = ap.parse_args()

    from lean_dojo import Dojo, Theorem
    import lean_dojo as ldj
    from audit.fingerprint import collect_fields, assemble, build_fingerprint
    from audit.replay import lean_git_repo, kill_orphan_lean, deadline

    run = Path(args.run)
    conf = [c for c in json.load(open(run / "confirmation.json"))
            if c.get("verdict") == "CONFIRMED"]
    # Stream the state file and keep ONLY the roots these witnesses need. Materialising all
    # 168k records (a 205MB file) as dicts costs several GB, and with a Dojo session on top the
    # process was SIGKILLed by the OS after printing its first header -- a silent death with no
    # traceback, which reads as a crash rather than as memory pressure.
    wanted = set()
    for f in _glob.glob(str(run / "stageb.shard*.jsonl")):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                wanted.add((r["file"], r["theorem"], r["tactic_index"]))
    prov = {}
    with open(args.states, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            pv = d["provenance"]
            k = (pv["file_path"], pv["theorem_full_name"], pv["tactic_index"])
            if k in wanted:
                prov[k] = d
    _log_n = len(prov)
    print(f"[mech] indexed {_log_n} roots of {len(wanted)} referenced", flush=True)
    # confirmation.json may predate the tactic_index field; recover it from the Stage B rows.
    idx = {}
    for f in _glob.glob(str(run / "stageb.shard*.jsonl")):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                idx[(r["file"], r["theorem"], r["pp_hash"])] = r["tactic_index"]

    # `phi_state` is the raw TacticState.pp and is not a Fingerprint field, so it is carried
    # alongside. The deep rung is `phi_state_deep`.
    LADDER = ["phi_state", "phi_all_shallow", "phi_state_deep", "phi_deep_proofs",
              "phi_full", "phi_full_proofs"]
    out = []
    for c in conf:
        ti = c.get("tactic_index", idx.get((c["file"], c["theorem"], c["pp_hash"])))
        s = prov.get((c["file"], c["theorem"], ti))
        if s is None:
            continue
        p = s["provenance"]
        print("=" * 78)
        print(f"{c['theorem']}\n  {c['file']}")
        thm = Theorem(lean_git_repo(ldj, p["repo_url"], p["repo_commit"]),
                      p["file_path"], p["theorem_full_name"])
        ctx = Dojo(thm, timeout=args.dojo_timeout)
        dojo, st0 = ctx.__enter__()
        rec = {"theorem": c["theorem"], "file": c["file"],
               "histories": c["histories"], "rungs": {}}
        try:
            for pre in s["proof_prefix"]:
                st0 = dojo.run_tac(st0, pre)
            fps = {}
            for hist in c["histories"]:
                st = st0
                ok = True
                for tac in hist:
                    with deadline(90, tac):
                        st = dojo.run_tac(st, tac)
                    if not isinstance(st, ldj.TacticState):
                        ok = False
                        break
                if not ok:
                    print(f"  replay failed for {hist}: {str(st)[:120]}")
                    continue
                try:
                    with deadline(300, "fingerprint"):
                        fp = build_fingerprint(assemble(collect_fields(dojo, st)),
                                               "mech", faithful=False)
                except BaseException as e:
                    print(f"  fingerprint failed for {hist}: {type(e).__name__}")
                    continue
                fp.phi_state = st.pp                      # the rung a prover actually keys on
                fps[" ; ".join(hist)] = fp
            if len(fps) < 2:
                print("  could not restore both members")
                ctx.__exit__(None, None, None)
                continue
            names = list(fps)
            for rung in LADDER:
                vals = {n: getattr(fps[n], rung, None) for n in names}
                vals = {n: v for n, v in vals.items() if v is not None} or vals
                if any(v is None for v in vals.values()):
                    rec["rungs"][rung] = "unavailable"
                    print(f"  {rung:16} unavailable")
                    continue
                sep = len({_h(v) for v in vals.values()}) > 1
                rec["rungs"][rung] = {"separates": sep,
                                      "hashes": {n: _h(v) for n, v in vals.items()},
                                      "bytes": {n: len(v) for n, v in vals.items()}}
                mark = "SEPARATES" if sep else "collides "
                print(f"  {rung:16} {mark}  " +
                      "  ".join(f"{_h(v)}({len(v)}B)" for v in vals.values()))
            # The separating rung differs by exactly the hidden content. Extract it verbatim:
            # a hash tells us THAT the states differ, this tells us WHAT the printer was hiding.
            a, b = (fps[names[0]].phi_all_shallow, fps[names[1]].phi_all_shallow)
            if a != b:
                import difflib
                dl = [l for l in difflib.unified_diff(a.split(), b.split(), n=6, lineterm="")
                      if l[:1] in "+-" and l[:3] not in ("+++", "---")]
                ctx_ = [l for l in difflib.unified_diff(a.split(), b.split(), n=6, lineterm="")]
                rec["diff_tokens"] = dl
                rec["diff_context"] = ctx_[:40]
                print(f"  HIDDEN DIFFERENCE ({len(b)-len(a):+d} bytes): {dl}")
                print("  context:")
                for l in ctx_[3:28]:
                    print(f"    {l}")

            # Derived fingerprints too: F_core is the elaborated-goal identity.
            for f in ("f_core", "f_proof", "f_env", "f_exec"):
                vals = {n: getattr(fps[n], f, None) for n in names}
                sep = len(set(vals.values())) > 1
                rec["rungs"][f] = {"separates": sep, "values": vals}
                print(f"  {f:16} {'SEPARATES' if sep else 'collides '}  "
                      + "  ".join(str(v)[:14] for v in vals.values()))
        except BaseException as e:
            print(f"  ERROR {type(e).__name__}: {e}")
            rec["error"] = type(e).__name__
        finally:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
        out.append(rec)

    (run / "mechanism.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {run / 'mechanism.json'}")
    kill_orphan_lean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
