"""Manual-audit form for one alias class (blueprint §8.3).

§8.3 caps the pilot at 10 audited classes and requires, for each:

  1. confirm exact model-input identity
  2. inspect full expression and environment differences
  3. identify the omitted information
  4. replay each discrepant tactic at least twice
  5. rule out timeout, version mismatch, and nondeterminism
  6. verify successor structure
  7. test whether retrieved premises or a repair exposes the distinction
  8. classify as mathematically relevant, operationally relevant, or irrelevant

This produces the evidence for 1-7 and leaves 8 to the auditor, because that judgement is the
point of a *manual* audit. Every replay here runs in a FRESH Dojo — the sessions that found the
divergence are not reused, so a wedged or stateful session cannot manufacture agreement.

Usage:
    python tools/audit_class.py --run runs/probe2 --key 8c33dbf195
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _show(label: str, text: str, limit: int = 1400) -> None:
    print(f"\n--- {label} ({len(text)} bytes) ---")
    print(text[:limit] + ("\n… [truncated]" if len(text) > limit else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--key", required=True, help="observation_key (prefix ok)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dojo-timeout", type=int, default=600)
    args = ap.parse_args()

    from lean_dojo import Dojo, LeanGitRepo, Theorem
    from audit.canon import alpha_normalize, env_fields
    from audit.fingerprint import collect_fields, assemble, build_fingerprint

    run = Path(args.run)
    states = [json.loads(l) for l in (run / "states.jsonl").read_text().splitlines()]
    members = [s for s in states if _h(s["observation"]["phi_state"]).startswith(args.key)]
    if len(members) < 2:
        print(f"no class with >=2 members for key {args.key!r} "
              f"(found {len(members)})", file=sys.stderr)
        return 1

    phi = members[0]["observation"]["phi_state"]
    print("=" * 78)
    print(f"MANUAL AUDIT (blueprint §8.3) — class {args.key}")
    print("=" * 78)

    # ---- 1. exact model-input identity ------------------------------------
    hashes = {_h(s["observation"]["phi_state"]) for s in members}
    raws = {s["observation"]["phi_state"] for s in members}
    print(f"\n[1] MODEL-INPUT IDENTITY")
    print(f"    members            : {len(members)}")
    print(f"    distinct φ_state   : {len(raws)}  (must be 1)")
    print(f"    distinct sha256    : {len(hashes)}  (must be 1)")
    print(f"    byte-identical     : {'YES' if len(raws) == 1 else 'NO — NOT AN ALIAS'}")
    _show("φ_state (what the model sees)", phi)
    for s in members:
        p = s["provenance"]
        print(f"    · {p['file_path']} :: {p['theorem_full_name']} "
              f"(tactic #{p['tactic_index']}, human={s['human_tactic'][:44]!r})")
    if len(raws) != 1:
        return 1

    # ---- 2/3. live re-fingerprint, then diff -------------------------------
    print(f"\n[2] RE-FINGERPRINT IN FRESH SESSIONS")
    fps = []
    for s in members:
        p = s["provenance"]
        repo = LeanGitRepo(p["repo_url"], p["repo_commit"])
        thm = Theorem(repo, p["file_path"], p["theorem_full_name"])
        try:
            ctx = Dojo(thm, timeout=args.dojo_timeout)
            dojo, st = ctx.__enter__()
        except Exception as e:
            print(f"    {p['theorem_full_name'][:40]}: DOJO FAIL {type(e).__name__}")
            fps.append(None)
            continue
        try:
            import lean_dojo as ldj
            ok = True
            for pre in s["proof_prefix"]:
                st = dojo.run_tac(st, pre)
                if not isinstance(st, ldj.TacticState):
                    print(f"    {p['theorem_full_name'][:40]}: PREFIX FAIL")
                    ok = False
                    break
            if not ok:
                fps.append(None)
                continue
            fields = collect_fields(dojo, st)
            fp = build_fingerprint(assemble(fields), "audit", faithful=False)
            fps.append(fp)
            print(f"    {p['theorem_full_name'][:40]}: {len(fields)}/11 fields  "
                  f"f_core={fp.f_core[:12]} f_env={fp.f_env[:12]}")
        finally:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass

    good = [f for f in fps if f is not None]
    if len(good) < 2:
        print("    INCONCLUSIVE: fewer than 2 members re-fingerprinted.")
        return 1

    a, b = good[0], good[1]
    print(f"\n[3] WHERE THE HIDDEN DIFFERENCE LIVES")
    print(f"    F_core differ      : {a.f_core != b.f_core}")
    print(f"    F_env  differ      : {a.f_env != b.f_env}   "
          f"({'SAME ENVIRONMENT — not recoverable from context' if a.f_env == b.f_env else 'environment differs'})")
    print(f"    α-normalized differ: {alpha_normalize(a.phi_full) != alpha_normalize(b.phi_full)}"
          f"   (if False, the difference is pure alpha-renaming)")
    print(f"    shallow differ     : {a.phi_all_shallow != b.phi_all_shallow}")
    print(f"    full differ        : {a.phi_full != b.phi_full}")
    print(f"    full+proofs differ : {a.phi_full_proofs != b.phi_full_proofs}")

    if a.phi_full != b.phi_full:
        diff = list(difflib.unified_diff(
            a.phi_full.splitlines(), b.phi_full.splitlines(),
            "member_A.phi_full", "member_B.phi_full", lineterm="", n=1))
        print(f"\n--- phi_full diff ({len(diff)} lines) ---")
        print("\n".join(diff[:60]))
    if a.f_env != b.f_env:
        fa, fb = env_fields(a.env_digest), env_fields(b.env_digest)
        print("\n--- env_behavior fields that differ ---")
        for k in sorted(set(fa) | set(fb)):
            if fa.get(k) != fb.get(k):
                print(f"    {k}: {str(fa.get(k))[:60]!r} vs {str(fb.get(k))[:60]!r}")

    # ---- 4/5/6. repeated replay in fresh sessions --------------------------
    cls = None
    for line in (run / "classes.jsonl").read_text().splitlines():
        c = json.loads(line)
        if c["observation_key"].startswith(args.key):
            cls = c
    tactics = (cls or {}).get("divergent_tactics") or []
    print(f"\n[4-6] REPEATED REPLAY OF DISCREPANT TACTICS (fresh Dojo per run)")
    if not tactics:
        print("    class has no recorded divergent tactics")
    for tac in tactics:
        print(f"\n    tactic: {tac!r}")
        for s in members:
            p = s["provenance"]
            repo = LeanGitRepo(p["repo_url"], p["repo_commit"])
            thm = Theorem(repo, p["file_path"], p["theorem_full_name"])
            seen = []
            for _ in range(args.repeats):
                try:
                    ctx = Dojo(thm, timeout=args.dojo_timeout)
                    dojo, st = ctx.__enter__()
                except Exception as e:
                    seen.append(f"DOJO-{type(e).__name__}")
                    continue
                try:
                    import lean_dojo as ldj
                    ok = True
                    for pre in s["proof_prefix"]:
                        st = dojo.run_tac(st, pre)
                        if not isinstance(st, ldj.TacticState):
                            ok = False
                            break
                    if not ok:
                        seen.append("PREFIX-FAIL")
                        continue
                    res = dojo.run_tac(st, tac)
                    name = type(res).__name__
                    msg = (getattr(res, "error", "") or "")[:60]
                    if "timeout" in msg.lower() or "Timeout" in name:
                        seen.append("TIMEOUT")
                    else:
                        seen.append(name if name != "TacticState" else
                                    f"SUCC({len(getattr(res, 'pp', '').splitlines())}L)")
                except Exception as e:
                    seen.append(f"EXC-{type(e).__name__}")
                finally:
                    try:
                        ctx.__exit__(None, None, None)
                    except Exception:
                        pass
            stable = len(set(seen)) == 1
            print(f"      {p['theorem_full_name'][:38]:40} {seen}  "
                  f"{'stable' if stable else 'NONDETERMINISTIC — invalidates witness (§7.7)'}")

    # ---- 7. which repair rung exposes it ----------------------------------
    print(f"\n[7] WHICH REPAIR EXPOSES THE DISTINCTION")
    for label, va, vb in [("R2_deep", a.phi_state_deep, b.phi_state_deep),
                          ("R3_proofs", a.phi_deep_proofs, b.phi_deep_proofs),
                          ("R4_struct", a.phi_full, b.phi_full),
                          ("R6_env", a.f_env, b.f_env)]:
        print(f"    {label:12} {'EXPOSES' if va != vb else 'blind'}")

    print(f"\n[8] CLASSIFICATION — auditor judgement required")
    print("    mathematically relevant / operationally relevant / irrelevant?")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
