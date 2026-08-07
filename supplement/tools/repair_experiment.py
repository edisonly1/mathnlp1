"""§7.9 step 5: does a REPAIRED observation change what the model proposes?

Alias regret is 0.0 on every class measured, and Amendment A5 explains why the §4.6 formulation
cannot capture the cost we actually observe: `V_full` maximises over the *same* fixed top-k list
that `V_φ` sees, so it has no way to express "the model would have proposed something else had
the observation distinguished the states". The cost shows up instead as absolute failure on one
member.

This measures the counterfactual directly. For each non-dismissible witness, generate ReProver's
top-k under three observations and replay each on BOTH members:

  A  default    φ_state                      — what the model sees today (identical per member)
  B  repaired   φ_full  (R4 structural)      — pp.all + deepTerms, DIFFERS per member
  C  control    φ_state + inert padding      — same length inflation as B, no information

C is the part that makes this interpretable. ReProver was trained on default `φ_state`, so
condition B is out of distribution: if B degrades, that could be the repair failing OR the model
choking on an unfamiliar format. C perturbs length and token distribution *without* adding
information, so:

    B helps and C does not      -> the repair carries real information
    B and C degrade alike       -> OOD artifact, B tells us nothing
    both help                   -> length/format effect, not the repair

Usage:
    python tools/repair_experiment.py --run runs/census_par --keys bf0dd9b888,eaf1ce5fd9,e08b68012a
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PAD = ("-- inert padding, carries no information about the goal; present only to match the "
       "token-count inflation of the repaired observation. ")


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--keys", required=True, help="comma-separated observation_key prefixes")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dojo-timeout", type=int, default=600)
    args = ap.parse_args()

    from lean_dojo import Dojo, LeanGitRepo, Theorem
    from audit.config import load_config
    from audit.fingerprint import collect_fields, assemble, build_fingerprint
    from audit.model_runner import ReProverGenerator
    from audit.observation import ObservationBuilder, StateOnlyRetriever
    from audit.replay import lean_git_repo
    import lean_dojo as ldj

    cfg = load_config(args.config)
    run = Path(args.run)
    model = ReProverGenerator(cfg)
    ob = ObservationBuilder(cfg.tokenizer, cfg.max_input_length, retriever=StateOnlyRetriever())

    states = [json.loads(l) for l in (run / "states.jsonl").read_text().splitlines()]
    by_hash: dict = {}
    for s in states:
        by_hash.setdefault(_h(s["observation"]["phi_state"]), []).append(s)

    results = []
    for key in args.keys.split(","):
        key = key.strip()
        members = next((v for k, v in by_hash.items() if k.startswith(key) and len(v) >= 2), None)
        if not members:
            print(f"  {key}: no class found", flush=True)
            continue
        phi = members[0]["observation"]["phi_state"]
        print(f"\n{'='*78}\nWITNESS {key}  ({len(members)} members, φ_state {len(phi)}B)\n{'='*78}",
              flush=True)

        # --- per-member repaired observation (this is what the model does NOT get today) ---
        repaired, sessions = {}, {}
        for idx, s in enumerate(members):
            p = s["provenance"]
            thm = Theorem(lean_git_repo(ldj, p["repo_url"], p["repo_commit"]),
                          p["file_path"], p["theorem_full_name"])
            ctx = Dojo(thm, timeout=args.dojo_timeout)
            dojo, st = ctx.__enter__()
            ok = True
            for pre in s["proof_prefix"]:
                st = dojo.run_tac(st, pre)
                if not isinstance(st, ldj.TacticState):
                    ok = False
                    break
            if not ok:
                ctx.__exit__(None, None, None)
                continue
            fp = build_fingerprint(assemble(collect_fields(dojo, st)), "repair", faithful=False)
            repaired[idx] = fp.phi_full
            sessions[idx] = (ctx, dojo, st)
            print(f"  member {idx}: φ_full {len(fp.phi_full)}B  f_core={fp.f_core[:10]}", flush=True)

        if len(sessions) < 2:
            for ctx, _, _ in sessions.values():
                ctx.__exit__(None, None, None)
            print("  fewer than 2 members restorable; skipping")
            continue

        # Pad the control to the median repaired length so B and C inflate alike.
        target = sorted(len(v) for v in repaired.values())[len(repaired) // 2]
        pad = PAD * max(1, (target - len(phi)) // len(PAD) + 1)
        conditions = {"A_default": {i: phi for i in sessions},
                      "B_repaired": {i: repaired[i] for i in sessions},
                      "C_control": {i: phi + "\n" + pad[:max(0, target - len(phi))]
                                    for i in sessions}}

        try:
            for cond, obs_per_member in conditions.items():
                # One generation per DISTINCT observation. Under A and C every member shares one
                # observation, so the model necessarily emits one list; under B they differ.
                gen: dict = {}
                for i, obs in obs_per_member.items():
                    k = _h(obs)
                    if k not in gen:
                        ids, trunc = ob.build_tok(ob.build_rag(obs, members[i]["provenance"]))
                        gen[k] = (model.top_k_tactics(list(ids)), trunc)
                distinct = len(gen)

                valid: dict = {}
                for i, (ctx, dojo, st) in sessions.items():
                    tactics, _ = gen[_h(obs_per_member[i])]
                    ok = []
                    for t in tactics[:max(cfg.top_k)]:
                        try:
                            r = dojo.run_tac(st, t)
                            if isinstance(r, ldj.ProofFinished) or isinstance(r, ldj.TacticState):
                                ok.append(t)
                        except Exception:
                            pass
                    valid[i] = ok

                solved = sum(1 for v in valid.values() if v)
                print(f"\n  [{cond}] {distinct} distinct model input(s), "
                      f"{len(obs_per_member)} members", flush=True)
                for i, v in valid.items():
                    obs_len = len(obs_per_member[i])
                    print(f"      member {i} (obs {obs_len:5}B): {len(v)} valid  {v[:2]}", flush=True)
                print(f"      members with >=1 working tactic: {solved}/{len(valid)}")
                results.append({"witness": key, "condition": cond,
                                "distinct_inputs": distinct, "members_solved": solved,
                                "n_members": len(valid),
                                "valid": {str(i): v for i, v in valid.items()}})
        finally:
            for ctx, _, _ in sessions.values():
                try:
                    ctx.__exit__(None, None, None)
                except Exception:
                    pass

    out = run / "repair_experiment.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n{'='*78}\nwrote {out}")
    print("READING: B helps and C does not -> repair carries information.")
    print("         B and C alike          -> out-of-distribution artifact, B is uninformative.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
