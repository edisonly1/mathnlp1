"""Phase 2 of the second-policy evaluation: BFS-Prover top-k generation.

The second policy is the miniCTX state-tactic model (deepseek-coder-1.3b base, decoder-only,
2024), architecturally and generationally distinct from ReProver (ByT5-small encoder-decoder,
2023) with different training data. Its documented prompt is the [STATE]/[TAC] instruction
template from the model card. (BFS-Prover, Qwen2.5-Math-7B, was the first choice; its 15GB
weights did not download on this connection.) Decoding matches the ReProver protocol exactly: deterministic beam search, 8 beams, 8
return sequences, so one ranked candidate list per shared observation.

Runs with no Lean session open: a 7B model in fp16 and Lean cannot share this machine's memory.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--renderings", default="runs/policy2/renderings.jsonl")
    ap.add_argument("--out", default="runs/policy2/candidates.jsonl")
    ap.add_argument("--model", default="l3lab/ntp-mathlib-st-deepseek-coder-1.3b")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[gen] loading {args.model} on {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(dev)
    model.eval()

    rows = [json.loads(l) for l in open(args.renderings) if l.strip()]
    done = set()
    outp = Path(args.out)
    if outp.exists():
        for l in open(outp):
            if l.strip():
                d = json.loads(l)
                done.add((d["theorem"], d["pp_hash"]))
    fh = open(outp, "a")
    for i, r in enumerate(rows, 1):
        if (r["theorem"], r["pp_hash"]) in done:
            continue
        if "BFS-Prover" in args.model:
            prompt = r["pp"] + ":::"
        else:
            # miniCTX state-tactic template, verbatim from the model card
            prompt = ("/- You are proving a theorem in Lean 4.\n"
                      "You are given the following information:\n"
                      "- The current proof state, inside [STATE]...[/STATE]\n\n"
                      "Your task is to generate the next tactic in the proof.\n"
                      "Put the next tactic inside [TAC]...[/TAC]\n-/\n"
                      "[STATE]\n" + r["pp"] + "\n[/STATE]\n[TAC]\n")
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                ids, num_beams=args.top_k, num_return_sequences=args.top_k,
                do_sample=False, max_new_tokens=args.max_new,
                output_scores=True, return_dict_in_generate=True,
                pad_token_id=tok.eos_token_id)
        cands, seen = [], set()
        scores = ([float(x) for x in out.sequences_scores]
                  if out.sequences_scores is not None else [0.0]*args.top_k)
        for seq, sc in zip(out.sequences, scores):
            text = tok.decode(seq[ids.shape[1]:], skip_special_tokens=True)
            tac = text.split("[/TAC]")[0].strip() if "[/TAC]" in text \
                  else text.split("\n")[0].strip()
            if tac and tac not in seen:
                seen.add(tac)
                cands.append({"tactic": tac, "score": sc})
        fh.write(json.dumps({"theorem": r["theorem"], "pp_hash": r["pp_hash"],
                             "file": r["file"], "tactic_index": r["tactic_index"],
                             "histories": r["histories"],
                             "candidates": cands}, ensure_ascii=False) + "\n")
        fh.flush()
        print(f"[gen] [{i}/{len(rows)}] {time.time()-t0:5.1f}s "
              f"{len(cands)} candidates  {r['theorem'][:40]}", flush=True)
    fh.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
