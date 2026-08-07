"""BFS-Prover-V2 tactic generator on Apple-silicon MPS.

The model card documents the interface exactly: input is `"{state}:::"` where {state} is a
Lean 4 tactic state, and the model was trained on Mathlib traced via LeanDojo, so our `pp`
renderings are in-distribution. Deterministic beam search is used, matching the protocol of
the other two policies (candidate identity across alias members is a control, so sampling is
not an option).
"""
from __future__ import annotations

import os


class BFSProverGenerator:
    def __init__(self, model_id: str = "ByteDance-Seed/BFS-Prover-V2-7B",
                 max_new_tokens: int = 64, device: str = "auto",
                 revision: str | None = None, local_files_only: bool = False):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        load_args = {"local_files_only": local_files_only}
        if revision:
            load_args["revision"] = revision
        self.tok = AutoTokenizer.from_pretrained(model_id, **load_args)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, **load_args).to(device).eval()
        self.max_new_tokens = max_new_tokens

    def top_k_scored(self, state_pp: str, k: int = 8):
        """Deterministic beam search; returns [(tactic, sequence logprob-score)]."""
        prompt = state_pp + ":::"
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **enc,
                num_beams=k, num_return_sequences=k, do_sample=False,
                max_new_tokens=self.max_new_tokens,
                early_stopping=True,
                output_scores=True, return_dict_in_generate=True,
                pad_token_id=self.tok.eos_token_id)
        n = enc["input_ids"].shape[1]
        seen, cands = set(), []
        for seq, score in zip(out.sequences, out.sequences_scores):
            tac = self.tok.decode(seq[n:], skip_special_tokens=True).strip()
            # The card says the model may echo the state; keep only the tactic line(s)
            # after any ::: it re-emits.
            if ":::" in tac:
                tac = tac.split(":::")[-1].strip()
            if tac and tac not in seen:
                seen.add(tac)
                cands.append((tac, float(score)))
        return cands

    def score_actions(self, items: list[tuple[str, str]], batch_size: int = 4):
        """Score tactic strings conditionally on already formatted state prompts.

        Each state is supplied without the model-card ``:::`` separator.  The return value
        contains the sum and mean token log probabilities of the tactic only.  Prompt tokens
        therefore cannot contribute directly to a paired member comparison.
        """
        results = [None] * len(items)
        pad_id = self.tok.pad_token_id
        if pad_id is None:
            pad_id = self.tok.eos_token_id
        encoded = []
        for original_index, (state_pp, action) in enumerate(items):
            prompt_ids = self.tok(
                state_pp + ":::", add_special_tokens=True)["input_ids"]
            action_ids = self.tok(action, add_special_tokens=False)["input_ids"]
            if not action_ids:
                raise ValueError("cannot score an empty tactic")
            encoded.append((original_index, prompt_ids, action_ids))
        # Length bucketing avoids padding a short proof state to the longest state in an
        # arbitrary corpus-order batch.  This materially reduces attention work on MPS.
        encoded.sort(key=lambda value: len(value[1]) + len(value[2]))
        for start in range(0, len(encoded), batch_size):
            batch = encoded[start:start + batch_size]
            width = max(len(prompt) + len(action) for _, prompt, action in batch)
            input_ids = self.torch.full(
                (len(batch), width), pad_id, dtype=self.torch.long,
                device=self.device)
            attention_mask = self.torch.zeros(
                (len(batch), width), dtype=self.torch.long, device=self.device)
            for index, (_, prompt, action) in enumerate(batch):
                values = self.torch.tensor(prompt + action, dtype=self.torch.long,
                                           device=self.device)
                input_ids[index, :len(values)] = values
                attention_mask[index, :len(values)] = 1
            with self.torch.no_grad():
                logits = self.model(
                    input_ids=input_ids, attention_mask=attention_mask).logits
            for index, (original_index, prompt, action) in enumerate(batch):
                positions = self.torch.arange(
                    len(prompt) - 1, len(prompt) + len(action) - 1,
                    device=self.device)
                action_logits = logits[index, positions, :].float()
                targets = input_ids[index, positions + 1]
                log_probs = self.torch.log_softmax(action_logits, dim=-1)
                token_scores = log_probs.gather(1, targets[:, None]).squeeze(1)
                total = float(token_scores.sum().cpu())
                results[original_index] = {
                    "sum_logprob": total,
                    "mean_logprob": total / len(action),
                    "action_tokens": len(action),
                    "prompt_tokens": len(prompt),
                }
            del logits
        return results
