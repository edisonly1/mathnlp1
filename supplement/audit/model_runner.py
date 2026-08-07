"""Component 6 — Model runner (blueprint §7.9, §11.1).

Runs the public ReProver tactic generator once per unique token input and caches top-k tactics
(blueprint §11.4: "cache model outputs by exact token sequence so identical inputs are evaluated
once"). Deterministic beam search means identical token input ⇒ identical output — so any
difference in downstream tactic validity across alias-class members is attributable to the hidden
state, not stochastic model variation (blueprint §7.9).

Validity of a generated tactic on a specific class member is NOT decided here; it is decided by
replaying the generated tactic on that member (Component 5). This module only produces the
candidate tactics.

Torch/transformers are imported lazily so the module stays importable off-box.
"""
from __future__ import annotations

from typing import Optional

from .config import Config


class ReProverGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_name = cfg.tacgen
        self.device = cfg.device
        self.num_beams = cfg.num_beams
        self.max_k = max(cfg.top_k)
        self._model = None
        self._tok = None
        self._cache: dict[tuple[int, ...], list[str]] = {}

    def _load(self):
        if self._model is None:
            import torch  # noqa
            from transformers import AutoTokenizer, T5ForConditionalGeneration
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            if self.device.startswith("cuda"):
                try:
                    self._model = self._model.to(self.device)
                except Exception:
                    self.device = "cpu"
            self._model.eval()

    def top_k_tactics(self, phi_tok_ids: list[int]) -> list[str]:
        """Deterministic top-k tactics for a token-id input. Cached by exact token sequence."""
        key = tuple(phi_tok_ids)
        if key in self._cache:
            return self._cache[key]
        self._load()
        import torch
        input_ids = torch.tensor([phi_tok_ids], dtype=torch.long)
        if self.device.startswith("cuda"):
            input_ids = input_ids.to(self.device)
        with torch.no_grad():
            out = self._model.generate(
                input_ids=input_ids,
                num_beams=self.num_beams,
                num_return_sequences=self.max_k,
                do_sample=False,               # deterministic (blueprint §7.9, model.do_sample=false)
                max_length=256,
                early_stopping=True,
            )
        tactics = [self._tok.decode(o, skip_special_tokens=True).strip() for o in out]
        # De-dup while preserving beam order.
        seen, uniq = set(), []
        for t in tactics:
            if t not in seen:
                seen.add(t); uniq.append(t)
        self._cache[key] = uniq
        return uniq

    def top_k_scored(self, phi_tok_ids: list[int], k: Optional[int] = None,
                     max_length: int = 128) -> list[tuple[str, float]]:
        """Top-k tactics WITH beam log-probabilities, for best-first search.

        `top_k_tactics` discards scores, which is fine for the §7.9 validity check but not for a
        search: best-first needs to order the frontier by cumulative log-probability the way
        ReProver does. Beam search returns length-normalised sequence scores, so the value here
        is directly usable as an additive node score.

        `max_length` is 128 rather than 256 because this is a byte-level tokenizer and tactics
        are short; halving it roughly halves CPU generation time, which dominates the search.
        """
        key = ("scored", tuple(phi_tok_ids), k, max_length)
        if key in self._cache:
            return self._cache[key]
        self._load()
        import torch
        n = k or self.max_k
        input_ids = torch.tensor([phi_tok_ids], dtype=torch.long)
        if self.device.startswith("cuda"):
            input_ids = input_ids.to(self.device)
        with torch.no_grad():
            out = self._model.generate(
                input_ids=input_ids,
                num_beams=max(n, self.num_beams),
                num_return_sequences=n,
                do_sample=False,
                max_length=max_length,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        texts = [self._tok.decode(o, skip_special_tokens=True).strip()
                 for o in out.sequences]
        scores = ([float(x) for x in out.sequences_scores]
                  if getattr(out, "sequences_scores", None) is not None
                  else [0.0] * len(texts))
        seen, uniq = set(), []
        for t, sc in zip(texts, scores):
            if t and t not in seen:
                seen.add(t)
                uniq.append((t, sc))
        self._cache[key] = uniq
        return uniq

    def cache_stats(self) -> dict:
        return {"unique_token_inputs_scored": len(self._cache)}
