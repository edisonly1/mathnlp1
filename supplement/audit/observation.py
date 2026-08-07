"""Component 3 — Observation builder (blueprint §4.1, §7.3).

Constructs the three observation layers for each state:
  φ_state = ppGoal                          (already captured in extract.py)
  φ_RAG   = retrieved premises ⊕ φ_state    (retriever-dependent)
  φ_tok   = truncate_L(tokenize(φ_RAG))     (the ACTUAL model input)

The scientifically primary object is φ_state; φ_RAG and φ_tok establish the downstream
consequence for the real ReProver (blueprint §4.1). φ_tok collisions are the strongest
model-level claim: a deterministic model MUST return the same distribution for identical
token input (blueprint §7.5).

Reproducing the *historical* ReProver retriever byte-for-byte is finicky; we ship a
`StateOnlyRetriever` default (φ_RAG = φ_state, i.e. no retrieval) and a documented interface
for the real retriever. Under StateOnly, φ_tok collisions are a **lower bound** on token-input
aliasing (retrieval can only split classes further, not merge them, unless it injects
identical premises — which is itself measured).
"""
from __future__ import annotations

from typing import Optional, Protocol

from .schema import Observation, StateRecord


class Retriever(Protocol):
    def retrieve(self, phi_state: str, provenance) -> str:
        """Return the premise block to be concatenated before φ_state (exact bytes)."""
        ...


class StateOnlyRetriever:
    """φ_RAG = φ_state. Lower-bound retriever (no premises). Fully deterministic."""

    def retrieve(self, phi_state: str, provenance) -> str:  # noqa: D401
        return ""


class ReProverRetriever:
    """Real ReProver retriever (blueprint §4.1 R_k). Reconstructs the exact premise block.

    TODO(runner): wire the public retriever checkpoint + the benchmark's premise corpus at the
    pinned commit, and reproduce the EXACT concatenation order and formatting the historical
    checkpoint consumed. Until validated against the checkpoint (Week 3 fidelity gate,
    blueprint §12), prefer StateOnly and report φ_tok results as a lower bound.
    """

    def __init__(self, model_name: str, corpus_revision: str, k: int, device: str = "cpu"):
        self.model_name = model_name
        self.corpus_revision = corpus_revision
        self.k = k
        self.device = device
        self._model = None  # lazily loaded

    def retrieve(self, phi_state: str, provenance) -> str:  # pragma: no cover - needs GPU + corpus
        raise NotImplementedError(
            "ReProverRetriever must be wired to the pinned premise corpus + retriever checkpoint. "
            "See blueprint §4.1 (φ_RAG) and §7.3 (record retrieved premises, scores, order, versions).")


class ObservationBuilder:
    def __init__(self, tokenizer_name: str, max_input_length: int,
                 retriever: Optional[Retriever] = None, concat_sep: str = "\n\n"):
        self.max_input_length = max_input_length
        self.retriever = retriever or StateOnlyRetriever()
        self.concat_sep = concat_sep
        self._tok = None
        self._tokenizer_name = tokenizer_name

    @property
    def tok(self):
        if self._tok is None:
            from transformers import AutoTokenizer  # lazy: keep module import-safe off-box
            self._tok = AutoTokenizer.from_pretrained(self._tokenizer_name)
        return self._tok

    def build_rag(self, phi_state: str, provenance) -> str:
        premises = self.retriever.retrieve(phi_state, provenance)
        if not premises:
            return phi_state
        # Exact concatenation order matters and is versioned (blueprint §7.3, §7.5).
        return premises + self.concat_sep + phi_state

    def build_tok(self, phi_rag: str) -> tuple[list[int], bool]:
        """Return (token_ids, truncated?). Deterministic; mirrors ReProver's truncation (§4.1)."""
        # Length WITHOUT truncation, to detect whether truncation dropped content.
        full = self.tok(phi_rag, add_special_tokens=True, truncation=False)["input_ids"]
        trunc = self.tok(phi_rag, add_special_tokens=True, truncation=True,
                         max_length=self.max_input_length)["input_ids"]
        return list(trunc), (len(full) > len(trunc))

    def enrich(self, rec: StateRecord) -> StateRecord:
        """Fill φ_RAG and φ_tok on a StateRecord in place; return it."""
        obs: Observation = rec.observation
        obs.phi_rag = self.build_rag(obs.phi_state, rec.provenance)
        ids, truncated = self.build_tok(obs.phi_rag)
        obs.phi_tok_ids = ids
        obs.truncated = truncated
        return rec
