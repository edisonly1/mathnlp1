"""Load and validate the version-locked run configuration (blueprint §7.1, §11.3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]

    # --- convenience typed accessors ---
    @property
    def repo_url(self) -> str: return self.raw["corpus"]["repo_url"]
    @property
    def repo_commit(self) -> str: return self.raw["corpus"]["repo_commit"]
    @property
    def lean_toolchain(self) -> str: return self.raw["corpus"]["lean_toolchain"]
    @property
    def gate0_toolchain(self) -> str:
        """Toolchain for standalone Gate-0 witnesses, distinct from corpus provenance."""
        return self.raw.get("regression", {}).get("lean_toolchain", self.lean_toolchain)
    @property
    def cache_dir(self) -> str: return self.raw["corpus"]["cache_dir"]

    @property
    def n_states(self) -> int: return int(self.raw["sampling"]["n_states"])
    @property
    def seed(self) -> int: return int(self.raw["sampling"]["seed"])
    @property
    def stratify_by(self) -> list[str]: return list(self.raw["sampling"]["stratify_by"])

    @property
    def pp_all(self) -> bool: return bool(self.raw["fingerprint"]["pp_all"])
    @property
    def fingerprint_channel(self) -> str: return self.raw["fingerprint"]["channel"]

    @property
    def tokenizer(self) -> str: return self.raw["observation"]["tokenizer"]
    @property
    def max_input_length(self) -> int: return int(self.raw["observation"]["max_input_length"])
    @property
    def retriever(self) -> str: return self.raw["observation"]["retriever"]
    @property
    def group_on(self) -> list[str]: return list(self.raw["observation"]["group_on"])

    @property
    def core_panel(self) -> list[str]: return list(self.raw["replay"]["core_panel"])
    @property
    def structural_panel(self) -> list[str]: return list(self.raw["replay"]["structural_panel"])
    @property
    def repeats(self) -> int: return int(self.raw["replay"]["repeats"])
    @property
    def hard_timeout_s(self) -> int: return int(self.raw["replay"]["hard_timeout_s"])
    @property
    def goal_order_sensitive(self) -> bool: return bool(self.raw["replay"]["goal_order_sensitive"])
    @property
    def reverify_fraction(self) -> float: return float(self.raw["replay"]["reverify_nondivergent_fraction"])
    @property
    def session_open_timeout_s(self) -> int:
        return int(self.raw["replay"].get("session_open_timeout_s", 240))
    @property
    def tactic_watchdog_s(self) -> int:
        return int(self.raw["replay"].get("tactic_watchdog_s", 90))
    @property
    def replay_isolation(self) -> str:
        """'session' reuses one Dojo per theorem (feasible at pilot scale); 'process' opens a
        fresh Dojo per repeat (blueprint §7.7 strict reading; use for the manual audit)."""
        return self.raw["replay"].get("isolation", "session")

    @property
    def tacgen(self) -> str: return self.raw["model"]["tacgen"]
    @property
    def top_k(self) -> list[int]: return list(self.raw["model"]["top_k"])
    @property
    def num_beams(self) -> int: return int(self.raw["model"]["num_beams"])
    @property
    def device(self) -> str: return self.raw["model"]["device"]

    @property
    def repair_rungs(self) -> list[str]: return list(self.raw["repair_ladder"]["rungs"])
    @property
    def output_dir(self) -> str: return self.raw["output_dir"]

    def validate(self) -> list[str]:
        """Return a list of blocking problems (empty ⇒ ok)."""
        problems: list[str] = []
        if "REPLACE" in self.repo_commit:
            problems.append(
                "corpus.repo_commit is a placeholder — pin the LeanDojo Benchmark 4 commit "
                "so traced states match the public ReProver checkpoint distribution.")
        if self.max_input_length <= 0:
            problems.append("observation.max_input_length must be positive.")
        if self.repeats < 2:
            problems.append("replay.repeats must be ≥2 (blueprint §7.7).")
        return problems


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw)
