"""Canonicalization policy (blueprint §7.4, §9).

Pure Python, no LeanDojo — fully unit-testable off-box. Everything that decides *what counts
as the same executable state* lives here, versioned by `SCHEMA_VERSION`, so a collision can be
re-audited from the released serialization rather than trusted as an opaque hash.

Three confounders from blueprint §9 are handled:

* **Alpha-renaming.** Inaccessible/hygienic names (`x✝`, `x✝¹`) are renamed to positional
  slots in order of first appearance, so two structurally identical states do not appear
  distinct merely because the elaborator picked different shadow names.
* **Unstable metavariable IDs.** `?m.4711` is a runtime counter; it is renumbered positionally
  for the same reason. §9 requires canonicalizing "by dependency order and structural content,
  not runtime IDs" — positional order of appearance in the rendered goal is the tractable
  approximation, and it is applied identically to every member of a class.
* **Rendering-only options.** `pp.*` / `format.* `/ `trace.*` cannot change tactic behavior, so
  they are stripped before the options digest. Leaving them in makes a `set_option pp.deepTerms`
  look like an `options_transparency` mechanism.
"""
from __future__ import annotations

import hashlib
import re

SCHEMA_VERSION = "2"

# `x✝`, `x✝¹`, `ω✝²` … — Lean's rendering of inaccessible / shadowed local names.
_INACCESSIBLE = re.compile(r"[^\s()\[\]{},:]*✝[¹²³⁰⁴-⁹]*")
# `?m.4711`, `?foo.12` — metavariables carrying a runtime counter.
_MVAR = re.compile(r"\?[A-Za-z_][A-Za-z0-9_.']*")
# `_uniq.4711` — internal unique names that occasionally survive into pp output.
_UNIQ = re.compile(r"_uniq\.\d+")

_RENDERING_OPTION_PREFIXES = ("pp.", "format.", "trace.")
_OPTION_ENTRY = re.compile(r"\(\s*([^(),]+?)\s*,\s*(.*?)\s*\)(?=\s*,\s*\(|\s*\]$|$)")


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _renumber(text: str, pattern: re.Pattern, fmt: str) -> str:
    """Replace every distinct match with `fmt % index`, indexed by first appearance."""
    mapping: dict[str, str] = {}

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok not in mapping:
            mapping[tok] = fmt % len(mapping)
        return mapping[tok]

    return pattern.sub(repl, text)


def alpha_normalize(pp: str) -> str:
    """Stable alpha-normalization of a rendered proof state (blueprint §7.4, §9).

    Order matters: `_uniq` first (it can appear inside an inaccessible token), then
    inaccessible names, then metavariables.
    """
    if not pp:
        return pp
    out = _renumber(pp, _UNIQ, "_uniq.%d")
    out = _renumber(out, _INACCESSIBLE, "_a%d")
    out = _renumber(out, _MVAR, "?m.%d")
    return out


def parse_options(raw: str) -> dict[str, str]:
    """Parse Lean's `toString (opts : Options)` — `[(k, v), (k, v)]` — into a dict."""
    if not raw or raw == "NA":
        return {}
    return {m.group(1): m.group(2) for m in _OPTION_ENTRY.finditer(raw.strip())}


def behavioral_options(raw: str) -> dict[str, str]:
    """Drop options that only affect rendering; those cannot change tactic behavior."""
    return {
        k: v for k, v in parse_options(raw).items()
        if not any(k.startswith(p) for p in _RENDERING_OPTION_PREFIXES)
    }


def options_digest(raw: str) -> str:
    opts = behavioral_options(raw)
    body = ";".join(f"{k}={opts[k]}" for k in sorted(opts))
    return f"optsN={len(opts)}|optsH={sha(body)[:16]}"


def canonical_env(env_behavior: str, raw_options: str) -> str:
    """The behavior-relevant ambient digest that goes into `F_exec`.

    `env_behavior` arrives from Lean already sorted and free of module identity; the options
    digest is appended here because the rendering-option filter is Python-side policy.
    """
    return f"{env_behavior}|{options_digest(raw_options)}"


def env_fields(env_digest: str) -> dict[str, str]:
    """Split a `k=v|k=v` digest into fields for mechanism attribution."""
    out: dict[str, str] = {}
    for part in env_digest.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out
