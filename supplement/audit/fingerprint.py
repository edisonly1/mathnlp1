"""Component 2 — Canonicalizer / execution fingerprint (blueprint §7.4).

Turns a live Dojo proof state into a `Fingerprint` by injecting a metaprogram that renders the
goal at several option levels and digests the ambient environment. The rendering ladder and the
two `pp` invariants that make it sound are documented in `lean_fingerprint/Fingerprint.lean`;
the canonicalization policy (α-normalization, rendering-option filtering) lives in `canon.py`.

Extraction channels (config `fingerprint.channel`):
  * `throwError` (default): inject a self-contained metaprogram via `Dojo.run_tac` and read the
    JSON back from the deliberately-thrown error. No repo modification. The inline program only
    references core `Lean.*` names (available in every Mathlib file); every field is try-wrapped
    so it always yields at least `phi_full`.
  * `frontend`: requires `lean_fingerprint/Fingerprint.lean` on the import path; injects the
    single tactic `fingerprint_throw`. Identical output, but maintained as real source.

If extraction fails entirely we fall back to a degraded fingerprint keyed on `φ_state` alone
(marked `faithful=False`, `degraded=True`) so the pipeline never crashes — but such members
cannot establish latency and are counted separately in the analysis.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from .schema import Fingerprint

# Metaprograms injected for channel='throwError'. MUST stay in sync with
# lean_fingerprint/Fingerprint.lean — guarded by tests in test_fingerprint_attribution.py.
#
# ---------------------------------------------------------------------------------------
# HARD CONSTRAINT: the submitted tactic string must stay under ~1 KB.
#
# `Dojo.run_tac` in lean-dojo 2.0.2 deadlocks on tactic strings of ~1000 bytes or more. Measured
# by padding ONE known-good program with a trailing Lean comment and changing nothing else
# (lean4-example, 35-byte goal):
#
#     400 / 800 / 950 chars ....... 0.6s   ok
#     1000 / 1024 / 1500 / 2000 ... DojoTacticTimeoutError
#
# The cliff sits between 950 and 1000 bytes and is independent of what the program does — it is
# a pipe-buffer limit in the pexpect-based REPL protocol, not a cost of the program itself.
#
# This is why the fingerprint is collected as SEVERAL SMALL PROBES rather than one program.
# Everything else about the payload is fine: a 64 KB response round-trips untruncated (in 18s),
# so the constraint is on what we SEND, not what comes back.
#
# (An earlier reading of the same symptom blamed `Lean.Json.mkObj`/`compress` for being slow on
# unicode-heavy goal text. That was wrong: the JSON variants happened to be ~1050 bytes and the
# delimiter variants ~950. JSON is avoided anyway — it costs bytes we cannot spare — but it was
# never the cause.)
#
# Two `pp` invariants that are also easy to get wrong (both were wrong in v1):
#   * `pp.all` does NOT imply `pp.deepTerms` — without it the fingerprint inherits the ambient
#     elision and is blind to overflow aliasing (blueprint §5.3 mechanism 2).
#   * `pp.all` switches `pp.proofs` ON — it must be switched back off explicitly, or the
#     shallow-vs-full pair differs in both elision and proof visibility and the mechanism
#     discriminator in `analysis.attribute_mechanisms` becomes unsound.
# ---------------------------------------------------------------------------------------

#: U+001F UNIT SEPARATOR — cannot occur in Lean pretty-printer output.
FIELD_SEP = "\x1f"

#: Refuse to submit anything at or above this; see the length-cliff measurement above. The
#: measured cliff is at 1000 bytes, so this leaves a margin.
MAX_TACTIC_BYTES = 950

#: Logical fields assembled into a Fingerprint.
MAIN_FIELDS = ("phi_deep", "phi_deep_proofs", "phi_all_shallow", "phi_full",
               "phi_full_proofs", "env_behavior", "env_provenance", "raw_options")


def _probe(body: str) -> str:
    """Wrap a body that binds `r` into a minimal `run_tac` that throws `r` back."""
    return ("run_tac (show Lean.Elab.Tactic.TacticM Unit from do\n"
            "  let g ← Lean.Elab.Tactic.getMainGoal\n"
            f"{body}\n"
            '  throwError s!"FPRINT_V2:{r}")')


def _render_probe(opts: str) -> str:
    return _probe(f"  let r := toString (← Lean.withOptions (fun o => {opts}) (Lean.Meta.ppGoal g))")


#: One small probe per field. Each is submitted independently, so a failure costs one field
#: rather than the whole state, and each stays well below MAX_TACTIC_BYTES.
PROBES: dict = {
    "phi_deep": _render_probe("(o.setBool `pp.deepTerms true).setBool `pp.proofs false"),
    "phi_deep_proofs": _render_probe("(o.setBool `pp.deepTerms true).setBool `pp.proofs true"),
    "phi_all_shallow": _render_probe(
        "((o.setBool `pp.all true).setBool `pp.notation false).setBool `pp.proofs false"),
    "phi_full": _render_probe(
        "(((o.setBool `pp.all true).setBool `pp.notation false).setBool `pp.deepTerms true)"
        ".setBool `pp.proofs false"),
    "phi_full_proofs": _render_probe(
        "(((o.setBool `pp.all true).setBool `pp.notation false).setBool `pp.deepTerms true)"
        ".setBool `pp.proofs true"),
    "ns": _probe('  let r ← (try do pure s!"ns={← Lean.getCurrNamespace}" catch _ => pure "ns=NA")'),
    "open": _probe(
        "  let r ← (try do\n"
        "     let ds ← Lean.getOpenDecls\n"
        '     pure s!"open={((ds.map toString).toArray.qsort (· < ·)).toList}"\n'
        '   catch _ => pure "open=NA")'),
    "linst": _probe(
        "  let r ← (try do\n"
        "     let insts ← Lean.Meta.getLocalInstances\n"
        "     let ss ← insts.toList.mapM (fun li => do\n"
        "       let t ← Lean.instantiateMVars (← Lean.Meta.inferType li.fvar)\n"
        '       return s!"{li.className}:{toString (← Lean.Meta.ppExpr t)}")\n'
        '     pure s!"linst={((ss.toArray.qsort (· < ·)).toList)}"\n'
        '   catch _ => pure "linst=NA")'),
    # Declared-constant count: distinguishes two points in one file, which every other
    # env field is blind to. See §8.3 audit of class 8c33dbf195.
    "nconsts": _probe(
        '  let r \u2190 (try do let e \u2190 Lean.getEnv; let c := e.constants.map₂.foldl (fun n _ _ => n + 1) 0; pure s!"nconsts={c}" '
        'catch _ => pure "nconsts=NA")'),
    "env_provenance": _probe(
        '  let r ← (try do let e ← Lean.getEnv; pure s!"mod={e.mainModule}" '
        'catch _ => pure "mod=NA")'),
    "raw_options": _probe(
        '  let r ← (try do let o ← Lean.getOptions; pure (toString o) catch _ => pure "NA")'),
    # Separate because `getSimpTheorems` can raise Lean's `Exception.internal`, which a
    # `try ... catch _` in TacticM does not reliably intercept.
    "simp": _probe(
        "  let s ← Lean.Meta.getSimpTheorems\n"
        "  let ns := ((s.lemmaNames.toList.map (fun o => toString o.key)).toArray.qsort (· < ·))\n"
        '  let r := s!"simpN={ns.size}|simpH={hash ns.toList}"'),
}

#: Values concatenated, in this order, into `env_behavior`.
ENV_FIELDS = ("ns", "open", "linst", "simp", "nconsts")

TACTIC_SRC_INJECTED = "fingerprint_throw"
TACTIC_SRC_INJECTED_SIMP = "fingerprint_simp_throw"

_MARKER = "FPRINT_V2:"



def _error_message(result: Any) -> Optional[str]:
    """Pull the human-readable error text out of a LeanDojo run_tac result."""
    for attr in ("error", "message"):
        v = getattr(result, attr, None)
        if isinstance(v, str) and v:
            return v
    # LeanError often stringifies usefully
    s = str(result)
    return s or None


def _payload(msg: str) -> Optional[str]:
    """Return the single-line payload after the marker, or None if absent."""
    if not msg:
        return None
    i = msg.find(_MARKER)
    if i < 0:
        return None
    body = msg[i + len(_MARKER):]
    # Renderings are multi-line, but everything after the payload is unrelated compiler output.
    # The Lean side emits exactly one payload per probe, so cut at the trailing marker-free tail
    # only when the probe is known to be single-line (handled by callers via `multiline`).
    return body


#: A Lean diagnostic header, e.g. `/tmp/W.lean:39:2: error: `. Rendered proof states are
#: multi-line, so a multi-line payload has to stop at the next diagnostic rather than at the
#: next newline — otherwise it absorbs file paths and line numbers, which differ per state and
#: make every field compare unequal.
_DIAG_LINE = re.compile(r"^\s*\S*\.lean:\d+:\d+:\s+(error|warning|info)\b")


def parse_probe(msg: str, multiline: bool) -> Optional[str]:
    """Extract one probe's value.

    Single-line probes are cut at the first newline; multi-line probes are cut at the first
    following Lean diagnostic header.
    """
    body = _payload(msg)
    if body is None:
        return None
    lines = body.splitlines()
    if not lines:
        return ""
    if not multiline:
        return lines[0].strip()
    kept = []
    for line in lines:
        if _DIAG_LINE.match(line):
            break
        kept.append(line)
    return "\n".join(kept).rstrip()


#: Probes whose value legitimately spans multiple lines (rendered proof states).
_MULTILINE = {"phi_deep", "phi_deep_proofs", "phi_all_shallow", "phi_full", "phi_full_proofs"}


def parse_main(msg: str) -> Optional[dict]:
    """Back-compat: parse a single unit-separated payload carrying every MAIN_FIELD."""
    body = _payload(msg)
    if body is None:
        return None
    parts = body.split(FIELD_SEP)
    if len(parts) < len(MAIN_FIELDS):
        return None
    return dict(zip(MAIN_FIELDS, parts[:len(MAIN_FIELDS)]))


def _extract_json(msg: str) -> Optional[dict]:
    """Deprecated alias kept so older diagnostics in tools/ keep working."""
    return parse_main(msg)


def build_fingerprint(data: dict, channel: str, faithful: bool) -> Fingerprint:
    """Assemble a Fingerprint from collected fields. Pure — unit-tested off-box."""
    return Fingerprint.from_parts(
        phi_full=data["phi_full"],
        env_digest=data.get("env_behavior", ""),
        channel=channel,
        faithful=faithful,
        phi_state_deep=data.get("phi_deep", ""),
        phi_deep_proofs=data.get("phi_deep_proofs", ""),
        phi_all_shallow=data.get("phi_all_shallow", ""),
        phi_full_proofs=data.get("phi_full_proofs", ""),
        env_provenance=data.get("env_provenance", ""),
        raw_options=data.get("raw_options", ""),
    )


def degraded_fingerprint(phi_state: str, channel: str) -> Fingerprint:
    """Fallback when extraction fails. Keyed on φ_state alone, so it can never *create*
    apparent latency — two degraded members of one class collapse to a single f_exec."""
    return Fingerprint.from_parts(
        phi_full="DEGRADED:" + phi_state,
        env_digest="NA",
        channel=channel + "+degraded",
        faithful=False,
    )


def is_degraded(fp: Optional[Fingerprint]) -> bool:
    return fp is None or fp.channel.endswith("+degraded")


def _run(dojo: Any, state: Any, src: str) -> Optional[str]:
    """Submit one probe and return the message it threw back (None on hard failure)."""
    if len(src.encode("utf-8")) >= MAX_TACTIC_BYTES:
        # Guard rather than deadlock: run_tac wedges the whole session past ~1 KB, which would
        # turn one oversized probe into a cascade of timeouts across every later state.
        raise ValueError(f"probe is {len(src.encode())}B, at or above the "
                         f"{MAX_TACTIC_BYTES}B run_tac limit")
    try:
        return _error_message(dojo.run_tac(state, src))
    except Exception as e:
        return str(e)


def collect_fields(dojo: Any, state: Any, probes: Optional[dict] = None) -> dict:
    """Run every probe against `state` and return the raw field values that succeeded.

    Probes are independent: a field that fails is simply absent, so one unavailable extension
    (e.g. `getSimpTheorems` in a file that never imports the simp extension) costs that field
    rather than the whole fingerprint.
    """
    probes = probes if probes is not None else PROBES
    out: dict = {}
    for name, src in probes.items():
        val = parse_probe(_run(dojo, state, src) or "", multiline=name in _MULTILINE)
        if val is not None:
            out[name] = val
    return out


def assemble(fields: dict) -> dict:
    """Fold per-probe values into the MAIN_FIELDS shape `build_fingerprint` expects."""
    env = "|".join(fields.get(k) or f"{k}=NA" for k in ENV_FIELDS)
    data = {k: fields.get(k, "") for k in MAIN_FIELDS}
    data["env_behavior"] = env
    return data


def fingerprint_state(dojo: Any, state: Any, channel: str, phi_state: str,
                      with_simp: bool = True) -> Fingerprint:
    """Compute the execution fingerprint of `state` inside an open `dojo`.

    Collected as ~11 small probes rather than one program: `Dojo.run_tac` deadlocks at ~1 KB
    of tactic text (see the measurement at the top of this module), and independent probes also
    mean a single unavailable field cannot cost the whole state.

    `dojo`/`state` are LeanDojo objects. This is import-safe off-box because the LeanDojo types
    are only touched here (never imported at module load).
    """
    if channel == "frontend":
        data = parse_main(_run(dojo, state, TACTIC_SRC_INJECTED) or "")
        if data and data.get("phi_full"):
            return build_fingerprint(data, channel, faithful=True)
        return degraded_fingerprint(phi_state, channel)

    probes = dict(PROBES)
    if not with_simp:
        probes.pop("simp", None)
    fields = collect_fields(dojo, state, probes)
    if not fields.get("phi_full"):
        return degraded_fingerprint(phi_state, channel)
    return build_fingerprint(assemble(fields), channel, faithful=False)
