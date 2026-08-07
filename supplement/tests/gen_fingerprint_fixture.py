"""Regenerate `fixtures/gate0_fingerprints.json` from a real Lean run.

Run manually (needs Lean on PATH); the committed fixture is what the test suite consumes:

    python tests/gen_fingerprint_fixture.py

This deliberately drives the *exact* metaprogram the harness injects at runtime
(`audit.fingerprint.TACTIC_SRC_INLINE`), so the fixture cannot drift from the tactic that
actually runs against LeanDojo. The tactic ends in `throwError`, so each probe surfaces as a
Lean error message — the same channel `fingerprint_state` reads — and is parsed back with the
same `_extract_json`.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit.fingerprint import PROBES, assemble, parse_probe, _MULTILINE  # noqa: E402

TOOLCHAIN = "leanprover/lean4:v4.32.0"

PREAMBLE = r"""import Lean
open Lean Elab Tactic Meta

-- Every probe reports by throwing, so the file deliberately produces hundreds of errors.
-- Lean stops elaborating at 100 by default, which silently truncates the fixture.
set_option maxErrors 100000

opaque f : Nat → Nat
axiom f_zero : f 0 = 0

structure Box where n : Nat
def x : Box := ⟨1⟩
def y : Box := ⟨1⟩
-- Two Add instances as NAMED constants. A `local instance` inside a per-witness file would be
-- the only one there and would render as `instAddBox` in both, making the pair indistinguishable
-- at pp.all: the fingerprint separates instances by name, not by value.
def addBoxSum : Add Box := ⟨fun p q => ⟨p.n + q.n⟩⟩
def addBoxZero : Add Box := ⟨fun _ _ => ⟨0⟩⟩

axiom a : Nat
axiom b : Nat
axiom c : Nat
namespace N1
  axiom lem : a = b
end N1
namespace N2
  axiom lem : a = c
end N2

class Tag where val : Nat
def tval [Tag] : Nat := Tag.val

axiom Q : Prop
axiom qf : Q → Q → Q
axiom qa : Q
axiom qb : Q
structure Wrap where
  n : Nat
  h : Q
"""

# (label, prelude lines, goal statement) — order matters: errors surface in source order.
WITNESSES = [
    ("A_with_simp",    ["section A", "attribute [local simp] f_zero"], "f 0 = 0", ["end A"]),
    ("A_without_simp", [], "f 0 = 0", []),
    # Instance supplied explicitly in the goal term: default pp hides instance-implicit
    # arguments, so both display as `(x + y).n = 2` while pp.all separates them.
    ("B_normal",       [], "(@HAdd.hAdd Box Box Box (@instHAdd Box addBoxSum) x y).n = 2", []),
    ("B_degenerate",   [], "(@HAdd.hAdd Box Box Box (@instHAdd Box addBoxZero) x y).n = 2", []),
    ("C_scope_N1",     ["section S1", "open N1"], "a = b", ["end S1"]),
    ("C_scope_N2",     ["section S2", "open N2"], "a = b", ["end S2"]),
    ("D_over_true",    ["set_option pp.deepTerms false in",
                        "set_option pp.deepTerms.threshold 2 in"],
                       "id (id (id (id (5 : Nat)))) = 5", []),
    ("D_over_false",   ["set_option pp.deepTerms false in",
                        "set_option pp.deepTerms.threshold 2 in"],
                       "id (id (id (id (7 : Nat)))) = 5", []),
    ("E_ingoal_2",     [], "@tval (Tag.mk 2) = 2", []),
    ("E_ingoal_0",     [], "@tval (Tag.mk 0) = 2", []),
    ("CTRL_proof_aa",  [], "(Wrap.mk 1 (qf qa qa)).n = 1", []),
    ("CTRL_proof_ab",  [], "(Wrap.mk 1 (qf qa qb)).n = 1", []),
]


SENTINEL = "__P__"


def _sentinel(name: str) -> str:
    return ('example : True := by\n'
            f'  run_tac (show Lean.Elab.Tactic.TacticM Unit from '
            f'throwError "FPRINT_V2:{SENTINEL}{name}")')


def build_witness_source(label: str, pre: list, goal: str, post: list) -> str:
    """A standalone Lean file for ONE witness: sentinel + probe, per probe.

    Probes are taken verbatim from `audit.fingerprint.PROBES`, so the fixture cannot drift from
    what the harness submits. The sentinel makes parsing explicit rather than positional.
    """
    parts = [PREAMBLE]
    parts.extend(pre)
    for name, src in PROBES.items():
        parts.append(_sentinel(f"{label}|{name}"))
        # `set_option ... in` binds only the next declaration, and the sentinel sits in
        # between, so the option lines are re-emitted before each probe.
        parts.extend(l for l in pre if l.startswith("set_option"))
        parts.append(f"example : {goal} := by")
        parts.extend("  " + l for l in src.splitlines())
    parts.extend(post)
    return "\n".join(parts) + "\n"


def parse_witness(label: str, stream: str) -> dict:
    """Sentinel-driven parse: a payload naming a field is followed by that field's value."""
    fields: dict = {}
    pending = None
    for chunk in stream.split("FPRINT_V2:")[1:]:
        head = chunk.splitlines()[0].strip() if chunk.splitlines() else ""
        if head.startswith(SENTINEL):
            tag = head[len(SENTINEL):]
            pending = tag.split("|", 1)[1] if "|" in tag else None
            continue
        if pending is None:
            continue
        val = parse_probe("FPRINT_V2:" + chunk, multiline=pending in _MULTILINE)
        if val is not None:
            fields[pending] = val
        pending = None
    return fields


def main() -> int:
    fixture = {}
    failures = []
    for label, pre, goal, post in WITNESSES:
        src = build_witness_source(label, pre, goal, post)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / f"{label}.lean"
            path.write_text(src, encoding="utf-8")
            proc = subprocess.run(["lean", f"+{TOOLCHAIN}", str(path)],
                                  capture_output=True, text=True, encoding="utf-8")
        fields = parse_witness(label, proc.stdout + "\n" + proc.stderr)
        missing = [k for k in PROBES if k not in fields]
        if not fields.get("phi_full"):
            failures.append((label, missing, (proc.stdout + proc.stderr)[:400]))
            continue
        if missing:
            print(f"  {label}: missing {missing}", file=sys.stderr)
        fixture[label] = assemble(fields)
        print(f"  {label}: {len(fields)}/{len(PROBES)} fields", file=sys.stderr)

    if failures:
        for label, missing, log in failures:
            print(f"FAILED {label}: missing {missing}\n{log}", file=sys.stderr)
        return 1

    out = Path(__file__).resolve().parent / "fixtures" / "gate0_fingerprints.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({len(fixture)} witnesses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
