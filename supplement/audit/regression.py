"""Component 8 — Regression suite (blueprint §11.1 item 8, §10.2 condition 6).

Runs the certified controlled witnesses under the pinned toolchain and checks that each
declared pair still exhibits the expected relationship between the model-visible observation
(`φ_state`) and the tactic outcome:

  * identical display  ⇔  equal byte length AND equal `String.hash` of φ_state
  * divergent outcome  ⇔  the two probes' `ok` flags differ

Also runs the file twice to confirm determinism (blueprint §7.7, §11.3). This runs `lean`
directly via subprocess, so — unlike extraction/replay — it works on ANY OS (including the
Windows dev box), giving a cheap version-lock + pipeline check before the LeanDojo audit.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_PROBEREC = re.compile(r"^PROBEREC\|(.*?)\|(true|false)\|(\d+)\|(\d+)\|(.*)$")

# Where the hidden difference lives. `in_goal` differences are NOT closable by an environment
# digest (repair rung R6) — they need the structural rung R4 — so the suite asserts that at
# least one survives, which is what keeps the result from being a pure interface note.
LOCUS_AMBIENT = "ambient"
LOCUS_IN_GOAL = "in_goal"
LOCUS_RENDERER = "renderer"
LOCUS_NONE = "none"

# (name, labelA, labelB, expect_identical_display, expect_divergent, overflow_only, locus)
MANIFEST = [
    ("A_active_env_simp",      "A_with_simp",   "A_without_simp", True, True,  False, LOCUS_AMBIENT),
    ("B_active_env_instance",  "B_normal",      "B_degenerate",   True, True,  False, LOCUS_IN_GOAL),
    ("C_scope_resolution",     "C_scope_N1",    "C_scope_N2",     True, True,  False, LOCUS_AMBIENT),
    ("D_print_overflow",       "D_over_true",   "D_over_false",   True, True,  True,  LOCUS_RENDERER),
    ("E_ingoal_instance",      "E_ingoal_2",    "E_ingoal_0",     True, True,  False, LOCUS_IN_GOAL),
    # Negative control: identical display, hidden difference is a proof term, NO divergence.
    ("CTRL_proof_irrelevance", "CTRL_proof_aa", "CTRL_proof_ab",  True, False, False, LOCUS_NONE),
]


@dataclass
class Probe:
    label: str
    ok: bool
    length: int
    hash: str
    quoted: str


def _run_lean(witness_file: Path, toolchain: str, lean_bin: str = "lean") -> str:
    cmd = [lean_bin, f"+{toolchain}", str(witness_file)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    # PROBEREC goes to stdout; sorry warnings to stderr. Return stdout only.
    return proc.stdout


def _parse(stdout: str) -> dict[str, Probe]:
    out: dict[str, Probe] = {}
    for line in stdout.splitlines():
        m = _PROBEREC.match(line.strip())
        if not m:
            continue
        label, ok, length, h, quoted = m.groups()
        out[label] = Probe(label=label, ok=(ok == "true"), length=int(length), hash=h, quoted=quoted)
    return out


def run_regression(toolchain: str, witness_file: Optional[Path] = None,
                   lean_bin: str = "lean") -> dict:
    witness_file = witness_file or (Path(__file__).resolve().parent.parent
                                    / "regression" / "CertifiedWitnesses.lean")
    if not witness_file.exists():
        return {"passed": False, "error": f"witness file not found: {witness_file}"}

    run1 = _parse(_run_lean(witness_file, toolchain, lean_bin))
    run2 = _parse(_run_lean(witness_file, toolchain, lean_bin))

    # Determinism: the two runs must agree bit-for-bit on every probe (§7.7).
    deterministic = all(
        lbl in run2 and run1[lbl].hash == run2[lbl].hash and run1[lbl].ok == run2[lbl].ok
        for lbl in run1) and (set(run1) == set(run2))

    results = []
    all_pass = deterministic and len(run1) > 0
    for name, la, lb, exp_ident, exp_div, overflow, locus in MANIFEST:
        pa, pb = run1.get(la), run1.get(lb)
        if pa is None or pb is None:
            results.append({"name": name, "passed": False, "reason": "probe(s) missing",
                            "found": [la in run1, lb in run1]})
            all_pass = False
            continue
        identical_display = (pa.hash == pb.hash and pa.length == pb.length and pa.quoted == pb.quoted)
        divergent = (pa.ok != pb.ok)
        ok = (identical_display == exp_ident) and (divergent == exp_div)
        # A witness whose goal failed to elaborate is not a witness. `sorry` in the displayed
        # state means the pair is comparing two failed elaborations, which is how the v1
        # negative control passed while controlling nothing.
        elaborated = "sorry" not in pa.quoted and "sorry" not in pb.quoted
        if not elaborated:
            ok = False
        results.append({
            "name": name, "passed": ok, "overflow_only": overflow, "locus": locus,
            "identical_display": identical_display, "expected_identical": exp_ident,
            "divergent": divergent, "expected_divergent": exp_div,
            "goal_elaborated": elaborated,
            "phi_state": pa.quoted, "hash": pa.hash,
            "ok_a": pa.ok, "ok_b": pb.ok,
        })
        all_pass = all_pass and ok

    def _verified(r) -> bool:
        return bool(r.get("passed") and r.get("divergent") and r.get("identical_display"))

    # A non-overflow, divergent, identical-display witness must survive (blueprint §10.2 cond 4).
    non_overflow_ok = any(_verified(r) and not r.get("overflow_only") for r in results)
    # ≥2 independent mechanisms (blueprint §5.4).
    mechanisms = sorted({r["name"] for r in results if _verified(r) and not r.get("overflow_only")})
    # ≥1 in-goal witness: a difference an environment digest (R6) cannot close.
    in_goal_ok = any(_verified(r) and r.get("locus") == LOCUS_IN_GOAL for r in results)
    # The negative control must be present AND inert — otherwise "no divergence" is untested.
    ctrl = next((r for r in results if r.get("locus") == LOCUS_NONE), None)
    control_ok = bool(ctrl and ctrl.get("passed") and ctrl.get("identical_display")
                      and not ctrl.get("divergent") and ctrl.get("goal_elaborated"))

    return {
        "passed": bool(all_pass and non_overflow_ok and in_goal_ok and control_ok
                       and len(mechanisms) >= 2),
        "deterministic": deterministic,
        "n_probes": len(run1),
        "non_overflow_witness_survives": non_overflow_ok,
        "in_goal_witness_survives": in_goal_ok,
        "inert_control_holds": control_ok,
        "verified_nonoverflow_mechanisms": mechanisms,
        "toolchain": toolchain,
        "pairs": results,
    }
