"""Current-version persistence test: do the confirmed witnesses survive on today's Mathlib?

For each witness the corresponding theorem is located in a CURRENT Mathlib checkout, its proof is
replaced in place with prefix + history + trace_state + probe (wrapped in `fail_if_success` on
members where the probe is expected to fail), the single file is compiled with `lake env lean`,
and the file is then restored via git. In-place replacement preserves namespace, variable, and
notation context, which standalone restatement cannot.

Classification per member:
  PERSISTS      file compiles (probe behaved as at the pinned commit; sorry warnings only)
  CHANGED       compile error at or after the probe line (probe behavior flipped)
  NOT_PORTABLE  theorem absent (renamed) or error before the probe line (prefix/API drift)

Usage:
    python tools/persistence_test.py --repo ../persist/mathlib4_current \\
        --witnesses coercion,NeNot,FinEta,instance,proofterm,residual,table1
"""
from __future__ import annotations
import argparse, glob, json, re, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DECL_START = re.compile(r"^(@\[|theorem |lemma |protected |private |noncomputable |instance |"
                        r"def |abbrev |structure |class |inductive |example|end |section|"
                        r"namespace |open |variable|set_option |attribute |#|/--|--)")

def _log(m): print(f"[persist] {m}", flush=True)

def find_decl(repo: Path, lastname: str, orig_file: str):
    """Locate `theorem/lemma <lastname>` in the current tree; prefer the original file path."""
    pat = re.compile(rf"^\s*(?:@\[[^\]]*\]\s*)?(?:protected\s+|private\s+|nonrec\s+)*"
                     rf"(?:theorem|lemma)\s+{re.escape(lastname)}\b")
    cands = []
    orig = repo / orig_file
    files = ([orig] if orig.exists() else []) + \
            [Path(f) for f in glob.glob(str(repo / "Mathlib/**/*.lean"), recursive=True)]
    seen = set()
    for f in files:
        if f in seen: continue
        seen.add(f)
        try: text = f.read_text()
        except Exception: continue
        for i, line in enumerate(text.splitlines()):
            if pat.match(line):
                cands.append((f, i))
        if cands and f == orig:
            break
    return cands[0] if cands else None

def splice(repo: Path, f: Path, line_no: int, tactics: list) -> tuple:
    """Replace the proof of the declaration starting at line_no with our tactic block.
    Returns (probe_line_number_in_file, ok)."""
    lines = f.read_text().splitlines()
    # find ':= by' at or after the decl start
    j = line_no
    while j < len(lines) and ":=" not in lines[j]:
        j += 1
        if j - line_no > 60: return None, False
    head = lines[j]
    k = head.index(":=")
    lines[j] = head[:k] + ":= by"
    # find end of proof body: next column-0 declaration
    e = j + 1
    while e < len(lines):
        L = lines[e]
        if L and not L[0].isspace() and DECL_START.match(L):
            break
        e += 1
    body = ["  " + t for t in tactics]
    probe_line = j + len(body)          # 1-based adjusted below
    new = lines[:j+1] + body + lines[e:]
    f.write_text("\n".join(new) + "\n")
    return probe_line + 1, True

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="runs/persistence/results.jsonl")
    ap.add_argument("--select", default="", help="comma-separated theorem substrings; empty=all")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()

    # witnesses: confirmed classes + prefix from states + expected member outcomes
    uniq = {}
    for run in ("runs/reconv_b2", "runs/hunt_enn", "runs/hunt_rare"):
        rows = {}
        for fpath in glob.glob(f"{run}/stageb.shard*.jsonl"):
            for l in open(fpath):
                if l.strip():
                    r = json.loads(l); rows[(r["theorem"], r["pp_hash"])] = r
        for c in json.load(open(f"{run}/confirmation.json")):
            if c.get("verdict") == "CONFIRMED":
                k = (c["theorem"], c["pp_hash"])
                if k in rows: uniq.setdefault(k, (rows[k], c))
    sel = [s for s in args.select.split(",") if s]
    items = [(k, v) for k, v in uniq.items()
             if not sel or any(s in k[0] for s in sel)]
    wanted = {(r["file"], r["theorem"], r["tactic_index"]) for _, (r, _) in items}
    prov = {}
    with open("runs/killtest/states.jsonl") as fh:
        for line in fh:
            if not line.strip(): continue
            d = json.loads(line); pv = d["provenance"]
            kk = (pv["file_path"], pv["theorem_full_name"], pv["tactic_index"])
            if kk in wanted: prov[kk] = d
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if outp.exists():
        for l in open(outp):
            if l.strip():
                d = json.loads(l); done.add((d["theorem"], d["pp_hash"], d["member"]))
    fh_out = open(outp, "a")

    for (tn, ph), (r, c) in items:
        s = prov.get((r["file"], r["theorem"], r["tactic_index"]))
        if s is None: continue
        prefix = s["proof_prefix"]
        probe = (c.get("confirmed_probes") or [None])[0]
        rt = c.get("repeat_table") or {}
        exp = {n: (v[0] if v else None) for n, v in (rt.get(probe) or {}).items()}
        lastname = tn.split(".")[-1]
        loc = find_decl(repo, lastname, r["file"])
        for hist in r["histories"]:
            member = " ; ".join(hist)
            if (tn, ph, member) in done: continue
            expected = exp.get(member)
            if expected is None: continue
            rec = {"theorem": tn, "pp_hash": ph, "member": member,
                   "probe": probe, "expected": expected}
            if loc is None:
                rec["verdict"] = "NOT_PORTABLE"; rec["reason"] = "declaration not found (renamed)"
                fh_out.write(json.dumps(rec) + "\n"); fh_out.flush()
                _log(f"{tn[:40]} [{member[:24]}]: NOT_PORTABLE (renamed)")
                continue
            f, ln = loc
            probe_t = (f"fail_if_success {probe}" if expected == "FAIL" else probe)
            tactics = list(prefix) + list(hist) + ["trace_state", probe_t, "all_goals sorry"]
            pl, ok = splice(repo, f, ln, tactics)
            if not ok:
                subprocess.run(["git", "checkout", "--", str(f.relative_to(repo))], cwd=repo)
                rec["verdict"] = "NOT_PORTABLE"; rec["reason"] = "splice failed"
                fh_out.write(json.dumps(rec) + "\n"); fh_out.flush(); continue
            t0 = time.time()
            try:
                pr = subprocess.run(["lake", "env", "lean", str(f.relative_to(repo))],
                                    cwd=repo, capture_output=True, text=True,
                                    timeout=args.timeout)
                errs = [m for m in re.finditer(rf"{re.escape(f.name)}:(\d+):\d+: error", pr.stdout + pr.stderr)]
                err_lines = [int(m.group(1)) for m in errs]
                traces = re.findall(r"(?s)trace: ?\n?(.*?)(?=\n\S|\Z)", pr.stdout)
                rec["compile_s"] = round(time.time() - t0, 1)
                rec["error_lines"] = err_lines[:5]
                rec["probe_file_line"] = pl
                if not err_lines:
                    rec["verdict"] = "PERSISTS"
                elif all(e < pl - 1 for e in err_lines):
                    rec["verdict"] = "NOT_PORTABLE"; rec["reason"] = "prefix/history fails (API drift)"
                else:
                    rec["verdict"] = "CHANGED"
                rec["trace_hash"] = (__import__("hashlib").sha256(
                    traces[-1].strip().encode()).hexdigest()[:12] if traces else None)
            except subprocess.TimeoutExpired:
                rec["verdict"] = "TIMEOUT"
            finally:
                subprocess.run(["git", "checkout", "--", str(f.relative_to(repo))], cwd=repo)
            fh_out.write(json.dumps(rec) + "\n"); fh_out.flush()
            _log(f"{tn[:40]} [{member[:24]}]: {rec['verdict']} ({rec.get('compile_s','-')}s)")
    fh_out.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
