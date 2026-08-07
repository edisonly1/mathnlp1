"""Sharded, supervised, parallel class processing.

Two problems this solves at once.

**CPU.** The serial pipeline processes one class at a time and spends almost all of it blocked
on a single `lake env lean` subprocess, so ~1 of 12 cores is busy. Classes are independent —
each opens its own Dojo sessions — so they shard cleanly.

**Hangs.** A SIGALRM watchdog only works when the block is signal-interruptible. Two census runs
hung anyway, with 24 threads live and Lean grandchildren leaked past the session cap. A parent
process enforcing a wall-clock deadline and then SIGKILLing a worker's whole process tree is a
*hard* guarantee that no in-process mechanism can provide: the worker dies, its Lean processes
die with it, and the remaining shards keep going.

Design
------
* The parent samples once, writes `states.jsonl`, groups into candidate classes, and shards them
  round-robin so slow files spread across workers rather than clumping.
* Each worker re-reads `states.jsonl`, takes its shard, and appends to `classes.shard<i>.jsonl`,
  flushing per class — so a killed worker keeps everything it finished.
* Workers do **not** call `trace()`. `Dojo` only needs `LeanGitRepo` + `Theorem`, so each worker
  skips the ~455s traced-repo load that dominates single-process startup.
* The parent watches each shard file's mtime as a heartbeat. A worker silent longer than
  `--stall-timeout` is killed with its whole tree and its shard is recorded as incomplete.

Usage:
    python tools/parallel_audit.py --config config.yaml --stamp census_par --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _log(msg: str) -> None:
    print(f"[parallel] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def run_worker(cfg_path: str, out_dir: str, shard: int, n_shards: int) -> int:
    # Optional target filter: only process these observation keys.
    from audit import analysis as A
    from audit.alias_index import candidate_classes, finalize_class
    from audit.config import load_config
    from audit.replay import DojoReplayer, kill_orphan_lean
    from audit.run_gate1 import _build_class_summary, _class_to_json
    from audit.schema import Fingerprint, Observation, Provenance, StateRecord

    cfg = load_config(cfg_path)
    out = Path(out_dir)

    records = []
    for line in (out / "states.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        records.append(StateRecord(
            provenance=Provenance(**d["provenance"]),
            observation=Observation(**d["observation"]),
            human_tactic=d.get("human_tactic", ""),
            proof_prefix=d.get("proof_prefix", [])))

    cands = candidate_classes(records, "phi_state", min_members=2)
    target_file = Path(out_dir) / "target_keys.json"
    if target_file.exists():
        targets = set(json.loads(target_file.read_text()))
        cands = [c for c in cands if c[0] in targets]
        _log(f"worker {shard}: target filter -> {len(cands)} classes of interest")
    mine = [c for i, c in enumerate(cands) if i % n_shards == shard]

    # Resume: skip any class already recorded by a previous run (this shard or another).
    # Without this, re-running after a worker is killed redoes work that is already on disk —
    # and the killed shards are exactly the ones worth retrying.
    done_keys = set()
    for f in Path(out_dir).glob("classes.shard*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done_keys.add(json.loads(line)["observation_key"])
                except Exception:
                    pass
    todo = [(k, m) for (k, m) in mine if k not in done_keys]
    _log(f"worker {shard}: {len(mine)} assigned, {len(mine)-len(todo)} already done, "
         f"{len(todo)} to do")
    mine = todo

    replayer = DojoReplayer(cfg)
    stats = {"replays": 0, "nonreproducible_replays": 0, "dropped_reps": 0,
             "degraded_fingerprints": 0}
    fh = open(out / f"classes.shard{shard}.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    try:
        for i, (obs_key, members) in enumerate(mine, 1):
            t = time.time()
            try:
                built = _build_class_summary(cfg, replayer, obs_key, "phi_state", members, stats)
            except BaseException as e:          # BaseException: WatchdogTimeout is one
                _log(f"worker {shard} [{i}/{len(mine)}] {obs_key[:10]} FAILED {type(e).__name__}")
                built = None
            if built is not None:
                cs, _ = built
                fh.write(json.dumps(_class_to_json(cs), ensure_ascii=False) + "\n")
            else:
                # Touch the file even on a skip: it is the parent's heartbeat.
                fh.write("")
            fh.flush()
            os.utime(out / f"classes.shard{shard}.jsonl", None)
            rate = (time.time() - t0) / i
            _log(f"worker {shard} [{i}/{len(mine)}] {obs_key[:10]} {time.time()-t:5.1f}s "
                 f"eta {(rate*(len(mine)-i))/60:.0f}m")
    finally:
        fh.close()
        try:
            replayer.reset_sessions()
        except Exception:
            pass
        kill_orphan_lean()
    return 0


# --------------------------------------------------------------------------- #
# Supervisor
# --------------------------------------------------------------------------- #
def kill_tree(pid: int) -> None:
    from audit.replay import _descendants
    for c in reversed(_descendants(pid)):
        try:
            os.kill(c, signal.SIGKILL)
        except Exception:
            pass
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--stall-timeout", type=int, default=900,
                    help="kill a worker whose shard file has been silent this long")
    ap.add_argument("--max-sessions", type=int, default=0,
                    help="open Dojo sessions per worker; workers x sessions Lean processes must "
                         "fit in RAM (6x4 exhausted 24GB and macOS killed two workers)")
    ap.add_argument("--n-files", type=int, default=0, help="0 = exhaustive census")
    ap.add_argument("--n-states", type=int, default=0)
    ap.add_argument("--worker", type=int, default=-1)   # internal
    ap.add_argument("--n-shards", type=int, default=1)  # internal
    ap.add_argument("--out", default="")                # internal
    args = ap.parse_args()

    if args.worker >= 0:
        if args.max_sessions:
            from audit import replay as _replay
            _replay.MAX_OPEN_SESSIONS = args.max_sessions
        return run_worker(args.config, args.out, args.worker, args.n_shards)

    from audit.config import load_config
    from audit.extract import sample_dense
    from audit.alias_index import candidate_classes
    from audit.schema import StateRecord  # noqa: F401

    cfg = load_config(args.config)
    out = Path(cfg.output_dir) / args.stamp
    out.mkdir(parents=True, exist_ok=True)

    if not (out / "states.jsonl").exists():
        _log("sampling (parent, one traced-repo load) …")
        t0 = time.time()
        recs = sample_dense(cfg, n_files=args.n_files or 10**9,
                            n=args.n_states or None, stats={})
        with open(out / "states.jsonl", "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(r.to_json() + "\n")
        _log(f"sampled {len(recs)} states in {time.time()-t0:.0f}s")
    else:
        _log("reusing existing states.jsonl")

    procs, started = {}, time.time()
    for i in range(args.workers):
        cmd = [sys.executable, __file__, "--config", args.config, "--stamp", args.stamp,
               "--worker", str(i), "--n-shards", str(args.workers), "--out", str(out),
               "--max-sessions", str(args.max_sessions)]
        logf = open(out / f"worker{i}.log", "w")
        procs[i] = (subprocess.Popen(cmd, stdout=logf, stderr=logf), logf)
        (out / f"classes.shard{i}.jsonl").touch()
    _log(f"launched {args.workers} workers; stall timeout {args.stall_timeout}s")

    killed, died = [], []
    expected = {}
    while any(p.poll() is None for p, _ in procs.values()):
        time.sleep(20)
        now = time.time()
        for i, (p, _) in list(procs.items()):
            rc = p.poll()
            if rc is not None:
                # A worker that exits without finishing its shard is a SILENT DEATH (macOS
                # killed workers under memory pressure with no traceback). Counting it as
                # "done" hides an incomplete shard, so record it explicitly.
                if i not in killed and i not in died:
                    log_txt = (out / f"worker{i}.log").read_text(errors="ignore")
                    last = [l for l in log_txt.splitlines() if "] " in l and "/" in l]
                    frag = last[-1].split("] ")[-1][:40] if last else "?"
                    if "assigned, " in log_txt and "0 to do" in log_txt:
                        pass                      # nothing was assigned: a clean no-op
                    elif rc != 0 or "[88/88]" not in log_txt:
                        died.append(i)
                        _log(f"worker {i} EXITED rc={rc} without finishing (last: {frag})")
                continue
            shard = out / f"classes.shard{i}.jsonl"
            quiet = now - shard.stat().st_mtime
            if quiet > args.stall_timeout:
                _log(f"STALL: worker {i} silent {quiet/60:.0f}m — killing its process tree")
                kill_tree(p.pid)
                killed.append(i)
        done = sum(1 for p, _ in procs.values() if p.poll() is not None)
        total = sum(sum(1 for _ in open(out / f"classes.shard{j}.jsonl"))
                    for j in range(args.workers))
        _log(f"  {done}/{args.workers} workers done, {total} classes, "
             f"{(now-started)/60:.0f}m elapsed")

    for _, logf in procs.values():
        logf.close()
    # Glob every shard, not range(args.workers): a resumed run with FEWER workers than the
    # original silently dropped shards 3-5, merging 173 of 346 classes that were on disk.
    merged = out / "classes.jsonl"
    n = 0
    with open(merged, "w", encoding="utf-8") as fh:
        for shard_file in sorted(out.glob("classes.shard*.jsonl")):
            for line in open(shard_file):
                if line.strip():
                    fh.write(line)
                    n += 1
    _log(f"merged {n} classes into {merged}")
    if killed:
        _log(f"WORKERS KILLED FOR STALLING: {killed} — their shards are incomplete")
    if died:
        _log(f"WORKERS DIED EARLY (likely memory pressure): {sorted(set(died))} — "
             f"re-run with fewer workers to cover the gap")
    _log("re-running this command resumes: workers skip classes already in any shard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
