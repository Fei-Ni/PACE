"""
Calibrate the failure alarm (commitment audit) on your own rollouts, and measure how well it detects failures.

The alarm has two thresholds (framework.progress in the checkpoint's config.yaml):
  audit_threshold      on the shortfall CUSUM (`audit_score`)
  audit_stall_chunks   on the action chunks since p_cur last rose (`audit_stall`)

Calibration sets each threshold so that, on its own, it flags `false_alarm / 2` of the successful episodes of the
calibration runs; together they flag at most `false_alarm` of them. Use runs of checkpoints other than the one you will
deploy (other seeds or training steps), then check the result on the deployed checkpoint with --test.

Inputs are evaluation output directories, searched recursively for
  progress_*.json           written by examples/LIBERO/eval_libero.sh (one query per action chunk)
  seed*_progress.json       written by examples/SimplerEnv/eval_simpler.sh with SIMPLER_RECORD_DIR set (every step)

    # calibrate on held-out runs, then test on the deployed checkpoint
    python -m tools.progress_viz.calibrate_audit --chunk_steps 8 --dev <eval_a> <eval_b> --test <eval_release>
    # evaluate fixed thresholds
    python -m tools.progress_viz.calibrate_audit --chunk_steps 8 --threshold 0.4 --stall_chunks 18 --test <eval_dir>

Detection is reported per run directory: false alarms (successful episodes flagged), detected failures (failed
episodes flagged), and the median share of the episode still left when the alarm fires on a failure.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List

import numpy as np

from pace.model.progress.audit import update_audit


def _num(v):
    return float("nan") if v is None else float(v)


def load_episode(path: str) -> Dict:
    d = json.load(open(path))
    if "query_steps" in d:  # LIBERO client: values per environment step, the server queried at query_steps
        T = len(d["progress"])
        steps = [s for s in d["query_steps"] if s < T]
        res = [_num(d["audit_delta"][s]) for s in steps]
        prog = [_num(d["progress"][s]) for s in steps]
    else:  # SimplerEnv recorder: one entry per environment step, every step is a query
        prog = [_num(v) for v in d.get("predicted_progress", [])]
        res = [_num(v) for v in d.get("progress_audit_delta", [math.nan] * len(prog))]
        T = int(d.get("steps", len(prog)))
        steps = list(range(len(prog)))
    return dict(path=path, success=bool(float(d["success"]) > 0), T=max(T, 1), steps=steps, res=res, prog=prog)


def trace(ep: Dict, chunk_steps: int, kappa: float):
    """Shortfall CUSUM and stall after every query, with both alarms disabled."""
    state, out = None, []
    for s, r, p in zip(ep["steps"], ep["res"], ep["prog"]):
        state, score, stall, _ = update_audit(state, s, r, p, chunk_steps, kappa=kappa, threshold=math.inf,
                                              stall_chunks=0)
        out.append((s, score, stall))
    return out


def first_alarm(tr, threshold: float, stall_chunks: float):
    for s, score, stall in tr:
        if score > threshold or (stall_chunks > 0 and stall > stall_chunks):
            return s
    return None


def find_episodes(root: str) -> List[str]:
    pats = ("**/progress_*.json", "**/seed*_progress.json")
    return sorted({p for pat in pats for p in glob.glob(os.path.join(root, pat), recursive=True)})


def load_runs(dirs, chunk_steps, kappa):
    runs = {}
    for root in dirs or []:
        paths = find_episodes(root)
        if not paths:
            raise SystemExit(f"no progress traces found under {root}")
        eps = [load_episode(p) for p in paths]
        for ep in eps:
            ep["trace"] = trace(ep, chunk_steps, kappa)
        runs[root] = eps
    return runs


def calibrate(eps: List[Dict], false_alarm: float):
    succ = [e for e in eps if e["success"]]
    if not succ:
        raise SystemExit("calibration runs contain no successful episodes")

    def pick(values):  # the (1 - false_alarm / 2) quantile of the per-episode maxima
        return float(np.quantile(np.asarray(values, float), 1.0 - false_alarm / 2))

    max_score = [max([s for _, s, _ in e["trace"]] or [0.0]) for e in succ]
    max_stall = [max([t for _, _, t in e["trace"]] or [0.0]) for e in succ]
    return pick(max_score), pick(max_stall), len(succ)


def evaluate(eps: List[Dict], threshold: float, stall_chunks: float) -> Dict:
    succ = [e for e in eps if e["success"]]
    fail = [e for e in eps if not e["success"]]
    fa = [first_alarm(e["trace"], threshold, stall_chunks) is not None for e in succ]
    left = []
    for e in fail:
        a = first_alarm(e["trace"], threshold, stall_chunks)
        if a is not None:
            left.append(1.0 - a / e["T"])
    return dict(n_succ=len(succ), n_fail=len(fail), false_alarm=float(np.mean(fa)) if fa else float("nan"),
                detected=len(left) / len(fail) if fail else float("nan"),
                remaining=float(np.median(left)) if left else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", nargs="+", help="evaluation directories used to set the thresholds")
    ap.add_argument("--test", nargs="+", help="evaluation directories to report on with the thresholds")
    ap.add_argument("--chunk_steps", type=int, required=True, help="action chunk length H (8 LIBERO, 16 SimplerEnv)")
    ap.add_argument("--false_alarm", type=float, default=0.05, help="target share of flagged successes (both signals)")
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, help="use this audit_threshold instead of calibrating")
    ap.add_argument("--stall_chunks", type=float, help="use this audit_stall_chunks instead of calibrating")
    ap.add_argument("--json", help="write the results to this file")
    a = ap.parse_args()
    if not a.dev and (a.threshold is None or a.stall_chunks is None):
        ap.error("give --dev to calibrate, or both --threshold and --stall_chunks")
    dev = load_runs(a.dev, a.chunk_steps, a.kappa)
    test = load_runs(a.test, a.chunk_steps, a.kappa)
    threshold, stall_chunks = a.threshold, a.stall_chunks
    if a.dev:
        pooled = [e for eps in dev.values() for e in eps]
        cal_threshold, cal_stall, n = calibrate(pooled, a.false_alarm)
        print(f"calibrated on {n} successful episodes from {len(dev)} run(s), {a.false_alarm:.1%} false-alarm budget")
        threshold = cal_threshold if threshold is None else threshold
        stall_chunks = cal_stall if stall_chunks is None else stall_chunks
    print(f"\nframework:\n  progress:\n    audit_kappa: {a.kappa:g}\n    audit_threshold: {threshold:.3f}\n"
          f"    audit_stall_chunks: {stall_chunks:.2f}\n")
    results = {"audit_kappa": a.kappa, "audit_threshold": threshold, "audit_stall_chunks": stall_chunks,
               "chunk_steps": a.chunk_steps, "runs": {}}
    print(f"{'run':<52} {'set':<5} {'successes flagged':>18} {'failures flagged':>17} {'episode left':>13}")
    for tag, runs in (("dev", dev), ("test", test)):
        for root, eps in runs.items():
            r = evaluate(eps, threshold, stall_chunks)
            results["runs"][root] = dict(set=tag, **r)
            left = "-" if r["remaining"] is None else f"{r['remaining']:.0%}"
            name = root if len(root) <= 52 else "…" + root[-51:]
            print(f"{name:<52} {tag:<5} {r['false_alarm']:>9.1%} of {r['n_succ']:<5} "
                  f"{r['detected']:>8.1%} of {r['n_fail']:<4} {left:>13}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)
        print("\nwrote", a.json)


if __name__ == "__main__":
    main()
