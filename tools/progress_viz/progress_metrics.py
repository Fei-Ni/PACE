"""
Summarise predicted-vs-true progress traces (traces.json from libero_progress_study / simpler_progress_study).

    python -m tools.progress_viz.progress_metrics <out>/traces.json [--out metrics.json]

Per group (LIBERO suite or SimplerEnv task) and overall: Spearman / Pearson correlation and MAE on successful
episodes, fraction of non-decreasing steps, final predicted progress of successes vs failures and the AUROC of the
final / maximum predicted progress for separating successes from failures.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import pearsonr, spearmanr


def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def summarise(rows, label: str, nondecreasing_tol: float = 0.02) -> dict:
    successes = [r for r in rows if r["success"]]
    failures = [r for r in rows if not r["success"]]
    sp, pe, mae, fin_s, mono = [], [], [], [], []
    for r in successes:
        p, t = np.asarray(r["pred"]), np.asarray(r["true"])
        if np.std(p) > 1e-6 and np.std(t) > 1e-6:
            sp.append(spearmanr(p, t).correlation)
            pe.append(pearsonr(p, t)[0])
        mae.append(np.abs(p - t).mean())
        fin_s.append(p[-1])
        dp = np.diff(p)
        mono.append((dp >= -nondecreasing_tol).mean() if len(dp) else 1.0)
    fin_f = [np.asarray(r["pred"])[-1] for r in failures]
    true_f = [np.asarray(r["true"])[-1] for r in failures if not np.isnan(np.asarray(r["true"])[-1])]
    maxp_f = [np.max(r["pred"]) for r in failures]
    gap_f = [
        np.abs(np.asarray(r["pred"]) - np.asarray(r["true"])).mean()
        for r in failures
        if not np.isnan(np.asarray(r["true"])).all()
    ]
    out = dict(
        label=label,
        n_success=len(successes),
        n_failure=len(failures),
        success_spearman_median=float(np.nanmedian(sp)) if sp else None,
        success_spearman_mean=float(np.nanmean(sp)) if sp else None,
        success_pearson_mean=float(np.nanmean(pe)) if pe else None,
        success_frac_spearman_gt_0p8=float(np.mean(np.asarray(sp) > 0.8)) if sp else None,
        success_mae_mean=float(np.mean(mae)) if mae else None,
        success_final_pred_mean=float(np.mean(fin_s)) if fin_s else None,
        success_nondecreasing_frac=float(np.mean(mono)) if mono else None,
        failure_final_pred_mean=float(np.mean(fin_f)) if fin_f else None,
        failure_final_true_mean=float(np.mean(true_f)) if true_f else None,
        failure_mae_mean=float(np.mean(gap_f)) if gap_f else None,
        auroc_final_pred_success_vs_failure=auroc(fin_s, fin_f),
        auroc_max_pred_success_vs_failure=auroc([np.max(r["pred"]) for r in successes], maxp_f),
    )
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", help="traces.json")
    parser.add_argument("--out", default=None, help="output JSON (default: metrics.json next to traces.json)")
    args = parser.parse_args()

    with open(args.traces) as f:
        rows = json.load(f)
    groups = {}
    for r in rows:
        groups.setdefault(r.get("suite") or r.get("task"), []).append(r)
    results = [summarise(rows, "ALL")] + [summarise(v, k) for k, v in sorted(groups.items())]
    for row in results:
        print(json.dumps(row))
    out_path = args.out or args.traces.replace("traces.json", "metrics.json")
    if out_path == args.traces:
        out_path = args.traces + ".metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print("saved", out_path)


if __name__ == "__main__":
    main()
