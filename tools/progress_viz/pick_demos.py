"""
Pick representative rollouts for progress demos from traces.json: for every group (suite / task) the successful
episode whose Spearman correlation is closest to the group median, and the failure episode(s) with the highest
correlation (only failures that have a true-progress trace).

    python -m tools.progress_viz.pick_demos <out>/traces.json [--failures_per_group 1]

Prints `SELECT i j ...`, to be passed to render_progress_demo.py --select.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces")
    parser.add_argument("--failures_per_group", type=int, default=1)
    args = parser.parse_args()

    with open(args.traces) as f:
        rows = json.load(f)
    pick = []
    groups = {}
    for i, r in enumerate(rows):
        p, t = np.asarray(r["pred"]), np.asarray(r["true"])
        r["_sp"] = spearmanr(p, t).correlation if np.std(p) > 1e-6 and np.std(t) > 1e-6 else np.nan
        groups.setdefault((r.get("suite") or r["task"], r["success"]), []).append(i)
    for (group, ok), idx in sorted(groups.items()):
        sps = np.array([rows[i]["_sp"] for i in idx])
        med = np.nanmedian(sps) if not np.isnan(sps).all() else np.nan
        if ok:
            if np.isnan(med):
                continue
            j = idx[int(np.nanargmin(np.abs(sps - med)))]
            pick.append(j)
            print("success", group, "median sp %.3f pick %d sp %.3f n %d" % (med, j, rows[j]["_sp"], rows[j]["n"]))
        else:
            cand = [i for i in idx if not np.isnan(np.asarray(rows[i]["true"])).all()]
            cand = sorted(cand, key=lambda i: -np.nan_to_num(rows[i]["_sp"]))[: args.failures_per_group]
            for j in cand:
                pick.append(j)
                print(
                    "failure",
                    group,
                    "pick %d sp %.3f final pred %.3f true %.3f"
                    % (j, rows[j]["_sp"], rows[j]["pred"][-1], rows[j]["true"][-1]),
                )
    print("SELECT", " ".join(map(str, pick)))


if __name__ == "__main__":
    main()
