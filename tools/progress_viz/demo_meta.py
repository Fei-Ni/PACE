"""
Write a small JSON index of rendered demo videos (file name, benchmark, task, outcome, length, Spearman correlation,
final predicted / true progress), e.g. for a project page.

    python -m tools.progress_viz.demo_meta --traces <out>/traces.json --bench LIBERO --select 84 195 ... \
        [--out demos.json]

File names follow render_progress_demo.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from tools.progress_viz.render_progress_demo import demo_filename


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--bench", required=True, help="label stored in the index, e.g. LIBERO or SimplerEnv")
    parser.add_argument("--select", type=int, nargs="+", required=True)
    parser.add_argument("--out", default=None, help="output JSON (default: print to stdout)")
    args = parser.parse_args()

    with open(args.traces) as f:
        rows = json.load(f)
    out = []
    for i in args.select:
        r = rows[i]
        p, t = np.asarray(r["pred"], float), np.asarray(r["true"], float)
        sp = spearmanr(p, t).correlation if np.std(p) > 1e-6 and np.std(t) > 1e-6 else None
        out.append(
            dict(
                file=demo_filename(i, r),
                bench=args.bench,
                group=r.get("suite") or r["task"],
                task=(r.get("instruction") or r["task"]).replace("_", " "),
                success=bool(r["success"]),
                steps=int(r["n"]),
                sp=None if sp is None else round(float(sp), 2),
                final_pred=round(float(p[-1]), 2),
                final_true=None if np.isnan(t[-1]) else round(float(t[-1]), 2),
            )
        )
    text = json.dumps(out, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print("saved", args.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
