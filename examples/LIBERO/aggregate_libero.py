"""
Aggregate LIBERO results written by eval_libero.py (one summary.json per suite) into per-suite and mean success.

    python examples/LIBERO/aggregate_libero.py --eval_dir <OUT_DIR>

expects <OUT_DIR>/<suite>/videos/summary.json for every suite (layout produced by eval_libero.sh) and writes
<OUT_DIR>/libero_summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def aggregate(eval_dir: Path, suites, videos_subdir: str = "videos", expected_episodes: int = 0) -> dict:
    per_suite = {}
    for suite in suites:
        summary_path = eval_dir / suite / videos_subdir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing suite summary: {summary_path}")
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        episodes = int(summary["total_episodes"])
        if expected_episodes and episodes != expected_episodes:
            raise ValueError(f"{suite}: expected {expected_episodes} episodes, found {episodes} in {summary_path}")
        per_suite[suite] = {
            "success_rate": float(summary["success_rate"]),
            "episodes": episodes,
            "successes": int(summary["total_successes"]),
            "seed": summary.get("seed"),
            "summary_path": str(summary_path),
        }
    mean = sum(v["success_rate"] for v in per_suite.values()) / len(per_suite)
    return {"suites": per_suite, "mean_success_rate": mean, "suite_order": list(suites)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_dir", required=True, type=Path)
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    parser.add_argument("--videos_subdir", default="videos")
    parser.add_argument(
        "--expected_episodes", type=int, default=0, help="episodes per suite to enforce (e.g. 500); 0 = no check"
    )
    args = parser.parse_args()

    result = aggregate(args.eval_dir, args.suites, args.videos_subdir, args.expected_episodes)
    out_path = args.eval_dir / "libero_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    for suite, row in result["suites"].items():
        print(f"{suite:16s} {100.0 * row['success_rate']:6.1f}  ({row['successes']}/{row['episodes']})")
    print(f"{'mean':16s} {100.0 * result['mean_success_rate']:6.1f}")
    print(f"LIBERO_SUMMARY_JSON={out_path}")


if __name__ == "__main__":
    main()
