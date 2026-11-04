"""
Aggregate SimplerEnv (ManiSkill3, WidowX) results written by start_simpler_env_ms3.py.

    python examples/SimplerEnv/aggregate_simpler.py --eval_dir <OUT_DIR> [--episodes_per_task 24]

expects <OUT_DIR>/<task>/summary.json for the four WidowX tasks (layout produced by eval_simpler.sh) and writes
<OUT_DIR>/simpler_summary.json with per-task success and the mean over the four tasks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Paper order: Put Spoon, Put Carrot, Stack Block, Put Eggplant.
TASKS = [
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
]


def aggregate(eval_dir: Path, tasks=TASKS, episodes_per_task: int = 24) -> dict:
    per_task = {}
    for task in tasks:
        summary_path = eval_dir / task / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing task summary: {summary_path}")
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        num_episodes = int(summary.get("num_episodes", -1))
        if episodes_per_task and num_episodes != episodes_per_task:
            raise ValueError(f"Unexpected num_episodes for {task}: got {num_episodes}, expected {episodes_per_task}")
        per_task[task] = {
            "mean_success": float(summary["mean_success"]),
            "num_episodes": num_episodes,
            "seed": summary.get("seed"),
            "action_ensemble": summary.get("action_ensemble"),
            "summary_path": str(summary_path),
        }
    mean = sum(v["mean_success"] for v in per_task.values()) / len(per_task)
    return {"task_order": list(tasks), "tasks": per_task, "widowx_mean_success": mean}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_dir", required=True, type=Path)
    parser.add_argument("--episodes_per_task", type=int, default=24, help="0 disables the episode-count check")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    args = parser.parse_args()

    result = aggregate(args.eval_dir, args.tasks, args.episodes_per_task)
    out_path = args.eval_dir / "simpler_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    for task, row in result["tasks"].items():
        print(f"{task:32s} {100.0 * row['mean_success']:6.1f}  ({row['num_episodes']} episodes)")
    print(f"{'mean':32s} {100.0 * result['widowx_mean_success']:6.1f}")
    print(f"SIMPLER_SUMMARY_JSON={out_path}")


if __name__ == "__main__":
    main()
