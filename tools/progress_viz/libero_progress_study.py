"""
Progress-quality study on LIBERO rollouts written by examples/LIBERO/eval_libero.py.

For each rollout (rollout_*.mp4 + progress_*.json) the predicted progress trace is compared with the "true" progress:
the PACE training-label definition (frozen DINOv2 CLS, L2-normalised; projection of the current frame onto the
start->goal displacement, clamped to [0, 1]). Success episodes use their own last frame as the goal. Failure episodes
use the last frame of a successful rollout of the same task and episode index (same LIBERO initial state) found in
one of --oracle_dirs ("oracle goal"); without one the true trace is NaN.

    python -m tools.progress_viz.libero_progress_study --eval_dir <OUT_DIR of eval_libero.sh> --out <DIR> \
        [--oracle_dirs <other OUT_DIR> ...] [--n_success 100]

Writes <out>/traces.json (input of progress_metrics.py, pick_demos.py and render_progress_demo.py).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

import numpy as np

from tools.progress_viz.common import ProgressEncoder, add_encoder_args, decode_video

DEFAULT_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_dir", required=True, help="directory with <suite>/<videos_subdir>/rollout_*.mp4")
    parser.add_argument("--out", required=True, help="output directory for traces.json")
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    parser.add_argument("--videos_subdir", default="videos")
    parser.add_argument("--n_success", type=int, default=100, help="successful rollouts sampled per suite")
    parser.add_argument("--oracle_dirs", nargs="*", default=[], help="other eval dirs to take failure goals from")
    parser.add_argument("--seed", type=int, default=0, help="seed of the success-episode sampling")
    add_encoder_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoder = ProgressEncoder(args)
    random.seed(args.seed)
    rows = []
    os.makedirs(args.out, exist_ok=True)
    for suite in args.suites:
        vids = sorted(glob.glob(os.path.join(args.eval_dir, suite, args.videos_subdir, "rollout_*.mp4")))
        succ = [v for v in vids if v.endswith("_success.mp4")]
        fail = [v for v in vids if v.endswith("_failure.mp4")]
        pick = random.sample(succ, min(args.n_success, len(succ))) + fail
        for video in pick:
            match = re.search(r"rollout_(.*)_episode(\d+)_(success|failure)\.mp4$", os.path.basename(video))
            task, ep, res = match.group(1), int(match.group(2)), match.group(3)
            progress_name = os.path.basename(video).replace("rollout_", "progress_")
            progress_json = os.path.join(os.path.dirname(video), progress_name)
            progress_json = progress_json[: -len(".mp4")] + ".json"
            with open(progress_json) as f:
                pred = np.asarray(json.load(f)["progress"], dtype=np.float32)
            frames = decode_video(video)
            n = min(len(frames), len(pred))
            frames, pred = frames[:n], pred[:n]
            z = encoder.encode(frames)
            goal_src, z_goal = None, None
            if res == "success":
                z_goal, goal_src = z[-1:], "self"
            else:
                for oracle_dir in args.oracle_dirs:
                    pattern = os.path.join(
                        oracle_dir, suite, args.videos_subdir, f"rollout_{task}_episode{ep}_success.mp4"
                    )
                    candidates = glob.glob(pattern)
                    if candidates:
                        z_goal = encoder.encode(decode_video(candidates[0])[-1:])
                        goal_src = os.path.basename(os.path.normpath(oracle_dir))
                        break
            true = encoder.progress(z, z[:1], z_goal) if z_goal is not None else np.full(n, np.nan, np.float32)
            rows.append(
                dict(
                    suite=suite,
                    task=task,
                    ep=ep,
                    success=(res == "success"),
                    n=int(n),
                    goal_src=goal_src,
                    pred=np.round(pred, 4).tolist(),
                    true=np.round(true, 4).tolist(),
                    video=video,
                )
            )
            print(suite, ep, res, n, goal_src, "final pred %.3f true %.3f" % (pred[-1], true[-1]), flush=True)
    with open(os.path.join(args.out, "traces.json"), "w") as f:
        json.dump(rows, f)
    print("saved", len(rows))


if __name__ == "__main__":
    main()
