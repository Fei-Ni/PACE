"""
Progress-quality study on SimplerEnv rollouts recorded with examples/SimplerEnv/eval_simpler.sh and
SIMPLER_RECORD_DIR set (per episode: <task>/seed<S>_<outcome>_frames.npz + _progress.json).

True progress: the PACE training-label definition (frozen DINOv2 CLS, L2-normalised; projection onto start->goal,
clamped to [0, 1]). Success: goal = the episode's own last frame. Failure: goal = last frame of a successful rollout
with the same task and seed (same scene layout) from one of --oracle_dirs; NaN if none.

    python -m tools.progress_viz.simpler_progress_study --rec_dir <SIMPLER_RECORD_DIR> --out <DIR> \
        [--oracle_dirs <other record dirs> ...]

Writes <out>/traces.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from tools.progress_viz.common import ProgressEncoder, add_encoder_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rec_dir", required=True, help="SIMPLER_RECORD_DIR of the evaluated checkpoint")
    parser.add_argument("--out", required=True, help="output directory for traces.json")
    parser.add_argument("--oracle_dirs", nargs="*", default=[], help="other record dirs to take failure goals from")
    add_encoder_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoder = ProgressEncoder(args)
    rows = []
    os.makedirs(args.out, exist_ok=True)
    for progress_json in sorted(glob.glob(os.path.join(args.rec_dir, "*", "*_progress.json"))):
        with open(progress_json) as f:
            meta = json.load(f)
        frames_npz = progress_json.replace("_progress.json", "_frames.npz")
        frames = np.load(frames_npz)["frames"]
        pred = np.asarray(meta.get("predicted_progress", []), np.float32)
        n = min(len(frames), len(pred))
        if n == 0:
            continue
        z = encoder.encode(frames[:n])
        ok = float(meta["success"]) > 0
        z_goal, goal_src = (z[-1:], "self") if ok else (None, None)
        if not ok:
            for oracle_dir in args.oracle_dirs:
                candidates = glob.glob(os.path.join(oracle_dir, meta["task"], f"seed{meta['seed']}_success_frames.npz"))
                if candidates:
                    z_goal = encoder.encode(np.load(candidates[0])["frames"][-1:])
                    goal_src = os.path.basename(os.path.normpath(oracle_dir))
                    break
        true = encoder.progress(z, z[:1], z_goal) if z_goal is not None else np.full(n, np.nan, np.float32)
        observed = np.asarray(meta.get("observed_progress_dino", [np.nan] * n)[:n], float)
        rows.append(
            dict(
                task=meta["task"],
                instruction=meta.get("instruction"),
                seed=meta["seed"],
                ep=meta["seed"],
                success=ok,
                n=int(n),
                goal_src=goal_src,
                pred=np.round(pred[:n], 4).tolist(),
                true=np.round(true, 4).tolist(),
                frames_npz=frames_npz,
                observed=np.round(observed, 4).tolist(),
            )
        )
        print(
            meta["task"],
            meta["seed"],
            "S" if ok else "F",
            n,
            goal_src,
            "final pred %.3f true %.3f" % (pred[n - 1], true[-1]),
            flush=True,
        )
    with open(os.path.join(args.out, "traces.json"), "w") as f:
        json.dump(rows, f)
    print("saved", len(rows))


if __name__ == "__main__":
    main()
