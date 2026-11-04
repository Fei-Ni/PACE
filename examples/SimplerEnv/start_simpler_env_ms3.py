"""
SimplerEnv (ManiSkill3) WidowX evaluation client for PACE: runs one task for N episodes with seeds
seed, seed+1, ..., against a running policy server and writes <logging-dir>/summary.json.

Optional rollout recording (does not change actions): with --record-dir (default: env SIMPLER_RECORD_DIR) every
episode's frames (`seed<S>_<outcome>_frames.npz`) and per-step progress outputs (`seed<S>_<outcome>_progress.json`)
are saved under <record-dir>/<task>/ for tools/progress_viz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval import *  # noqa: F401,F403
from mani_skill.utils import common

from examples.SimplerEnv.model2simpler_interface import PACESimplerClient
from simpler_env.utils.env.observation_utils import get_image_from_maniskill3_obs_dict


RECORDED_PROGRESS_KEYS = (
    "predicted_progress",
    "predicted_delta_progress",
    "observed_progress_dino",
    "observed_dp_prev",
    "progress_audit_delta",
    "audit_score",
    "audit_stall",
    "audit_flag",
)

TASK_TO_ENV_ID = {
    "widowx_spoon_on_towel": "PutSpoonOnTableClothInScene-v1",
    "widowx_carrot_on_plate": "PutCarrotOnPlateInScene-v1",
    "widowx_stack_cube": "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
    "widowx_put_eggplant_in_basket": "PutEggplantInBasketScene-v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PACE policy in SimplerEnv ManiSkill3 (WidowX).")
    parser.add_argument("--ckpt-path", required=True, help="checkpoint reference (config + action statistics only)")
    parser.add_argument("--task", default="widowx_put_eggplant_in_basket", choices=sorted(TASK_TO_ENV_ID))
    parser.add_argument("--num-episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shader", default="default")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--logging-dir", required=True)
    parser.add_argument("--save-tag", default="")
    parser.add_argument(
        "--action-ensemble",
        type=int,
        default=1,
        help="1 = adaptive action ensembling over the last 7 chunks (default), 0 = first action of each chunk.",
    )
    parser.add_argument(
        "--pass-step-to-server",
        type=int,
        default=1,
        help="Send the env step and a new-episode flag so the progress state is reset per episode (keep 1).",
    )
    parser.add_argument(
        "--record-dir",
        default=os.environ.get("SIMPLER_RECORD_DIR", ""),
        help="Optional: save frames + per-step progress outputs per episode (default: env SIMPLER_RECORD_DIR).",
    )
    return parser.parse_args()


def to_numpy_image(obs_image: torch.Tensor) -> np.ndarray:
    image = obs_image
    if image.ndim == 4:
        image = image[0]
    image = image[..., :3].to(torch.uint8).cpu().numpy()
    return image


def main() -> None:
    args = parse_args()

    env_id = TASK_TO_ENV_ID[args.task]
    logging_dir = Path(args.logging_dir).expanduser().resolve()
    logging_dir.mkdir(parents=True, exist_ok=True)

    sensor_configs = {"shader_pack": args.shader}
    env = gym.make(
        env_id,
        obs_mode="rgb+segmentation",
        num_envs=1,
        sensor_configs=sensor_configs,
    )

    model = PACESimplerClient(
        policy_ckpt_path=args.ckpt_path,
        port=args.port,
        host=args.host,
        action_scale=1.0,
        action_ensemble=bool(args.action_ensemble),
        pass_step_to_server=bool(args.pass_step_to_server),
    )

    success_values: list[float] = []
    episode_summaries: list[dict[str, object]] = []
    started_at = time.time()

    for episode_offset in range(args.num_episodes):
        episode_seed = args.seed + episode_offset
        reset_options = {"episode_id": torch.tensor([episode_seed])}
        obs, _ = env.reset(seed=episode_seed, options=reset_options)
        instruction = env.unwrapped.get_language_instruction()[0]
        model.reset(instruction)

        predicted_terminated = False
        truncated = False
        step_count = 0
        success = 0.0
        # Optional rollout recording for progress demos (off unless --record-dir is set; does not change actions).
        record_dir = args.record_dir
        rec_frames: list[np.ndarray] = []
        rec_progress: dict[str, list[float]] = {}

        while not (predicted_terminated or truncated):
            image = to_numpy_image(get_image_from_maniskill3_obs_dict(env, obs))
            _, processed_action = model.step(
                image,
                instruction,
                step=step_count,
                progress_reset_episode=(step_count == 0),
            )
            action = np.concatenate(
                [
                    processed_action["world_vector"],
                    processed_action["rot_axangle"],
                    processed_action["gripper"],
                ],
                axis=0,
            ).astype(np.float32)
            predicted_terminated = bool(processed_action["terminate_episode"][0] > 0)
            if record_dir:
                rec_frames.append(np.asarray(image, dtype=np.uint8))
                payload = getattr(model, "latest_server_payload", None) or {}
                for key in RECORDED_PROGRESS_KEYS:
                    if key in payload:
                        value = float(np.asarray(payload[key], dtype=np.float64).ravel()[0])
                        rec_progress.setdefault(key, []).append(value)

            obs, reward, terminated, truncated_tensor, info = env.step(torch.from_numpy(action)[None, :])
            info = common.to_numpy(info)
            truncated = bool(np.asarray(common.to_numpy(truncated_tensor)).any())
            terminated = bool(np.asarray(common.to_numpy(terminated)).any())
            success_arr = np.asarray(info.get("success", [0]), dtype=np.float32).reshape(-1)
            success = float(success_arr[0]) if success_arr.size else 0.0
            step_count += 1

            new_instruction = env.unwrapped.get_language_instruction()[0]
            if new_instruction != instruction:
                instruction = new_instruction

            if terminated:
                break

        success_values.append(success)
        if record_dir:
            rec_path = Path(record_dir) / args.task
            rec_path.mkdir(parents=True, exist_ok=True)
            outcome = "success" if success > 0 else "failure"
            stem = rec_path / f"seed{episode_seed}_{outcome}"
            np.savez_compressed(f"{stem}_frames.npz", frames=np.stack(rec_frames) if rec_frames else np.zeros((0,)))
            with open(f"{stem}_progress.json", "w", encoding="utf-8") as f_rec:
                json.dump(
                    {
                        "task": args.task,
                        "instruction": instruction,
                        "seed": episode_seed,
                        "success": success,
                        "steps": step_count,
                        **rec_progress,
                    },
                    f_rec,
                )
        episode_summary = {
            "episode_id": episode_offset,
            "seed": episode_seed,
            "success": success,
            "steps": step_count,
        }
        episode_summaries.append(episode_summary)

    summary = {
        "task": args.task,
        "env_id": env_id,
        "num_episodes": args.num_episodes,
        "seed": args.seed,
        "save_tag": args.save_tag,
        "ckpt_path": args.ckpt_path,
        "host": args.host,
        "port": args.port,
        "action_ensemble": bool(args.action_ensemble),
        "pass_step_to_server": bool(args.pass_step_to_server),
        "mean_success": float(np.mean(success_values)) if success_values else 0.0,
        "elapsed_sec": time.time() - started_at,
        "episodes": episode_summaries,
    }

    summary_path = logging_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    env.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
