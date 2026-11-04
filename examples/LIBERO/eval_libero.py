"""
LIBERO evaluation client for PACE (run against `deployment.model_server.server_policy`).

For every episode it writes a rollout video and a per-step progress trace
(`progress_<task>_episode<i>_<success|failure>.json`: predicted progress, commitment, observed progress, audit
residual, failure-alarm score / stall / flag) into `--args.video-out-path`, and a `summary.json` with per-task and total success rates.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import pathlib
import time

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.model2libero_interface import PACELiberoClient  # noqa: E402

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
MAX_STEPS = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}
def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [224, 224]

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"  # Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    task_id: int = -1  # Optional: run only a single task id within the suite
    episode_idx: int = -1  # Optional: run only a single init-state episode index

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    # Checkpoint reference used to read config.yaml + dataset_statistics.json
    # (.pt file, checkpoint directory or Hugging Face repo id); must match the checkpoint served by the server.
    pretrained_path: str = ""


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name not in MAX_STEPS:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    max_steps = MAX_STEPS[args.task_suite_name]

    model = PACELiberoClient(
        policy_ckpt_path=args.pretrained_path,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )

    # Start evaluation
    total_episodes, total_successes = 0, 0
    per_task = {}
    started_at = time.time()
    task_ids = [args.task_id] if args.task_id >= 0 else list(range(num_tasks_in_suite))
    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Start episodes
        task_episodes, task_successes = 0, 0
        if args.episode_idx >= 0:
            episode_ids = [args.episode_idx]
        else:
            episode_ids = list(range(args.num_trials_per_task))

        for episode_idx in tqdm.tqdm(episode_ids):
            if episode_idx >= len(initial_states):
                raise IndexError(
                    f"episode_idx={episode_idx} out of range for task_id={task_id} "
                    f"(available init states={len(initial_states)})"
                )
            # Recreate env per episode to avoid EGL/render context accumulation crash.
            # Long-lived OffScreenRenderEnv accumulates MuJoCo/EGL state that causes
            # SIGABRT in robosuite.utils.binding_utils.read_pixels after many episodes.
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

            logging.info(f"\nTask: {task_description}")

            # Reset environment
            model.reset(task_description=task_description)  # reset client state; the server resets at step 0
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            progress_trace = {}  # key -> per-step list (latest server output, repeated within an action chunk)
            query_steps = []

            logging.info(f"Starting episode {task_episodes + 1}...")
            step = 0
            done = False

            while t < max_steps + args.num_steps_wait:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                )

                # Save preprocessed image for replay video
                replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                observation = {
                    "observation.primary": np.expand_dims(
                        img, axis=0
                    ),  # (H, W, C), dtype=unit8, range(0-255)
                    "observation.wrist_image": np.expand_dims(
                        wrist_img, axis=0
                    ),  # (H, W, C)
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }

                # align key with model API
                obs_input = {
                    "images": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "task_description": observation["instruction"][0],
                    "step": step,
                }

                response = model.step(**obs_input)

                raw_action = response["raw_action"]
                for key, value in response["progress"].items():
                    progress_trace.setdefault(key, []).append(value)
                if response["queried"]:
                    query_steps.append(step)

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                # see ../robosuite/controllers/controller_factory.py
                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")

            # Save the per-step progress trace (the server is queried every action chunk; values are repeated
            # for the steps executed open loop within the chunk).
            if progress_trace:
                progress_path = (
                    pathlib.Path(args.video_out_path) / f"progress_{task_segment}_episode{episode_idx}_{suffix}.json"
                )
                payload = {
                    "task": task_description,
                    "episode": episode_idx,
                    "success": bool(done),
                    "progress_update_stride": int(model.action_chunk_size),
                    "query_steps": query_steps,
                    # p_cur readout; kept under "progress" for tools/progress_viz
                    "progress": [float(p) for p in progress_trace.get("predicted_progress", [])],
                    "commitment": progress_trace.get("predicted_delta_progress", []),
                    "observed_progress": progress_trace.get("observed_progress_dino", []),
                    "observed_dp_prev": progress_trace.get("observed_dp_prev", []),
                    "audit_delta": progress_trace.get("progress_audit_delta", []),
                    "audit_score": progress_trace.get("audit_score", []),
                    "audit_stall": progress_trace.get("audit_stall", []),
                    "audit_flag": [bool(v) for v in progress_trace.get("audit_flag", [])],
                }
                with open(progress_path, "w") as _f:
                    json.dump(payload, _f, allow_nan=True)

            imageio.mimwrite(
                pathlib.Path(args.video_out_path)
                / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )

            # Close env to release EGL/render context before next episode
            try:
                env.close()
            except Exception:
                pass

        # Log final results
        per_task[str(task.language)] = {
            "task_id": int(task_id),
            "episodes": int(task_episodes),
            "successes": int(task_successes),
            "success_rate": float(task_successes) / float(task_episodes),
        }
        logging.info(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}"
        )
        logging.info(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}"
        )

    logging.info(
        f"Total success rate: {float(total_successes) / float(total_episodes)}"
    )
    logging.info(f"Total episodes: {total_episodes}")

    summary = {
        "task_suite_name": args.task_suite_name,
        "num_trials_per_task": args.num_trials_per_task,
        "seed": args.seed,
        "pretrained_path": args.pretrained_path,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": float(total_successes) / float(total_episodes) if total_episodes else 0.0,
        "elapsed_sec": time.time() - started_at,
        "tasks": per_task,
    }
    summary_path = pathlib.Path(args.video_out_path) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Summary written to {summary_path}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(eval_libero)
