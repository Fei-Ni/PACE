"""
LIBERO client-side policy wrapper: queries the PACE policy server once per action chunk (open loop within a chunk),
un-normalises the actions with the checkpoint's dataset statistics and keeps the latest progress readouts.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from pace.model.checkpoint_io import read_checkpoint_meta

# Per-query progress outputs of the PACE server that are kept for the per-step trace.
PROGRESS_KEYS = (
    "predicted_progress",
    "predicted_delta_progress",
    "observed_progress_dino",
    "observed_dp_prev",
    "progress_audit_delta",
    "audit_score",
    "audit_stall",
    "audit_flag",
)


class PACELiberoClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        image_size: List[int] = [224, 224],
        host: str = "127.0.0.1",
        port: int = 10093,
    ) -> None:
        # The checkpoint reference is only used to read config.yaml + dataset_statistics.json
        # (a .pt file, a checkpoint directory or a Hugging Face repo id).
        model_config, norm_stats, _ = read_checkpoint_meta(policy_ckpt_path, require_weights=False)
        self.unnorm_key = self._check_unnorm_key(norm_stats, unnorm_key)
        self.action_norm_stats = norm_stats[self.unnorm_key]["action"]
        self.action_chunk_size = int(model_config["framework"]["action_model"]["future_action_window_size"]) + 1
        self.image_size = image_size
        print(f"*** unnorm_key: {self.unnorm_key}, action chunk: {self.action_chunk_size} ***")

        self.client = WebsocketClientPolicy(host, port)
        self.task_description = None
        self.raw_actions = None
        self.latest_progress: Dict[str, float] = {}

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.raw_actions = None
        self.latest_progress = {}

    def step(self, images, task_description: Optional[str] = None, step: int = 0, **kwargs) -> dict:
        """
        Return the action for environment step `step`; the server is queried every `action_chunk_size` steps.

        :param images: list of views, each (H, W, 3) uint8 (LIBERO: [agentview, wrist])
        :return: {"raw_action": {...}, "progress": {latest progress outputs}, "queried": bool}
        """
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        queried = False
        if step % self.action_chunk_size == 0:
            images = [self._resize_image(image) for image in images]
            vla_input = {
                "batch_images": [images],
                "instructions": [self.task_description],
                "do_sample": False,
                "step": step,
            }
            if step == 0:
                vla_input["progress_reset_episode"] = True

            response = self.client.infer(vla_input)
            if not isinstance(response, dict):
                raise RuntimeError(f"Unexpected inference response type: {type(response)!r}")
            if response.get("ok") is False:
                raise RuntimeError(f"Policy server inference failed: {response.get('error', response)!r}")

            response_data = response.get("data", response)
            if "normalized_actions" not in response_data:
                raise RuntimeError(f"Malformed inference response without normalized_actions: {response!r}")

            normalized_actions = response_data["normalized_actions"][0]  # [chunk, D]
            self.raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats
            )
            self.latest_progress = {
                key: float(np.asarray(response_data[key], dtype=np.float64).flatten()[0])
                for key in PROGRESS_KEYS
                if key in response_data
            }
            queried = True

        raw_actions = self.raw_actions[step % self.action_chunk_size][None]
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }
        return {"raw_action": raw_action, "progress": dict(self.latest_progress), "queried": queried}

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """Min-max un-normalisation (LIBERO statistics); the gripper channel is binarised at 0.5."""
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                "The model was trained on more than one dataset; pass `unnorm_key` from: " f"{list(norm_stats.keys())}"
            )
            unnorm_key = next(iter(norm_stats.keys()))
        assert unnorm_key in norm_stats, f"unnorm_key {unnorm_key!r} not in {list(norm_stats.keys())}"
        return unnorm_key

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
