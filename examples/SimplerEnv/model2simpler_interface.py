"""
SimplerEnv (WidowX / Bridge) client-side policy wrapper: queries the PACE policy server at every environment step,
un-normalises the action chunk with the checkpoint's q01/q99 statistics and (by default) applies the StarVLA
adaptive action ensemble over the last 7 chunks.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Dict, List, Optional

import cv2 as cv
import numpy as np
from transforms3d.euler import euler2axangle

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.SimplerEnv.adaptive_ensemble import AdaptiveEnsembler
from pace.model.checkpoint_io import read_checkpoint_meta


class PACESimplerClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        horizon: int = 0,
        action_ensemble_horizon: Optional[int] = None,
        image_size: List[int] = [224, 224],
        action_scale: float = 1.0,
        action_ensemble: bool = True,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "127.0.0.1",
        port: int = 10093,
        pass_step_to_server: bool = True,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        unnorm_key = "oxe_bridge" if unnorm_key is None else unnorm_key
        if action_ensemble_horizon is None:
            # WidowX: ensemble window of 7 chunks (StarVLA default for Bridge).
            action_ensemble_horizon = 7
        self.sticky_gripper_num_repeat = 1

        # Only config.yaml + dataset_statistics.json are read (.pt file, checkpoint directory or HF repo id).
        _, norm_stats, _ = read_checkpoint_meta(policy_ckpt_path, require_weights=False)
        self.unnorm_key = self.resolve_unnorm_key(norm_stats, unnorm_key)
        print(f"*** policy_setup: widowx_bridge, unnorm_key: {self.unnorm_key} ***")

        self.client = WebsocketClientPolicy(host, port)
        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = norm_stats[self.unnorm_key]["action"]
        self.pass_step_to_server = bool(pass_step_to_server)
        self.latest_server_payload: Optional[dict] = None

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.latest_server_payload = None

    def step(
        self,
        image: np.ndarray,
        task_description: Optional[str] = None,
        step: Optional[int] = None,
        progress_reset_episode: Optional[bool] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: task description; if it differs from the previous one, the client state is reset
            step / progress_reset_episode: sent to the server so its per-episode progress state is aligned
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action for the ManiSkill environment with keys
                'world_vector' (3,), 'rot_axangle' (3,), 'gripper' (1,), 'terminate_episode' (1,)
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        self._add_image_to_history(self._resize_image(image))

        image = self._resize_image(image)
        vla_input = {
            "batch_images": [[image]],
            "instructions": [self.task_description],
            "do_sample": False,
        }
        if self.pass_step_to_server and step is not None:
            vla_input["step"] = int(step)
        if self.pass_step_to_server and progress_reset_episode is not None:
            vla_input["progress_reset_episode"] = bool(progress_reset_episode)

        response = self.client.infer(vla_input)
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected inference response type: {type(response)!r}")
        if response.get("ok") is False:
            raise RuntimeError(f"Policy server inference failed: {response.get('error', response)!r}")
        response_data = response.get("data", response)
        if "normalized_actions" not in response_data:
            raise RuntimeError(f"Malformed inference response without normalized_actions: {response!r}")
        self.latest_server_payload = response_data

        normalized_actions = response_data["normalized_actions"][0]  # [chunk, D]
        raw_actions = self.unnormalize_actions(
            normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats
        )

        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)

        roll, pitch, yaw = action_rotation_delta
        axes, angles = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = axes * angles
        action["rot_axangle"] = action_rotation_axangle * self.action_scale
        action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        action["terminate_episode"] = np.array([0.0])
        return raw_action, action

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """q01/q99 un-normalisation (Bridge statistics); the gripper channel is binarised at 0.5."""
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions

    @staticmethod
    def resolve_unnorm_key(norm_stats: Dict[str, dict], requested_key: Optional[str]) -> str:
        if requested_key in norm_stats:
            return requested_key
        if len(norm_stats) == 1:
            fallback_key = next(iter(norm_stats.keys()))
            print(
                f"*** requested unnorm_key {requested_key!r} not found; "
                f"falling back to sole available key {fallback_key!r} ***"
            )
            return fallback_key
        raise KeyError(f"unnorm_key {requested_key!r} not found; available keys: {list(norm_stats.keys())}")

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
