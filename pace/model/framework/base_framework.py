"""
Base framework abstraction providing:
- Pretrained loading (config + normalization stats + weights)
- Action space utilities (dimension, stats, (un)normalization)
- Trainable module discovery helper
Note: No device placement or optimizer concerns handled here (delegated to trainer).
"""

from typing import Dict, List

import numpy as np
import torch
from transformers import PretrainedConfig, PreTrainedModel

from pace.model.checkpoint_io import read_checkpoint_meta
from pace.model.framework import build_framework
from pace.model.framework.share_tools import dict_to_namespace
from pace.model.tools import auto_get_trainable_modules
from pace.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class baseframework(PreTrainedModel):
    """
    Lightweight base class for higher-level VLA model assemblies.
    Subclasses are expected to:
      - Accept a structured config
      - Register components in __init__
      - Use provided helpers for action normalization handling
    """

    def __init__(self, hf_config=PretrainedConfig()) -> None:
        super().__init__(hf_config)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: str,
        **kwargs,
    ):
        """
        Restore a model instance from a saved checkpoint.

        Args:
            pretrained_checkpoint: a `.pt` file (flat Hugging Face layout or `<run>/checkpoints/*.pt`), a checkpoint
                directory, or a Hugging Face repo id (see `pace.model.checkpoint_io`).

        Returns:
            The framework model with weights loaded strictly (left on CPU/GPU as built; caller decides device).
        """
        model_config, norm_stats, resolved = read_checkpoint_meta(pretrained_checkpoint, require_weights=True)
        logger.info(f"Loading checkpoint `{resolved.weights}` (layout: {resolved.layout})")

        model_config = dict_to_namespace(model_config)
        model_config.trainer.pretrained_checkpoint = None
        framework_model = build_framework(cfg=model_config)
        # set for action un-normalization
        framework_model.norm_stats = norm_stats
        model_state_dict = torch.load(resolved.weights, map_location="cpu")
        model_keys = set(framework_model.state_dict().keys())
        checkpoint_keys = set(model_state_dict.keys())
        try:
            framework_model.load_state_dict(model_state_dict, strict=True)
        except RuntimeError:
            common_keys = model_keys.intersection(checkpoint_keys)
            missing_keys = model_keys - common_keys
            unexpected_keys = checkpoint_keys - common_keys
            if missing_keys:
                logger.warning(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")
            raise
        return framework_model

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Infer or validate the dataset stats key used for un-normalization.

        Args:
            norm_stats: Dict[str, dict] mapping dataset key -> stats block.
            unnorm_key: Optional explicit dataset key.

        Returns:
            str: Resolved key.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                "Your model was trained on more than one dataset, "
                "please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            "The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_stats(self, unnorm_key=None, norm_stats=None):
        """Return the action normalization statistics for `unnorm_key`."""
        if norm_stats is None:
            norm_stats = self.norm_stats
        unnorm_key = self._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @property
    def trainable_module_keys(self, max_depth=1) -> List[str]:
        """Enumerate trainable submodule names up to a depth."""
        return auto_get_trainable_modules(self, max_depth=max_depth)

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Map normalized actions (~[-1, 1]) back to the original range with q01/q99 statistics
        (channel 6, the gripper, is thresholded to {0, 1}).
        """
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
