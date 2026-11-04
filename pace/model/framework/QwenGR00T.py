# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
# Modified for PACE: trimmed to the base class used by `pace.model.framework.pace.PACE`.
"""
Qwen-GR00T base framework: Qwen3-VL + GR00T-style flow-matching action head.

PACE subclasses this class and overrides `forward` / `predict_action`; the base class builds the VLM interface and a
default flow-matching head (which PACE replaces with its token-conditioned head).
"""

from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from pace.model.framework.base_framework import baseframework
from pace.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from pace.model.modules.vlm import get_vlm_model
from pace.training.trainer_utils import initialize_overwatch
from pace.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


class Qwen_GR00T(baseframework):
    """
    Vision-language-action model: Qwen3-VL features conditioning a flow-matching DiT action head.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # The DiT cross-attends to VLM hidden states; its cross-attention width follows the VLM hidden size.
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ):
        batch_images = [example["image"] for example in examples]  # [B, [PIL]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # [B, len, action_dim]

        state = [example["state"] for example in examples] if "state" in examples[0] else None

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        with torch.autocast("cuda", dtype=torch.float32):
            # Align labels: keep the last chunk_len steps.
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -(self.future_action_window_size + 1) :, :]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(last_hidden_repeated, actions_target_repeated, state_repeated)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs: str,
    ) -> dict:
        """
        Sample a normalized action chunk with the flow-matching head.

        Returns:
            dict: normalized_actions (np.ndarray) of shape [B, T, action_dim].
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )
        do_sample = kwargs.get("do_sample", True)
        if isinstance(do_sample, str):
            do_sample = do_sample.lower() in {"1", "true", "yes"}
        sample_seed = kwargs.get("sample_seed")

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden,
                state,
                do_sample=do_sample,
                sample_seed=sample_seed,
            )

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}
