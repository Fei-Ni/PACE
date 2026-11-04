"""
PACE: a progress-conditioned vision-language-action policy.

The policy reads three progress signals from K + 2 learnable query tokens of a Qwen3-VL backbone and a scalar
history token, and conditions a flow-matching action DiT on them:

  - foresight anchor   K goal queries (mean-pooled) -> z_goal_pred, supervised in frozen DINOv2 CLS space
  - current progress   q_pcur  -> p_cur_pred
  - forward commitment q_dpcur -> dp_cur_pred (predicted progress increment over the next Delta steps)
  - historical pace    dp_prev, the realised increment over the previous Delta steps, enters the VLM input as an
                       extra token (observed from frozen DINOv2 features at inference time)

The action DiT receives detached (p_cur_pred, dp_cur_pred[, z_goal_pred]) as prepended conditioning tokens produced
by zero-initialised MLPs. At inference a per-episode commitment audit compares the realised increment with the
commitment made Delta steps earlier and runs an online failure detector on it (shortfall CUSUM + progress stall, see
`pace.model.progress.audit`), reported as `audit_score`, `audit_stall` and `audit_flag` (monitoring only, actions are
unchanged).

The progress target is the geometric projection of the current frame onto the start->goal displacement in frozen,
L2-normalised DINOv2 CLS space, clamped to [0, 1] (see `pace.model.progress.progress_modules`).
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from pace.model.framework.QwenGR00T import Qwen_GR00T
from pace.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from pace.model.progress.audit import (
    DEFAULT_AUDIT_KAPPA,
    DEFAULT_AUDIT_STALL_CHUNKS,
    DEFAULT_AUDIT_THRESHOLD,
    update_audit,
)
from pace.model.progress.dinov2_assets import resolve_dinov2_repo_root, resolve_dinov2_weights
from pace.model.progress.dinov2_encoder import (
    batched_dinov2_cls_features,
    load_frozen_dinov2,
    module_param_dtype,
    normalize_dinov2_variant,
    select_primary_frames,
    validate_dinov2_target_dim,
)
from pace.model.progress.progress_head import LatentPredictionHead
from pace.model.progress.progress_modules import compute_goal_projection_progress
from pace.model.progress.query_token_utils import append_query_token_ids
from pace.model.tools import FRAMEWORK_REGISTRY
from pace.training.trainer_utils.trainer_tools import resize_images


# ---------------------------------------------------------------------------
# Action head that accepts prepended conditioning tokens
# ---------------------------------------------------------------------------


class ExtraTokenConditionedActionHead(FlowmatchingActionHead):
    """
    Variant of the flow-matching action head that accepts pre-projected
    conditioning tokens and prepends them before the future/action tokens.
    """

    def __init__(self, full_config):
        super().__init__(full_config)

    def _stack_extra_tokens(
        self,
        extra_prepend_tokens: Sequence[torch.Tensor],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not extra_prepend_tokens:
            return None

        tokens = []
        for token in extra_prepend_tokens:
            if token is None:
                continue
            if token.ndim == 2:
                token = token.unsqueeze(1)
            if token.ndim != 3 or token.shape[1] != 1:
                raise ValueError(
                    "Each extra prepend token must have shape [B, D] or [B, 1, D], " f"got {tuple(token.shape)}"
                )
            tokens.append(token.to(device))

        if not tokens:
            return None
        return torch.cat(tokens, dim=1)

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        extra_prepend_tokens: Sequence[torch.Tensor],
        state: torch.Tensor = None,
    ):
        device = vl_embs.device
        noise = torch.randn(actions.shape, device=device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=device, dtype=actions.dtype)
        t = t[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()

        action_features = self.action_encoder(noisy_trajectory, t_discretized)
        state_features = self.state_encoder(state) if state is not None else None
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        prepend_tokens = self._stack_extra_tokens(extra_prepend_tokens, device)
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)

        seq_parts = []
        if state_features is not None:
            seq_parts.append(state_features)
        if prepend_tokens is not None:
            seq_parts.append(prepend_tokens)
        seq_parts.extend([future_tokens, action_features])
        sa_embs = torch.cat(seq_parts, dim=1)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]
        return ((pred_actions - velocity) ** 2).mean()

    @torch.no_grad()
    def predict_action_with_extra_tokens(
        self,
        vl_embs: torch.Tensor,
        extra_prepend_tokens: Sequence[torch.Tensor],
        state: torch.Tensor = None,
        do_sample: bool = True,
        sample_seed: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size = vl_embs.shape[0]
        device = vl_embs.device

        generator = None
        if sample_seed is not None or not do_sample:
            seed = 0 if sample_seed is None else int(sample_seed)
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
            generator=generator,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        state_features = self.state_encoder(state) if state is not None else None
        future_tokens_base = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        prepend_tokens = self._stack_extra_tokens(extra_prepend_tokens, device)

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full((batch_size,), fill_value=t_discretized, device=device)

            action_features = self.action_encoder(actions, timesteps_tensor)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            seq_parts = []
            if state_features is not None:
                seq_parts.append(state_features)
            if prepend_tokens is not None:
                seq_parts.append(prepend_tokens)
            seq_parts.extend([future_tokens_base, action_features])
            sa_embs = torch.cat(seq_parts, dim=1)

            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
                return_all_hidden_states=False,
            )
            pred = self.action_decoder(model_output)
            velocity_pred = pred[:, -self.config.action_horizon :]
            actions = actions + velocity_pred * dt

        return actions


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_layer_hidden(qwenvl_outputs, layer_idx: int):
    """
    Extract an intermediate hidden state from the VLM outputs.

    HuggingFace convention: hidden_states[0] = embedding output, hidden_states[i+1] = transformer layer i output,
    so layer_idx=18 reads index 19.
    """
    return qwenvl_outputs.hidden_states[layer_idx + 1]  # [B, seq_len, hidden_dim]


def _parse_do_sample(kwargs):
    do_sample = kwargs.get("do_sample", True)
    if isinstance(do_sample, str):
        do_sample = do_sample.lower() in {"1", "true", "yes"}
    return do_sample


def _repeat_actions_and_state(self_obj, last_hidden, examples, state):
    actions = [e["action"] for e in examples]
    actions_t = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
    actions_target = actions_t[:, -(self_obj.future_action_window_size + 1) :, :]
    # Number of noise/time samples per example in the flow-matching loss (trainer.repeated_diffusion_steps).
    K = self_obj.config.trainer.get("repeated_diffusion_steps", 4)
    actions_rep = actions_target.repeat(K, 1, 1)
    last_hidden_rep = last_hidden.repeat(K, 1, 1)
    state_rep = None
    if state is not None:
        state_rep = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
        state_rep = state_rep.repeat(K, 1, 1)
    return actions_target, actions_rep, last_hidden_rep, state_rep, K


def _extra_token_action_forward(self_obj, last_hidden, extra_tokens, examples, state):
    _, actions_rep, last_hidden_rep, state_rep, K = _repeat_actions_and_state(self_obj, last_hidden, examples, state)
    repeated_tokens = []
    for token in extra_tokens:
        if token is None:
            repeated_tokens.append(None)
            continue
        repeated_tokens.append(token.repeat(K, 1))
    return self_obj.action_model(last_hidden_rep, actions_rep, repeated_tokens, state_rep)


def _as_bool_flag(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _should_reset_online_start(kwargs) -> bool:
    if _as_bool_flag(kwargs.get("progress_reset_episode", False)):
        return True
    step = kwargs.get("step", None)
    if step is None:
        return False
    try:
        return int(step) == 0
    except (TypeError, ValueError):
        return False


def _zero_init_last_linear(module: nn.Module):
    last_linear = None
    for child in module.modules():
        if isinstance(child, nn.Linear):
            last_linear = child
    if last_linear is None:
        raise ValueError("Expected at least one Linear layer to zero-initialise.")
    nn.init.zeros_(last_linear.weight)
    nn.init.zeros_(last_linear.bias)


# ---------------------------------------------------------------------------
# Conditioning MLPs
# ---------------------------------------------------------------------------


class VectorTokenConditionBridge(nn.Module):
    """
    Map a vector (the predicted foresight anchor in frozen-DINO space) into a DiT conditioning token with a
    zero-initialised last layer, so the action head starts from its unconditioned behaviour.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = int(hidden_dim or output_dim)
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, int(output_dim)),
        )
        _zero_init_last_linear(self.net)

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        first_linear = self.net[0]
        vector = vector.to(device=first_linear.weight.device, dtype=first_linear.weight.dtype)
        return self.net(vector)


class ScalarTokenConditionBridge(nn.Module):
    """
    Map a scalar into a token embedding with a zero-initialised last layer.

    Used for the DiT conditioning tokens (p_cur, dp_cur) and for the observed `dp_prev` token that is added to the
    VLM input sequence.
    """

    def __init__(self, output_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = int(hidden_dim or output_dim)
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        _zero_init_last_linear(self.net)

    def forward(self, scalar: torch.Tensor) -> torch.Tensor:
        if scalar.ndim == 1:
            scalar = scalar.unsqueeze(-1)
        first_linear = self.net[0]
        scalar = scalar.to(
            device=first_linear.weight.device,
            dtype=first_linear.weight.dtype,
        )
        return self.net(scalar).float()


# ---------------------------------------------------------------------------
# PACE
# ---------------------------------------------------------------------------


@FRAMEWORK_REGISTRY.register("PACE")
class PACE(Qwen_GR00T):
    """
    PACE policy: Qwen3-VL backbone + progress query tokens + progress-conditioned flow-matching action DiT.

    Readouts (hidden states of layer `layer_idx`):
      - K goal queries (mean-pooled) -> z_goal_pred (L2-normalised, frozen DINOv2 CLS space)
      - `<|q_pcur|>`  -> p_cur_pred
      - `<|q_dpcur|>` -> dp_cur_pred
    Inputs besides the current observation: one history frame at t - Delta and the `<|dp_prev|>` token carrying the
    realised progress increment over [t - Delta, t].

    Module attribute names (`long_target_head`, `pcur_head`, `dpcur_head`, `dp_prev_token_bridge`, `pcur_to_dit`,
    `dpcur_to_dit`, `goal_to_dit`, `dino_model`) and the special-token order are part of the checkpoint format.
    """

    QUERY_LONG_TOKEN = "<|q_long|>"
    QUERY_PCUR_TOKEN = "<|q_pcur|>"
    QUERY_DPCUR_TOKEN = "<|q_dpcur|>"
    DP_PREV_TOKEN = "<|dp_prev|>"

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        progress_cfg = getattr(config.framework, "progress", None)
        self.layer_idx = int(getattr(progress_cfg, "layer_idx", 18))
        self.lambda_long = float(getattr(progress_cfg, "lambda_long", 0.1))
        self.lambda_pcur = float(getattr(progress_cfg, "lambda_pcur", 0.1))
        self.lambda_dpcur = float(getattr(progress_cfg, "lambda_dpcur", 0.1))
        self.query_init_std = float(getattr(progress_cfg, "query_init_std", 0.01))
        self.train_query_token_rows_only = bool(getattr(progress_cfg, "train_query_token_rows_only", True))
        self.dino_variant = normalize_dinov2_variant(getattr(progress_cfg, "dino_variant", "vitb14"))
        self.dino_target_dim = int(getattr(progress_cfg, "dino_target_dim", 768))
        validate_dinov2_target_dim(self.dino_variant, self.dino_target_dim)
        self.dino_head_hidden_dim = int(getattr(progress_cfg, "dino_head_hidden_dim", 1024))
        # Delta: history / commitment horizon in environment steps (the dataloader uses the same offset).
        self.future_step_offset = int(getattr(progress_cfg, "progress_future_offset", 7))
        self.use_history_images = bool(getattr(progress_cfg, "use_history_images", True))
        self.use_dp_prev_input = bool(getattr(progress_cfg, "use_dp_prev_input", True))
        if not bool(getattr(progress_cfg, "use_full_goal_anchor_at_inference", True)):
            raise ValueError(
                f"[{self.__class__.__name__}] PACE measures online progress against the full predicted foresight "
                "anchor; framework.progress.use_full_goal_anchor_at_inference=false is not supported."
            )
        #   num_goal_queries   K goal query tokens, mean-pooled into the anchor head
        #   use_goal_dit_token also feed the (detached) foresight anchor to the DiT as a conditioning token
        #   lambda_cons        consistency loss between the p_cur readout and the projection onto the predicted anchor
        #   infer_history_mode "step_aligned": history frame / dp_prev window chosen by environment step (t - Delta),
        #                      matching the training definition regardless of how often the client queries
        self.num_goal_queries = max(1, int(getattr(progress_cfg, "num_goal_queries", 4)))
        self.use_goal_dit_token = bool(getattr(progress_cfg, "use_goal_dit_token", True))
        self.lambda_cons = float(getattr(progress_cfg, "lambda_cons", 0.1))
        self.cons_detach_anchor = bool(getattr(progress_cfg, "cons_detach_anchor", True))
        self.infer_history_mode = str(getattr(progress_cfg, "infer_history_mode", "step_aligned")).lower()
        if self.infer_history_mode != "step_aligned":
            raise ValueError(
                f"[{self.__class__.__name__}] infer_history_mode must be 'step_aligned', got {self.infer_history_mode!r}"
            )
        # Commitment audit (monitoring only): alarm when the shortfall CUSUM exceeds audit_threshold or p_cur has not
        # risen for more than audit_stall_chunks action chunks (<= 0 disables the stall signal).
        self.audit_kappa = float(getattr(progress_cfg, "audit_kappa", DEFAULT_AUDIT_KAPPA))
        self.audit_threshold = float(getattr(progress_cfg, "audit_threshold", DEFAULT_AUDIT_THRESHOLD))
        self.audit_stall_chunks = float(getattr(progress_cfg, "audit_stall_chunks", DEFAULT_AUDIT_STALL_CHUNKS))

        self.dino_repo_root = resolve_dinov2_repo_root(getattr(progress_cfg, "dino_repo_root", None))
        self.dino_weights = resolve_dinov2_weights(
            self.dino_variant,
            configured=getattr(progress_cfg, "dino_weights", None),
        )

        # K=1 keeps the single <|q_long|> token.
        if self.num_goal_queries == 1:
            goal_query_tokens = [self.QUERY_LONG_TOKEN]
        else:
            goal_query_tokens = [f"<|q_goal_{idx}|>" for idx in range(self.num_goal_queries)]
        token_ids = self.qwen_vl_interface.ensure_special_tokens(
            goal_query_tokens
            + [
                self.QUERY_PCUR_TOKEN,
                self.QUERY_DPCUR_TOKEN,
                self.DP_PREV_TOKEN,
            ],
            init_std=self.query_init_std,
        )
        if self.train_query_token_rows_only and hasattr(self.qwen_vl_interface, "set_trainable_token_rows"):
            self.qwen_vl_interface.set_trainable_token_rows(token_ids)
        token_ids = [int(token_id) for token_id in token_ids]
        num_goal = self.num_goal_queries
        self.goal_query_ids = token_ids[:num_goal]
        self.q_long_id = self.goal_query_ids[0]
        self.q_pcur_id, self.q_dpcur_id, self.dp_prev_token_id = token_ids[num_goal:]
        # Rows that must keep their fresh initialisation when a checkpoint with a
        # different vocabulary size is row-merged at warm start.
        self.progress_special_token_ids = list(token_ids)

        d_vlm = int(self.qwen_vl_interface.model.config.hidden_size)
        self.long_target_head = LatentPredictionHead(
            input_dim=d_vlm,
            output_dim=self.dino_target_dim,
            hidden_dim=self.dino_head_hidden_dim,
            normalize_output=True,
        )
        self.pcur_head = nn.Linear(d_vlm, 1)
        self.dpcur_head = nn.Linear(d_vlm, 1)
        self.dp_prev_token_bridge = ScalarTokenConditionBridge(d_vlm, hidden_dim=d_vlm)

        self.action_model = ExtraTokenConditionedActionHead(full_config=config)
        d_dit = int(self.action_model.input_embedding_dim)
        self.pcur_to_dit = ScalarTokenConditionBridge(d_dit, hidden_dim=d_dit)
        self.dpcur_to_dit = ScalarTokenConditionBridge(d_dit, hidden_dim=d_dit)
        self.goal_to_dit = (
            VectorTokenConditionBridge(self.dino_target_dim, d_dit, hidden_dim=d_dit)
            if self.use_goal_dit_token
            else None
        )
        vlm_device = next(self.qwen_vl_interface.model.parameters()).device
        if self.goal_to_dit is not None:
            self.goal_to_dit.to(vlm_device)
        self.long_target_head.to(vlm_device)
        self.pcur_head.to(vlm_device)
        self.dpcur_head.to(vlm_device)
        self.dp_prev_token_bridge.to(vlm_device)
        self.action_model.to(vlm_device)
        self.pcur_to_dit.to(vlm_device)
        self.dpcur_to_dit.to(vlm_device)

        self.dino_model, self.dino_preprocess = load_frozen_dinov2(
            self.dino_repo_root,
            self.dino_weights,
            variant=self.dino_variant,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self._infer_rollout_state: Dict[str, Dict[str, object]] = {}

        print(
            f"[{self.__class__.__name__}] dino_variant={self.dino_variant}, "
            f"dino_dim={self.dino_target_dim}, "
            f"lambda_long={self.lambda_long}, lambda_pcur={self.lambda_pcur}, "
            f"lambda_dpcur={self.lambda_dpcur}, history={self.use_history_images}, "
            f"dp_prev_token={self.use_dp_prev_input}, "
            f"future_offset={self.future_step_offset}, "
            f"layer={self.layer_idx}, goal_queries={self.num_goal_queries}, "
            f"goal_dit_token={self.use_goal_dit_token}, lambda_cons={self.lambda_cons}, "
            f"infer_history_mode={self.infer_history_mode}, "
            f"audit_threshold={self.audit_threshold}, audit_stall_chunks={self.audit_stall_chunks}, "
            f"audit_kappa={self.audit_kappa}"
        )

    def _require_anchor_images(self, examples, key: str):
        if key not in examples[0]:
            raise ValueError(
                f"[{self.__class__.__name__}] Batch missing {key!r}. "
                "Set datasets.vla_data.return_progress_anchors=true so the dataloader "
                "emits prev/start/goal/future frames."
            )
        return [e[key] for e in examples]

    def _encode_dino_primary_groups(
        self,
        batch_image_groups: Sequence[Sequence[Sequence[Image.Image]]],
        device: torch.device,
    ) -> List[torch.Tensor]:
        flat_frames: List[Image.Image] = []
        lengths: List[int] = []
        for batch_images in batch_image_groups:
            frames = select_primary_frames(batch_images)
            lengths.append(len(frames))
            flat_frames.extend(frames)

        if not flat_frames:
            raise ValueError(f"[{self.__class__.__name__}] Empty image group for frozen DINO encoding.")

        all_feats = batched_dinov2_cls_features(
            self.dino_model,
            self.dino_preprocess,
            flat_frames,
            device=device,
        )
        outputs: List[torch.Tensor] = []
        start = 0
        for length in lengths:
            outputs.append(all_feats[start : start + length])
            start += length
        return outputs

    def _combine_history_and_current(
        self,
        prev_images: Sequence[Sequence[Image.Image]],
        current_images: Sequence[Sequence[Image.Image]],
    ) -> List[List[Image.Image]]:
        if not self.use_history_images:
            return [list(images) for images in current_images]

        merged: List[List[Image.Image]] = []
        for prev_row, current_row in zip(prev_images, current_images):
            merged.append(list(prev_row) + list(current_row))
        return merged

    def _build_qwenvl_inputs_with_progress_context(
        self,
        current_images: Sequence[Sequence[Image.Image]],
        prev_images: Sequence[Sequence[Image.Image]],
        instructions: Sequence[str],
        dp_prev_scalar: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], int, int]:
        model_images = self._combine_history_and_current(prev_images, current_images)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=model_images,
            instructions=instructions,
        )

        tail_token_ids = list(self.goal_query_ids) + [self.q_pcur_id, self.q_dpcur_id]
        query_offset = 0
        if self.use_dp_prev_input:
            tail_token_ids = [self.dp_prev_token_id] + tail_token_ids
            query_offset = 1
        qwen_inputs, base_seq_len = append_query_token_ids(qwen_inputs, tail_token_ids)

        if self.use_dp_prev_input:
            input_ids = qwen_inputs["input_ids"]
            embed_tokens = self.qwen_vl_interface.model.get_input_embeddings()
            inputs_embeds = embed_tokens(input_ids).clone()
            dp_prev_embed = self.dp_prev_token_bridge(dp_prev_scalar).to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            inputs_embeds[:, base_seq_len, :] = inputs_embeds[:, base_seq_len, :] + dp_prev_embed
            position_ids, _ = self.qwen_vl_interface.model.model.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=qwen_inputs.get("image_grid_thw"),
                video_grid_thw=qwen_inputs.get("video_grid_thw"),
                attention_mask=qwen_inputs.get("attention_mask"),
            )
            qwen_inputs["position_ids"] = position_ids
            qwen_inputs["inputs_embeds"] = inputs_embeds
            qwen_inputs.pop("input_ids", None)

        return qwen_inputs, base_seq_len, query_offset

    def _forward_vlm_with_progress_context(
        self,
        current_images: Sequence[Sequence[Image.Image]],
        prev_images: Sequence[Sequence[Image.Image]],
        instructions: Sequence[str],
        dp_prev_scalar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qwen_inputs, base_seq_len, query_offset = self._build_qwenvl_inputs_with_progress_context(
            current_images,
            prev_images,
            instructions,
            dp_prev_scalar,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
            hidden_readout = _get_layer_hidden(qwenvl_outputs, self.layer_idx)

        query_start = base_seq_len + query_offset
        num_goal = self.num_goal_queries
        # K goal queries are mean-pooled into one anchor representation (K=1 reads the single q_long position).
        h_long = hidden_readout[:, query_start : query_start + num_goal, :].float().mean(dim=1)
        h_pcur = hidden_readout[:, query_start + num_goal, :].float()
        h_dpcur = hidden_readout[:, query_start + num_goal + 1, :].float()
        last_hidden_base = last_hidden[:, :base_seq_len, :]
        return last_hidden_base, h_long, h_pcur, h_dpcur

    def _build_action_condition_tokens(
        self,
        p_cur: torch.Tensor,
        dp_cur: torch.Tensor,
        z_goal: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        pcur_dtype = module_param_dtype(self.pcur_to_dit)
        dpcur_dtype = module_param_dtype(self.dpcur_to_dit)
        tokens = [
            self.pcur_to_dit(p_cur.detach().to(dtype=pcur_dtype)),
            self.dpcur_to_dit(dp_cur.detach().to(dtype=dpcur_dtype)),
        ]
        if self.goal_to_dit is not None:
            if z_goal is None:
                raise ValueError(f"[{self.__class__.__name__}] use_goal_dit_token=True requires z_goal.")
            goal_dtype = module_param_dtype(self.goal_to_dit)
            tokens.append(self.goal_to_dit(z_goal.detach().to(dtype=goal_dtype)))
        return tokens

    def _dino_device(self) -> torch.device:
        return next(self.dino_model.parameters()).device

    def _compute_observed_progress_scalar(
        self,
        z_cur: torch.Tensor,
        z0: torch.Tensor,
        z_goal_anchor: torch.Tensor,
    ) -> float:
        return float(
            compute_goal_projection_progress(
                z_cur.unsqueeze(0),
                z0.to(z_cur.device).unsqueeze(0),
                z_goal_anchor.to(z_cur.device).unsqueeze(0),
            )
            .squeeze()
            .item()
        )

    def reset_rollout_state(self) -> None:
        """Forget all per-episode inference state (history frames, anchors, audit detector)."""
        self._infer_rollout_state = {}

    def forward(self, examples=None, **kwargs):
        current_images = [e["image"] for e in examples]
        prev_images = self._require_anchor_images(examples, "prev_image")
        start_images = self._require_anchor_images(examples, "start_image")
        goal_images = self._require_anchor_images(examples, "goal_image")
        future_images = self._require_anchor_images(examples, "future_image")
        instructions = [e["lang"] for e in examples]
        state = [e["state"] for e in examples] if "state" in examples[0] else None

        with torch.no_grad():
            z_prev_target, z_start_target, z_current_target, z_future_target, z_goal_target = (
                self._encode_dino_primary_groups(
                    [prev_images, start_images, current_images, future_images, goal_images],
                    self._dino_device(),
                )
            )
            p_prev_gt = compute_goal_projection_progress(
                z_prev_target,
                z_start_target,
                z_goal_target,
            )
            p_cur_gt = compute_goal_projection_progress(
                z_current_target,
                z_start_target,
                z_goal_target,
            )
            p_future_gt = compute_goal_projection_progress(
                z_future_target,
                z_start_target,
                z_goal_target,
            )
            dp_prev_gt = p_cur_gt - p_prev_gt
            dp_cur_gt = p_future_gt - p_cur_gt

        last_hidden, h_long, h_pcur, h_dpcur = self._forward_vlm_with_progress_context(
            current_images,
            prev_images,
            instructions,
            dp_prev_gt,
        )
        if z_goal_target.device != last_hidden.device:
            z_goal_target = z_goal_target.to(last_hidden.device)
            p_cur_gt = p_cur_gt.to(last_hidden.device)
            dp_cur_gt = dp_cur_gt.to(last_hidden.device)
        long_head_dtype = module_param_dtype(self.long_target_head)
        pcur_head_dtype = module_param_dtype(self.pcur_head)
        dpcur_head_dtype = module_param_dtype(self.dpcur_head)
        z_goal_pred = self.long_target_head(h_long.to(dtype=long_head_dtype)).float()
        p_cur_pred = self.pcur_head(h_pcur.to(dtype=pcur_head_dtype)).float()
        dp_cur_pred = self.dpcur_head(h_dpcur.to(dtype=dpcur_head_dtype)).float()

        loss_long = F.mse_loss(z_goal_pred, z_goal_target.detach())
        loss_pcur = F.mse_loss(p_cur_pred, p_cur_gt.detach())
        loss_dpcur = F.mse_loss(dp_cur_pred, dp_cur_gt.detach())

        # Consistency: the deployed policy only sees its own predicted anchor, so tie the p_cur readout to the
        # projection onto that anchor.
        if self.lambda_cons > 0.0:
            anchor = z_goal_pred.detach() if self.cons_detach_anchor else z_goal_pred
            p_anchor = compute_goal_projection_progress(
                z_current_target.to(anchor.device),
                z_start_target.to(anchor.device),
                anchor,
            )
            if self.cons_detach_anchor:
                p_anchor = p_anchor.detach()
            loss_cons = F.mse_loss(p_cur_pred, p_anchor)
        else:
            p_anchor = None
            loss_cons = torch.zeros((), device=p_cur_pred.device, dtype=torch.float32)

        condition_tokens = self._build_action_condition_tokens(
            p_cur_pred,
            dp_cur_pred,
            z_goal_pred,
        )

        with torch.autocast("cuda", dtype=torch.float32):
            action_loss = _extra_token_action_forward(
                self,
                last_hidden,
                condition_tokens,
                examples,
                state,
            )
            total_loss = (
                action_loss
                + self.lambda_long * loss_long
                + self.lambda_pcur * loss_pcur
                + self.lambda_dpcur * loss_dpcur
                + self.lambda_cons * loss_cons
            )

        extra_metrics = {}
        if p_anchor is not None:
            extra_metrics["pcur_anchor_mean"] = p_anchor.mean()
        return {
            **extra_metrics,
            "action_loss": total_loss,
            "_action_dit_loss": action_loss,
            "loss_long": loss_long,
            "loss_pcur": loss_pcur,
            "loss_dpcur": loss_dpcur,
            "loss_cons": loss_cons,
            "cos_long": F.cosine_similarity(z_goal_pred, z_goal_target, dim=-1).mean(),
            "pcur_gt_mean": p_cur_gt.mean(),
            "pcur_pred_mean": p_cur_pred.mean(),
            "pcur_mae": (p_cur_pred - p_cur_gt).abs().mean(),
            "dpcur_gt_mean": dp_cur_gt.mean(),
            "dpcur_pred_mean": dp_cur_pred.mean(),
            "dpcur_mae": (dp_cur_pred - dp_cur_gt).abs().mean(),
            "dpcur_neg_frac": (dp_cur_gt < 0).float().mean(),
            "dp_prev_gt_mean": dp_prev_gt.mean(),
            "future_step_offset": torch.tensor(
                float(self.future_step_offset),
                device=last_hidden.device,
                dtype=torch.float32,
            ),
        }

    @torch.inference_mode()
    def predict_action(self, batch_images, instructions, state=None, **kwargs):
        """
        One policy query.

        Online state is kept per instruction and reset when the client sends `progress_reset_episode=True` or
        `step=0`. Clients should send the environment step (`step`) with every query; the history frame and the
        observed dp_prev are then taken at step t - Delta regardless of how often the client queries.

        Returns a dict of numpy arrays (batch dimension first):
            normalized_actions        [B, H, action_dim]  action chunk in normalised action space
            predicted_progress        [B, 1]  p_cur readout (current progress)
            predicted_delta_progress  [B, 1]  dp_cur readout (forward commitment over the next Delta steps)
            observed_progress_dino    [B, 1]  projection of the current frame onto start -> previous anchor
            observed_dp_prev          [B, 1]  realised increment over [t - Delta, t] fed to the VLM (historical pace)
            progress_audit_delta      [B, 1]  audit residual: realised increment - commitment made at t - Delta
                                              (NaN when no commitment is available yet)
            audit_score               [B, 1]  shortfall CUSUM of the audit residuals (weighted by action chunks)
            audit_stall               [B, 1]  action chunks since p_cur last rose by more than 0.02
            audit_flag                [B, 1]  alarm: audit_score > audit_threshold or audit_stall > audit_stall_chunks
        """
        train_obs = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs:
            batch_images = resize_images(batch_images, train_obs)

        reset_episode = _should_reset_online_start(kwargs)
        current_images = [list(images) for images in batch_images]
        current_dino = self._encode_dino_primary_groups([current_images], self._dino_device())[0]
        raw_step = kwargs.get("step", None)
        try:
            client_step = None if raw_step is None else int(raw_step)
        except (TypeError, ValueError):
            client_step = None
        try:
            update_stride = max(int(kwargs.get("progress_update_stride", 1) or 1), 1)
        except (TypeError, ValueError):
            update_stride = 1

        prev_images: List[List[Image.Image]] = []
        dp_prev_values: List[float] = []
        audit_values: List[float] = []
        observed_progress_values: List[float] = []
        prior_states: List[Optional[Dict[str, object]]] = []
        env_steps: List[int] = []
        for task_text, image_row, z_cur in zip(instructions, current_images, current_dino):
            key = str(task_text)
            prior = None if reset_episode else self._infer_rollout_state.get(key)
            prior_states.append(prior)

            if client_step is not None:
                env_step = client_step
            elif prior is not None:
                env_step = int(prior.get("last_env_step", 0)) + update_stride
            else:
                env_step = 0
            env_steps.append(env_step)

            if prior is None:
                prev_images.append(list(image_row))
                dp_prev_values.append(0.0)
                audit_values.append(float("nan"))
                observed_progress_values.append(0.0)
                continue

            z_goal_anchor = prior.get("goal_anchor_full", None)
            z0 = prior["z0_dino"].detach().cpu().float().clone()

            # Same window as training: history frame / dp_prev at t - Delta
            # (clamped to the episode start), independent of query cadence.
            history = prior.get("history", [])
            target_step = max(env_step - self.future_step_offset, 0)
            ref = None
            for entry in history:
                if int(entry["env_step"]) <= target_step:
                    ref = entry
            if ref is None and history:
                ref = history[0]
            if ref is None:
                prev_images.append(list(prior["last_images"]))
            else:
                prev_images.append(list(ref["images"]))
            if z_goal_anchor is None or ref is None:
                dp_prev_values.append(0.0)
                observed_progress_values.append(0.0)
                audit_values.append(float("nan"))
                continue
            z_goal = z_goal_anchor.detach().cpu().float().clone()
            p_cur_dino = self._compute_observed_progress_scalar(z_cur, z0, z_goal)
            p_ref_dino = self._compute_observed_progress_scalar(ref["z_cur"].to(z_cur.device), z0, z_goal)
            observed_progress_values.append(p_cur_dino)
            dp_prev_values.append(p_cur_dino - p_ref_dino)
            ref_commit = ref.get("dpcur_pred", None)
            audit_values.append(float("nan") if ref_commit is None else dp_prev_values[-1] - float(ref_commit))

        dp_prev_tensor = torch.tensor(
            dp_prev_values,
            device=current_dino.device,
            dtype=torch.float32,
        ).unsqueeze(-1)
        last_hidden, h_long, h_pcur, h_dpcur = self._forward_vlm_with_progress_context(
            current_images,
            prev_images,
            instructions,
            dp_prev_tensor,
        )
        long_head_dtype = module_param_dtype(self.long_target_head)
        pcur_head_dtype = module_param_dtype(self.pcur_head)
        dpcur_head_dtype = module_param_dtype(self.dpcur_head)
        z_goal_pred = self.long_target_head(h_long.to(dtype=long_head_dtype)).float()
        p_cur_pred = self.pcur_head(h_pcur.to(dtype=pcur_head_dtype)).float()
        dp_cur_pred = self.dpcur_head(h_dpcur.to(dtype=dpcur_head_dtype)).float()
        condition_tokens = self._build_action_condition_tokens(
            p_cur_pred,
            dp_cur_pred,
            z_goal_pred,
        )

        state_t = None
        if state is not None:
            state_t = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
        do_sample = _parse_do_sample(kwargs)
        sample_seed = kwargs.get("sample_seed")

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action_with_extra_tokens(
                last_hidden,
                condition_tokens,
                state_t,
                do_sample=do_sample,
                sample_seed=sample_seed,
            )

        # Update the per-episode online state (does not affect the actions returned by this call).
        audit_scores: List[float] = []
        audit_stalls: List[float] = []
        audit_flags: List[bool] = []
        for idx, (task_text, image_row, z_cur, z_goal_row, dp_cur_row, prior) in enumerate(
            zip(instructions, current_images, current_dino, z_goal_pred, dp_cur_pred, prior_states)
        ):
            key = str(task_text)
            if prior is None:
                z0 = z_cur.detach().cpu().float().clone()
            else:
                z0 = prior["z0_dino"].detach().cpu().float().clone()
            history = [] if prior is None else list(prior.get("history", []))
            history.append(
                {
                    "env_step": int(env_steps[idx]),
                    "images": list(image_row),
                    "z_cur": z_cur.detach().cpu().float().clone(),
                    "dpcur_pred": float(dp_cur_row.item()),
                }
            )
            audit_state, audit_score, audit_stall, audit_flag = update_audit(
                None if prior is None else prior.get("audit"),
                env_steps[idx],
                audit_values[idx],
                float(p_cur_pred[idx].item()),
                chunk_steps=self.future_step_offset + 1,
                kappa=self.audit_kappa,
                threshold=self.audit_threshold,
                stall_chunks=self.audit_stall_chunks,
            )
            audit_scores.append(audit_score)
            audit_stalls.append(audit_stall)
            audit_flags.append(audit_flag)
            self._infer_rollout_state[key] = {
                "z0_dino": z0,
                "last_images": list(image_row),
                "last_env_step": int(env_steps[idx]),
                # Full predicted foresight anchor of this query; the next query measures progress against it.
                "goal_anchor_full": z_goal_row.detach().cpu().float().clone(),
                # Keep enough entries to reach back Delta steps at any query cadence
                # (one entry per query; per-step clients need Delta + 1).
                "history": history[-(self.future_step_offset + 2) :],
                "audit": audit_state,
            }

        return {
            "normalized_actions": pred_actions.detach().cpu().numpy(),
            "predicted_progress": p_cur_pred.detach().cpu().float().numpy(),
            "predicted_delta_progress": dp_cur_pred.detach().cpu().float().numpy(),
            "observed_progress_dino": np.asarray(observed_progress_values, dtype=np.float32)[:, None],
            "observed_dp_prev": np.asarray(dp_prev_values, dtype=np.float32)[:, None],
            "progress_audit_delta": np.asarray(audit_values, dtype=np.float32)[:, None],
            "audit_score": np.asarray(audit_scores, dtype=np.float32)[:, None],
            "audit_stall": np.asarray(audit_stalls, dtype=np.float32)[:, None],
            "audit_flag": np.asarray(audit_flags, dtype=bool)[:, None],
        }
