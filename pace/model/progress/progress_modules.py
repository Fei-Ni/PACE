"""Progress geometry shared by training (labels) and inference (observed progress)."""

from __future__ import annotations

import torch


def compute_goal_projection_progress(
    h_current: torch.Tensor,  # [B, D]
    h_start_anchor: torch.Tensor,  # [B, D]
    h_goal_anchor: torch.Tensor,  # [B, D]
    eps: float = 1e-8,
) -> torch.Tensor:  # [B, 1] in [0, 1]
    """
    Scalar projection progress along the start->goal displacement:

        delta_t = h_t - h_start
        delta_g = h_goal - h_start
        progress = clamp( <delta_t, delta_g> / ||delta_g||^2 , 0, 1 )

    In PACE the inputs are L2-normalised frozen DINOv2 CLS features. For a zero-norm goal displacement the
    progress is 0.
    """
    delta_t = h_current.float() - h_start_anchor.float()
    delta_g = h_goal_anchor.float() - h_start_anchor.float()
    goal_norm_sq = (delta_g * delta_g).sum(dim=-1, keepdim=True)
    out = torch.zeros(delta_t.shape[0], 1, device=delta_t.device, dtype=torch.float32)
    valid = goal_norm_sq.squeeze(-1) > eps
    if valid.any():
        numer = (delta_t[valid] * delta_g[valid]).sum(dim=-1, keepdim=True)
        out[valid] = (numer / goal_norm_sq[valid].clamp_min(eps)).clamp(0.0, 1.0)
    return out
