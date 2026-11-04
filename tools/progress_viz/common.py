"""
Shared helpers for the progress studies: the progress definition used here is the training-label definition of PACE
(frozen DINOv2 CLS features, L2-normalised; projection of the current frame onto the start->goal displacement,
clamped to [0, 1]).
"""

from __future__ import annotations

import argparse

import numpy as np


def add_encoder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dinov2_variant", default="vitb14", help="DINOv2 backbone (released models: vitb14)")
    parser.add_argument(
        "--dinov2_weights",
        default=None,
        help="dinov2_<variant>_pretrain.pth; default: env DINOV2_WEIGHTS or the $PACE_CACHE download",
    )
    parser.add_argument("--dinov2_repo_root", default=None, help="DINOv2 code root (default: third_party/dinov2)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=256, help="frames per DINOv2 forward pass")


class ProgressEncoder:
    """Frozen DINOv2 encoder + geometric progress (same code path as the PACE policy)."""

    def __init__(self, args: argparse.Namespace):
        import torch

        from pace.model.progress.dinov2_assets import resolve_dinov2_repo_root, resolve_dinov2_weights
        from pace.model.progress.dinov2_encoder import load_frozen_dinov2

        self.torch = torch
        self.device = torch.device(args.device)
        self.batch_size = int(args.batch_size)
        repo_root = resolve_dinov2_repo_root(args.dinov2_repo_root)
        weights = resolve_dinov2_weights(args.dinov2_variant, configured=args.dinov2_weights)
        self.model, self.preprocess = load_frozen_dinov2(
            repo_root, weights, variant=args.dinov2_variant, device=args.device
        )

    def encode(self, frames):
        """frames: sequence of (H, W, 3) uint8 arrays -> [N, D] L2-normalised CLS features."""
        from PIL import Image

        from pace.model.progress.dinov2_encoder import batched_dinov2_cls_features

        out = []
        for i in range(0, len(frames), self.batch_size):
            batch = [Image.fromarray(np.asarray(x, dtype=np.uint8)) for x in frames[i : i + self.batch_size]]
            out.append(batched_dinov2_cls_features(self.model, self.preprocess, batch, device=self.device))
        return self.torch.cat(out)

    def progress(self, z, z_start, z_goal) -> np.ndarray:
        """Geometric progress of every row of z w.r.t. z_start -> z_goal (both [1, D])."""
        from pace.model.progress.progress_modules import compute_goal_projection_progress

        return (
            compute_goal_projection_progress(z, z_start.expand_as(z), z_goal.expand_as(z)).squeeze(-1).cpu().numpy()
        )


def decode_video(path: str):
    import av

    with av.open(path) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
