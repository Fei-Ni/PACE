"""
Frozen DINOv2 encoder that defines PACE's progress geometry: L2-normalised CLS features of the primary camera view
(224x224, ImageNet normalisation). Shared by the policy (training labels, online progress) and tools/progress_viz.
"""

import os
import sys
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as tv_transforms


def module_param_dtype(module: nn.Module) -> torch.dtype:
    for param in module.parameters():
        return param.dtype
    for buffer in module.buffers():
        return buffer.dtype
    return torch.float32


DINOV2_VARIANT_DIMS: Dict[str, int] = {
    "vits14": 384,
    "vitb14": 768,
    "vitl14": 1024,
    "vitg14": 1536,
}

DINOV2_VARIANT_BUILDERS: Dict[str, str] = {
    "vits14": "dinov2_vits14",
    "vitb14": "dinov2_vitb14",
    "vitl14": "dinov2_vitl14",
    "vitg14": "dinov2_vitg14",
}


def normalize_dinov2_variant(variant: str) -> str:
    normalized = str(variant).strip()
    if normalized.startswith("dinov2_"):
        normalized = normalized[len("dinov2_") :]
    if normalized not in DINOV2_VARIANT_DIMS:
        valid = ", ".join(sorted(DINOV2_VARIANT_DIMS))
        raise ValueError(f"Unsupported DINOv2 variant {variant!r}; expected one of: {valid}")
    return normalized


def validate_dinov2_target_dim(variant: str, target_dim: int) -> None:
    normalized = normalize_dinov2_variant(variant)
    expected_dim = DINOV2_VARIANT_DIMS[normalized]
    if int(target_dim) != expected_dim:
        raise ValueError(
            f"DINOv2 {normalized} outputs {expected_dim} dims, but "
            f"framework.progress.dino_target_dim={target_dim}. "
            "Set dino_variant and dino_target_dim consistently."
        )


def load_frozen_dinov2(
    repo_root: str,
    weights_path: str,
    *,
    variant: str = "vitb14",
    device: str = "cuda",
):
    """Build a DINOv2 backbone from the (vendored) DINOv2 code, load the official weights and freeze it."""
    repo_root = os.path.expanduser(str(repo_root))
    weights_path = os.path.expanduser(str(weights_path))
    variant = normalize_dinov2_variant(variant)
    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"DINOv2 code root not found: {repo_root}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"DINOv2 weights not found: {weights_path}")

    # The DINOv2 code uses absolute `dinov2.*` imports, so its root has to be importable.
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from dinov2.hub import backbones as dinov2_backbones  # type: ignore

    builder_name = DINOV2_VARIANT_BUILDERS[variant]
    model = getattr(dinov2_backbones, builder_name)(pretrained=False)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    preprocess = tv_transforms.Compose(
        [
            tv_transforms.Resize((224, 224)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return model, preprocess


def select_primary_frames(batch_images: Sequence[Sequence[Image.Image]]) -> List[Image.Image]:
    primary_frames: List[Image.Image] = []
    for images in batch_images:
        if not images:
            raise ValueError("[PACE] Expected at least one image per sample for DINOv2 extraction.")
        primary_frames.append(images[0])
    return primary_frames


def batched_dinov2_cls_features(
    model,
    preprocess,
    frames: Sequence[Image.Image],
    *,
    device: torch.device,
) -> torch.Tensor:
    """L2-normalised DINOv2 CLS features of a list of frames."""
    if not frames:
        raise ValueError("[PACE] Empty frame batch for DINOv2 feature extraction.")
    model_dtype = module_param_dtype(model)
    tensor = torch.stack([preprocess(frame) for frame in frames], dim=0).to(device=device, dtype=model_dtype)
    with torch.inference_mode():
        feats = model.forward_features(tensor)["x_norm_clstoken"]
    return F.normalize(feats.float(), dim=-1)
