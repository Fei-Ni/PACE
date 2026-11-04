# DINOv2 (vendored subset)

Minimal, unmodified subset of https://github.com/facebookresearch/dinov2 needed to build the frozen DINOv2
backbones used by PACE (`dinov2.hub.backbones.dinov2_vitb14(pretrained=False)`):

- `dinov2/__init__.py`
- `dinov2/hub/__init__.py`, `dinov2/hub/backbones.py`, `dinov2/hub/utils.py`
- `dinov2/models/__init__.py`, `dinov2/models/vision_transformer.py`
- `dinov2/layers/*.py`

License: Apache License 2.0 ([LICENSE](LICENSE)), Copyright (c) Meta Platforms, Inc. and affiliates. The rest of the
upstream repository (training / evaluation code, notebooks, and the separately licensed Cell-DINO and X-Ray DINO
components) is not included.

The files are byte-identical (same MD5) to the DINOv2 copy used to train and evaluate the released PACE checkpoints,
which was taken from the upstream `main` branch in 2026. The attention layer uses
`torch.nn.functional.scaled_dot_product_attention`; if xFormers is installed, DINOv2 uses its memory-efficient
attention instead (xFormers was not installed in the environment used for the released models; set
`XFORMERS_DISABLED=1` to match it).

PACE adds this directory to `sys.path` when it builds the encoder. Set `DINO_REPO_ROOT` to use another checkout of
facebookresearch/dinov2 instead. The pretrained weights (`dinov2_vitb14_pretrain.pth`) are not vendored: they are
downloaded from https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth into
`$PACE_CACHE/dinov2/` (default `~/.cache/pace/dinov2/`) on first use, or taken from `DINOV2_WEIGHTS`.
