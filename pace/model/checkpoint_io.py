"""
Checkpoint discovery shared by the policy server (`from_pretrained`) and the evaluation clients.

A checkpoint reference may be any of:
  (a) a `.pt` file with `config.yaml` + `dataset_statistics.json` in the same directory (flat Hugging Face layout)
  (b) a `.pt` file in the training layout `<run>/checkpoints/steps_N_pytorch_model.pt` (or `<run>/final_model/...`),
      with `config.yaml` + `dataset_statistics.json` in `<run>`
  (c) a directory containing `pytorch_model.pt` + `config.yaml` + `dataset_statistics.json`
      (a training run directory with `final_model/pytorch_model.pt` also works)
  (d) a Hugging Face model repo id, e.g. `org/PACE-LIBERO` (optionally `org/name@revision`), fetched with
      `huggingface_hub.snapshot_download` and then treated as (c)

Clients only need the config and the normalisation statistics, so they call `resolve_checkpoint(...,
require_weights=False)`, which downloads just those two files for a repo id.

This module has no torch dependency.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Tuple

CONFIG_NAME = "config.yaml"
STATS_NAME = "dataset_statistics.json"
WEIGHTS_NAME = "pytorch_model.pt"
DEFAULT_BASE_VLM = "Qwen/Qwen3-VL-4B-Instruct"
REPO_ROOT = Path(__file__).resolve().parents[2]

_HF_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*(@[A-Za-z0-9._/-]+)?$")


@dataclass(frozen=True)
class ResolvedCheckpoint:
    weights: Optional[Path]
    config: Path
    stats: Path
    layout: str  # "flat", "training", "directory" or "hf:<repo_id>"


def looks_like_hf_repo_id(ref: str) -> bool:
    ref = str(ref)
    if ref.startswith((".", "/", "~")) or ref.endswith(".pt") or os.path.exists(ref):
        return False
    return bool(_HF_REPO_ID.match(ref))


def _has_meta(directory: Path) -> bool:
    return (directory / CONFIG_NAME).is_file() and (directory / STATS_NAME).is_file()


def _resolve_pt_file(pt: Path) -> ResolvedCheckpoint:
    if pt.suffix != ".pt":
        raise ValueError(f"Expected a .pt checkpoint file, got {pt}")
    if _has_meta(pt.parent):
        return ResolvedCheckpoint(pt, pt.parent / CONFIG_NAME, pt.parent / STATS_NAME, "flat")
    if len(pt.parents) > 1 and _has_meta(pt.parents[1]):
        run_dir = pt.parents[1]
        return ResolvedCheckpoint(pt, run_dir / CONFIG_NAME, run_dir / STATS_NAME, "training")
    searched = [str(pt.parent)] + ([str(pt.parents[1])] if len(pt.parents) > 1 else [])
    raise FileNotFoundError(
        f"No {CONFIG_NAME} + {STATS_NAME} found next to {pt} (searched: {', '.join(searched)})"
    )


def _find_weights_in_dir(directory: Path) -> Optional[Path]:
    for candidate in (directory / WEIGHTS_NAME, directory / "final_model" / WEIGHTS_NAME):
        if candidate.is_file():
            return candidate
    pts = sorted(directory.glob("*.pt"))
    if len(pts) == 1:
        return pts[0]
    if len(pts) > 1:
        raise FileNotFoundError(
            f"{directory} contains several .pt files ({', '.join(p.name for p in pts)}); pass the file explicitly"
        )
    return None


def _resolve_directory(directory: Path, require_weights: bool, layout: str) -> ResolvedCheckpoint:
    if not _has_meta(directory):
        raise FileNotFoundError(f"{directory} does not contain {CONFIG_NAME} and {STATS_NAME}")
    weights = _find_weights_in_dir(directory)
    if require_weights and weights is None:
        raise FileNotFoundError(f"No {WEIGHTS_NAME} (or single *.pt) found in {directory}")
    return ResolvedCheckpoint(weights, directory / CONFIG_NAME, directory / STATS_NAME, layout)


def _default_snapshot_download(**kwargs) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def resolve_checkpoint(
    ref: str | os.PathLike,
    require_weights: bool = True,
    snapshot_download: Optional[Callable[..., str]] = None,
) -> ResolvedCheckpoint:
    """Resolve a checkpoint reference (see module docstring) to weights / config / stats paths."""
    ref_str = os.fspath(ref)
    path = Path(ref_str).expanduser()
    if path.is_file():
        return _resolve_pt_file(path)
    if path.is_dir():
        return _resolve_directory(path, require_weights, "directory")
    if looks_like_hf_repo_id(ref_str):
        repo_id, _, revision = ref_str.partition("@")
        patterns = [CONFIG_NAME, STATS_NAME] + (["*.pt"] if require_weights else [])
        download = snapshot_download or _default_snapshot_download
        local_dir = download(repo_id=repo_id, revision=revision or None, allow_patterns=patterns)
        return _resolve_directory(Path(local_dir), require_weights, f"hf:{repo_id}")
    raise FileNotFoundError(
        f"Checkpoint `{ref_str}` is neither an existing .pt file, a checkpoint directory, nor a Hugging Face repo id"
    )


def load_config_file(config_path: Path) -> dict:
    """Load a checkpoint config.yaml into a plain dict (OmegaConf interpolations are resolved if available)."""
    try:
        from omegaconf import OmegaConf
    except ImportError:  # clients may run without omegaconf
        import yaml

        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return OmegaConf.to_container(OmegaConf.load(str(config_path)), resolve=True)


def read_checkpoint_meta(
    ref: str | os.PathLike,
    require_weights: bool = False,
    snapshot_download: Optional[Callable[..., str]] = None,
) -> Tuple[dict, dict, ResolvedCheckpoint]:
    """Return (config dict, dataset statistics dict, resolved paths) for a checkpoint reference."""
    resolved = resolve_checkpoint(ref, require_weights=require_weights, snapshot_download=snapshot_download)
    cfg = load_config_file(resolved.config)
    with open(resolved.stats, "r") as f:
        norm_stats = json.load(f)
    return cfg, norm_stats, resolved


def resolve_base_vlm(configured: Optional[str], env: Optional[Mapping[str, str]] = None) -> str:
    """
    Decide where to load the Qwen3-VL base model from.

    1. env BASE_VLM (a local directory, or a Hugging Face repo id)
    2. the configured `framework.qwenvl.base_vlm` if it is an existing local directory
    3. otherwise the Hugging Face id `Qwen/Qwen3-VL-4B-Instruct`
    """
    env = os.environ if env is None else env
    env_value = env.get("BASE_VLM")
    if env_value:
        env_path = Path(env_value).expanduser()
        if env_path.is_dir():
            return str(env_path.resolve())
        if looks_like_hf_repo_id(env_value):
            return env_value
        raise FileNotFoundError(f"BASE_VLM={env_value} is neither a directory nor a Hugging Face repo id")
    if configured:
        configured_path = Path(str(configured)).expanduser()
        if configured_path.is_dir():
            return str(configured_path.resolve())
        repo_relative = REPO_ROOT / configured_path
        if not configured_path.is_absolute() and repo_relative.is_dir():
            return str(repo_relative.resolve())
    return DEFAULT_BASE_VLM
