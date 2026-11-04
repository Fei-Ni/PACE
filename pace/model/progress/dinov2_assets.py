"""
Locate the frozen DINOv2 encoder used by PACE (code + weights).

Code (the `dinov2` package, absolute imports):
    1. env DINO_REPO_ROOT
    2. `framework.progress.dino_repo_root` if it is an existing directory
    3. the vendored subset in <repo>/third_party/dinov2 (requires a source checkout / `pip install -e .`)

Weights (`dinov2_<variant>_pretrain.pth`, official LVD-142M release):
    1. env DINOV2_WEIGHTS (DINO_WEIGHTS is accepted as a legacy alias); must exist
    2. `framework.progress.dino_weights` if it is an existing file
    3. <cache>/dinov2/dinov2_<variant>_pretrain.pth, downloaded from dl.fbaipublicfiles.com on first use;
       <cache> = env PACE_CACHE, default ~/.cache/pace

Environment variables take precedence over the config. This module is pure Python (no torch import).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Mapping, Optional

DINOV2_URL_TEMPLATE = "https://dl.fbaipublicfiles.com/dinov2/dinov2_{variant}/dinov2_{variant}_pretrain.pth"
REPO_ROOT = Path(__file__).resolve().parents[3]
VENDORED_DINOV2_ROOT = REPO_ROOT / "third_party" / "dinov2"


def default_cache_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get("PACE_CACHE") or Path.home() / ".cache" / "pace").expanduser()


def dinov2_weights_url(variant: str) -> str:
    return DINOV2_URL_TEMPLATE.format(variant=variant)


def dinov2_weights_filename(variant: str) -> str:
    return f"dinov2_{variant}_pretrain.pth"


def resolve_dinov2_repo_root(
    configured: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    vendored: Path = VENDORED_DINOV2_ROOT,
) -> str:
    env = os.environ if env is None else env
    env_root = env.get("DINO_REPO_ROOT")
    if env_root:
        path = Path(env_root).expanduser()
        if not (path / "dinov2").is_dir():
            raise FileNotFoundError(f"DINO_REPO_ROOT={env_root} does not contain a `dinov2` package")
        return str(path)
    if configured:
        path = Path(str(configured)).expanduser()
        if (path / "dinov2").is_dir():
            return str(path)
    if (Path(vendored) / "dinov2").is_dir():
        return str(vendored)
    raise FileNotFoundError(
        f"DINOv2 code not found (looked for {vendored}/dinov2). Use a source checkout (`pip install -e .`) "
        "or set DINO_REPO_ROOT to a facebookresearch/dinov2 checkout."
    )


def _urllib_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part", dir=str(dest.parent))
    try:
        with os.fdopen(fd, "wb") as tmp, urllib.request.urlopen(url) as response:  # noqa: S310 (fixed https URL)
            shutil.copyfileobj(response, tmp, length=1 << 20)
        os.replace(tmp_name, dest)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def resolve_dinov2_weights(
    variant: str,
    configured: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    cache_dir: Optional[Path] = None,
    download: Optional[Callable[[str, Path], None]] = None,
) -> str:
    env = os.environ if env is None else env
    for key in ("DINOV2_WEIGHTS", "DINO_WEIGHTS"):
        value = env.get(key)
        if value:
            path = Path(value).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"{key}={value} is not a file")
            return str(path)
    if configured:
        path = Path(str(configured)).expanduser()
        if path.is_file():
            return str(path)
    cache_dir = default_cache_dir(env) if cache_dir is None else Path(cache_dir)
    dest = cache_dir / "dinov2" / dinov2_weights_filename(variant)
    if dest.is_file():
        return str(dest)
    url = dinov2_weights_url(variant)
    print(f"[PACE] Downloading DINOv2 {variant} weights: {url} -> {dest}")
    (download or _urllib_download)(url, dest)
    if not dest.is_file():
        raise FileNotFoundError(f"DINOv2 weights download did not produce {dest}")
    return str(dest)
