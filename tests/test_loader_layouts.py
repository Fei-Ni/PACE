"""
Checkpoint / asset resolution used by `from_pretrained` and the evaluation clients (pure Python, no torch):
flat Hugging Face layout, training layout, checkpoint directory, Hugging Face repo id; base VLM and DINOv2 weights.
"""

import json
from pathlib import Path

import pytest

from pace.model import checkpoint_io
from pace.model.checkpoint_io import (
    DEFAULT_BASE_VLM,
    looks_like_hf_repo_id,
    read_checkpoint_meta,
    resolve_base_vlm,
    resolve_checkpoint,
)
from pace.model.progress import dinov2_assets

CONFIG_TEXT = """
framework:
  name: PACE
  action_model:
    future_action_window_size: 7
trainer:
  max_train_steps: 10
"""
STATS = {"franka": {"action": {"min": [0.0] * 7, "max": [1.0] * 7}}}


def write_meta(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(CONFIG_TEXT)
    (directory / "dataset_statistics.json").write_text(json.dumps(STATS))


def test_flat_hf_layout_pt_file(tmp_path):
    write_meta(tmp_path / "PACE-LIBERO")
    pt = tmp_path / "PACE-LIBERO" / "pytorch_model.pt"
    pt.write_bytes(b"")
    resolved = resolve_checkpoint(pt)
    assert resolved.layout == "flat"
    assert resolved.weights == pt
    assert resolved.config == tmp_path / "PACE-LIBERO" / "config.yaml"
    assert resolved.stats == tmp_path / "PACE-LIBERO" / "dataset_statistics.json"


def test_training_layout_config_two_levels_up(tmp_path):
    run = tmp_path / "runs" / "pace_libero"
    write_meta(run)
    pt = run / "checkpoints" / "steps_30000_pytorch_model.pt"
    pt.parent.mkdir(parents=True)
    pt.write_bytes(b"")
    resolved = resolve_checkpoint(pt)
    assert resolved.layout == "training"
    assert resolved.weights == pt
    assert resolved.config == run / "config.yaml"


def test_final_model_in_training_run(tmp_path):
    run = tmp_path / "run"
    write_meta(run)
    pt = run / "final_model" / "pytorch_model.pt"
    pt.parent.mkdir(parents=True)
    pt.write_bytes(b"")
    assert resolve_checkpoint(pt).config == run / "config.yaml"
    # the run directory itself resolves to final_model/pytorch_model.pt
    assert resolve_checkpoint(run).weights == pt


def test_directory_layout(tmp_path):
    write_meta(tmp_path / "ckpt")
    pt = tmp_path / "ckpt" / "pytorch_model.pt"
    pt.write_bytes(b"")
    resolved = resolve_checkpoint(tmp_path / "ckpt")
    assert resolved.layout == "directory"
    assert resolved.weights == pt


def test_directory_single_other_pt_and_ambiguous(tmp_path):
    write_meta(tmp_path / "ckpt")
    (tmp_path / "ckpt" / "steps_5_pytorch_model.pt").write_bytes(b"")
    assert resolve_checkpoint(tmp_path / "ckpt").weights.name == "steps_5_pytorch_model.pt"
    (tmp_path / "ckpt" / "steps_6_pytorch_model.pt").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="several"):
        resolve_checkpoint(tmp_path / "ckpt")


def test_directory_without_weights_ok_for_clients(tmp_path):
    write_meta(tmp_path / "meta_only")
    resolved = resolve_checkpoint(tmp_path / "meta_only", require_weights=False)
    assert resolved.weights is None
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path / "meta_only", require_weights=True)


def test_missing_meta_is_reported(tmp_path):
    pt = tmp_path / "a" / "b" / "model.pt"
    pt.parent.mkdir(parents=True)
    pt.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        resolve_checkpoint(pt)


def test_hf_repo_id(tmp_path):
    calls = []

    def fake_snapshot_download(repo_id, revision=None, allow_patterns=None):
        calls.append((repo_id, revision, tuple(allow_patterns)))
        local = tmp_path / "hf" / repo_id.replace("/", "--")
        write_meta(local)
        if "*.pt" in allow_patterns:
            (local / "pytorch_model.pt").write_bytes(b"")
        return str(local)

    resolved = resolve_checkpoint("org/PACE-LIBERO", snapshot_download=fake_snapshot_download)
    assert resolved.layout == "hf:org/PACE-LIBERO"
    assert resolved.weights.name == "pytorch_model.pt"
    # clients only download config + statistics
    cfg, stats, _ = read_checkpoint_meta("org/PACE-LIBERO@v1", snapshot_download=fake_snapshot_download)
    assert calls[-1] == ("org/PACE-LIBERO", "v1", ("config.yaml", "dataset_statistics.json"))
    assert cfg["framework"]["action_model"]["future_action_window_size"] == 7
    assert stats == STATS


def test_nonexistent_reference(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint("./not/a/repo/id")


def test_repo_id_detection(tmp_path):
    assert looks_like_hf_repo_id("Qwen/Qwen3-VL-4B-Instruct")
    assert looks_like_hf_repo_id("org/name@main")
    assert not looks_like_hf_repo_id("/abs/path")
    assert not looks_like_hf_repo_id("./rel/path")
    assert not looks_like_hf_repo_id("a/b/c")
    assert not looks_like_hf_repo_id("ckpt/model.pt")


def test_model_tools_read_mode_config_accepts_all_layouts(tmp_path):
    """The legacy helper used by older client code resolves the flat layout too (SemanticVLA parents[1] bug)."""
    yaml = pytest.importorskip("yaml")  # noqa: F841
    from pace.model.tools import read_mode_config

    write_meta(tmp_path / "flat")
    (tmp_path / "flat" / "pytorch_model.pt").write_bytes(b"")
    cfg, stats = read_mode_config(str(tmp_path / "flat" / "pytorch_model.pt"))
    assert cfg["framework"]["name"] == "PACE"
    assert stats == STATS


# --- base VLM -----------------------------------------------------------------------------------------------------


def test_base_vlm_env_directory_wins(tmp_path):
    local = tmp_path / "Qwen3-VL-4B-Instruct"
    local.mkdir()
    assert resolve_base_vlm("/does/not/exist", env={"BASE_VLM": str(local)}) == str(local.resolve())


def test_base_vlm_env_repo_id_and_invalid(tmp_path):
    assert resolve_base_vlm(None, env={"BASE_VLM": "Qwen/Qwen3-VL-4B-Instruct"}) == "Qwen/Qwen3-VL-4B-Instruct"
    with pytest.raises(FileNotFoundError):
        resolve_base_vlm(None, env={"BASE_VLM": str(tmp_path / "missing" / "dir" / "x")})


def test_base_vlm_existing_configured_path(tmp_path):
    local = tmp_path / "qwen"
    local.mkdir()
    assert resolve_base_vlm(str(local), env={}) == str(local.resolve())


def test_base_vlm_missing_configured_path_falls_back_to_hf():
    # e.g. an absolute cluster path stored in a released config.yaml
    assert resolve_base_vlm("/nonexistent/models/Qwen/Qwen3-VL-4B-Instruct", env={}) == DEFAULT_BASE_VLM
    assert resolve_base_vlm(DEFAULT_BASE_VLM, env={}) == DEFAULT_BASE_VLM
    assert resolve_base_vlm(None, env={}) == DEFAULT_BASE_VLM


def test_base_vlm_repo_relative_path(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_io, "REPO_ROOT", tmp_path)
    (tmp_path / "models" / "qwen").mkdir(parents=True)
    assert resolve_base_vlm("models/qwen", env={}) == str((tmp_path / "models" / "qwen").resolve())


# --- DINOv2 code and weights ---------------------------------------------------------------------------------------


def test_dinov2_weights_env_wins(tmp_path):
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"")
    env = {"DINOV2_WEIGHTS": str(weights)}
    assert dinov2_assets.resolve_dinov2_weights("vitb14", configured="/x.pth", env=env) == str(weights)
    with pytest.raises(FileNotFoundError):
        dinov2_assets.resolve_dinov2_weights("vitb14", env={"DINOV2_WEIGHTS": str(tmp_path / "missing.pth")})


def test_dinov2_weights_legacy_env_alias(tmp_path):
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"")
    assert dinov2_assets.resolve_dinov2_weights("vitb14", env={"DINO_WEIGHTS": str(weights)}) == str(weights)


def test_dinov2_weights_configured_existing_file(tmp_path):
    weights = tmp_path / "configured.pth"
    weights.write_bytes(b"")
    assert dinov2_assets.resolve_dinov2_weights("vitb14", configured=str(weights), env={}) == str(weights)


def test_dinov2_weights_download_into_cache(tmp_path):
    downloads = []

    def fake_download(url, dest):
        downloads.append((url, dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"weights")

    env = {"PACE_CACHE": str(tmp_path / "cache")}
    path = dinov2_assets.resolve_dinov2_weights(
        "vitb14", configured="/cluster/only.pth", env=env, download=fake_download
    )
    expected = tmp_path / "cache" / "dinov2" / "dinov2_vitb14_pretrain.pth"
    assert path == str(expected)
    assert downloads == [
        ("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth", expected)
    ]
    # second call hits the cache
    assert dinov2_assets.resolve_dinov2_weights("vitb14", env=env, download=fake_download) == str(expected)
    assert len(downloads) == 1


def test_dinov2_default_cache_dir(tmp_path):
    assert dinov2_assets.default_cache_dir({"PACE_CACHE": str(tmp_path)}) == tmp_path
    assert dinov2_assets.default_cache_dir({}) == Path.home() / ".cache" / "pace"


def test_dinov2_repo_root_resolution(tmp_path):
    vendored = dinov2_assets.VENDORED_DINOV2_ROOT
    assert (vendored / "dinov2" / "hub" / "backbones.py").is_file()
    assert dinov2_assets.resolve_dinov2_repo_root(None, env={}) == str(vendored)
    other = tmp_path / "dinov2_checkout"
    (other / "dinov2").mkdir(parents=True)
    assert dinov2_assets.resolve_dinov2_repo_root(None, env={"DINO_REPO_ROOT": str(other)}) == str(other)
    assert dinov2_assets.resolve_dinov2_repo_root(str(other), env={}) == str(other)
    # a stale configured path (e.g. from a released config.yaml) falls back to the vendored copy
    assert dinov2_assets.resolve_dinov2_repo_root("/nonexistent/dinov2", env={}) == str(vendored)
    with pytest.raises(FileNotFoundError):
        dinov2_assets.resolve_dinov2_repo_root(None, env={"DINO_REPO_ROOT": str(tmp_path / "nope")})
