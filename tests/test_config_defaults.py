"""
The release YAMLs must encode exactly the effective recipe of the released checkpoints, contain no dead keys, no
absolute / cluster paths, and agree with the training launchers (pure Python; needs PyYAML or OmegaConf).
"""

from pathlib import Path

import pytest

from pace.training.dotlist import normalize_dotlist_args
from pace.training.effective_config import check_consistency, format_effective, load_merged_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "pace" / "config" / "training"
CONFIGS = {name: CONFIG_DIR / f"{name}.yaml" for name in ("pace_libero", "pace_bridge")}

COMMON = {
    "framework.name": "PACE",
    "framework.qwenvl.base_vlm": "Qwen/Qwen3-VL-4B-Instruct",
    "framework.qwenvl.attn_implementation": "flash_attention_2",
    "framework.action_model.action_model_type": "DiT-B",
    "framework.action_model.num_inference_timesteps": 4,
    "framework.action_model.diffusion_model_cfg.num_layers": 16,
    "framework.progress.layer_idx": 18,
    "framework.progress.dino_variant": "vitb14",
    "framework.progress.dino_target_dim": 768,
    "framework.progress.num_goal_queries": 4,
    "framework.progress.use_goal_dit_token": True,
    "framework.progress.lambda_cons": 0.1,
    "framework.progress.cons_detach_anchor": True,
    "framework.progress.lambda_long": 0.1,
    "framework.progress.lambda_pcur": 0.1,
    "framework.progress.lambda_dpcur": 0.1,
    "framework.progress.infer_history_mode": "step_aligned",
    "framework.progress.use_full_goal_anchor_at_inference": True,
    "framework.progress.use_history_images": True,
    "framework.progress.use_dp_prev_input": True,
    "framework.progress.audit_kappa": 0.0,
    "datasets.vla_data.per_device_batch_size": 16,
    "datasets.vla_data.return_progress_anchors": True,
    "trainer.gradient_accumulation_steps": 1,
    "trainer.repeated_diffusion_steps": 4,
    "trainer.freeze_modules": "",
    "trainer.lr_scheduler_type": "cosine_with_min_lr",
    "trainer.optimizer.betas": [0.9, 0.95],
    "trainer.gradient_clipping": 1.0,
    "trackers": ["jsonl"],
    "wandb_entity": None,
}

EXPECTED = {
    "pace_libero": {
        "seed": 43,
        "framework.progress.audit_threshold": 0.4,
        "framework.progress.audit_stall_chunks": 18,
        "datasets.vla_data.data_mix": "libero_all",
        "trainer.max_train_steps": 30000,
        "trainer.num_warmup_steps": 5000,
        "trainer.learning_rate.base": 4e-5,
        "trainer.learning_rate.qwen_vl_interface": 1e-5,
        "trainer.learning_rate.action_model": 1e-4,
        "trainer.scheduler_specific_kwargs.min_lr": 1e-6,
        "framework.action_model.future_action_window_size": 7,
        "framework.action_model.action_horizon": 8,
        "framework.progress.progress_future_offset": 7,
        "trainer.pretrained_checkpoint": None,
    },
    "pace_bridge": {
        "framework.progress.audit_threshold": 0.4,
        "framework.progress.audit_stall_chunks": 2.12,
        "seed": 42,
        "datasets.vla_data.data_mix": "bridge",
        "datasets.vla_data.image_size": [224, 224],
        "trainer.max_train_steps": 100000,
        "trainer.num_warmup_steps": 5000,
        "trainer.learning_rate.base": 1e-5,
        "trainer.learning_rate.qwen_vl_interface": 1e-5,
        "trainer.learning_rate.action_model": 1e-4,
        "trainer.scheduler_specific_kwargs.min_lr": 5e-7,
        "framework.action_model.future_action_window_size": 15,
        "framework.action_model.action_horizon": 16,
        "framework.progress.progress_future_offset": 15,
        "trainer.reload_modules": "qwen_vl_interface,action_model",
        "trainer.reload_strict": False,
    },
}

# Every key in the release YAMLs must be read by the code (no dead configuration).
ALLOWED_KEYS = {
    "run_id", "run_root_dir", "seed", "trackers", "wandb_entity", "wandb_project",
    "framework.name", "framework.qwenvl.base_vlm", "framework.qwenvl.attn_implementation",
    "framework.action_model.action_model_type", "framework.action_model.hidden_size",
    "framework.action_model.add_pos_embed", "framework.action_model.max_seq_len", "framework.action_model.action_dim",
    "framework.action_model.state_dim", "framework.action_model.future_action_window_size",
    "framework.action_model.action_horizon", "framework.action_model.past_action_window_size",
    "framework.action_model.noise_beta_alpha", "framework.action_model.noise_beta_beta",
    "framework.action_model.noise_s", "framework.action_model.num_timestep_buckets",
    "framework.action_model.num_inference_timesteps", "framework.action_model.num_target_vision_tokens",
    "framework.action_model.diffusion_model_cfg.dropout", "framework.action_model.diffusion_model_cfg.final_dropout",
    "framework.action_model.diffusion_model_cfg.interleave_self_attention",
    "framework.action_model.diffusion_model_cfg.norm_type", "framework.action_model.diffusion_model_cfg.num_layers",
    "framework.action_model.diffusion_model_cfg.output_dim",
    "framework.action_model.diffusion_model_cfg.positional_embeddings",
    "framework.progress.layer_idx", "framework.progress.num_goal_queries", "framework.progress.use_goal_dit_token",
    "framework.progress.lambda_long", "framework.progress.lambda_pcur", "framework.progress.lambda_dpcur",
    "framework.progress.lambda_cons", "framework.progress.cons_detach_anchor", "framework.progress.query_init_std",
    "framework.progress.train_query_token_rows_only", "framework.progress.progress_future_offset",
    "framework.progress.use_history_images", "framework.progress.use_dp_prev_input",
    "framework.progress.use_full_goal_anchor_at_inference", "framework.progress.infer_history_mode",
    "framework.progress.dino_variant", "framework.progress.dino_target_dim", "framework.progress.dino_head_hidden_dim",
    "framework.progress.dino_repo_root", "framework.progress.dino_weights", "framework.progress.audit_kappa",
    "framework.progress.audit_threshold", "framework.progress.audit_stall_chunks",
    "datasets.vla_data.dataset_py", "datasets.vla_data.data_root_dir", "datasets.vla_data.data_mix",
    "datasets.vla_data.CoT_prompt", "datasets.vla_data.per_device_batch_size", "datasets.vla_data.strict_sample_errors",
    "datasets.vla_data.sample_max_retries", "datasets.vla_data.return_progress_anchors",
    "datasets.vla_data.progress_future_offset", "datasets.vla_data.image_size",
    "trainer.max_train_steps", "trainer.num_warmup_steps", "trainer.save_interval", "trainer.learning_rate.base",
    "trainer.learning_rate.qwen_vl_interface", "trainer.learning_rate.action_model", "trainer.lr_scheduler_type",
    "trainer.scheduler_specific_kwargs.min_lr", "trainer.freeze_modules", "trainer.logging_frequency",
    "trainer.gradient_clipping", "trainer.gradient_accumulation_steps", "trainer.repeated_diffusion_steps",
    "trainer.optimizer.betas", "trainer.optimizer.eps", "trainer.optimizer.weight_decay",
    "trainer.pretrained_checkpoint", "trainer.reload_modules", "trainer.reload_strict",
    "trainer.state_save_interval", "trainer.resume_from_latest",
}  # fmt: skip

def flatten(node, prefix=""):
    out = {}
    for key, value in node.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, full))
        else:
            out[full] = value
    return out


@pytest.fixture(scope="module", params=sorted(CONFIGS))
def nested_config(request):
    return request.param, load_merged_config(str(CONFIGS[request.param]), [])


@pytest.fixture(scope="module")
def config(nested_config):
    name, cfg = nested_config
    return name, flatten(cfg)


def test_effective_values(config):
    name, cfg = config
    for key, expected in {**COMMON, **EXPECTED[name]}.items():
        assert key in cfg, f"{name}: missing {key}"
        value = cfg[key]
        if isinstance(expected, float):
            assert value == pytest.approx(expected), f"{name}: {key}={value!r}, expected {expected!r}"
        else:
            assert value == expected, f"{name}: {key}={value!r}, expected {expected!r}"


def test_no_dead_keys(config):
    name, cfg = config
    unknown = sorted(set(cfg) - ALLOWED_KEYS)
    assert not unknown, f"{name}: keys not read by the code: {unknown}"
    assert "framework.action_model.repeated_diffusion_steps" not in cfg


def test_image_size_only_for_widowx(config):
    name, cfg = config
    assert ("datasets.vla_data.image_size" in cfg) == (name != "pace_libero")


def test_no_absolute_or_cluster_paths(config):
    name, cfg = config
    for key, value in cfg.items():
        if isinstance(value, str):
            assert not value.startswith(("/", "~")), f"{name}: {key} is an absolute path: {value}"
            assert "${" not in value, f"{name}: {key} uses interpolation (not expanded from the shell): {value}"


def test_consistency_and_effective_line(nested_config):
    name, cfg = nested_config
    assert check_consistency(cfg) == []
    line = format_effective(cfg, num_processes=4)
    assert line.startswith("EFFECTIVE: ")
    assert "framework=PACE" in line
    assert "global_batch=64" in line
    assert "dino_variant=vitb14" in line
    assert "K_goal_queries=4" in line
    assert "goal_dit_token=True" in line
    assert "lambda_cons=0.1" in line


def test_consistency_detects_mismatched_delta():
    cfg = load_merged_config(str(CONFIGS["pace_libero"]), ["--datasets.vla_data.progress_future_offset", "15"])
    assert any("progress_future_offset" in problem for problem in check_consistency(cfg))


def test_cli_overrides_are_applied():
    cfg = load_merged_config(
        str(CONFIGS["pace_bridge"]),
        ["--datasets.vla_data.data_root_dir", "my/data", "--trainer.max_train_steps", "7", "--run_id", "x"],
    )
    assert cfg["datasets"]["vla_data"]["data_root_dir"] == "my/data"
    assert cfg["trainer"]["max_train_steps"] == 7
    assert cfg["run_id"] == "x"


def test_dotlist_parsing():
    assert normalize_dotlist_args(["--a.b", "1", "--flag", "--c=2", "--l", "x", "y"]) == [
        "a.b=1",
        "flag=true",
        "c=2",
        "l=[x,y]",
    ]


def test_framework_registered_in_source():
    source = (REPO_ROOT / "pace" / "model" / "framework" / "pace.py").read_text()
    assert '@FRAMEWORK_REGISTRY.register("PACE")' in source


def test_launchers_use_matching_configs():
    pairs = {
        "examples/LIBERO/train_libero.sh": "pace_libero.yaml",
        "examples/SimplerEnv/train_bridge.sh": "pace_bridge.yaml",
    }
    for script, yaml_name in pairs.items():
        text = (REPO_ROOT / script).read_text()
        assert f'CONFIG_YAML="pace/config/training/{yaml_name}"' in text, script
        assert "pace/config/deepspeed/zero2.yaml" in text, script


