"""
Print the effective training configuration (YAML + CLI overrides) as a single `EFFECTIVE:` line.

Used by the training launchers (also in `--dry-run` mode) and by `train_pace.py` at start-up, so the values a run
actually uses can be checked against the intended recipe from the log.

    python -m pace.training.effective_config --config_yaml pace/config/training/pace_libero.yaml \
        --num_processes 4 [--trainer.max_train_steps 100 ...]

Only needs PyYAML or OmegaConf (no torch).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, List, Mapping, Optional

from pace.training.dotlist import normalize_dotlist_args


def _get(cfg: Any, dotted: str, default: Any = None) -> Any:
    node = cfg
    for part in dotted.split("."):
        if node is None:
            return default
        if isinstance(node, Mapping) or hasattr(node, "get"):
            node = node.get(part, None)
        else:
            node = getattr(node, part, None)
    return default if node is None else node


def format_effective(cfg: Any, num_processes: Optional[int] = None) -> str:
    """Return the `EFFECTIVE:` summary line for a (merged) training config."""
    per_device = _get(cfg, "datasets.vla_data.per_device_batch_size")
    grad_acc = _get(cfg, "trainer.gradient_accumulation_steps", 1)
    global_batch = "?"
    if per_device is not None and num_processes is not None:
        global_batch = int(per_device) * int(num_processes) * int(grad_acc)
    lr = _get(cfg, "trainer.learning_rate", {}) or {}
    lr_str = "/".join(str(_get(lr, key, "-")) for key in ("base", "qwen_vl_interface", "action_model"))
    fields = [
        ("framework", _get(cfg, "framework.name")),
        ("data_mix", _get(cfg, "datasets.vla_data.data_mix")),
        ("data_root", _get(cfg, "datasets.vla_data.data_root_dir")),
        ("max_train_steps", _get(cfg, "trainer.max_train_steps")),
        ("per_device_batch", per_device),
        ("num_gpus", num_processes if num_processes is not None else "?"),
        ("grad_acc", grad_acc),
        ("global_batch", global_batch),
        ("lr(base/vlm/action)", lr_str),
        ("warmup", _get(cfg, "trainer.num_warmup_steps")),
        ("seed", _get(cfg, "seed")),
        ("dino_variant", _get(cfg, "framework.progress.dino_variant")),
        ("dino_dim", _get(cfg, "framework.progress.dino_target_dim")),
        ("K_goal_queries", _get(cfg, "framework.progress.num_goal_queries")),
        ("goal_dit_token", _get(cfg, "framework.progress.use_goal_dit_token")),
        ("lambda_cons", _get(cfg, "framework.progress.lambda_cons")),
        (
            "lambda_long/pcur/dpcur",
            "/".join(
                str(_get(cfg, f"framework.progress.{key}", "-"))
                for key in ("lambda_long", "lambda_pcur", "lambda_dpcur")
            ),
        ),
        ("chunk", _get(cfg, "framework.action_model.action_horizon")),
        ("delta", _get(cfg, "framework.progress.progress_future_offset")),
        ("denoise_steps", _get(cfg, "framework.action_model.num_inference_timesteps")),
        ("repeated_diffusion_steps", _get(cfg, "trainer.repeated_diffusion_steps", 4)),
        ("init", _get(cfg, "trainer.pretrained_checkpoint", "none")),
        ("reload_modules", _get(cfg, "trainer.reload_modules", "all")),
        ("run", f"{_get(cfg, 'run_root_dir')}/{_get(cfg, 'run_id')}"),
    ]
    return "EFFECTIVE: " + " ".join(f"{key}={value}" for key, value in fields)


def check_consistency(cfg: Any) -> List[str]:
    """Return a list of problems in values that must agree across config sections."""
    problems = []
    model_delta = _get(cfg, "framework.progress.progress_future_offset")
    data_delta = _get(cfg, "datasets.vla_data.progress_future_offset")
    if model_delta is not None and data_delta is not None and int(model_delta) != int(data_delta):
        problems.append(
            f"framework.progress.progress_future_offset ({model_delta}) != "
            f"datasets.vla_data.progress_future_offset ({data_delta})"
        )
    window = _get(cfg, "framework.action_model.future_action_window_size")
    horizon = _get(cfg, "framework.action_model.action_horizon")
    if window is not None and horizon is not None and int(window) + 1 != int(horizon):
        problems.append(f"action_horizon ({horizon}) != future_action_window_size + 1 ({int(window) + 1})")
    return problems


def load_merged_config(config_yaml: str, overrides: List[str]):
    """Load a YAML config and apply `--a.b value` style overrides (OmegaConf if available, else PyYAML)."""
    dotlist = normalize_dotlist_args(overrides)
    try:
        from omegaconf import OmegaConf
    except ImportError:
        OmegaConf = None
    if OmegaConf is not None:
        cfg = OmegaConf.merge(OmegaConf.load(config_yaml), OmegaConf.from_dotlist(dotlist))
        return OmegaConf.to_container(cfg, resolve=True)

    import yaml

    with open(config_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    for item in dotlist:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value) if value else value
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--num_processes", type=int, default=None)
    args, overrides = parser.parse_known_args(argv)
    cfg = load_merged_config(args.config_yaml, overrides)
    print(format_effective(cfg, args.num_processes))
    problems = check_consistency(cfg)
    for problem in problems:
        print(f"CONFIG ERROR: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
