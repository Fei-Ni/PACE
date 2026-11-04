"""
PACE training entry point.

    accelerate launch --config_file pace/config/deepspeed/zero2.yaml --num_processes 4 \
        pace/training/train_pace.py --config_yaml pace/config/training/pace_libero.yaml [--a.b value ...]

Run from the repository root with the package importable (`pip install -e .` or PYTHONPATH=<repo>).
Any `--a.b value` argument after the known flags is merged into the YAML config (OmegaConf dotlist). The merged
config is written to `<run_root_dir>/<run_id>/config.yaml` together with `dataset_statistics.json`; checkpoints go to
`<run>/checkpoints/steps_<N>_pytorch_model.pt` and `<run>/final_model/pytorch_model.pt`.

Loggers (`trackers`): "jsonl" appends metrics to `<run>/metrics.jsonl`; "wandb" logs to Weights & Biases
(`wandb_project`, entity from `wandb_entity` or the WANDB_ENTITY environment variable).

Resumable state: `trainer.state_save_interval > 0` writes the full accelerate/DeepSpeed state (model, optimizer,
scheduler, RNG) to `<run>/latest_state` (newest copy only); `trainer.resume_from_latest=true` restores it at start.
The data order is re-sampled after a resume.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.distributed as dist
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from pace.dataloader import build_dataloader
from pace.model.framework import build_framework
from pace.training.dotlist import normalize_dotlist_args
from pace.training.effective_config import check_consistency, format_effective
from pace.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = get_logger(__name__)


def build_accelerator(cfg):
    grad_acc_steps = int(cfg.trainer.gradient_accumulation_steps)
    deepspeed_plugin = DeepSpeedPlugin(gradient_accumulation_steps=grad_acc_steps)
    grad_acc_plugin = GradientAccumulationPlugin(
        num_steps=grad_acc_steps,
        sync_each_batch=True,
    )
    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_plugin=grad_acc_plugin,
    )
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", accelerator.local_process_index))
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
    accelerator.print(f"[build_accelerator] gradient_accumulation_steps={grad_acc_steps}")
    accelerator.print(accelerator.state)
    return accelerator


def synchronize_processes(accelerator: Accelerator | None = None) -> None:
    """Synchronize all ranks without relying on NCCL's implicit device inference."""
    if accelerator is not None:
        if dist.is_available() and dist.is_initialized() and torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", accelerator.local_process_index))
            torch.cuda.set_device(local_rank % torch.cuda.device_count())
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            accelerator.wait_for_everyone()
        return
    if dist.is_available() and dist.is_initialized():
        barrier_kwargs = {}
        if torch.cuda.is_available():
            barrier_kwargs["device_ids"] = [torch.cuda.current_device()]
        dist.barrier(**barrier_kwargs)


def setup_directories(cfg) -> Path:
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.yaml")
        with open(output_dir / "config.json", "w") as f_json:
            json.dump(OmegaConf.to_container(cfg, resolve=True), f_json, indent=2)
    return output_dir


def prepare_data(cfg, accelerator) -> DataLoader:
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    accelerator.dataloader_config.dispatch_batches = False
    synchronize_processes(accelerator)
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple:
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )
    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )
    return optimizer, lr_scheduler


def _cfg_flag(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return bool(value)


class PACETrainer(TrainerUtils):
    """Training loop (accelerate + DeepSpeed) that also logs the PACE sub-losses returned by the model."""

    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.completed_steps = 0
        self.total_batch_size = (
            cfg.datasets.vla_data.per_device_batch_size
            * accelerator.num_processes
            * accelerator.gradient_accumulation_steps
        )
        self.trackers = [str(t).lower() for t in (getattr(cfg, "trackers", None) or [])]
        self._wandb = None

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        if hasattr(self.config.trainer, "pretrained_checkpoint") and self.config.trainer.pretrained_checkpoint:
            reload_strict = getattr(self.config.trainer, "reload_strict", True)
            if isinstance(reload_strict, str):
                reload_strict = reload_strict.lower() not in {"false", "0", "no"}
            self.model = self.load_pretrained_backbones(
                self.model,
                self.config.trainer.pretrained_checkpoint,
                reload_modules=getattr(self.config.trainer, "reload_modules", None),
                strict=reload_strict,
            )

        freeze_modules = getattr(self.config.trainer, "freeze_modules", None)
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)
        # Rebuild optimizer/scheduler after checkpoint loading and any module freezes
        # so DeepSpeed only sees the final trainable parameter set.
        self.optimizer, self.lr_scheduler = setup_optimizer_and_scheduler(
            model=self.model,
            cfg=self.config,
        )

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )
        # The scheduler is not passed through accelerator.prepare, so register it
        # explicitly to have it saved/restored together with model + optimizer.
        self.accelerator.register_for_checkpointing(self.lr_scheduler)

        self._init_trackers()
        self._init_checkpointing()
        self._maybe_resume_from_latest_state()

    def _init_trackers(self):
        if not self.accelerator.is_main_process or "wandb" not in self.trackers:
            return
        try:
            import wandb

            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=getattr(self.config, "wandb_project", "pace"),
                entity=getattr(self.config, "wandb_entity", None) or os.environ.get("WANDB_ENTITY") or None,
                group=getattr(self.config, "wandb_group", None),
            )
            self._wandb = wandb
        except Exception as e:  # logging must never stop training
            logger.warning(f"WandB init failed ({e}); continuing without WandB.")

    def _init_checkpointing(self):
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _latest_state_dir(self) -> str:
        return os.path.join(self.config.output_dir, "latest_state")

    def _maybe_resume_from_latest_state(self):
        if not _cfg_flag(getattr(self.config.trainer, "resume_from_latest", False)):
            return
        state_dir = self._latest_state_dir()
        meta_path = os.path.join(state_dir, "pace_resume_meta.json")
        if not os.path.isfile(meta_path):
            logger.info(f"resume_from_latest: no saved state under {state_dir}; starting from step 0.")
            return
        with open(meta_path) as f:
            meta = json.load(f)
        self.accelerator.load_state(state_dir)
        self.completed_steps = int(meta["completed_steps"])
        logger.info(f"Resumed full training state from {state_dir} at step {self.completed_steps}.")

    def _save_latest_state(self):
        state_dir = self._latest_state_dir()
        tmp_dir = state_dir + ".tmp"
        old_dir = state_dir + ".old"
        if self.accelerator.is_main_process:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(tmp_dir)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            with open(os.path.join(tmp_dir, "pace_resume_meta.json"), "w") as f:
                json.dump({"completed_steps": int(self.completed_steps)}, f)
            shutil.rmtree(old_dir, ignore_errors=True)
            if os.path.isdir(state_dir):
                os.rename(state_dir, old_dir)
            os.rename(tmp_dir, state_dir)
            shutil.rmtree(old_dir, ignore_errors=True)
            self.accelerator.print(f"Resumable state saved at step {self.completed_steps}: {state_dir}")
        self.accelerator.wait_for_everyone()

    def _save_checkpoint(self):
        if self.accelerator.is_main_process:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps({"steps": self.completed_steps}) + "\n")
            self.accelerator.print(f"Checkpoint saved at {checkpoint_path}_pytorch_model.pt")
        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        if self.completed_steps % self.config.trainer.logging_frequency != 0:
            return
        if not self.accelerator.is_main_process:
            return
        metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
        metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
        if self._wandb is not None and self._wandb.run is not None:
            self._wandb.log(metrics, step=self.completed_steps)
        if "jsonl" in self.trackers:
            with open(os.path.join(self.config.output_dir, "metrics.jsonl"), "a") as f:
                f.write(json.dumps({"step": self.completed_steps, **metrics}) + "\n")
        logger.info(f"Step {self.completed_steps}, Loss: {metrics}")

    def _create_data_iterators(self):
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        try:
            return next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            return next(self.vla_iter)

    def _train_step(self, batch_vla):
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(
                    batch_vla,
                    global_step=self.completed_steps,
                    max_train_steps=self.config.trainer.max_train_steps,
                )
                action_loss = output_dict["action_loss"]
                total_loss = action_loss

            self.accelerator.backward(total_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            self.lr_scheduler.step()

        # total_loss = flow matching + weighted progress losses; the model also returns each term separately.
        metrics = {"total_loss": total_loss.item()}
        for key, value in output_dict.items():
            if key == "action_loss":
                continue
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                metrics[key.lstrip("_")] = value.item()
            elif isinstance(value, (float, int)):
                metrics[key.lstrip("_")] = float(value)

        return metrics

    def train(self):
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimisation steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size    = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation    = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size         = {self.total_batch_size}")

        self._create_data_iterators()

        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        if self.completed_steps > 0:
            progress_bar.update(self.completed_steps)
        state_save_interval = int(getattr(self.config.trainer, "state_save_interval", 0) or 0)

        while self.completed_steps < self.config.trainer.max_train_steps:
            t0_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t1_data = time.perf_counter()

            t0_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t1_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_t": f"{t1_data - t0_data:.3f}",
                        "model_t": f"{t1_model - t0_model:.3f}",
                    }
                )

            step_metrics["data_time"] = t1_data - t0_data
            step_metrics["model_time"] = t1_model - t0_model
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if (
                state_save_interval > 0
                and self.completed_steps > 0
                and self.completed_steps % state_save_interval == 0
                and self.completed_steps < self.config.trainer.max_train_steps
            ):
                self._save_latest_state()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        skip_final_save = _cfg_flag(getattr(self.config.trainer, "skip_final_save", False))
        if self.accelerator.is_main_process and not skip_final_save:
            final_dir = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_dir, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_dir, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_dir}")
        elif self.accelerator.is_main_process:
            logger.info("Training complete. Final model save skipped by trainer.skip_final_save.")
        if self.accelerator.is_main_process and self._wandb is not None and self._wandb.run is not None:
            self._wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg):
    logger.info("PACE training :: warming up")
    accelerator = build_accelerator(cfg)
    if accelerator.is_main_process:
        print(format_effective(OmegaConf.to_container(cfg, resolve=True), accelerator.num_processes), flush=True)

    setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator)

    trainer = PACETrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=None,
        lr_scheduler=None,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("Training finished.")
    synchronize_processes(accelerator)
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_config(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="pace/config/training/pace_libero.yaml")
    args, clipargs = parser.parse_known_args(argv)

    cfg = OmegaConf.load(args.config_yaml)
    cli_cfg = OmegaConf.from_dotlist(normalize_dotlist_args(clipargs))
    cfg = OmegaConf.merge(cfg, cli_cfg)
    problems = check_consistency(OmegaConf.to_container(cfg, resolve=True))
    if problems:
        raise ValueError("Inconsistent config: " + "; ".join(problems))
    return cfg


if __name__ == "__main__":
    main(parse_config())
