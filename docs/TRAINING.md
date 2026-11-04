# Training

The launchers train PACE with the YAML recipes as shipped; nothing has to be overridden. Every launcher prints an `EFFECTIVE:` line (framework, data mix, steps, batch, DINOv2 variant, K, goal token, λ<sub>cons</sub>) before training starts, and `DRY_RUN=1` stops right after it.

## 0. Environment

Install PACE as in [INFERENCE.md](INFERENCE.md#1-install). Training uses `accelerate` with DeepSpeed ZeRO-2 ([pace/config/deepspeed/zero2.yaml](../pace/config/deepspeed/zero2.yaml)), bf16 mixed precision and gradient checkpointing. We train on one node with 4 × NVIDIA GH200 (96 GB), 16 samples per GPU, no gradient accumulation (global batch 64).

With `trainer.state_save_interval > 0` and `trainer.resume_from_latest: true` (the defaults of `pace_bridge.yaml`; add `--trainer.state_save_interval 5000 --trainer.resume_from_latest true` for LIBERO), the full training state (model, optimizer, scheduler, RNG) is kept in `<run>/latest_state/` and a job that hits a wall-clock limit continues where it stopped when resubmitted.

## 1. Data

All training data is in LeRobot format.

```bash
# LIBERO: the four *_no_noops LeRobot datasets from IPEC-COMMUNITY, plus meta/modality.json
DATA_ROOT=/path/to/LEROBOT_LIBERO_DATA bash examples/LIBERO/prepare_libero_lerobot_data.sh

# BridgeData V2 (IPEC-COMMUNITY/bridge_orig_lerobot), plus meta/modality.json
DATA_ROOT=/path/to/OXE_LEROBOT_DATASET bash examples/SimplerEnv/prepare_oxe_lerobot_data.sh

# Warm start for SimplerEnv: StarVLA's Qwen3VL-GR00T-Bridge-RT-1 checkpoint
huggingface-cli download StarVLA/Qwen3VL-GR00T-Bridge-RT-1 --local-dir checkpoints/Qwen3VL-GR00T-Bridge-RT-1
```

The images are resized to 224 × 224. No proprioceptive state is used. Progress labels need no annotation: they are computed on the fly from the frozen DINOv2 encoder (first and last frame of each episode).

## 2. LIBERO

```bash
DATA_ROOT=/path/to/LEROBOT_LIBERO_DATA bash examples/LIBERO/train_libero.sh    # run dir: runs/pace_libero
```

One model for all four suites (`data_mix: libero_all`), initialized from `Qwen/Qwen3-VL-4B-Instruct`; the action head, progress heads and query tokens start from scratch. 30K steps ≈ 7 passes over the 273K training frames; about 11.5 h on 4 × GH200.

## 3. SimplerEnv (WidowX)

```bash
# BridgeData V2, warm-started from StarVLA's Bridge checkpoint (100K steps)
DATA_ROOT=/path/to/OXE_LEROBOT_DATASET bash examples/SimplerEnv/train_bridge.sh    # run dir: runs/pace_bridge
```

`INIT_CKPT` overrides the initial checkpoint (default: the StarVLA checkpoint downloaded above); `RUN_ROOT_DIR`, `RUN_ID` and `NUM_GPUS` are read from the environment as well, and extra `--key.path value` arguments override the YAML. Only the backbone and the action head are loaded from the StarVLA checkpoint (`reload_modules: qwen_vl_interface,action_model`); the new query-token rows, progress heads and conditioning MLPs are initialized fresh. 100K steps ≈ 3.4 passes over BridgeData V2, about 28 h on 4 × GH200; a checkpoint is saved every 10K steps.

## 4. Hyperparameters

| | LIBERO | SimplerEnv (BridgeData V2) |
|---|---|---|
| Config | `pace_libero.yaml` | `pace_bridge.yaml` |
| Initialization | Qwen3-VL-4B-Instruct | StarVLA Bridge-RT-1 (backbone + action head) |
| Trainable | all parameters (embedding table: new token rows only) | same |
| Steps / warmup | 30K / 5K | 100K / 5K |
| Learning rate: base / backbone / action head | 4e-5 / 1e-5 / 1e-4 | 1e-5 / 1e-5 / 1e-4 |
| Schedule | cosine to 1e-6 | cosine to 5e-7 |
| Optimizer | AdamW (β 0.9, 0.95; wd 1e-8), grad clip 1.0 | same |
| Batch | 16 × 4 GPUs = 64 | 64 |
| Camera views | third-person + wrist | third-person |
| Action chunk H / progress offset Δ | 8 / 7 | 16 / 15 |
| Flow-matching steps (train repeats / inference) | 4 / 4 | 4 / 4 |
| DINOv2 | ViT-B/14, frozen | same |
| Goal queries K / query layer | 4 / 18 | same |
| λ<sub>goal</sub>, λ<sub>progress</sub>, λ<sub>commit</sub>, λ<sub>cons</sub> | 0.1 each | same |
| Seed | 43 | 42 |
