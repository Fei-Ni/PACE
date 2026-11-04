#!/usr/bin/env bash
# SimplerEnv (WidowX): train PACE on BridgeData V2 (data_mix=bridge) for 100k steps, 4 GPUs x 16 (global batch 64),
# warm-started from the StarVLA Qwen3VL-GR00T-Bridge-RT-1 checkpoint (steps_20000; only the VLM and the action head
# are reloaded, non-strict). The recipe lives in pace/config/training/pace_bridge.yaml; this script only passes paths.
#
# Usage: bash examples/SimplerEnv/train_bridge.sh [--dry-run] [--key.path value ...]
# Environment: DATA_ROOT (default datasets/OXE_LEROBOT_DATASET),
#   INIT_CKPT (default checkpoints/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt, from the
#   Hugging Face repo StarVLA/Qwen3VL-GR00T-Bridge-RT-1), RUN_ROOT_DIR (default runs), RUN_ID (default pace_bridge),
#   NUM_GPUS (default 4), MAIN_PROCESS_PORT (default 29500), PYTHON / ACCELERATE, BASE_VLM, DINOV2_WEIGHTS,
#   PACE_CACHE, WANDB_ENTITY.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_YAML="pace/config/training/pace_bridge.yaml"
DATA_ROOT="${DATA_ROOT:-datasets/OXE_LEROBOT_DATASET}"
INIT_CKPT="${INIT_CKPT:-checkpoints/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-runs}"
RUN_ID="${RUN_ID:-pace_bridge}"
NUM_GPUS="${NUM_GPUS:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
PYTHON="${PYTHON:-python}"
ACCELERATE="${ACCELERATE:-accelerate}"
DRY_RUN="${DRY_RUN:-0}"

EXTRA=()
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,13p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) EXTRA+=("${arg}") ;;
  esac
done

OVERRIDES=(
  --datasets.vla_data.data_root_dir "${DATA_ROOT}"
  --trainer.pretrained_checkpoint "${INIT_CKPT}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --run_id "${RUN_ID}"
)

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" -m pace.training.effective_config --config_yaml "${CONFIG_YAML}" --num_processes "${NUM_GPUS}" \
  "${OVERRIDES[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
if [[ "${DRY_RUN}" == "1" ]]; then exit 0; fi

if [[ ! -f "${INIT_CKPT}" ]]; then
  echo "Initial checkpoint not found: ${INIT_CKPT} (set INIT_CKPT)" >&2
  exit 1
fi
export TOKENIZERS_PARALLELISM=false
"${ACCELERATE}" launch \
  --config_file pace/config/deepspeed/zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  pace/training/train_pace.py \
  --config_yaml "${CONFIG_YAML}" \
  "${OVERRIDES[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
