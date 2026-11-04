#!/usr/bin/env bash
# Train the PACE LIBERO checkpoint: 4 suites jointly (libero_all), 30k steps, 4 GPUs x 16 (global batch 64).
# The recipe lives in pace/config/training/pace_libero.yaml; this script only passes paths.
#
# Usage: bash examples/LIBERO/train_libero.sh [--dry-run] [--key.path value ...]
#   Extra --key.path value pairs are OmegaConf overrides (they are printed in the EFFECTIVE line).
# Environment: DATA_ROOT (default datasets/LEROBOT_LIBERO_DATA), RUN_ROOT_DIR (default runs),
#   RUN_ID (default pace_libero), NUM_GPUS (default 4), MAIN_PROCESS_PORT (default 29500),
#   PYTHON / ACCELERATE (default python / accelerate), BASE_VLM, DINOV2_WEIGHTS, PACE_CACHE, WANDB_ENTITY.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_YAML="pace/config/training/pace_libero.yaml"
DATA_ROOT="${DATA_ROOT:-datasets/LEROBOT_LIBERO_DATA}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-runs}"
RUN_ID="${RUN_ID:-pace_libero}"
NUM_GPUS="${NUM_GPUS:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
PYTHON="${PYTHON:-python}"
ACCELERATE="${ACCELERATE:-accelerate}"
DRY_RUN="${DRY_RUN:-0}"

EXTRA=()
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) EXTRA+=("${arg}") ;;
  esac
done

OVERRIDES=(
  --datasets.vla_data.data_root_dir "${DATA_ROOT}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --run_id "${RUN_ID}"
)

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" -m pace.training.effective_config --config_yaml "${CONFIG_YAML}" --num_processes "${NUM_GPUS}" \
  "${OVERRIDES[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
if [[ "${DRY_RUN}" == "1" ]]; then exit 0; fi

export TOKENIZERS_PARALLELISM=false
"${ACCELERATE}" launch \
  --config_file pace/config/deepspeed/zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  pace/training/train_pace.py \
  --config_yaml "${CONFIG_YAML}" \
  "${OVERRIDES[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
