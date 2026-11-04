#!/usr/bin/env bash
# Evaluate PACE on SimplerEnv WidowX with ManiSkill3 (client side; start examples/SimplerEnv/run_server.sh first).
# Protocol: 4 tasks (spoon, carrot, stack block, eggplant) x 24 episodes, seeds 0-23, adaptive action ensemble.
#
# Usage: bash examples/SimplerEnv/eval_simpler.sh [CKPT] [--dry-run]
#   CKPT  same checkpoint reference the server loads (used here for the action statistics only)
# Environment:
#   SIM_PYTHON       python of a standard SimplerEnv-ManiSkill3 install (mani_skill, simpler_env) (default python)
#   SIMPLERENV_PATH  optional SimplerEnv checkout added to PYTHONPATH (not needed if simpler_env is pip-installed)
#   POLICY_HOST / POLICY_PORT      policy server address (default 127.0.0.1 / 10093)
#   EPISODES (24)  SEED (first seed, 0)  ACTION_ENSEMBLE (1)  SHADER (default)
#   TASKS            space-separated task names (default: the four WidowX tasks)
#   OUT_DIR          output directory (default results/simpler_eval/<timestamp>)
#   SIMPLER_RECORD_DIR  optional: record frames + per-step progress outputs for tools/progress_viz
#   MS_ASSET_DIR     optional ManiSkill asset directory (ManiSkill default otherwise)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CKPT="${CKPT:-checkpoints/PACE-SimplerEnv}"
SIM_PYTHON="${SIM_PYTHON:-python}"
SIMPLERENV_PATH="${SIMPLERENV_PATH:-}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-10093}"
EPISODES="${EPISODES:-24}"
SEED="${SEED:-0}"
ACTION_ENSEMBLE="${ACTION_ENSEMBLE:-1}"
SHADER="${SHADER:-default}"
TASKS="${TASKS:-widowx_spoon_on_towel widowx_carrot_on_plate widowx_stack_cube widowx_put_eggplant_in_basket}"
OUT_DIR="${OUT_DIR:-results/simpler_eval/$(date +%Y%m%d_%H%M%S)}"
SIMPLER_RECORD_DIR="${SIMPLER_RECORD_DIR:-}"
DRY_RUN="${DRY_RUN:-0}"

POSITIONAL=()
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) POSITIONAL+=("${arg}") ;;
  esac
done
if [[ ${#POSITIONAL[@]} -ge 1 ]]; then CKPT="${POSITIONAL[0]}"; fi

LAST_SEED=$((SEED + EPISODES - 1))
echo "EFFECTIVE: role=client benchmark=SimplerEnv-WidowX(ManiSkill3) ckpt=${CKPT} server=${POLICY_HOST}:${POLICY_PORT}" \
  "tasks=[${TASKS}] episodes_per_task=${EPISODES} seeds=${SEED}-${LAST_SEED} action_ensemble=${ACTION_ENSEMBLE}" \
  "shader=${SHADER} out_dir=${OUT_DIR} record_dir=${SIMPLER_RECORD_DIR:-off} python=${SIM_PYTHON}"
if [[ "${DRY_RUN}" == "1" ]]; then exit 0; fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${SIMPLERENV_PATH:+:${SIMPLERENV_PATH}}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${OUT_DIR}"
for task in ${TASKS}; do
  task_dir="${OUT_DIR}/${task}"
  mkdir -p "${task_dir}"
  echo "=== ${task}: ${EPISODES} episodes, seeds ${SEED}-${LAST_SEED} ==="
  "${SIM_PYTHON}" examples/SimplerEnv/start_simpler_env_ms3.py \
    --ckpt-path "${CKPT}" \
    --task "${task}" \
    --num-episodes "${EPISODES}" \
    --seed "${SEED}" \
    --shader "${SHADER}" \
    --host "${POLICY_HOST}" \
    --port "${POLICY_PORT}" \
    --action-ensemble "${ACTION_ENSEMBLE}" \
    --pass-step-to-server 1 \
    --record-dir "${SIMPLER_RECORD_DIR}" \
    --logging-dir "${task_dir}" 2>&1 | tee "${task_dir}/eval.log"
done

# shellcheck disable=SC2086
"${SIM_PYTHON}" examples/SimplerEnv/aggregate_simpler.py --eval_dir "${OUT_DIR}" --episodes_per_task "${EPISODES}" \
  --tasks ${TASKS}
