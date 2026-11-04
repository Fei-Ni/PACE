#!/usr/bin/env bash
# Evaluate PACE on LIBERO (client side; start examples/LIBERO/run_server.sh first).
# Default protocol: 4 suites (spatial, object, goal, 10/long) x 10 tasks x 50 trials, seed 7.
#
# Usage: bash examples/LIBERO/eval_libero.sh [CKPT] [--dry-run]
#   CKPT  same checkpoint reference the server loads (used here for config + action statistics only)
# Environment:
#   LIBERO_PYTHON  python of the LIBERO environment (default python)
#   LIBERO_HOME    optional LIBERO checkout added to PYTHONPATH (not needed if LIBERO is pip-installed)
#   POLICY_HOST / POLICY_PORT    policy server address (default 127.0.0.1 / 10093)
#   SUITES         space-separated suites (default "libero_spatial libero_object libero_goal libero_10")
#   NUM_TRIALS     trials per task (default 50)   SEED  env seed (default 7)
#   OUT_DIR        output directory (default results/libero_eval/<timestamp>)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CKPT="${CKPT:-checkpoints/PACE-LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
LIBERO_HOME="${LIBERO_HOME:-}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-10093}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEED="${SEED:-7}"
OUT_DIR="${OUT_DIR:-results/libero_eval/$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

POSITIONAL=()
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) POSITIONAL+=("${arg}") ;;
  esac
done
if [[ ${#POSITIONAL[@]} -ge 1 ]]; then CKPT="${POSITIONAL[0]}"; fi

echo "EFFECTIVE: role=client benchmark=LIBERO ckpt=${CKPT} server=${POLICY_HOST}:${POLICY_PORT} suites=[${SUITES}]" \
  "trials_per_task=${NUM_TRIALS} seed=${SEED} out_dir=${OUT_DIR} python=${LIBERO_PYTHON}"
if [[ "${DRY_RUN}" == "1" ]]; then exit 0; fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${LIBERO_HOME:+:${LIBERO_HOME}}${PYTHONPATH:+:${PYTHONPATH}}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# MuJoCo requires the EGL device to be one of the visible GPUs: default to the first entry of CUDA_VISIBLE_DEVICES.
FIRST_VISIBLE_GPU="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${FIRST_VISIBLE_GPU%%,*}}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HOME}/.libero}"
export TOKENIZERS_PARALLELISM=false

# LIBERO asks interactively for its data paths on first import; write the default config if it is missing.
if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  LIBERO_PKG="$("${LIBERO_PYTHON}" -c 'import importlib.util as u; print(u.find_spec("libero").submodule_search_locations[0])')"
  BENCH_ROOT="${LIBERO_PKG}/libero"
  mkdir -p "${LIBERO_CONFIG_PATH}"
  cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<CFG
benchmark_root: ${BENCH_ROOT}
bddl_files: ${BENCH_ROOT}/bddl_files
init_states: ${BENCH_ROOT}/init_files
datasets: ${BENCH_ROOT}/../datasets
assets: ${BENCH_ROOT}/assets
CFG
  echo "Wrote default LIBERO config to ${LIBERO_CONFIG_PATH}/config.yaml"
fi

mkdir -p "${OUT_DIR}"
for suite in ${SUITES}; do
  suite_dir="${OUT_DIR}/${suite}"
  mkdir -p "${suite_dir}/videos"
  echo "=== ${suite}: ${NUM_TRIALS} trials/task, seed ${SEED} ==="
  "${LIBERO_PYTHON}" examples/LIBERO/eval_libero.py \
    --args.pretrained-path "${CKPT}" \
    --args.host "${POLICY_HOST}" \
    --args.port "${POLICY_PORT}" \
    --args.task-suite-name "${suite}" \
    --args.num-trials-per-task "${NUM_TRIALS}" \
    --args.seed "${SEED}" \
    --args.video-out-path "${suite_dir}/videos" 2>&1 | tee "${suite_dir}/eval.log"
done

# shellcheck disable=SC2086
"${LIBERO_PYTHON}" examples/LIBERO/aggregate_libero.py --eval_dir "${OUT_DIR}" --suites ${SUITES}
