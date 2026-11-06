#!/usr/bin/env bash
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"
source "${SCRIPT_DIR}/train_common.sh"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
ACCELERATOR="${ACCELERATOR:-npu}"
if [[ "${ACCELERATOR}" != "npu" ]]; then
  printf '[shadow-grasp] ERROR: Only NPU training is supported; ACCELERATOR must be npu, got: %s\n' \
    "${ACCELERATOR}" >&2
  exit 2
fi
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}"
log() {
  printf '[shadow-grasp] %s\n' "$*"
}
die() {
  printf '[shadow-grasp] ERROR: %s\n' "$*" >&2
  exit 2
}
on_exit() {
  status=$?
  if (( status != 0 )); then
    printf '[shadow-grasp] FAILED (exit=%d). Log: %s\n' \
      "${status}" "${LOG_FILE:-not-created}" >&2
  fi
}
trap on_exit EXIT
EMBODIMENT="${EMBODIMENT:-shadow_grasp}"
EMBODIMENT_DATASET="${EMBODIMENT_DATASET:-shadow_grasp_0725_core_bottle_1071}"
DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/outputs/shadow_grasp_bottle_1071"
if [[ "${EMBODIMENT_DATASET}" != "shadow_grasp_0725_core_bottle_1071" ]]; then
  DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/outputs/${EMBODIMENT_DATASET}"
fi
PRETRAIN_MODEL="${PRETRAIN_MODEL:-${REPO_ROOT}/ckpts/InternVL3_5-2B}"
EXPERT_MODEL="${EXPERT_MODEL:-${REPO_ROOT}/ckpts/Qwen3-0.6B}"
RESUME_PATH="${RESUME_PATH:-${REPO_ROOT}/ckpts/Being-H05-2B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun)}"
BEINGH_ENV="${BEINGH_ENV:-$(dirname "$(dirname "${PYTHON_BIN}")")}"
NORMALIZATION="${NORMALIZATION:-q99}"
case "${NORMALIZATION}" in
  q99|minmax|wrist_minmax_zraw|wrist_euler_minmax_zraw|wrist_rot6d_minmax_zraw|wrist_rot6d_minmax_joints)
    DEFAULT_DATASET_CONFIG_FILE="${REPO_ROOT}/configs/posttrain/${EMBODIMENT}/${EMBODIMENT_DATASET}_${NORMALIZATION}.yaml" ;;
  none) DEFAULT_DATASET_CONFIG_FILE="${REPO_ROOT}/configs/posttrain/${EMBODIMENT}/${EMBODIMENT_DATASET}.yaml" ;;
  *)
    die "NORMALIZATION must be 'q99', 'minmax', 'wrist_minmax_zraw', 'wrist_euler_minmax_zraw', 'wrist_rot6d_minmax_zraw', 'wrist_rot6d_minmax_joints', or 'none', got: ${NORMALIZATION}"
    ;;
esac
DATASET_CONFIG_FILE="${DATASET_CONFIG_FILE:-${DEFAULT_DATASET_CONFIG_FILE}}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/data/${EMBODIMENT_DATASET}}"
# dataset_info.py resolves named datasets below BEINGH_DATA_ROOT.  Keep the
# runtime registry aligned when DATASET_PATH is overridden for external data.
export BEINGH_DATA_ROOT="${BEINGH_DATA_ROOT:-$(dirname "${DATASET_PATH}")}"
SAVE_MERGED_META="${SAVE_MERGED_META:-True}"
GENERATE_STATS="${GENERATE_STATS:-True}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export BEINGH_DEVICE_TYPE=npu
export BEINGH_DIST_BACKEND=hccl
VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}"
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29107}"
MAX_STEPS="${MAX_STEPS:-40000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_STEPS_START="${SAVE_STEPS_START:-15000}"
SAVE_MODEL_ONLY="${SAVE_MODEL_ONLY:-True}"
RESUME_MODEL_ONLY="${RESUME_MODEL_ONLY:-True}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
FUSED_OPTIMIZER="${FUSED_OPTIMIZER:-False}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SHARDING_STRATEGY="${SHARDING_STRATEGY:-SHARD_GRAD_OP}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-8704}"
EXPECTED_NUM_TOKENS="${EXPECTED_NUM_TOKENS:-8192}"
PREFER_BUFFER_BEFORE="${PREFER_BUFFER_BEFORE:-4096}"
MAX_BUFFER_SIZE="${MAX_BUFFER_SIZE:-4}"
ATTN_MODE="${ATTN_MODE:-causal}"
FORCE_IMAGE_SIZE="${FORCE_IMAGE_SIZE:-224}"
MAX_VIEW_NUM="${MAX_VIEW_NUM:--1}"
USE_FIXED_VIEW="${USE_FIXED_VIEW:-False}"
DOWN_SAMPLE_RATIO="${DOWN_SAMPLE_RATIO:-0.5}"
ACTION_CHUNK_LENGTH="${ACTION_CHUNK_LENGTH:-16}"
WRIST_ACTION_LOSS_WEIGHT="${WRIST_ACTION_LOSS_WEIGHT:-1.0}"
DEFAULT_WRIST_ACTION_DIM=6
if [[ "${NORMALIZATION}" == "wrist_rot6d_minmax_zraw" \
   || "${NORMALIZATION}" == "wrist_rot6d_minmax_joints" ]]; then
  DEFAULT_WRIST_ACTION_DIM=9
fi
WRIST_ACTION_DIM="${WRIST_ACTION_DIM:-${DEFAULT_WRIST_ACTION_DIM}}"
TEMPORAL_DELTA_LOSS_WEIGHT="${TEMPORAL_DELTA_LOSS_WEIGHT:-0.0}"
TEMPORAL_DELTA_HUBER_BETA="${TEMPORAL_DELTA_HUBER_BETA:-1.0}"
FREEZE_MLLM="${FREEZE_MLLM:-True}"
FREEZE_VIT_MLP="${FREEZE_VIT_MLP:-True}"
USE_MPG="${USE_MPG:-True}"
MPG_LAMBDA="${MPG_LAMBDA:-0.1}"
MPG_NUM_PROJECTIONS="${MPG_NUM_PROJECTIONS:-32}"
MPG_REFINEMENT_ITERS="${MPG_REFINEMENT_ITERS:-1}"
MPG_GATE_TEMPERATURE="${MPG_GATE_TEMPERATURE:-1.0}"
MPG_USE_STOP_GRADIENT="${MPG_USE_STOP_GRADIENT:-True}"
USE_TRAINING_TIME_RTC="${USE_TRAINING_TIME_RTC:-False}"
SIMULATED_DELAY="${SIMULATED_DELAY:-0}"
RTC_DELAY_EXP_WEIGHT="${RTC_DELAY_EXP_WEIGHT:-True}"
USE_INFERENCE_PREFIX_OVERWRITE="${USE_INFERENCE_PREFIX_OVERWRITE:-True}"
SMOKE_TEST="${SMOKE_TEST:-False}"
PERF_TEST="${PERF_TEST:-False}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-False}"
if [[ "${SMOKE_TEST}" == "True" || "${SMOKE_TEST}" == "true" ]]; then
  MAX_STEPS="${SMOKE_MAX_STEPS:-20}"
  SAVE_STEPS="${SMOKE_SAVE_STEPS:-20}"
  SAVE_STEPS_START="${SMOKE_SAVE_STEPS_START:-0}"
  SAVE_MODEL_ONLY="${SMOKE_SAVE_MODEL_ONLY:-True}"
  NUM_WORKERS="${SMOKE_NUM_WORKERS:-2}"
  PREFETCH_FACTOR="${SMOKE_PREFETCH_FACTOR:-2}"
fi
if [[ "${PERF_TEST}" == "True" || "${PERF_TEST}" == "true" ]]; then
  MAX_STEPS="${PERF_MAX_STEPS:-30}"
  SAVE_STEPS="${PERF_SAVE_STEPS:-30}"
  SAVE_STEPS_START="${PERF_SAVE_STEPS_START:-30}"
  LOGGING_STEPS="${PERF_LOGGING_STEPS:-5}"
  SAVE_MODEL_ONLY=True
fi
command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
  || die "Python executable not found: ${PYTHON_BIN}. Activate the beingh environment."
command -v "${TORCHRUN_BIN}" >/dev/null 2>&1 \
  || die "torchrun executable not found: ${TORCHRUN_BIN}. Activate the beingh environment."
for model_dir in "${PRETRAIN_MODEL}" "${EXPERT_MODEL}" "${RESUME_PATH}"; do
  [[ -d "${model_dir}" ]] || die "Model directory not found: ${model_dir}"
  [[ -f "${model_dir}/config.json" ]] || die "Missing ${model_dir}/config.json"
  model_weight_found=False
  for model_weight in "${model_dir}"/*.safetensors; do
    if [[ -f "${model_weight}" ]]; then
      model_weight_found=True
      break
    fi
  done
  [[ "${model_weight_found}" == "True" ]] \
    || die "No safetensors found in ${model_dir}"
done
[[ -f "${PRETRAIN_MODEL}/tokenizer_config.json" ]] \
  || die "Missing ${PRETRAIN_MODEL}/tokenizer_config.json"
[[ -f "${EXPERT_MODEL}/tokenizer_config.json" ]] \
  || die "Missing ${EXPERT_MODEL}/tokenizer_config.json"
[[ -f "${RESUME_PATH}/tokenizer_config.json" ]] \
  || die "Missing ${RESUME_PATH}/tokenizer_config.json"
[[ -f "${DATASET_CONFIG_FILE}" ]] \
  || die "Dataset config not found: ${DATASET_CONFIG_FILE}"
for meta_file in info.json episodes.jsonl episodes_stats.jsonl tasks.jsonl; do
  [[ -f "${DATASET_PATH}/meta/${meta_file}" ]] \
    || die "Dataset metadata not found: ${DATASET_PATH}/meta/${meta_file}"
done
PARQUET_COUNT="$(find "${DATASET_PATH}/data" -type f -name '*.parquet' | wc -l)"
VIDEO_COUNT="$(find "${DATASET_PATH}/videos" -type f -name '*.mp4' | wc -l)"
(( PARQUET_COUNT > 0 )) || die "No parquet files found under ${DATASET_PATH}/data"
(( VIDEO_COUNT > 0 )) || die "No MP4 files found under ${DATASET_PATH}/videos"
IFS=',' read -r -a SELECTED_GPUS <<< "${VISIBLE_DEVICES}"
(( NUM_GPUS > 0 )) || die "NUM_GPUS must be positive"
(( NUM_GPUS <= ${#SELECTED_GPUS[@]} )) \
  || die "NUM_GPUS=${NUM_GPUS}, but visible devices=${VISIBLE_DEVICES}"
if [[ "${ACCELERATOR}" == "npu" ]]; then
  "${PYTHON_BIN}" -c \
    'import torch, torch_npu; assert torch.npu.is_available(), "NPU unavailable"' \
    || die "torch_npu is unavailable in ${PYTHON_BIN}"
fi
for integer_value in \
  "${MAX_STEPS}" "${SAVE_STEPS}" "${SAVE_STEPS_START}" \
  "${NUM_WORKERS}" "${PREFETCH_FACTOR}" "${ACTION_CHUNK_LENGTH}" "${WRIST_ACTION_DIM}" \
  "${GRADIENT_ACCUMULATION_STEPS}" "${MAX_NUM_TOKENS}" \
  "${EXPECTED_NUM_TOKENS}"; do
  [[ "${integer_value}" =~ ^[0-9]+$ ]] \
    || die "Expected a non-negative integer, got: ${integer_value}"
done
(( MAX_STEPS > 0 )) || die "MAX_STEPS must be positive"
(( SAVE_STEPS > 0 )) || die "SAVE_STEPS must be positive"
(( ACTION_CHUNK_LENGTH > 0 )) || die "ACTION_CHUNK_LENGTH must be positive"
(( WRIST_ACTION_DIM > 0 )) || die "WRIST_ACTION_DIM must be positive"
(( GRADIENT_ACCUMULATION_STEPS > 0 )) \
  || die "GRADIENT_ACCUMULATION_STEPS must be positive"
(( EXPECTED_NUM_TOKENS <= MAX_NUM_TOKENS )) \
  || die "EXPECTED_NUM_TOKENS cannot exceed MAX_NUM_TOKENS"
# Calculate statistics exactly once before torchrun. Otherwise every rank can
# race while creating the same meta/stats.json.
STATS_FILE="${DATASET_PATH}/meta/stats.json"
if [[ ! -f "${STATS_FILE}" ]]; then
  if [[ "${GENERATE_STATS}" == "True" || "${GENERATE_STATS}" == "true" ]]; then
    log "Generating ${STATS_FILE} from ${PARQUET_COUNT} parquet files..."
    "${PYTHON_BIN}" - "${DATASET_PATH}" <<'PY'
import json
import os
import sys
from pathlib import Path
from BeingH.dataset.parquet_utils import calculate_dataset_statistics
dataset_path = Path(sys.argv[1])
parquet_files = sorted(dataset_path.glob("data/**/*.parquet"))
if not parquet_files:
    raise RuntimeError(f"No parquet files found in {dataset_path / 'data'}")
statistics = calculate_dataset_statistics(parquet_files)
output_path = dataset_path / "meta/stats.json"
temporary_path = output_path.with_suffix(".json.tmp")
temporary_path.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
os.replace(temporary_path, output_path)
print(f"Saved statistics to {output_path}")
PY
  else
    die "Missing ${STATS_FILE}; set GENERATE_STATS=True or create it first."
  fi
fi
for message in \
  "Preflight passed" \
  "Dataset: ${DATASET_PATH} (${PARQUET_COUNT} parquet, ${VIDEO_COUNT} videos)" \
  "Data config: ${DATASET_CONFIG_FILE}; normalization=${NORMALIZATION}" \
  "InternVL: ${PRETRAIN_MODEL}" "Expert: ${EXPERT_MODEL}" "Resume: ${RESUME_PATH}" \
  "Accelerator=${ACCELERATOR}; visible_devices=${VISIBLE_DEVICES}; ranks=${NUM_GPUS}" \
  "Steps=${MAX_STEPS}; save_every=${SAVE_STEPS}; save_start=${SAVE_STEPS_START}" \
  "Checkpoint: save_model_only=${SAVE_MODEL_ONLY}; resume_model_only=${RESUME_MODEL_ONLY}" \
  "Packed tokens: target=${EXPECTED_NUM_TOKENS}; hard_max=${MAX_NUM_TOKENS}" \
  "Workers/rank=${NUM_WORKERS}; prefetch=${PREFETCH_FACTOR}; fused AdamW=${FUSED_OPTIMIZER}" \
  "FSDP=${SHARDING_STRATEGY}; grad_accum=${GRADIENT_ACCUMULATION_STEPS}" \
  "Freeze MLLM=${FREEZE_MLLM}; freeze connector=${FREEZE_VIT_MLP}" \
  "Wrist action loss weight=${WRIST_ACTION_LOSS_WEIGHT} (unified slots 0:${WRIST_ACTION_DIM})" \
  "Temporal delta loss weight=${TEMPORAL_DELTA_LOSS_WEIGHT}; Huber beta=${TEMPORAL_DELTA_HUBER_BETA}"; do log "${message}"; done
if [[ "${PREFLIGHT_ONLY}" == "True" || "${PREFLIGHT_ONLY}" == "true" ]]; then
  log "PREFLIGHT_ONLY=True; torchrun was not launched."
  trap - EXIT
  exit 0
fi
RESUME_MODEL="$(basename "${RESUME_PATH}")"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_KIND="train"
if [[ "${SMOKE_TEST}" == "True" || "${SMOKE_TEST}" == "true" ]]; then
  RUN_KIND="smoke"
elif [[ "${PERF_TEST}" == "True" || "${PERF_TEST}" == "true" ]]; then
  RUN_KIND="perf"
fi
DEFAULT_RUN_NAME="${RUN_KIND}-${EMBODIMENT_DATASET}_${RESUME_MODEL}_freeze-mllm-${FREEZE_MLLM}_chunk-${ACTION_CHUNK_LENGTH}_tok-${EXPECTED_NUM_TOKENS}_norm-${NORMALIZATION}_wristw-${WRIST_ACTION_LOSS_WEIGHT}_tdelta-${TEMPORAL_DELTA_LOSS_WEIGHT}_mpg-${USE_MPG}_${RUN_TIMESTAMP}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/tensorboard}"
LOG_FILE="${OUTPUT_DIR}/training.log"
SNAPSHOT_DIR="${OUTPUT_DIR}/run_config"
mkdir -p "${SNAPSHOT_DIR}" "${LOG_DIR}"
for snapshot in "${SCRIPT_DIR}/train_shadow_grasp.sh" "${SCRIPT_DIR}/train_shadow_grasp.args" "${DATASET_CONFIG_FILE}" \
  "${REPO_ROOT}/configs/data_config.py" "${REPO_ROOT}/configs/dataset_info.py"; do
  if [[ -f "${snapshot}" ]]; then
    cp "${snapshot}" "${SNAPSHOT_DIR}/"
  else
    log "Skipping unavailable snapshot: ${snapshot}"
  fi
done
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse HEAD > "${SNAPSHOT_DIR}/git_commit.txt"
  git status --short > "${SNAPSHOT_DIR}/git_status.txt"
  git diff -- BeingH configs scripts/train \
    > "${SNAPSHOT_DIR}/local_changes.patch"
fi
log "Output: ${OUTPUT_DIR}"
log "Log: ${LOG_FILE}"
TRAIN_ARGS=()
while read -r flag value_source; do
  [[ -n "${flag}" && "${flag}" != \#* ]] || continue
  if [[ "${value_source}" == =* ]]; then value="${value_source#=}"; else value="${!value_source}"; fi
  TRAIN_ARGS+=("${flag}" "${value}")
done < "${SCRIPT_DIR}/train_shadow_grasp.args"
"${TORCHRUN_BIN}" \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  BeingH/train/train.py \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"
trap - EXIT
printf '%s\n' \
  "==========================================" \
  "Training complete" \
  "Output: ${OUTPUT_DIR}" \
  "Log: ${LOG_FILE}" \
  "=========================================="
