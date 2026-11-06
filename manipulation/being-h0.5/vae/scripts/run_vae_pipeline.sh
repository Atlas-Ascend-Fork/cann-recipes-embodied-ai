#!/usr/bin/env bash
# Copyright (c) 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0.
set -euo pipefail
BEINGH_DIR="${BEINGH_DIR:?Set BEINGH_DIR to the prepared Being-H0.5 checkout}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ASCEND_DRIVER_ENV="${ASCEND_DRIVER_ENV:-/usr/local/Ascend/driver/bin/setenv.bash}"
ASCEND_CANN_ENV="${ASCEND_CANN_ENV:?Set ASCEND_CANN_ENV to the CANN set_env.sh path}"
VAE_DEVICE="${VAE_DEVICE:-npu:0}"
TRAIN_VAE="${TRAIN_VAE:-1}"
RUN_POSTTRAIN="${RUN_POSTTRAIN:-1}"
MAX_STEPS="${MAX_STEPS:-40000}"
SOURCE_DATASET="${SOURCE_DATASET:?Set SOURCE_DATASET to the source LeRobot dataset}"
OUTPUT_DATASET="${OUTPUT_DATASET:?Set OUTPUT_DATASET to a new latent-z dataset path}"
DATASET_NAME="${DATASET_NAME:?Set DATASET_NAME to the output dataset directory name}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-}"
set +u
source "${ASCEND_DRIVER_ENV}"
source "${ASCEND_CANN_ENV}"
set -u
cd "${BEINGH_DIR}"
if [[ "${TRAIN_VAE}" == "1" ]]; then
  "${PYTHON_BIN}" vae/scripts/generate_random_data.py \
    --config vae/configs/train_native_n2.yaml \
    --device "${VAE_DEVICE}"
  "${PYTHON_BIN}" vae/scripts/train.py \
    --config vae/configs/train_native_n2.yaml \
    --device "${VAE_DEVICE}" \
    --no_wandb
fi
if [[ -z "${VAE_CHECKPOINT}" ]]; then
  VAE_CHECKPOINT="$(find vae/runs -path '*/checkpoints/inference.pt' -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
fi
[[ -f "${VAE_CHECKPOINT}" ]] || { echo "VAE checkpoint not found: ${VAE_CHECKPOINT}" >&2; exit 3; }
[[ ! -e "${OUTPUT_DATASET}" ]] || { echo "Refusing to overwrite ${OUTPUT_DATASET}" >&2; exit 4; }
"${PYTHON_BIN}" vae/scripts/build_lerobot_z_dataset.py \
  --source-dataset "${SOURCE_DATASET}" \
  --output-dataset "${OUTPUT_DATASET}" \
  --vae-checkpoint "${VAE_CHECKPOINT}" \
  --device "${VAE_DEVICE}" \
  --batch-size "${Z_BATCH_SIZE:-4096}"
"${PYTHON_BIN}" vae/scripts/verify_lerobot_z_dataset.py \
  --source-dataset "${SOURCE_DATASET}" \
  --converted-dataset "${OUTPUT_DATASET}" \
  --output "${OUTPUT_DATASET}.verification.json" \
  --vae-checkpoint "${VAE_CHECKPOINT}" \
  --device "${VAE_DEVICE}" \
  --atol "${Z_VERIFY_ATOL:-0.002}" \
  --rtol 0
if [[ "${RUN_POSTTRAIN}" == "1" ]]; then
  ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}" \
  NUM_GPUS="${NUM_GPUS:-2}" \
  MAX_STEPS="${MAX_STEPS}" \
  GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}" \
  MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-4096}" \
  EXPECTED_NUM_TOKENS="${EXPECTED_NUM_TOKENS:-4096}" \
  EMBODIMENT_DATASET="${DATASET_NAME}" \
  NORMALIZATION=wrist_rot6d_minmax_zraw \
  BEINGH_ENV="${BEINGH_ENV:-${CONDA_PREFIX:-}}" \
  bash scripts/train/train_shadow_grasp_npu.sh
fi
