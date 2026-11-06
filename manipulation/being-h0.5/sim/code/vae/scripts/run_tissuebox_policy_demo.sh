#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  cat <<'EOF'
Run one closed-loop Being-H0.5 + NativeVAE TissueBox episode.

Required:
  --model-path PATH                  Being-H0.5 post-trained checkpoint
  --vae-checkpoint PATH              NativeVAE inference checkpoint
  --control-reference-dataset PATH  Matching Shadow LeRobot dataset

Optional:
  --device DEVICE       Torch device (default: npu:0)
  --output-dir PATH     Result root (default: <vae>/results/tissuebox_demo)
  --python COMMAND      Python executable (default: python)
  -- EXTRA_ARGS         Forward remaining options to the evaluator
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VAE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH=""
VAE_CHECKPOINT=""
CONTROL_REFERENCE_DATASET=""
DEVICE="npu:0"
OUTPUT_DIR="${VAE_ROOT}/results/tissuebox_demo"
PYTHON_BIN="python"
EXTRA_ARGS=()

while (($#)); do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --vae-checkpoint) VAE_CHECKPOINT="$2"; shift 2 ;;
    --control-reference-dataset) CONTROL_REFERENCE_DATASET="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value_name in MODEL_PATH VAE_CHECKPOINT CONTROL_REFERENCE_DATASET; do
  if [[ -z "${!value_name}" ]]; then
    echo "Missing required option for ${value_name}" >&2
    usage >&2
    exit 2
  fi
done

exec "${PYTHON_BIN}" "${VAE_ROOT}/examples/beingh_shadow_grasp_eval.py" \
  --model-path "${MODEL_PATH}" \
  --vae-checkpoint "${VAE_CHECKPOINT}" \
  --control-reference-dataset "${CONTROL_REFERENCE_DATASET}" \
  --dataset "${VAE_ROOT}/evaluation/object_episodes/tissuebox_idx83_shadow.jsonl" \
  --scene-xml "${VAE_ROOT}/sim/assets/demos/tissuebox_idx83_shadow/scene.xml" \
  --scene-contains-object \
  --episode 0 \
  --hand shadow_hand_right \
  --instruction "Lift the tissue box from the table." \
  --success-profile dex-demo \
  --deployment-profile safe_smooth \
  --latent-observation-mode commanded \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
