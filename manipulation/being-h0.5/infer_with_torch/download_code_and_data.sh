#!/usr/bin/env bash
# Copyright (c) 2026 CANN Contributors
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

UPSTREAM_COMMIT="9443a4082bce013cc61e6523596940134179b23e"
MODEL_REPO="BeingBeyond/Being-H05-2B_libero"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="$(pwd)"
BEINGH_DIR=""
MODEL_DIR=""
DOWNLOAD_MODEL=0

usage() {
  cat <<'EOF'
Usage: download_code_and_data.sh [options]

Options:
  --work-root PATH      Parent directory for source and model downloads.
  --beingh-dir PATH     Being-H checkout path. Default: WORK_ROOT/Being-H.
  --model-dir PATH      Model download path. Default: WORK_ROOT/Being-H05-2B_libero.
  --download-model      Download the official LIBERO checkpoint with the hf CLI.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --beingh-dir) BEINGH_DIR="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --download-model) DOWNLOAD_MODEL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

BEINGH_DIR="${BEINGH_DIR:-${WORK_ROOT}/Being-H}"
MODEL_DIR="${MODEL_DIR:-${WORK_ROOT}/Being-H05-2B_libero}"
BEINGH_SOURCE_DIR="${BEINGH_DIR}/Being-H05"
UPSTREAM_REPO_DIR="${SCRIPT_DIR}/../third_party/Being-H"
UPSTREAM_CODE_DIR="${UPSTREAM_REPO_DIR}/Being-H05"
COMMON_CODE_DIR="${SCRIPT_DIR}/../common/code"
COMMON_PATCH_DIR="${SCRIPT_DIR}/../common/patches"
INFER_CODE_DIR="${SCRIPT_DIR}/code"
INFER_PATCH_DIR="${SCRIPT_DIR}/patches"
RUNTIME_IMPORT_SLIM_PATCH="${SCRIPT_DIR}/patches/runtime_import_slim.patch"
RUNTIME_REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"
NPU_INFERENCE_PATCH="${INFER_PATCH_DIR}/npu_inference.patch"

[[ -f "${UPSTREAM_CODE_DIR}/BeingH/model/beingvla.py" ]] || {
  echo "Being-H submodule is not initialized: ${UPSTREAM_REPO_DIR}" >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 1
}
ACTUAL_UPSTREAM_COMMIT="$(git -C "${UPSTREAM_REPO_DIR}" rev-parse HEAD 2>/dev/null || true)"
[[ "${ACTUAL_UPSTREAM_COMMIT}" == "${UPSTREAM_COMMIT}" ]] || {
  echo "Unexpected Being-H submodule commit: ${ACTUAL_UPSTREAM_COMMIT:-unknown}" >&2
  echo "Expected: ${UPSTREAM_COMMIT}" >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 1
}
[[ -d "${COMMON_CODE_DIR}" ]] || { echo "Missing common overlays: ${COMMON_CODE_DIR}" >&2; exit 1; }
[[ -d "${INFER_CODE_DIR}" ]] || { echo "Missing inference code: ${INFER_CODE_DIR}" >&2; exit 1; }
[[ -f "${RUNTIME_IMPORT_SLIM_PATCH}" ]] || { echo "Missing runtime import slim patch: ${RUNTIME_IMPORT_SLIM_PATCH}" >&2; exit 1; }
[[ -f "${RUNTIME_REQUIREMENTS}" ]] || { echo "Missing runtime requirements: ${RUNTIME_REQUIREMENTS}" >&2; exit 1; }
grep -Eq '^(diff --git |--- )' "${NPU_INFERENCE_PATCH}" || { echo "Invalid NPU inference patch: ${NPU_INFERENCE_PATCH}" >&2; exit 1; }

if [[ -e "${BEINGH_DIR}" ]]; then
  echo "Refusing to modify an existing path: ${BEINGH_DIR}" >&2
  echo "Pass --beingh-dir with a new path." >&2
  exit 1
fi

mkdir -p "${WORK_ROOT}"
mkdir -p "${BEINGH_DIR}"
mkdir -p "${BEINGH_SOURCE_DIR}"

# Copy only shared modules imported by the production inference path.  A
# positive allowlist prevents future training, evaluation, VAE or profiling
# files added to common/code from leaking into this delivery.
COMMON_RUNTIME_FILES=(
  "BeingH/dataset/transform/__init__.py"
  "BeingH/dataset/transform/base.py"
  "BeingH/dataset/transform/state_action.py"
  "BeingH/inference/beingh_policy.py"
  "BeingH/inference/service.py"
  "BeingH/model/beingvla.py"
  "BeingH/model/cache_utils/taylorseer.py"
  "BeingH/model/layers.py"
  "BeingH/model/llm/qwen2/__init__.py"
  "BeingH/model/llm/qwen2/configuration_qwen2.py"
  "BeingH/model/llm/qwen2/modeling_qwen2.py"
  "BeingH/model/llm/qwen2/qwen2_navit_base.py"
  "BeingH/model/llm/qwen2/tokenization_qwen2.py"
  "BeingH/model/llm/qwen2/tokenization_qwen2_fast.py"
  "BeingH/model/llm/qwen2_navit.py"
  "BeingH/model/llm/qwen3/configuration_qwen3.py"
  "BeingH/model/llm/qwen3/modeling_qwen3.py"
  "BeingH/model/llm/qwen3_navit.py"
  "BeingH/model/vit_model/internvit/configuration_intern_vit.py"
  "BeingH/model/vit_model/internvit_navit.py"
  "BeingH/utils/schema.py"
  "configs/data_config.py"
)
for relative_path in "${COMMON_RUNTIME_FILES[@]}"; do
  source_path="${COMMON_CODE_DIR}/${relative_path}"
  if [[ ! -f "${source_path}" ]]; then
    source_path="${UPSTREAM_CODE_DIR}/${relative_path}"
  fi
  target_path="${BEINGH_SOURCE_DIR}/${relative_path}"
  [[ -f "${source_path}" ]] || { echo "Missing shared runtime file: ${source_path}" >&2; exit 1; }
  mkdir -p "$(dirname "${target_path}")"
  cp -a "${source_path}" "${target_path}"
done

patch --batch --forward -l -F 0 -V none -d "${BEINGH_SOURCE_DIR}" -p2 \
  < "${COMMON_PATCH_DIR}/qwen2_npu_import.patch"
patch --batch --forward -l -F 0 -V none -d "${BEINGH_SOURCE_DIR}" -p2 \
  < "${COMMON_PATCH_DIR}/data_config.patch"
patch --batch --forward -l -F 0 -V none -d "${BEINGH_SOURCE_DIR}" -p2 \
  < "${INFER_PATCH_DIR}/internvit_eager.patch"
mkdir -p "${BEINGH_SOURCE_DIR}/BeingH"
cp -a "${INFER_CODE_DIR}/BeingH/." "${BEINGH_SOURCE_DIR}/BeingH/"
patch --batch --forward -l -F 0 -V none -d "${BEINGH_SOURCE_DIR}" -p2 \
  < "${COMMON_PATCH_DIR}/beingvla.patch"
patch --batch --forward -l -F 0 -V none -d "${BEINGH_SOURCE_DIR}" -p2 \
  < "${INFER_PATCH_DIR}/beingvla_prefix_torchair.patch"
patch --batch --forward -l -F 0 -V none -d "${BEINGH_SOURCE_DIR}" -p2 \
  < "${NPU_INFERENCE_PATCH}"
patch --batch --forward -l -F 0 -V none -d "${BEINGH_SOURCE_DIR}" -p2 \
  < "${INFER_PATCH_DIR}/beingvla_policy_compat.patch"
patch --batch --forward -F 0 -V none -d "${BEINGH_SOURCE_DIR}" -p1 < "${RUNTIME_IMPORT_SLIM_PATCH}"
cp -a "${RUNTIME_REQUIREMENTS}" "${BEINGH_SOURCE_DIR}/requirements.txt"

echo "Being-H0.5 source prepared at ${BEINGH_SOURCE_DIR}"
echo "Official Being-H source: ${UPSTREAM_REPO_DIR} (${UPSTREAM_COMMIT})"
echo "Local common overlays: ${COMMON_CODE_DIR}"
echo "Inference code: ${INFER_CODE_DIR}"

if (( DOWNLOAD_MODEL == 1 )); then
  command -v hf >/dev/null || {
    echo "The hf CLI is required for --download-model." >&2
    exit 1
  }
  if [[ -e "${MODEL_DIR}" ]]; then
    echo "Refusing to overwrite an existing model path: ${MODEL_DIR}" >&2
    exit 1
  fi
  hf download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"
  echo "Model downloaded to ${MODEL_DIR}"
else
  echo "Model download skipped. Re-run with --download-model when required."
fi
