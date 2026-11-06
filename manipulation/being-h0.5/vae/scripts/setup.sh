#!/usr/bin/env bash
# Copyright (c) 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VAE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RECIPE_ROOT="$(cd -- "${VAE_ROOT}/.." && pwd)"
WORK_DIR="${WORK_DIR:-$(pwd)}"
BEINGH_DIR="${BEINGH_DIR:-${WORK_DIR}/beingh05_vae}"
UPSTREAM_COMMIT="9443a4082bce013cc61e6523596940134179b23e"
UPSTREAM_REPO_DIR="${RECIPE_ROOT}/third_party/Being-H"
UPSTREAM_CODE_DIR="${UPSTREAM_REPO_DIR}/Being-H05"
UPSTREAM_MANIFEST="${RECIPE_ROOT}/common/upstream_files.txt"
COMMON_CODE_DIR="${RECIPE_ROOT}/common/code"
COMMON_PATCH_DIR="${RECIPE_ROOT}/common/patches"
VAE_CODE_DIR="${VAE_ROOT}/code"
VAE_PATCH_DIR="${VAE_ROOT}/patches"
INFER_CODE_DIR="${RECIPE_ROOT}/infer_with_torch/code"
INFER_PATCH_DIR="${RECIPE_ROOT}/infer_with_torch/patches"
SIM_CODE_DIR="${RECIPE_ROOT}/sim/code"

apply_delta() {
  local delta="$1"
  [[ -s "${delta}" ]] || { echo "Missing source delta: ${delta}" >&2; exit 1; }
  patch --batch --forward -l -F 0 -V none -d "${BEINGH_DIR}" -p2 < "${delta}"
}
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
[[ -f "${UPSTREAM_MANIFEST}" ]] || { echo "Missing upstream manifest: ${UPSTREAM_MANIFEST}" >&2; exit 1; }
[[ -d "${COMMON_CODE_DIR}" ]] || { echo "Missing common code: ${COMMON_CODE_DIR}" >&2; exit 1; }
[[ -d "${VAE_CODE_DIR}" ]] || { echo "Missing VAE code: ${VAE_CODE_DIR}" >&2; exit 1; }
[[ -d "${INFER_CODE_DIR}" ]] || { echo "Missing inference code: ${INFER_CODE_DIR}" >&2; exit 1; }
[[ -d "${SIM_CODE_DIR}" ]] || { echo "Missing simulation code: ${SIM_CODE_DIR}" >&2; exit 1; }
[[ -f "${VAE_CODE_DIR}/vae/native_vae/api.py" ]] || { echo "Missing vendored NativeVAE implementation" >&2; exit 1; }
if [[ -e "${BEINGH_DIR}" ]] && [[ -n "$(find "${BEINGH_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite a non-empty Being-H0.5 path: ${BEINGH_DIR}" >&2
  echo "Set BEINGH_DIR to a new or empty directory." >&2
  exit 2
fi
# Verify resources before creating the source workspace. Code-only assembly is
# useful for Docker builds and static checks; execution still needs runtime assets.
RESOURCE_STAGE=""
if [[ "${SKIP_ASSETS:-0}" != "1" ]]; then
  RESOURCE_STAGE="$(mktemp -d)"
  trap 'rm -rf -- "${RESOURCE_STAGE}"' EXIT
  resource_args=(--dest "${RESOURCE_STAGE}")
  [[ -z "${BEINGH_ASSET_ARCHIVE:-}" ]] || resource_args+=(--archive "${BEINGH_ASSET_ARCHIVE}")
  [[ -z "${BEINGH_ASSET_URL:-}" ]] || resource_args+=(--url "${BEINGH_ASSET_URL}")
  python3 "${RECIPE_ROOT}/resources/download_assets.py" "${resource_args[@]}"
fi
mkdir -p "${BEINGH_DIR}"
# Copy the allowlisted files from the pinned public submodule.
while IFS= read -r relative_path; do
  [[ -z "${relative_path}" || "${relative_path}" == \#* ]] && continue
  source_path="${UPSTREAM_CODE_DIR}/${relative_path}"
  target_path="${BEINGH_DIR}/${relative_path}"
  [[ -f "${source_path}" ]] || { echo "Missing pinned upstream file: ${source_path}" >&2; exit 1; }
  mkdir -p "$(dirname "${target_path}")"
  cp -a "${source_path}" "${target_path}"
done < "${UPSTREAM_MANIFEST}"
# Only project-specific files and files modified from upstream remain here.
cp -a "${COMMON_CODE_DIR}/." "${BEINGH_DIR}/"
apply_delta "${COMMON_PATCH_DIR}/qwen2_npu_import.patch"
apply_delta "${COMMON_PATCH_DIR}/siglip_npu_import.patch"
apply_delta "${COMMON_PATCH_DIR}/data_config.patch"
apply_delta "${COMMON_PATCH_DIR}/video_backend_fallback.patch"
apply_delta "${INFER_PATCH_DIR}/internvit_eager.patch"
cp -a "${INFER_CODE_DIR}/." "${BEINGH_DIR}/"
# Preserve the complete training-only layer classes before the inference patch
# replaces layers.py with its optimized runtime subset.
cp -a "${UPSTREAM_CODE_DIR}/BeingH/model/layers.py" \
  "${BEINGH_DIR}/BeingH/model/layers_upstream.py"
apply_delta "${INFER_PATCH_DIR}/npu_inference.patch"
# The inference overlay carries slim standalone copies of these shared files.
# A combined workspace needs their complete upstream APIs for post-training;
# the NPU runtime itself lives in separate modules and remains untouched.
for shared_file in \
  BeingH/dataset/preprocess.py \
  BeingH/model/__init__.py \
  BeingH/utils/constants.py \
  BeingH/utils/conversation.py; do
  cp -a "${UPSTREAM_CODE_DIR}/${shared_file}" "${BEINGH_DIR}/${shared_file}"
done
# Copy the NativeVAE implementation and its post-training runtime.
cp -a "${VAE_CODE_DIR}/." "${BEINGH_DIR}/"
apply_delta "${VAE_PATCH_DIR}/layers_training_compat.patch"
apply_delta "${VAE_PATCH_DIR}/posttrain.patch"
apply_delta "${VAE_PATCH_DIR}/dataset_info_local.patch"
# The combined workspace needs the model variant that retains both forward()
# for post-training and get_action() for accelerated evaluation.
cp -a "${UPSTREAM_CODE_DIR}/BeingH/model/beingvla.py" \
  "${BEINGH_DIR}/BeingH/model/beingvla.py"
apply_delta "${COMMON_PATCH_DIR}/beingvla.patch"
apply_delta "${INFER_PATCH_DIR}/beingvla_prefix_torchair.patch"
apply_delta "${INFER_PATCH_DIR}/beingvla_policy_compat.patch"
# MuJoCo is an independent source layer. It is copied last because it adds
# evaluation entry points and assets without replacing the NativeVAE runtime.
cp -a "${SIM_CODE_DIR}/." "${BEINGH_DIR}/"
if [[ -n "${RESOURCE_STAGE}" ]]; then
  cp -a "${RESOURCE_STAGE}/." "${BEINGH_DIR}/"
else
  echo "Code-only workspace: install resources/download_assets.py before running VAE/simulation."
fi
echo "Being-H0.5 VAE Ascend source is ready at ${BEINGH_DIR}"
echo "Official Being-H source: ${UPSTREAM_REPO_DIR} (${UPSTREAM_COMMIT})"
echo "Local common overlays: ${COMMON_CODE_DIR}"
echo "VAE post-training code: ${VAE_CODE_DIR}"
echo "Inference code: ${INFER_CODE_DIR}"
echo "MuJoCo simulation code: ${SIM_CODE_DIR}"
echo "Install dependencies with: python -m pip install -r ${BEINGH_DIR}/requirements.txt"
echo "Install VAE package with:  python -m pip install -e ${BEINGH_DIR}/vae"
echo "Install MuJoCo extras with: python -m pip install -r ${SIM_CODE_DIR}/../requirements.txt"
