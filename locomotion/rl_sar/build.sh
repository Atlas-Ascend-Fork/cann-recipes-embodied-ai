#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${RL_SAR_UPSTREAM_DIR:-${ROOT_DIR}/upstream}"
WORK_DIR="${ROOT_DIR}/cmake_build"
# Generated source, binaries, verification artifacts and logs share this root.
SOURCE_DIR="${WORK_DIR}/rl_sar-src"
BUILD_DIR="${WORK_DIR}/build"
BUILD_TYPE="Release"
ENABLE_OM="ON"

if [[ -L "${WORK_DIR}" ]]; then
    echo "Refusing a symlinked build directory: ${WORK_DIR}" >&2
    exit 1
fi

usage() {
    cat <<'EOF'
Usage: ./build.sh [--debug|--release] [--without-om] [--clean]

  --debug       Build with debug symbols
  --release     Build optimized binaries (default)
  --without-om  Build the upstream ONNX/PyTorch path without CANN
  --clean       Remove cmake_build/ (builds, checks and logs) and exit
EOF
}

while (($#)); do
    case "$1" in
        --debug) BUILD_TYPE="Debug" ;;
        --release) BUILD_TYPE="Release" ;;
        --without-om) ENABLE_OM="OFF" ;;
        --clean)
            rm -rf "${WORK_DIR}"
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ ! -f "${UPSTREAM_DIR}/src/rl_sar/library/core/rl_sdk/rl_sdk.cpp" ]]; then
    echo "Incomplete rl_sar submodule: ${UPSTREAM_DIR}" >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi
if [[ ! -f "${UPSTREAM_DIR}/src/rl_sar/library/thirdparty/unitree_sdk2-2.0.0/CMakeLists.txt" ]]; then
    echo "Missing nested unitree_sdk2 submodule under ${UPSTREAM_DIR}" >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

EXPECTED_UPSTREAM="2c26c1c0c3c1102b6097aef2ee44035f0f07eaa5"
EXPECTED_SDK="65e19c625b76b697f94e0b953e296ecc0ce7b4a1"
SDK_DIR="${UPSTREAM_DIR}/src/rl_sar/library/thirdparty/unitree_sdk2-2.0.0"
if [[ "$(git -C "${UPSTREAM_DIR}" rev-parse HEAD)" != "${EXPECTED_UPSTREAM}" ||
      "$(git -C "${SDK_DIR}" rev-parse HEAD)" != "${EXPECTED_SDK}" ]]; then
    echo "Upstream revisions differ from the pinned rl_sar dependencies" >&2
    exit 1
fi
if ! git -C "${UPSTREAM_DIR}" diff --quiet HEAD -- src/rl_sar; then
    echo "Upstream source has local changes; restore it before building" >&2
    exit 1
fi

rm -rf "${SOURCE_DIR}" "${BUILD_DIR}"
mkdir -p "${WORK_DIR}"
cmake -E copy_directory "${UPSTREAM_DIR}/src/rl_sar" "${SOURCE_DIR}"
patch --batch --forward --fuzz=0 -l -d "${SOURCE_DIR}" -p1 \
    < "${ROOT_DIR}/patches/0001-external-inference-backend.patch"
if [[ "${ENABLE_OM}" == "ON" ]]; then
    cmake -E copy \
        "${ROOT_DIR}/config/g1/robomimic/beyonddance/config.yaml" \
        "${SOURCE_DIR}/policy/g1/robomimic/beyonddance/config.yaml"
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DRL_SAR_SOURCE_DIR="${SOURCE_DIR}" \
    -DRL_SAR_ENABLE_OM="${ENABLE_OM}" \
    -DUSE_CMAKE=ON \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "${BUILD_DIR}" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-4}"

echo "Built: ${BUILD_DIR}/bin/rl_real_g1"
