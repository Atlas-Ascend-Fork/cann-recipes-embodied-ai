#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export REPO_ROOT
SHARED_ROOT="${REPO_ROOT}/beingh_shared_scripts"
exec bash "${SHARED_ROOT}/train/run_full_vae_npu_pipeline.sh" "$@"
