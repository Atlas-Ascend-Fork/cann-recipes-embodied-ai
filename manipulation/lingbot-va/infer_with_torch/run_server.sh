#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LINGBOT_VA_ROOT:?Set LINGBOT_VA_ROOT to the pinned official checkout}"
: "${MODEL_ROOT:?Set MODEL_ROOT to the suite checkpoint directory}"
TASK_PYTHON="${PYTHON:-python3}"
CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}"
set +u
source "${CANN_ROOT}/set_env.sh"
set -u
"${TASK_PYTHON}" "${SCRIPT_DIR}/prepare.py" source "${LINGBOT_VA_ROOT}"
export PYTHONPATH="${SCRIPT_DIR}/src:${LINGBOT_VA_ROOT}/wan_va:${LINGBOT_VA_ROOT}:${PYTHONPATH:-}"
export LINGBOT_VA_MODEL_ROOT="${MODEL_ROOT}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
if [[ -n "${NORM_STATS:-}" ]]; then
    export LINGBOT_VA_NORM_STATS="${NORM_STATS}"
else
    unset LINGBOT_VA_NORM_STATS
fi
exec "${TASK_PYTHON}" -m torch.distributed.run --nproc_per_node "${NPROC_PER_NODE:-1}" \
    --master_port "${MASTER_PORT:-29881}" "${SCRIPT_DIR}/src/launch_server.py" \
    --config-name libero --port "${PORT:-29976}" --save_root "${SAVE_ROOT:-outputs/server}"
