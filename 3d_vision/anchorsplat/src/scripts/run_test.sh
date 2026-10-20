#!/bin/bash
#
# Copyright (c) 2026 AnchorSplat Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$RECIPE_REPO_ROOT/.." && pwd)"
ANCHORSPLAT_ROOT="${ANCHORSPLAT_ROOT:-$WORKSPACE_ROOT/AnchorSplat}"
CONFIG="${CONFIG:-$ANCHORSPLAT_ROOT/src/depth_anything_3/configs/anchorsplat_mix.yaml}"
OUTPUT="${OUTPUT:-$ANCHORSPLAT_ROOT/outputs/anchorsplat_mix}"
NPUS="${NPUS:-8}"

: "${CKPT:?Set CKPT to an AnchorSplat training checkpoint}"

TRAINER="$ANCHORSPLAT_ROOT/src/train/trainer_mix.py"
[[ -f "$TRAINER" ]] || {
    echo "[ERROR] AnchorSplat trainer not found: $TRAINER" >&2
    echo "[ERROR] Run ./3d_vision/anchorsplat/src/scripts/setup.sh first." >&2
    exit 1
}
[[ -f "$CONFIG" ]] || {
    echo "[ERROR] Config not found: $CONFIG" >&2
    exit 1
}
[[ -f "$CKPT" ]] || {
    echo "[ERROR] Checkpoint not found: $CKPT" >&2
    exit 1
}

export PYTHONPATH="$ANCHORSPLAT_ROOT/src:${PYTHONPATH:-}"
command=(
    torchrun
    "--nproc_per_node=$NPUS"
    "$TRAINER"
    --config "$CONFIG"
    --output "$OUTPUT"
    --ckpt "$CKPT"
    --test
    "$@"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[DRY_RUN]'
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

command -v torchrun >/dev/null 2>&1 || {
    echo "[ERROR] Required command not found: torchrun" >&2
    exit 1
}

cd "$ANCHORSPLAT_ROOT"
exec "${command[@]}"
