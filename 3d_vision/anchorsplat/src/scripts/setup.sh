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
ANCHORSPLAT_REPO_URL="${ANCHORSPLAT_REPO_URL:-https://github.com/Zhang-Xiaoxue/AnchorSplat.git}"
ANCHORSPLAT_COMMIT="7f1625fccc431bd53c961649312a1b2a077e39f4"

SKIP_DEPS=false
SKIP_RUNTIME_CHECK=false

usage() {
    cat <<USAGE
Usage: $0 [OPTIONS]

Clone AnchorSplat, pin it to the recipe revision and install its
platform-independent Python dependencies.

Options:
  --skip-deps           Do not install Python dependencies.
  --skip-runtime-check  Skip torch_npu/ACL/Gaussian operator import checks.
  -h, --help            Show this help message.

Environment variables:
  ANCHORSPLAT_ROOT      Checkout path (default: $WORKSPACE_ROOT/AnchorSplat).
  ANCHORSPLAT_REPO_URL  Upstream Git URL.

Install matching torch, torchvision and torch-npu packages before running this
script. Source the CANN set_env.sh in the current shell as well.
USAGE
}

info() {
    echo "[INFO] $*"
}

error() {
    echo "[ERROR] $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --skip-runtime-check)
            SKIP_RUNTIME_CHECK=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown argument: $1"
            ;;
    esac
done

command -v git >/dev/null 2>&1 || error "Required command not found: git"
command -v python >/dev/null 2>&1 || error "Required command not found: python"

if [[ ! -d "$ANCHORSPLAT_ROOT/.git" ]]; then
    info "Cloning AnchorSplat into $ANCHORSPLAT_ROOT"
    git clone "$ANCHORSPLAT_REPO_URL" "$ANCHORSPLAT_ROOT"
fi

cd "$ANCHORSPLAT_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
    error "AnchorSplat checkout has local changes; refusing to switch revisions."
fi

git fetch origin "$ANCHORSPLAT_COMMIT" --depth=1 2>/dev/null || git fetch origin
git checkout --detach "$ANCHORSPLAT_COMMIT"
info "AnchorSplat pinned to $ANCHORSPLAT_COMMIT"

if [[ "$SKIP_DEPS" == false ]]; then
    python -c "import torch, torch_npu" >/dev/null 2>&1 || {
        error "Install matching torch and torch-npu packages before setup."
    }
    info "Installing AnchorSplat platform-independent dependencies"
    dependencies=(
        addict einops omegaconf opencv-python pillow imageio matplotlib tqdm
        typer huggingface_hub torchmetrics plyfile trimesh moviepy gradio
        fastapi uvicorn requests scipy scikit-learn open3d pycolmap evo
        easydict pillow-heif
    )
    python -m pip install "${dependencies[@]}"
fi

if [[ "$SKIP_RUNTIME_CHECK" == false ]]; then
    python -c "import torch, torch_npu, acl, meta_gauss_render"
    info "Verified torch, torch_npu, acl and meta_gauss_render imports"
fi

cat <<MSG

AnchorSplat workspace is ready.
Root:   $ANCHORSPLAT_ROOT
Commit: $ANCHORSPLAT_COMMIT

Export the source path when running commands manually:
  export PYTHONPATH=$ANCHORSPLAT_ROOT/src:\${PYTHONPATH:-}
MSG
