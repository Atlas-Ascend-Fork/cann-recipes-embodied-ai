#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# 第 1 步：拉 InternNav 上游、建模型推理用的 venv、下载权重。
#
# 前提：宿主机已装好 CANN 与匹配的 torch / torch_npu / torchvision（系统 Python）。
# 本样例验证环境：CANN 9.0.0 / 9.2.0 + torch 2.7.1 + torch_npu 2.7.1.post4 + torchvision 0.22.1，
# 系统 Python 3.11.4，aarch64。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# ---------- 1. 上游仓库 ----------
if [ ! -d "$INTERNNAV_ROOT/.git" ]; then
    info "克隆 InternNav 到 $INTERNNAV_ROOT"
    git clone "$GITCODE_MIRROR/in/InternNav.git" "$INTERNNAV_ROOT"
fi
git -C "$INTERNNAV_ROOT" fetch --all --tags --quiet || true
git -C "$INTERNNAV_ROOT" checkout --quiet "$INTERNNAV_COMMIT"
info "InternNav @ $(git -C "$INTERNNAV_ROOT" rev-parse --short HEAD)"

# internnav/model/encoder/__init__.py 会 eager import image_clip_encoder -> LongCLIP，
# 即使 InternVLA-N1 本身不用 LongCLIP，不初始化这个子模块也 import 不进去。
if [ ! -f "$INTERNNAV_ROOT/internnav/model/basemodel/LongCLIP/model/longclip.py" ]; then
    info "初始化 LongCLIP 子模块"
    git -C "$INTERNNAV_ROOT" submodule init internnav/model/basemodel/LongCLIP
    git -C "$INTERNNAV_ROOT" config \
        submodule.internnav/model/basemodel/LongCLIP.url "$GITCODE_MIRROR/lo/Long-CLIP.git"
    git -C "$INTERNNAV_ROOT" submodule update internnav/model/basemodel/LongCLIP
fi

# ---------- 2. 推理 venv ----------
# 用 venv 而不是 conda：torch_npu 已经和本机 CANN、驱动配好，--system-site-packages
# 直接继承即可，不必在新环境里重装难找的 aarch64 torch_npu wheel。
if [ ! -d "$INTERNNAV_ROOT/.venv" ]; then
    info "创建推理 venv（继承系统 torch / torch_npu）"
    python3 -m venv --system-site-packages "$INTERNNAV_ROOT/.venv"
fi
PIP="$INTERNNAV_ROOT/.venv/bin/pip"
info "安装推理依赖"
$PIP install -q -r "$SAMPLE_SRC/../requirements-npu.txt"
# --no-deps 跳过 core_requirements 里 ray / tyro 等只有评测服务模式才用到的重依赖
$PIP install -q -e "$INTERNNAV_ROOT" --no-deps

# ---------- 3. 权重 ----------
mkdir -p "$MODEL_STORE"
# internvla_n1_arch.py 里 MODEL_PATH_TO = "checkpoints"，是相对 CWD 的路径
ln -sfn "$MODEL_STORE" "$INTERNNAV_ROOT/checkpoints"

DA_CKPT="$MODEL_STORE/depth_anything_v2_metric_hypersim_vits.pth"
if [ ! -f "$DA_CKPT" ]; then
    info "下载 Depth-Anything-V2-Metric-Hypersim-Small（约 95MB）"
    curl -fsSL --retry 3 -o "$DA_CKPT" \
      "$HF_ENDPOINT/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth"
fi

if [ ! -f "$MODEL_STORE/InternVLA-N1-DualVLN/model-00004-of-00004.safetensors" ]; then
    info "下载 InternVLA-N1-DualVLN（约 16GB）"
    "$INTERNNAV_ROOT/.venv/bin/hf" download InternRobotics/InternVLA-N1-DualVLN \
      --local-dir "$MODEL_STORE/InternVLA-N1-DualVLN"
fi

info "完成。下一步：bash $SAMPLE_SRC/scripts/setup_habitat_env.sh"
