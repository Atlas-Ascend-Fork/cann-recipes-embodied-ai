#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# 本样例的公共路径与镜像配置。其他脚本都 source 这个文件。
# 需要改路径时，直接在外部 export 同名变量即可覆盖默认值。

# 样例源码根目录（本文件所在目录的上一级）
SAMPLE_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SAMPLE_SRC

# 上游 InternNav 仓库。按社区约定放在 cann-recipes-embodied-ai 的同级目录。
# $SAMPLE_SRC 是 <仓库根>/navigation/internvla-n1/infer_with_torch/src，
# 往上 4 级是仓库根，5 级才是仓库根的父目录
export INTERNNAV_ROOT="${INTERNNAV_ROOT:-$(cd "$SAMPLE_SRC/../../../../.." && pwd)/InternNav}"
# 上游 commit：本样例验证过的版本
export INTERNNAV_COMMIT="${INTERNNAV_COMMIT:-7a5c62400ac45b313d9b709c740b64191556a242}"

# habitat 栈的独立 venv 与桥接目录（见 src/compat/hvenv_tail.py 的说明）
export HABITAT_ENV_ROOT="${HABITAT_ENV_ROOT:-$HOME/internvla-n1-habitat}"
export HABITAT_VENV="${HABITAT_VENV:-$HABITAT_ENV_ROOT/venv}"
export HBRIDGE="${HBRIDGE:-$HABITAT_ENV_ROOT/bridge}"
export HABITAT_SIM_SRC="${HABITAT_SIM_SRC:-$HABITAT_ENV_ROOT/habitat-sim}"
export HABITAT_LAB_SRC="${HABITAT_LAB_SRC:-$HABITAT_ENV_ROOT/habitat-lab}"
export HABITAT_SIM_TAG="${HABITAT_SIM_TAG:-v0.3.3}"
export HABITAT_LAB_TAG="${HABITAT_LAB_TAG:-v0.2.4}"

# 权重与数据。都放在大盘上，不要放进代码仓
export MODEL_STORE="${MODEL_STORE:-$HOME/models}"
export DATA_ROOT="${DATA_ROOT:-$HOME/data}"

# 网络：github / huggingface 直连不通时走公开镜像；能直连的环境可 export 覆盖这两项
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export GITCODE_MIRROR="${GITCODE_MIRROR:-https://gitcode.com/gh_mirrors}"

info()  { echo -e "\033[0;32m[INFO]\033[0m  $*"; }
warn()  { echo -e "\033[0;33m[WARN]\033[0m  $*"; }
die()   { echo -e "\033[0;31m[ERROR]\033[0m $*" >&2; exit 1; }
