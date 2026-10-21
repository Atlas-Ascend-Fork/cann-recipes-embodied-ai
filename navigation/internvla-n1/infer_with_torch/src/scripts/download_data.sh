#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# 第 3 步：下载 VLN-CE 场景与 episode 数据。
#
# 数据来自 HuggingFace 上两个 gated 数据集，需要先用自己的账号申请访问
# （两个都是 gated=auto，点一下 "Agree and access repository" 即刻放行）：
#   https://huggingface.co/datasets/InternRobotics/Scene-N1      场景（mp3d_ce）
#   https://huggingface.co/datasets/InternRobotics/InternData-N1 episode
# 然后在 https://huggingface.co/settings/tokens 建一个 read token，
# 写到 ~/.cache/huggingface/token 或 export HF_TOKEN。
#
# 注意：数据集不会、也不应被提交进代码仓。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

: "${HF_TOKEN:=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
[ -n "$HF_TOKEN" ] || die "需要 HF_TOKEN（见本脚本头部说明）"

# hf-mirror 必须直连：走代理时它会 308 跳到 huggingface.co，
# 而 curl 跨域重定向会丢掉 Authorization 头，gated 仓库随即 401。
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

# token 不放进命令行参数：/proc 没开 hidepid，argv 对同机任何用户都可见。
# 写进一个 600 权限的 curl 配置文件，用 -K 读。
CURL_CFG="$(mktemp "${TMPDIR:-/tmp}/hfauth.XXXXXX")"
chmod 600 "$CURL_CFG"
trap 'rm -f "$CURL_CFG"' EXIT INT TERM
printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN" > "$CURL_CFG"
AUTH=(-K "$CURL_CFG")

SCENE_DIR="$DATA_ROOT/scene_data/mp3d_ce"
VLNCE_DIR="$DATA_ROOT/vln_ce/raw_data"
mkdir -p "$SCENE_DIR" "$VLNCE_DIR" "$DATA_ROOT/dl"

# ---------- 场景 ----------
# mp3d_ce.tar.gz 是整包 16GB，解开后按场景分目录。R2R / RxR 的 val_unseen
# 只用到其中 11 个场景，但官方只提供整包，没法按需下载。
if [ ! -d "$SCENE_DIR/mp3d" ]; then
    TARBALL="$DATA_ROOT/dl/mp3d_ce.tar.gz"
    SCENE_URL="$HF_ENDPOINT/datasets/InternRobotics/Scene-N1/resolve/main/mp3d_ce.tar.gz"

    # 按 Content-Length 判断是否已经下全，而不是用 [ -s ]：
    # 16GB 的下载中断得很频繁，半截文件也是「非空」的，用 [ -s ] 会直接把它交给 tar，
    # 得到的是一个静默损坏的场景目录。
    WANT=$(curl -fsSL -I --retry 3 "${AUTH[@]}" "$SCENE_URL" \
           | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n+0}')
    HAVE=$(stat -c %s "$TARBALL" 2>/dev/null || echo 0)
    if [ "$WANT" -gt 0 ] && [ "$HAVE" -eq "$WANT" ]; then
        info "mp3d_ce.tar.gz 已完整（$HAVE 字节），跳过下载"
    else
        info "下载 mp3d_ce.tar.gz（约 16GB，已有 $HAVE / $WANT 字节，断点续传）"
        # -C - 要求目标文件已存在才能续传；不存在时 curl 自动从 0 开始。
        curl -fL --retry 5 --retry-delay 5 -C - "${AUTH[@]}" -o "$TARBALL" "$SCENE_URL"
        HAVE=$(stat -c %s "$TARBALL" 2>/dev/null || echo 0)
        [ "$WANT" -le 0 ] || [ "$HAVE" -eq "$WANT" ] || die "下载不完整：$HAVE / $WANT 字节"
    fi

    # 包的顶层目录名不确定：若它自带 mp3d_ce/ 一层，直接 -C "$SCENE_DIR" 会解成
    # $SCENE_DIR/mp3d_ce/mp3d/，比脚本后面（以及 habitat 配置的 scenes_dir）期望的深一层。
    # 注意 `set -o pipefail`：`tar | head -1` 里 head 读完一行就退出，tar 随即吃到
    # SIGPIPE(141)，pipefail 把它当成整条管道的返回值，脚本会直接被 set -e 干掉。
    # 所以这一条必须在子 shell 里临时关掉 pipefail。
    TOP=$( set +o pipefail; tar -tzf "$TARBALL" 2>/dev/null | head -1 | cut -d/ -f1 )
    # 实测官方包的顶层就是 mp3d/，解到 $SCENE_DIR 正好得到 $SCENE_DIR/mp3d；
    # 万一哪天换成自带 mp3d_ce/ 一层，就多剥一层。
    STRIP=0
    [ "$TOP" = "mp3d_ce" ] && STRIP=1
    info "解压场景（顶层目录 '$TOP'，--strip-components=$STRIP）"
    tar -xzf "$TARBALL" -C "$SCENE_DIR" --strip-components="$STRIP"
    [ -d "$SCENE_DIR/mp3d" ] || die "解压后没有 $SCENE_DIR/mp3d，包结构与预期不符：顶层是 '$TOP'"
fi

# ---------- R2R episode ----------
# vln_r2r.yaml 的 data_path 是 data/vln_ce/raw_data/r2r/{split}/{split}.json.gz，
# 只需要 val_unseen.json.gz 这一个文件。
# R2R 不算 nDTW（vln_r2r.yaml 的 measurements 里没有 ndtw），因此**不需要** GT 文件；
# InternData-N1 的 r2r/val_unseen/ 下也确实只有 val_unseen.json.gz 和 val_unseen1.json，
# 没有 val_unseen_gt.json.gz。只有 RxR 要 GT，它在 rxr.zip 里，见下面。
R2R_DIR="$VLNCE_DIR/r2r/val_unseen"
mkdir -p "$R2R_DIR"
for f in val_unseen.json.gz; do
    [ -s "$R2R_DIR/$f" ] && continue
    info "下载 r2r/$f"
    curl -fsSL --retry 5 "${AUTH[@]}" -o "$R2R_DIR/$f" \
      "$HF_ENDPOINT/datasets/InternRobotics/InternData-N1/resolve/main/vln_ce/raw_data/r2r/val_unseen/$f"
done

# ---------- RxR episode ----------
# 官方是一个 RxR_VLNCE_v0 整包。注意包里**没有** _guide_en，
# 英文子集要自己筛，顺带还得转成 R2RVLN-v1 schema，见下面这步。
RXR_DIR="$VLNCE_DIR/rxr/val_unseen"
mkdir -p "$RXR_DIR"
if [ ! -s "$RXR_DIR/val_unseen_guide.json.gz" ]; then
    ZIP="$DATA_ROOT/dl/rxr.zip"
    [ -s "$ZIP" ] || { info "下载 rxr.zip（364MB）"; curl -fsSL --retry 5 "${AUTH[@]}" -o "$ZIP" \
      "$HF_ENDPOINT/datasets/InternRobotics/InternData-N1/resolve/main/vln_ce/raw_data/rxr/rxr.zip"; }
    info "解压 RxR val_unseen"
    unzip -joq "$ZIP" "RxR_VLNCE_v0/val_unseen/*" -d "$RXR_DIR"
fi

# 转换成 InternNav 的 vln_rxr.yaml 期望的 R2RVLN-v1 格式（同时筛英文子集）
RXR_CONV="$VLNCE_DIR/rxr_converted/val_unseen"
if [ ! -s "$RXR_CONV/val_unseen_guide_en.json.gz" ]; then
    mkdir -p "$RXR_CONV"
    info "转换 RxR -> R2RVLN-v1 schema（英文子集）"
    python3 "$SAMPLE_SRC/tools/convert_rxr_to_r2rce.py" \
        "$RXR_DIR/val_unseen_guide.json.gz" \
        "$RXR_CONV/val_unseen_guide_en.json.gz"
fi
# NDTW 度量在 measures.py 里写死了读 rxr/val_unseen/val_unseen_guide_gt.json.gz，
# 保持原路径即可，无需复制

# ---------- 挂到 InternNav ----------
# habitat 配置里的 scenes_dir / data_path 都是相对 InternNav 根目录的
ln -sfn "$DATA_ROOT" "$INTERNNAV_ROOT/data"

info "完成。数据落在 $DATA_ROOT，已软链到 $INTERNNAV_ROOT/data"
info "校验："
ls "$SCENE_DIR/mp3d" | wc -l | xargs echo "  场景数："
