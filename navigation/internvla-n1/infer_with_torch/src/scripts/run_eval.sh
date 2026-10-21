#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# 跑 VLN-CE 评测。默认把 episode 均分到所有可见 NPU 上，各跑一个独立进程，
# 结束后用 tools/merge_results.py 合并出总指标。
#
# 用法：
#   bash run_eval.sh r2r                    # R2R val_unseen 全量
#   bash run_eval.sh r2r --sample 110       # 分层采样 110 个（快速验证用）
#   bash run_eval.sh rxr --sample 220       # RxR val_unseen 英文子集
#
# 断点续跑：OUT_ROOT 默认带时间戳，每次都是新目录，因此**默认不会续跑**。
# 要接着上次的结果跑，必须显式指定同一个 OUT_ROOT（各分片的 progress.json
# 就在 $OUT_ROOT/shard<i>/progress.json）：
#   OUT_ROOT=$INTERNNAV_ROOT/logs/habitat/rxr_full bash run_eval.sh rxr
# 续跑要与首次运行用同样的分片数（NUM_NPU）、同样的种子（--seed，默认 0），
# 且保持融合视觉注意力开启（默认开），否则前后两半走的数值路径不同。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

TASK="${1:-r2r}"; shift || true
NUM_NPU="${NUM_NPU:-$(python3 -c 'import torch,torch_npu;print(torch.npu.device_count())' 2>/dev/null || echo 1)}"
OUT_ROOT="${OUT_ROOT:-$INTERNNAV_ROOT/logs/habitat/$TASK-$(date +%Y%m%d-%H%M%S)}"

case "$TASK" in
  r2r) HAB_CFG="scripts/eval/configs/vln_r2r.yaml"; DATA_ARG=() ;;
  rxr) HAB_CFG="scripts/eval/configs/vln_rxr.yaml"
       # 用转换后的英文子集，见 download_data.sh
       DATA_ARG=(--data-path "data/vln_ce/raw_data/rxr_converted/{split}/{split}_guide_en.json.gz") ;;
  *)   die "未知任务 $TASK，应为 r2r 或 rxr" ;;
esac

info "任务 $TASK | $NUM_NPU 张卡 | 输出 $OUT_ROOT"
mkdir -p "$OUT_ROOT"
PIDS=()
for i in $(seq 0 $((NUM_NPU - 1))); do
    ASCEND_RT_VISIBLE_DEVICES=$i \
    INTERNNAV_ROOT="$INTERNNAV_ROOT" HABITAT_VENV="$HABITAT_VENV" \
    PYTHONPATH="$SAMPLE_SRC:$HBRIDGE" \
    nohup "$INTERNNAV_ROOT/.venv/bin/python" "$SAMPLE_SRC/npu_eval.py" \
        --habitat-config "$HAB_CFG" "${DATA_ARG[@]}" \
        --shard "$i/$NUM_NPU" --output-path "$OUT_ROOT/shard$i" "$@" \
        > "$OUT_ROOT/shard$i.log" 2>&1 &
    PIDS+=($!)
    info "  卡 $i -> PID ${PIDS[-1]}"
done

for pid in "${PIDS[@]}"; do wait "$pid" || warn "PID $pid 非零退出"; done

info "合并结果"
python3 "$SAMPLE_SRC/tools/merge_results.py" "$OUT_ROOT"/shard* --per-scene \
    | tee "$OUT_ROOT/summary.txt"
