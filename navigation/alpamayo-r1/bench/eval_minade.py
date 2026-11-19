# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""100 clips minADE 评测：Physical-AI-AV 的 official chunk（默认 chunk id 3116）。

示例：
    python bench/eval_minade.py --clips 100 --tag a2
    python bench/eval_minade.py --clips 100 --graph --prefetch --tag a2-combo
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401

import _common

LOGGER = logging.getLogger(__name__)

CHUNK_ID = 3116


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="model_ckpt")
    parser.add_argument("--clips", type=int, default=100)
    parser.add_argument("--chunk", type=int, default=CHUNK_ID)
    parser.add_argument("--num-traj", type=int, default=10)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--dataset-revision", default="25.10")
    parser.add_argument("--ood-parquet", default="", help="逐 clip t0 表，默认全部 clip 用 5.1 s")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--out-dir", default="")
    return parser


def load_t0_map(path):
    """逐 clip 起始时间：reasoning/ood_reasoning.parquet 的 event_start_timestamp。"""
    import pandas as pd

    table = pd.read_parquet(path)
    t0_map = {}
    for clip_id, group in table.groupby(level=0):
        events = group["events"].iloc[0]
        try:
            if isinstance(events, str):
                events = json.loads(events)
            first = events[0]
            if isinstance(first, dict) and "event_start_timestamp" in first:
                t0_map[clip_id] = int(first["event_start_timestamp"])
        except (ValueError, KeyError, IndexError):
            continue
    return t0_map


def select_clips(args):
    """按 chunk 取出前 N 个 clip，并准备数据集接口与逐 clip t0。"""
    import physical_ai_av

    _common.patch_dataset_download()
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(revision=args.dataset_revision or None)
    index = avdi.clip_index
    clips = list(index[index["chunk"] == args.chunk].index)[: args.clips]
    t0_map = load_t0_map(args.ood_parquet) if args.ood_parquet else {}
    LOGGER.info("chunk=%s clips=%d t0_map=%d", args.chunk, len(clips), len(t0_map))
    return clips, t0_map, avdi


def evaluate(args, clips, t0_map, avdi):
    """逐 clip 推理并累计 minADE，返回 (ades, records, 总耗时秒)。"""
    model, processor = _common.build_model(args.model_path)
    ades = []
    records = []
    started = time.time()
    for position, clip_id in enumerate(clips):
        try:
            model_inputs, data = _common.prepare_inputs(
                model, processor, clip_id, t0_map.get(clip_id, _common.DEFAULT_T0_US), avdi
            )
            pred_xyz, _, _, seconds = _common.run_inference(
                model, model_inputs, args.graph, args.prefetch, args.num_traj
            )
            ade = _common.min_ade(pred_xyz, _common.ground_truth_xy(data))
        except RuntimeError as error:  # 单个 clip 失败时记录下来，不中断整批
            LOGGER.warning("clip %03d %s ERROR %r", position, clip_id[:8], error)
            break
        ades.append(ade)
        records.append({"clip": clip_id, "min_ade": ade, "seconds": round(seconds, 2)})
        LOGGER.info(
            "clip %03d %s minADE=%.4f t=%.2fs mean=%.4f",
            position,
            clip_id[:8],
            ade,
            seconds,
            float(np.mean(ades)),
        )
    return ades, records, time.time() - started


def summarize(args, ades, records, elapsed):
    summary = {
        "tag": args.tag,
        "graph": args.graph,
        "prefetch": args.prefetch,
        "num_traj": args.num_traj,
        "clips": len(ades),
        "mean_min_ade": float(np.mean(ades)) if ades else None,
        "total_seconds": round(elapsed),
        "records": records,
    }
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        target = os.path.join(args.out_dir, f"minade_{args.tag}.json")
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    clips, t0_map, avdi = select_clips(args)
    ades, records, elapsed = evaluate(args, clips, t0_map, avdi)
    summary = summarize(args, ades, records, elapsed)
    LOGGER.info(
        "EVAL tag=%s clips=%d mean_minADE=%s total=%ss graph=%s prefetch=%s",
        args.tag,
        summary["clips"],
        summary["mean_min_ade"],
        summary["total_seconds"],
        args.graph,
        args.prefetch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
