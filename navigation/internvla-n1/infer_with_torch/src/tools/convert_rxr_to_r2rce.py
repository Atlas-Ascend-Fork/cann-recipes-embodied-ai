# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""把 RxR-Habitat 原始 schema 的 VLN-CE episode 文件转成 habitat-lab 的 R2RVLN-v1 schema。

为什么需要：InternNav 的 scripts/eval/configs/vln_rxr.yaml 里 dataset.type 写的是
R2RVLN-v1，而官方 RxR_VLNCE_v0 包里的 val_unseen_guide.json.gz 是 RxR 原始格式 ——
没有顶层 instruction_vocab，instruction 里还带着 R2RVLN-v1 不认识的字段
（annotator_id / language / timed_instruction 等）。直接加载会报
`KeyError: 'instruction_vocab'`。

官方包里也**没有** `_guide_en` 这个文件，英文子集需要自己按 language 前缀筛。
本脚本一次做完这两件事。

转换只动 instruction 字段和顶层词表：
  - instruction_text 原样保留（InternVLA-N1 只读这个字段）
  - instruction_tokens 按新建的词表重新生成（模型不用，仅为满足 habitat 的 schema）
  - scene_id / start_position / start_rotation / reference_path / goals / info 一字不动
因此对评测指标没有任何影响。
"""
import argparse
import gzip
import json
import logging
import re
import sys
from collections import Counter

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]
# 与 R2R-VLN-CE 词表风格保持一致：单词（含撇号缩写）单独成词，标点各自成词
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]", re.IGNORECASE)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="RxR 原始 json.gz，例如 val_unseen_guide.json.gz")
    ap.add_argument("dst", help="输出的 R2RVLN-v1 格式 json.gz")
    ap.add_argument("--languages", default="en",
                    help="按 instruction.language 前缀过滤，逗号分隔；留空表示不过滤。默认 en")
    ap.add_argument("--min-count", type=int, default=1, help="入词表的最低词频")
    args = ap.parse_args()

    with gzip.open(args.src, "rt") as f:
        data = json.load(f)
    episodes = data["episodes"]
    total = len(episodes)

    if args.languages:
        prefixes = tuple(x.strip() for x in args.languages.split(",") if x.strip())
        episodes = [e for e in episodes
                    if e["instruction"].get("language", "").startswith(prefixes)]

    counter = Counter()
    for e in episodes:
        counter.update(tokenize(e["instruction"]["instruction_text"]))
    words = sorted(w for w, c in counter.items() if c >= args.min_count)
    word_list = SPECIAL_TOKENS + [w for w in words if w not in SPECIAL_TOKENS]
    stoi = {w: i for i, w in enumerate(word_list)}
    unk = stoi["<unk>"]

    for e in episodes:
        text = e["instruction"]["instruction_text"]
        e["instruction"] = {
            "instruction_text": text,
            "instruction_tokens": [stoi.get(t, unk) for t in tokenize(text)],
        }

    with gzip.open(args.dst, "wt") as f:
        json.dump({"episodes": episodes, "instruction_vocab": {"word_list": word_list}}, f)
    logger.info("%d -> %d episodes | 词表 %d | 写入 %s", total, len(episodes), len(word_list), args.dst)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    main()
