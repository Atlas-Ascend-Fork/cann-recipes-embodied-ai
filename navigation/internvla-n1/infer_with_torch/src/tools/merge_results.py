# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""合并多个分片的 progress.json，输出 VLN-CE 总指标，并给出 bootstrap 置信区间。

npu_eval.py 的 --shard 让每张卡跑一个独立进程、各写各的 progress.json。
指标都是逐 episode 的算术平均，所以离线合并与单进程整体计算完全等价。
"""
import argparse
import json
import logging
import math
import os
import random
import sys
from collections import defaultdict

logger = logging.getLogger(__name__)

METRICS = [
    # (progress.json 里的键, 展示名, 缩放, 越大越好)
    ("ne", "NE", 1, False),
    ("os", "OS", 100, True),
    ("success", "SR", 100, True),
    ("spl", "SPL", 100, True),
    ("ndtw", "nDTW", 100, True),
]


class MergeError(RuntimeError):
    """结果缺失或条数不符。CLI 入口据此以非零码退出。"""


def _read_progress(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load(dirs):
    rows, seen = [], set()
    for d in dirs:
        path = os.path.join(d, "progress.json")
        if not os.path.exists(path):
            logger.warning("[warn] 缺少 %s，跳过", path)
            continue
        for r in _read_progress(path):
            key = (r.get("scene_id"), r.get("episode_id"))
            if key in seen:  # 分片重叠或重复 resume，去重
                continue
            seen.add(key)
            rows.append(r)
    return rows


def bootstrap_ci(values, iters=4000, seed=0):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", help="各分片的输出目录")
    ap.add_argument("--per-scene", action="store_true", help="额外按场景拆分")
    ap.add_argument("--no-ci", action="store_true", help="不算置信区间")
    ap.add_argument("--expect", type=int, default=0,
                    help=">0 时断言 episode 总数等于该值，不等则非零退出。"
                         "用来挡住「跑少了却打印出一张好看的表」这种情况")
    ap.add_argument("--drop-nonfinite", action="store_true",
                    help="把任一指标为 inf/nan 的 episode 整条剔除，"
                         "保证各指标分母一致（默认只按指标逐项剔除并打印剔除数）")
    return ap.parse_args(argv)


def _finite(r, k):
    v = r.get(k)
    return isinstance(v, (int, float)) and math.isfinite(v)


def _handle_nonfinite(rows, drop):
    # RxR 的 val_unseen 里有一批 episode 的目标点在 navmesh 上与起点不连通
    # （实测 oLBMNvg9in8 场景 345 条里有 99 条），habitat 的 DistanceToGoal
    # 对这些返回 inf、SPL 返回 nan。不能放任不管：
    #   - inf 会让 NE 的均值直接变成 inf；
    #   - 若只静默跳过 nan，SPL 的分母就和 SR 不一样，两个数不可比。
    nonfinite = [r for r in rows
                 if any(k in r and not _finite(r, k) for k, _, _, _ in METRICS)]
    if not nonfinite:
        return rows
    scenes = defaultdict(int)
    for r in nonfinite:
        scenes[r.get("scene_id", "?")] += 1
    detail = "，".join(f"{s} {n}" for s, n in sorted(scenes.items()))
    logger.warning("[注意] %d 条 episode 含 inf/nan 指标（目标点与起点在 "
                   "navmesh 上不连通）：%s", len(nonfinite), detail)
    if drop:
        dropped = {(r.get("scene_id"), r.get("episode_id")) for r in nonfinite}
        rows = [r for r in rows if (r.get("scene_id"), r.get("episode_id")) not in dropped]
        logger.warning("[注意] --drop-nonfinite：已整条剔除，剩余 %d 条", len(rows))
    return rows


def _report_overall(rows, no_ci):
    logger.info("episodes = %d", len(rows))
    for key, name, scale, _ in METRICS:
        present = [r for r in rows if key in r]
        vals = [r[key] for r in present if _finite(r, key)]
        if not vals:
            continue
        skipped = len(present) - len(vals)
        note = f"   （{skipped} 条非有限值未计入，分母 {len(vals)}）" if skipped else ""
        mean = sum(vals) / len(vals) * scale
        if no_ci:
            logger.info("%-5s = %6.2f%s", name, mean, note)
        else:
            lo, hi = (x * scale for x in bootstrap_ci(vals))
            logger.info("%-5s = %6.2f   95%%CI [%6.2f, %6.2f]%s", name, mean, lo, hi, note)


def _scene_avg(rows, k, scale=1.0):
    # 与总表同样按 isfinite 过滤，否则含 inf/nan 的场景整行都是 nan
    vals = [x[k] for x in rows if k in x and _finite(x, k)]
    return (sum(vals) / len(vals) * scale, len(vals)) if vals else (float("nan"), 0)


def _report_per_scene(rows):
    by = defaultdict(list)
    for r in rows:
        by[r.get("scene_id", "?")].append(r)
    logger.info("\n分场景：")
    for scene in sorted(by):
        v = by[scene]
        n = len(v)
        sr, _ = _scene_avg(v, "success", 100)
        spl, n_spl = _scene_avg(v, "spl", 100)
        ne, n_ne = _scene_avg(v, "ne")
        # NE 与 SPL 各自剔除非有限值，分母可能都不等于 n，两个都要报出来
        extra = ""
        if n_ne != n or n_spl != n:
            extra = f"  [NE 分母 {n_ne}，SPL 分母 {n_spl}]"
        logger.info("  %-16s n=%4d  SR %5.1f  SPL %5.1f  NE %5.2f%s",
                    scene, n, sr, spl, ne, extra)


def run(argv=None):
    """合并并打印指标；结果缺失或条数不符时抛 MergeError。"""
    args = parse_args(argv)
    rows = load(args.dirs)
    if not rows:
        raise MergeError("没有读到任何结果")
    rows = _handle_nonfinite(rows, args.drop_nonfinite)
    _report_overall(rows, args.no_ci)
    if args.expect and len(rows) != args.expect:
        raise MergeError(f"\n[错误] episode 数 {len(rows)} != 预期 {args.expect}，结果不完整")
    if args.per_scene:
        _report_per_scene(rows)


def main():
    try:
        run()
    except MergeError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    main()
