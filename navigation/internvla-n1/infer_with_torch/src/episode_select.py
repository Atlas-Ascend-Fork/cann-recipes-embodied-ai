# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""评测 episode 的采样、分片与断点续跑过滤。

从 npu_eval.py 里抽出来单独成文件，因为这段是整条链路上唯一会**静默**出错的逻辑
（顺序搞反不会报错，只会让分片划分变样），需要能在没有 NPU / habitat 的机器上
直接跑单测。本模块只用标准库，不 import torch、habitat 或 InternNav。

对应单测：tests/test_episode_select.py
"""
import json
import logging
import os

logger = logging.getLogger(__name__)


def read_done(output_path):
    """读回本分片 progress.json 里已完成的 (scene_id, episode_id)。

    键的构造方式与上游 habitat_env.generate_episodes 保持一致。
    """
    done = set()
    path = os.path.join(output_path, "progress.json")
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            res = json.loads(line)
            if "scene_id" in res:
                done.add((res["scene_id"], int(res["episode_id"])))
    return done


def episode_key(ep):
    """episode 对象 -> progress.json 里那把键。

    上游的 scene_id 是形如 `.../mp3d/<场景>/<场景>.glb` 的路径，倒数第二段才是场景名。
    """
    return (ep.scene_id.split("/")[-2], int(ep.episode_id))


def select_episodes(episodes, sample=0, limit=0, shard=None, done=None):
    """按 采样 -> 切片 -> 过滤已完成 的顺序挑出本进程要跑的 episode。

    这三步的**顺序不能换**。上游 generate_episodes() 自带断点续跑：它先读
    output_path/progress.json 把已完成的剔掉再返回。如果在过滤后的列表上做
    eps[idx::num]，续跑时每个分片看到的列表长度都不一样（各自只认得自己那份
    progress.json），跨步落点随之偏移，得到的划分与首次运行**不是同一套**：
    已经跑完的会被另一个分片重跑，另有一部分 episode 谁都不跑、被静默丢掉。
    实测 3669 条、两分片各完成 963 / 929 条的场景下，按错误顺序续跑会跑 2723 条
    （多跑 963 条已完成的），同时丢掉 17 条。

    因此调用方要先让上游按**原始顺序**返回完整列表（把 output_path 临时指到一个
    不存在的目录，使其 done_res 为空），再交给本函数采样、切片，最后才用本分片
    自己的 progress.json 过滤。这样首次运行与续跑的分片划分完全一致。
    """
    total = len(episodes)
    eps = list(episodes)
    if sample > 0 and sample < total:
        # generate_episodes 是按场景排序输出的，直接取前 N 会全落在头一个场景里，
        # 等间隔取样才能按各场景的 episode 占比均匀覆盖。
        stride = total / sample
        eps = [eps[int(i * stride)] for i in range(sample)]
        logger.info("[sample] %d -> %d episodes（等间隔）", total, len(eps))
    elif limit > 0:
        eps = eps[:limit]
        logger.info("[limit] %d -> %d episodes（前 N 个，场景不均匀）", total, len(eps))
    if shard:
        idx, num = (int(x) for x in shard.split("/"))
        eps = eps[idx::num]
        logger.info("[shard %d/%d] -> 本进程 %d episodes", idx, num, len(eps))

    if done:
        before = len(eps)
        eps = [e for e in eps if episode_key(e) not in done]
        logger.info("[resume] progress.json 已完成 %d 个，本分片跳过 %d 个，剩余 %d",
                    len(done), before - len(eps), len(eps))
    return eps
