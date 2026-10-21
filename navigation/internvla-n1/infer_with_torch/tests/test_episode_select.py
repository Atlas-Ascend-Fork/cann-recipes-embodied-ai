# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""src/episode_select.py 的单测：采样、分片、断点续跑。

重点是那条唯一会**静默**出错的不变式：续跑不能改变分片划分。
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import episode_select  # noqa: E402


class Ep:
    """最小化的 episode 替身，只保留 select_episodes 用到的两个字段。"""

    def __init__(self, scene, episode_id):
        # 上游 scene_id 是 `.../mp3d/<场景>/<场景>.glb`，倒数第二段才是场景名
        self.scene_id = f"data/scene_data/mp3d_ce/mp3d/{scene}/{scene}.glb"
        self.episode_id = episode_id

    def __repr__(self):
        return f"Ep({self.scene_id.split('/')[-2]},{self.episode_id})"

    def __eq__(self, other):
        return (self.scene_id, self.episode_id) == (other.scene_id, other.episode_id)

    def __hash__(self):
        return hash((self.scene_id, self.episode_id))


def make_episodes(per_scene):
    """按场景排序生成 episode 列表，模拟上游 generate_episodes 的输出顺序。"""
    eps = []
    for scene, n in sorted(per_scene.items()):
        eps.extend(Ep(scene, i) for i in range(n))
    return eps


class TestShardPartition(unittest.TestCase):
    def setUp(self):
        self.eps = make_episodes({"aaa": 615, "bbb": 51, "ccc": 345, "ddd": 528})

    def test_shards_partition_exactly(self):
        """各分片无交集，并集等于全集。"""
        for num in (1, 2, 3, 8):
            got = []
            for i in range(num):
                got.extend(episode_select.select_episodes(
                    self.eps, shard=f"{i}/{num}"))
            self.assertEqual(len(got), len(self.eps), f"num={num} 条数对不上")
            self.assertEqual(set(got), set(self.eps), f"num={num} 集合对不上")

    def test_shard_is_stride_slice(self):
        self.assertEqual(
            episode_select.select_episodes(self.eps, shard="1/2"),
            self.eps[1::2])


class TestResumeKeepsPartition(unittest.TestCase):
    """续跑必须落在与首跑完全相同的划分上。

    每个分片只认得自己那份 progress.json。如果先按它过滤、再 eps[idx::num]，
    各分片看到的列表长度就不一样，跨步落点随之偏移，划分与首跑不是同一套：
    已跑完的会被另一个分片重跑，另有一部分 episode 谁都不跑、被静默丢掉。
    """

    def setUp(self):
        # 规模照 RxR val_unseen 英文子集（3669 条、11 个场景）
        self.eps = make_episodes({
            "2azQ1b91cZZ": 615, "8194nk5LbLH": 51, "EU6Fwq7SyZv": 345,
            "QUCTc6BB5sX": 528, "TbHJrupSAjP": 543, "X7HyMhZNoso": 468,
            "Z6MFQCViBuw": 288, "oLBMNvg9in8": 345, "pLe4wQe7qrG": 6,
            "x8F5xyUWy9e": 96, "zsNo4HB9uLZ": 384})
        self.num = 2
        # 真实中断时两个分片各完成的条数
        self.finished_counts = [963, 929]

    def test_total_is_3669(self):
        self.assertEqual(len(self.eps), 3669)

    def test_resume_covers_everything_exactly_once(self):
        first, finished, todo = self._resume()
        covered = []
        for i in range(self.num):
            covered.extend(finished[i])
            covered.extend(todo[i])
        self.assertEqual(len(covered), len(set(covered)), "有 episode 被重复跑")
        self.assertEqual(set(covered), set(self.eps), "有 episode 被静默丢掉")
        self.assertEqual(len(covered), 3669)

    def test_resume_is_exact_remainder_of_first_pass(self):
        first, finished, todo = self._resume()
        for i in range(self.num):
            # 本分片续跑的 = 本分片首跑划分里还没做完的那些，一条不多一条不少
            self.assertEqual(todo[i], first[i][self.finished_counts[i]:])
        self.assertEqual([len(t) for t in todo], [872, 905])

    def test_resume_is_logged(self):
        """续跑跳过了多少条要打进日志，否则断点续跑是否生效无从查起。"""
        done = {episode_select.episode_key(e) for e in self.eps[:10]}
        with self.assertLogs(episode_select.logger, "INFO") as cm:
            episode_select.select_episodes(self.eps, shard="0/2", done=done)
        self.assertIn("[shard 0/2] -> 本进程 1835 episodes", cm.output[0])
        self.assertIn("本分片跳过 5 个，剩余 1830", cm.output[1])

    def test_resume_does_not_move_shard_membership(self):
        """续跑时某分片的成员集合始终是它首跑划分的子集，不会串到别的分片去。"""
        first, _, todo = self._resume()
        for i in range(self.num):
            self.assertTrue(set(todo[i]).issubset(set(first[i])))
            for j in range(self.num):
                if i != j:
                    self.assertTrue(set(todo[i]).isdisjoint(set(first[j])))

    def _first_pass(self):
        return [episode_select.select_episodes(self.eps, shard=f"{i}/{self.num}")
                for i in range(self.num)]

    def _resume(self):
        """模拟整体续跑：返回 (本次要跑的, 断点前已跑的)，都含各分片的合并结果。"""
        first = self._first_pass()
        finished = [first[i][:self.finished_counts[i]] for i in range(self.num)]
        todo = []
        for i in range(self.num):
            done = {episode_select.episode_key(e) for e in finished[i]}
            todo.append(episode_select.select_episodes(
                self.eps, shard=f"{i}/{self.num}", done=done))
        return first, finished, todo


class TestSampleAndLimit(unittest.TestCase):
    def setUp(self):
        self.per_scene = {"aaa": 615, "bbb": 51, "ccc": 345, "ddd": 528, "eee": 96}
        self.eps = make_episodes(self.per_scene)

    def scene_of(self, ep):
        return ep.scene_id.split("/")[-2]

    def test_sample_covers_every_scene_proportionally(self):
        n = 110
        got = episode_select.select_episodes(self.eps, sample=n)
        self.assertEqual(len(got), n)
        total = len(self.eps)
        for scene, cnt in self.per_scene.items():
            share = sum(1 for e in got if self.scene_of(e) == scene) / n
            self.assertLess(abs(share - cnt / total), 0.01,
                            f"{scene} 的采样占比偏离全量占比超过 1 个百分点")

    def test_limit_falls_into_first_scenes(self):
        """--limit 的已知缺陷：只取前 N 条，会集中在字典序靠前的场景。"""
        got = episode_select.select_episodes(self.eps, limit=50)
        self.assertEqual(got, self.eps[:50])
        self.assertEqual({self.scene_of(e) for e in got}, {"aaa"})

    def test_sample_not_smaller_than_total_is_noop(self):
        self.assertEqual(
            episode_select.select_episodes(self.eps, sample=len(self.eps) + 1),
            self.eps)

    def test_sample_is_deterministic(self):
        a = episode_select.select_episodes(self.eps, sample=37)
        b = episode_select.select_episodes(self.eps, sample=37)
        self.assertEqual(a, b)

    def test_sample_applies_before_shard(self):
        """采样发生在分片之前：两个分片合起来正好是那 110 条。"""
        n, num = 110, 2
        full = episode_select.select_episodes(self.eps, sample=n)
        merged = []
        for i in range(num):
            merged.extend(episode_select.select_episodes(
                self.eps, sample=n, shard=f"{i}/{num}"))
        self.assertEqual(set(merged), set(full))
        self.assertEqual(len(merged), n)


class TestReadDone(unittest.TestCase):
    def test_reads_keys_and_tolerates_blank_lines(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "progress.json"), "w") as f:
                f.write(json.dumps({"scene_id": "aaa", "episode_id": 3}) + "\n")
                f.write("\n")
                f.write(json.dumps({"scene_id": "bbb", "episode_id": "7"}) + "\n")
                # 没有 scene_id 的行（上游偶尔写汇总行）应被忽略
                f.write(json.dumps({"note": "summary"}) + "\n")
            self.assertEqual(episode_select.read_done(d), {("aaa", 3), ("bbb", 7)})

    def test_missing_file_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(episode_select.read_done(d), set())

    def test_read_done_matches_episode_key(self):
        """read_done 的键与 episode_key 必须同构，否则过滤永远命中不了。"""
        ep = Ep("aaa", 3)
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "progress.json"), "w") as f:
                f.write(json.dumps({"scene_id": "aaa", "episode_id": 3}) + "\n")
            self.assertIn(episode_select.episode_key(ep), episode_select.read_done(d))


if __name__ == "__main__":
    unittest.main()
