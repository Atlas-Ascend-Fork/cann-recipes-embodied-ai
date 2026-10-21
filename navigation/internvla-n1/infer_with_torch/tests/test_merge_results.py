# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""src/tools/merge_results.py 的单测。

覆盖：分片合并与去重、inf/nan 指标的处理、
bootstrap 置信区间的确定性、--expect 的兜底、分场景明细。
"""
import contextlib
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest

TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "tools")
sys.path.insert(0, TOOLS_DIR)

import merge_results  # noqa: E402

INF = float("inf")
NAN = float("nan")


def write_shard(root, name, rows):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "progress.json"), "w") as f:
        for r in rows:
            # json.dump 不认 inf/nan 的标准写法，这里照 habitat 实际落盘的样子写
            f.write(json.dumps(r) + "\n")
    return d


DEFAULT_METRICS = {"success": 0.0, "spl": 0.0, "os": 0.0, "ne": 5.0}


def row(scene, eid, **metrics):
    """一条 progress.json 记录；未给出的指标取 DEFAULT_METRICS，ndtw 只在显式给出时出现。"""
    return {"scene_id": scene, "episode_id": eid, **DEFAULT_METRICS, **metrics}


@contextlib.contextmanager
def capture_log():
    """把 merge_results 的日志收进一个 StringIO，格式与 CLI 下的 stdout 输出相同。"""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    lg = merge_results.logger
    old_level = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    try:
        yield buf
    finally:
        lg.removeHandler(handler)
        lg.setLevel(old_level)


def run_script(*argv):
    """以子进程跑真正的命令行入口（含 __main__ 里的日志配置）。"""
    return subprocess.run(
        [sys.executable, os.path.join(TOOLS_DIR, "merge_results.py"), *argv],
        capture_output=True, text=True, check=False)


def run_cli(*argv):
    """跑一次 merge_results.run()，返回 (输出, MergeError 或 None)。"""
    exc = None
    with capture_log() as buf:
        try:
            merge_results.run(list(argv))
        except merge_results.MergeError as e:
            exc = e
    return buf.getvalue(), exc


def metric(out, name):
    """从输出里取某个指标的均值。"""
    for line in out.splitlines():
        if line.startswith(name):
            return float(line.split("=")[1].split()[0])
    raise AssertionError(f"输出里没有 {name}：\n{out}")


class TestLoad(unittest.TestCase):
    def test_dedup_across_shards(self):
        with tempfile.TemporaryDirectory() as t:
            a = write_shard(t, "shard0", [row("s", 1, success=1.0), row("s", 2)])
            b = write_shard(t, "shard1", [row("s", 2), row("s", 3)])
            rows = merge_results.load([a, b])
            self.assertEqual(len(rows), 3)
            self.assertEqual(sorted(r["episode_id"] for r in rows), [1, 2, 3])

    def test_missing_dir_is_skipped(self):
        with tempfile.TemporaryDirectory() as t:
            a = write_shard(t, "shard0", [row("s", 1)])
            with capture_log() as buf:
                rows = merge_results.load([a, os.path.join(t, "nope")])
            self.assertEqual(len(rows), 1)
            self.assertIn("[warn] 缺少", buf.getvalue())

    def test_blank_lines_ignored(self):
        with tempfile.TemporaryDirectory() as t:
            d = write_shard(t, "shard0", [row("s", 1)])
            with open(os.path.join(d, "progress.json"), "a") as f:
                f.write("\n\n")
            self.assertEqual(len(merge_results.load([d])), 1)


class TestBootstrapCI(unittest.TestCase):
    def test_deterministic(self):
        vals = [float(i % 2) for i in range(500)]
        self.assertEqual(merge_results.bootstrap_ci(vals), merge_results.bootstrap_ci(vals))

    def test_seed_changes_result(self):
        vals = [float(i % 3 == 0) for i in range(500)]
        self.assertNotEqual(merge_results.bootstrap_ci(vals, seed=0),
                            merge_results.bootstrap_ci(vals, seed=1))

    def test_ci_brackets_the_mean(self):
        vals = [1.0] * 60 + [0.0] * 40
        lo, hi = merge_results.bootstrap_ci(vals)
        self.assertLess(lo, 0.6)
        self.assertGreater(hi, 0.6)

    def test_constant_values_give_degenerate_ci(self):
        lo, hi = merge_results.bootstrap_ci([2.0] * 50)
        self.assertAlmostEqual(lo, 2.0)
        self.assertAlmostEqual(hi, 2.0)


class TestAggregation(unittest.TestCase):
    def test_plain_means(self):
        with tempfile.TemporaryDirectory() as t:
            d = write_shard(t, "shard0", [
                row("s", 1, success=1.0, spl=0.8, os=1.0, ne=1.0),
                row("s", 2, success=0.0, spl=0.0, os=1.0, ne=3.0),
                row("s", 3, success=1.0, spl=0.6, os=1.0, ne=2.0),
                row("s", 4, success=0.0, spl=0.0, os=0.0, ne=6.0),
            ])
            out, exc = run_cli(d, "--no-ci")
            self.assertIsNone(exc)
            self.assertIn("episodes = 4", out)
            self.assertAlmostEqual(metric(out, "SR"), 50.00, places=2)
            self.assertAlmostEqual(metric(out, "SPL"), 35.00, places=2)
            self.assertAlmostEqual(metric(out, "OS"), 75.00, places=2)
            self.assertAlmostEqual(metric(out, "NE"), 3.00, places=2)

    def test_shard_merge_equals_single_run(self):
        """分片再合并与整体计算必须等价 —— 这是手工分片方案成立的前提。"""
        rows = [row("s", i, success=float(i % 3 == 0), spl=i / 20.0, ne=i / 3.0)
                for i in range(20)]
        with tempfile.TemporaryDirectory() as t:
            whole = write_shard(t, "whole", rows)
            a = write_shard(t, "shard0", rows[0::2])
            b = write_shard(t, "shard1", rows[1::2])
            out_whole, _ = run_cli(whole, "--no-ci")
            out_split, _ = run_cli(a, b, "--no-ci")
            for name in ("SR", "SPL", "NE"):
                self.assertAlmostEqual(metric(out_whole, name), metric(out_split, name),
                                       places=6)

    def test_ndtw_only_reported_when_present(self):
        with tempfile.TemporaryDirectory() as t:
            d = write_shard(t, "shard0", [row("s", 1), row("s", 2)])
            out, _ = run_cli(d, "--no-ci")
            self.assertNotIn("nDTW", out)
            d2 = write_shard(t, "shard1", [row("s", 1, ndtw=0.7), row("s", 2, ndtw=0.5)])
            out2, _ = run_cli(d2, "--no-ci")
            self.assertAlmostEqual(metric(out2, "nDTW"), 60.00, places=2)


class TestNonFinite(unittest.TestCase):
    """inf 会让 NE 的均值变成 inf，nan 被静默跳过会让 SPL 与 SR 分母不一致。"""

    def test_inf_does_not_poison_ne(self):
        with tempfile.TemporaryDirectory() as t:
            d = write_shard(t, "shard0", self._rows())
            out, _ = run_cli(d, "--no-ci")
            self.assertAlmostEqual(metric(out, "NE"), 2.00, places=2)
            self.assertNotIn("inf", out.split("NE")[1].splitlines()[0])

    def test_denominator_is_printed(self):
        with tempfile.TemporaryDirectory() as t:
            d = write_shard(t, "shard0", self._rows())
            out, _ = run_cli(d, "--no-ci")
            self.assertIn("4 条 episode 含 inf/nan 指标", out)
            self.assertIn("oLBMNvg9in8 4", out)
            self.assertIn("分母 6", out)          # NE / SPL 的分母被显式打出来
            self.assertIn("episodes = 10", out)   # SR 仍按 10 条算

    def test_sr_denominator_differs_by_default(self):
        with tempfile.TemporaryDirectory() as t:
            d = write_shard(t, "shard0", self._rows())
            out, _ = run_cli(d, "--no-ci")
            self.assertAlmostEqual(metric(out, "SR"), 60.00, places=2)   # 6/10
            self.assertAlmostEqual(metric(out, "SPL"), 50.00, places=2)  # 3.0/6

    def test_drop_nonfinite_aligns_all_denominators(self):
        with tempfile.TemporaryDirectory() as t:
            d = write_shard(t, "shard0", self._rows())
            out, _ = run_cli(d, "--no-ci", "--drop-nonfinite")
            self.assertIn("已整条剔除，剩余 6 条", out)
            self.assertIn("episodes = 6", out)
            self.assertAlmostEqual(metric(out, "SR"), 100.00, places=2)
            self.assertAlmostEqual(metric(out, "SPL"), 50.00, places=2)
            self.assertNotIn("未计入", out)

    def test_no_note_when_all_finite(self):
        with tempfile.TemporaryDirectory() as t:
            d = write_shard(t, "shard0", [row("s", 1), row("s", 2)])
            out, _ = run_cli(d, "--no-ci")
            self.assertNotIn("inf/nan", out)
            self.assertNotIn("未计入", out)

    def _rows(self):
        good = [row("good", i, success=1.0, spl=0.5, os=1.0, ne=2.0) for i in range(6)]
        bad = [row("oLBMNvg9in8", i, success=0.0, spl=NAN, os=0.0, ne=INF)
               for i in range(4)]
        return good + bad


class TestExpect(unittest.TestCase):
    def test_expect_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as t:
            d = write_shard(t, "shard0", [row("s", 1), row("s", 2)])
            _, exc = run_cli(d, "--no-ci", "--expect", "3")
            self.assertIsInstance(exc, merge_results.MergeError)
            self.assertIn("结果不完整", str(exc))

    def test_cli_exits_nonzero_on_mismatch(self):
        """命令行入口把 MergeError 转成非零退出码，run_eval.sh 等脚本靠它判断失败。"""
        with tempfile.TemporaryDirectory() as t:
            d = write_shard(t, "shard0", [row("s", 1), row("s", 2)])
            proc = run_script(d, "--no-ci", "--expect", "3")
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(proc.stdout.splitlines()[0], "episodes = 2")
            self.assertIn("结果不完整", proc.stderr)


class TestCliOutput(unittest.TestCase):
    """真实命令行的 stdout 就是 run_eval.sh 用 tee 落盘的 summary.txt，逐行核对格式。"""

    def test_stdout_lines(self):
        with tempfile.TemporaryDirectory() as t:
            d = write_shard(t, "shard0", [
                row("aaa", 1, success=1.0, spl=0.8, os=1.0, ne=1.0),
                row("aaa", 2, success=0.0, spl=0.0, os=1.0, ne=3.0),
            ])
            proc = run_script(d, "--no-ci", "--per-scene")
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(proc.stdout.splitlines(), [
            "episodes = 2",
            "NE    =   2.00",
            "OS    = 100.00",
            "SR    =  50.00",
            "SPL   =  40.00",
            "",
            "分场景：",
            "  aaa              n=   2  SR  50.0  SPL  40.0  NE  2.00",
        ])

    def test_expect_match_passes(self):
        with tempfile.TemporaryDirectory() as t:
            d = write_shard(t, "shard0", [row("s", 1), row("s", 2)])
            _, exc = run_cli(d, "--no-ci", "--expect", "2")
            self.assertIsNone(exc)

    def test_empty_input_raises(self):
        with tempfile.TemporaryDirectory() as t:
            _, exc = run_cli(os.path.join(t, "nope"))
            self.assertIsInstance(exc, merge_results.MergeError)
            self.assertIn("没有读到任何结果", str(exc))


class TestPerScene(unittest.TestCase):
    def test_per_scene_rows_and_denominator_note(self):
        with tempfile.TemporaryDirectory() as t:
            d = write_shard(t, "shard0", [
                row("aaa", 1, success=1.0, spl=0.9, ne=1.0),
                row("aaa", 2, success=0.0, spl=0.0, ne=5.0),
                row("bbb", 1, success=0.0, spl=NAN, ne=INF),
                row("bbb", 2, success=1.0, spl=0.4, ne=2.0),
            ])
            out, _ = run_cli(d, "--no-ci", "--per-scene")
            self.assertIn("分场景：", out)
            aaa = [ln for ln in out.splitlines() if "aaa" in ln][0]
            bbb = [ln for ln in out.splitlines() if ln.strip().startswith("bbb")][0]
            self.assertIn("n=   2", aaa)
            self.assertNotIn("分母", aaa)
            # bbb 有一条 inf/nan，NE 与 SPL 的分母都只剩 1，必须显式标出来
            self.assertIn("NE 分母 1", bbb)
            self.assertIn("SPL 分母 1", bbb)


if __name__ == "__main__":
    unittest.main()
