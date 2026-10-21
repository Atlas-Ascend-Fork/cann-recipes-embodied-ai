# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""src/tools/convert_rxr_to_r2rce.py 的单测：在一份极小的合成样本上验证 schema 转换。

关注两点：
  - 转换后能被 R2RVLN-v1 的读取方式消费（有顶层 instruction_vocab、
    instruction 里只剩它认识的两个字段）；
  - 除 instruction 外的字段一字不动，所以对评测指标零影响。
"""
import contextlib
import gzip
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "tools"))

import convert_rxr_to_r2rce as conv  # noqa: E402


def rxr_episode(eid, language, text):
    """RxR 原始 schema 的一条 episode（字段名照官方 RxR_VLNCE_v0）。"""
    return {
        "episode_id": eid,
        "trajectory_id": eid * 10,
        "scene_id": "mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb",
        "start_position": [1.0, 0.5, -2.0],
        "start_rotation": [0.0, 0.7, 0.0, 0.7],
        "reference_path": [[1.0, 0.5, -2.0], [2.0, 0.5, -3.0]],
        "goals": [{"position": [2.0, 0.5, -3.0], "radius": 3.0}],
        "info": {"geodesic_distance": 7.5},
        "instruction": {
            "instruction_id": str(eid),
            "instruction_text": text,
            "language": language,
            "annotator_id": "ann-1",
            "edit_distance": 0.0,
            "timed_instruction": [{"word": "walk", "start_time": 0.1}],
        },
    }


@contextlib.contextmanager
def capture_log():
    """把转换脚本的日志收进一个 StringIO，格式与 CLI 下的 stdout 输出相同。"""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    old_level = conv.logger.level
    conv.logger.addHandler(handler)
    conv.logger.setLevel(logging.INFO)
    try:
        yield buf
    finally:
        conv.logger.removeHandler(handler)
        conv.logger.setLevel(old_level)


def write_src(path, episodes):
    with gzip.open(path, "wt") as f:
        json.dump({"episodes": episodes}, f)


def run_conv(src, dst, *extra):
    old = sys.argv
    sys.argv = ["convert_rxr_to_r2rce.py", src, dst, *extra]
    try:
        with capture_log() as buf:
            conv.main()
    finally:
        sys.argv = old
    with gzip.open(dst, "rt") as f:
        return json.load(f), buf.getvalue()


class TestTokenize(unittest.TestCase):
    def test_lowercases_and_splits_punctuation(self):
        self.assertEqual(conv.tokenize("Walk Into the room."),
                         ["walk", "into", "the", "room", "."])

    def test_keeps_apostrophe_contractions(self):
        self.assertEqual(conv.tokenize("don't stop"), ["don't", "stop"])

    def test_digits_are_words(self):
        self.assertEqual(conv.tokenize("turn 90 degrees"), ["turn", "90", "degrees"])


class TestConversion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "val_unseen_guide.json.gz")
        self.dst = os.path.join(self.tmp.name, "val_unseen_guide_en.json.gz")
        self.episodes = [
            rxr_episode(1, "en-IN", "Walk into the living room."),
            rxr_episode(2, "en-US", "Turn left and stop."),
            rxr_episode(3, "hi-IN", "kamre mein jaao."),
            rxr_episode(4, "te-IN", "gadi loki vellandi."),
        ]
        write_src(self.src, self.episodes)

    def tearDown(self):
        self.tmp.cleanup()

    def test_filters_to_english_subset(self):
        out, log = run_conv(self.src, self.dst)
        self.assertEqual([e["episode_id"] for e in out["episodes"]], [1, 2])
        self.assertIn("4 -> 2 episodes", log)

    def test_cli_prints_summary_to_stdout(self):
        proc = subprocess.run([sys.executable, conv.__file__, self.src, self.dst],
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stderr, "")
        self.assertTrue(proc.stdout.startswith("4 -> 2 episodes | 词表 "), proc.stdout)

    def test_schema_has_top_level_vocab(self):
        """直接加载 RxR 原始包会 KeyError: 'instruction_vocab'，转换后必须有。"""
        out, _ = run_conv(self.src, self.dst)
        self.assertIn("instruction_vocab", out)
        self.assertIn("word_list", out["instruction_vocab"])
        self.assertEqual(out["instruction_vocab"]["word_list"][:4], conv.SPECIAL_TOKENS)

    def test_instruction_keeps_only_r2rce_fields(self):
        out, _ = run_conv(self.src, self.dst)
        for e in out["episodes"]:
            self.assertEqual(set(e["instruction"]), {"instruction_text", "instruction_tokens"})

    def test_instruction_text_is_byte_identical(self):
        out, _ = run_conv(self.src, self.dst)
        self.assertEqual([e["instruction"]["instruction_text"] for e in out["episodes"]],
                         [self.episodes[0]["instruction"]["instruction_text"],
                          self.episodes[1]["instruction"]["instruction_text"]])

    def test_other_fields_untouched(self):
        """模型只读 instruction_text，轨迹字段必须原样保留，否则指标会变。"""
        out, _ = run_conv(self.src, self.dst)
        keys = ["scene_id", "start_position", "start_rotation",
                "reference_path", "goals", "info", "trajectory_id"]
        want = self.episodes[:2]
        self.assertEqual(len(out["episodes"]), len(want))
        for i, got in enumerate(out["episodes"]):
            for k in keys:
                self.assertEqual(got[k], want[i][k], k)

    def test_tokens_round_trip_through_vocab(self):
        out, _ = run_conv(self.src, self.dst)
        word_list = out["instruction_vocab"]["word_list"]
        for e in out["episodes"]:
            words = [word_list[i] for i in e["instruction"]["instruction_tokens"]]
            self.assertEqual(words, conv.tokenize(e["instruction"]["instruction_text"]))

    def test_min_count_maps_rare_words_to_unk(self):
        # "living" 只出现一次，min-count 2 时应落到 <unk>
        out, _ = run_conv(self.src, self.dst, "--min-count", "2")
        word_list = out["instruction_vocab"]["word_list"]
        self.assertNotIn("living", word_list)
        unk = word_list.index("<unk>")
        first = out["episodes"][0]
        words = conv.tokenize(first["instruction"]["instruction_text"])
        self.assertEqual(first["instruction"]["instruction_tokens"][words.index("living")],
                         unk)

    def test_languages_option(self):
        out, _ = run_conv(self.src, self.dst, "--languages", "hi,te")
        self.assertEqual([e["episode_id"] for e in out["episodes"]], [3, 4])

    def test_empty_languages_keeps_all(self):
        out, _ = run_conv(self.src, self.dst, "--languages", "")
        self.assertEqual(len(out["episodes"]), 4)

    def test_special_tokens_not_duplicated(self):
        write_src(self.src, [rxr_episode(1, "en-US", "<pad> walk <unk>")])
        out, _ = run_conv(self.src, self.dst)
        word_list = out["instruction_vocab"]["word_list"]
        self.assertEqual(len(word_list), len(set(word_list)))
        self.assertEqual(word_list[:4], conv.SPECIAL_TOKENS)


if __name__ == "__main__":
    unittest.main()
