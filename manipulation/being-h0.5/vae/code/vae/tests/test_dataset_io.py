# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class DatasetIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script_impl"))
        try:
            cls.builder = importlib.import_module("build_lerobot_z_dataset")
        finally:
            sys.path.pop(0)

    def test_fixed_size_action_columns_survive_parquet_roundtrip(self):
        rng = np.random.default_rng(17)
        values = rng.normal(size=(177, 52)).astype(np.float32)
        field_type = pa.list_(pa.float32(), 52)
        table = pa.table({"action": pa.array(np.zeros_like(values).tolist(), type=field_type)})
        table = self.builder.replace_column(table, "action", values)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.parquet"
            pq.write_table(table, path, compression="zstd")
            restored = np.asarray(pq.read_table(path)["action"].to_pylist(), dtype=np.float32)
        np.testing.assert_array_equal(restored, values)

    def test_episode_statistics_remain_independent_json_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            path = root / "meta/episodes_stats.jsonl"
            original = {"episode_index": 1, "stats": {"video": {"count": [177]}}}
            path.write_text(json.dumps(original) + "\n", encoding="utf-8")
            self.builder.write_episode_stats(root, {
                1: {"action": {"count": [177]}},
                0: {"action": {"count": [100]}},
            })
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([record["episode_index"] for record in records], [0, 1])
        self.assertEqual(records[1]["stats"]["video"], original["stats"]["video"])
        self.assertEqual(records[0]["stats"]["action"]["count"], [100])


if __name__ == "__main__":
    unittest.main()
