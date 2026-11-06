# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for finite latency aggregation."""

import unittest

import numpy as np

from vae.integration.benchmark_beingh_retargeting import percentile_summary


class BenchmarkMetricsTests(unittest.TestCase):
    def test_multiple_values_ignore_nonfinite_samples(self):
        result = percentile_summary(np.array([1.0, 2.0, 3.0, np.nan, np.inf]))
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["total_s"], 6.0)
        self.assertEqual(result["p50_s"], 2.0)
        self.assertAlmostEqual(result["p95_s"], 2.9)

    def test_zero_latency_is_a_valid_sample(self):
        result = percentile_summary(np.array([0.0]))
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["mean_s"], 0.0)

    def test_empty_or_nonfinite_only_has_no_percentiles(self):
        for values in ([], [np.nan, np.inf]):
            with self.subTest(values=values):
                result = percentile_summary(np.asarray(values))
                self.assertEqual(result["count"], 0)
                self.assertIsNone(result["p95_s"])


if __name__ == "__main__":
    unittest.main()
