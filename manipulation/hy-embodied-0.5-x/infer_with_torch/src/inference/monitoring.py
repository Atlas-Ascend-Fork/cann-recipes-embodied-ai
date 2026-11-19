# Copyright (c) 2026 li12345abc123. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional generation timing and memory observation and timing summaries."""

import threading
import time


class HostMemory:
    """Sample the process RSS every 50 ms during one operation."""

    def __init__(self):
        import psutil

        self.process = psutil.Process()
        self.peak_bytes = self.process.memory_info().rss
        self.stop_event = threading.Event()
        self.thread = None

    def __enter__(self):
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

    def _sample(self):
        while not self.stop_event.wait(0.05):
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)


class TokenTiming:
    """Observe the first output token; optional profiling stops after three output steps."""

    def __init__(self, start, profiler=None):
        self.start = start
        self.prompt_seen = False
        self.first_token_seconds = None
        self.profiler = profiler
        self.steps = 0

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        if self.first_token_seconds is None and value.numel():
            self.first_token_seconds = time.monotonic() - self.start
        self.steps += 1
        if self.profiler is not None and self.steps >= 3:
            self.close()

    def close(self):
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None

    def end(self):
        self.close()


def timing_summary(records):
    """Summarize recorded timings using the existing sample's measurement definitions."""

    def stats(values):
        values = sorted(values)
        if not values:
            return {'count': 0, 'mean': None, 'p50': None, 'p95': None}

        def percentile(p):
            pos = (len(values) - 1) * p
            low = int(pos)
            high = min(low + 1, len(values) - 1)
            return values[low] + (values[high] - values[low]) * (pos - low)

        return {
            'count': len(values),
            'mean': sum(values) / len(values),
            'p50': percentile(0.5),
            'p95': percentile(0.95),
        }

    generated = [r for r in records if not r.get('error') and r.get('generationSeconds', 0) > 0]
    total_seconds = sum(r['generationSeconds'] for r in generated)
    return {
        'ttftSeconds': stats([r['ttftSeconds'] for r in records if r.get('ttftSeconds') is not None]),
        'endToEndSeconds': stats([r['endToEndSeconds'] for r in records if r.get('endToEndSeconds') is not None]),
        'outputTokensPerGenerationSecond': (
            sum(r.get('output_tokens', 0) for r in generated) / total_seconds if total_seconds else None
        ),
        'generationThroughputPopulation': 'error-free records with positive generationSeconds',
        'peakHostRssBytes': max(
            (r['peakHostRssBytes'] for r in records if r.get('peakHostRssBytes') is not None), default=None
        ),
        'peakNpuAllocatedBytes': max(
            (r['peakNpuAllocatedBytes'] for r in records if r.get('peakNpuAllocatedBytes') is not None), default=None
        ),
    }
