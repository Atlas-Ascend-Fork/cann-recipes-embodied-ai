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

"""Single-image NPU generation shared by the CLI, evaluator and model tests."""

from dataclasses import dataclass
from contextlib import nullcontext
import gc
import math
import random
import time

from .monitoring import HostMemory, TokenTiming


def _is_finite_number(value):
    """Accept finite integer/float settings while excluding boolean values."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class GenerationOptions:
    max_new_tokens: int = 32768
    temperature: float = 0.05
    top_p: float = 1.0
    seed: int = 42
    thinking: bool = False
    timeout_seconds: float = 7200.0
    verify_finite: bool = False
    observe: bool = False

    def __post_init__(self):
        if not isinstance(self.max_new_tokens, int) or isinstance(self.max_new_tokens, bool) or self.max_new_tokens < 1:
            raise ValueError('max_new_tokens must be a positive integer')
        if not _is_finite_number(self.temperature) or self.temperature < 0:
            raise ValueError('temperature must be finite and nonnegative')
        if not _is_finite_number(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError('top_p must be finite and in (0, 1]')
        if not _is_finite_number(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError('timeout_seconds must be finite and positive')
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        if any(not isinstance(value, bool) for value in [self.thinking, self.verify_finite, self.observe]):
            raise ValueError('thinking, verify_finite and observe must be booleans')


def seed_generators(seed):
    """Use the same Python, NumPy, PyTorch and NPU RNG initialization as the baseline."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)


def prepare_inputs(pipe, image, text, thinking):
    """Apply Tencent Hunyuan's processor and template on the pipeline's NPU device."""
    from PIL import Image

    if isinstance(image, Image.Image):
        opened = image.convert('RGB')
    else:
        with Image.open(image) as source:
            opened = source.convert('RGB')
    try:
        messages = pipe.build_messages(text, image=opened)
        inputs = pipe.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt',
            enable_thinking=thinking,
        )
        return inputs.to(pipe.model.device)
    finally:
        opened.close()


def generation_hooks(options, start):
    """Create per-call timeout and optional finite-logits checks."""
    import torch
    from transformers import LogitsProcessor, StoppingCriteria

    class Deadline(StoppingCriteria):
        expired = False

        def __call__(self, *args, **kwargs):
            self.expired = time.monotonic() - start > options.timeout_seconds
            return self.expired

    class FiniteScores(LogitsProcessor):
        checks = 0

        def __call__(self, input_ids, scores):
            if not torch.isfinite(scores).all():
                raise FloatingPointError('Nonfinite generation logits')
            self.checks += 1
            return scores

    return Deadline(), FiniteScores()


def finish_output(record, last_token, eos_token_ids, deadline_expired, token_limit):
    """Distinguish a normal EOS from a timeout or a truncated token-limit result."""
    if deadline_expired:
        record['error'] = 'GenerationTimeout'
    elif record['output_tokens'] >= token_limit and last_token not in eos_token_ids:
        record['error'] = 'GenerationTokenLimit'


def generate_completion(pipe, inputs, options, record, profiler):
    import torch
    from transformers import LogitsProcessorList, StoppingCriteriaList

    observed = options.observe or profiler is not None
    if observed:
        torch.npu.synchronize()
    start = time.monotonic()
    timer = TokenTiming(start, profiler) if observed else None
    deadline, finite = generation_hooks(options, start)
    extra = {'logits_processor': LogitsProcessorList([finite])} if options.verify_finite else {}
    if timer is not None:
        extra['streamer'] = timer
    try:
        if profiler is not None:
            profiler.start()
        with torch.inference_mode():
            output = pipe.model.generate(
                **inputs,
                max_new_tokens=options.max_new_tokens,
                use_cache=True,
                temperature=options.temperature,
                top_p=options.top_p,
                do_sample=options.temperature > 0,
                stopping_criteria=StoppingCriteriaList([deadline]),
                **extra,
            )
        if observed:
            torch.npu.synchronize()
    finally:
        if timer is not None:
            timer.close()
            record['generationSeconds'] = time.monotonic() - start
            record['ttftSeconds'] = timer.first_token_seconds
    response = output[:, record['input_tokens']:]
    record['output_tokens'] = int(response.shape[1])
    record['raw_output'] = pipe.processor.batch_decode(
        response,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    if options.verify_finite:
        record['finiteLogitsChecks'] = finite.checks
    eos = pipe.model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    last_token = int(response[0, -1]) if record['output_tokens'] else None
    finish_output(record, last_token, eos, deadline.expired, options.max_new_tokens)


def generate_one(pipe, image, text, options, profiler=None):
    """Generate one completion; optional measurements are enabled by the evaluator/tests."""
    import torch

    record = dict(
        seed=options.seed,
        thinking=options.thinking,
        raw_output='',
        input_tokens=0,
        output_tokens=0,
        error=None,
    )
    observed = options.observe or profiler is not None
    if observed:
        record.update(ttftSeconds=None, generationSeconds=0.0)
    started = time.monotonic()
    seed_generators(options.seed)
    if observed:
        torch.npu.reset_peak_memory_stats()
    inputs = None
    context = HostMemory() if observed else nullcontext()
    with context as memory:
        try:
            inputs = prepare_inputs(pipe, image, text, options.thinking)
            record['input_tokens'] = int(inputs['input_ids'].shape[1])
            generate_completion(pipe, inputs, options, record, profiler)
        except Exception as error:
            record['error'] = f'{type(error).__name__}: {error}'
    if observed:
        record.update(
            endToEndSeconds=time.monotonic() - started,
            peakHostRssBytes=memory.peak_bytes,
            peakNpuAllocatedBytes=torch.npu.max_memory_allocated(),
        )
    inputs = None
    if observed:
        gc.collect()
        torch.npu.synchronize()
        record.update(
            npuAllocatedAfterCleanupBytes=torch.npu.memory_allocated(),
            npuReservedAfterCleanupBytes=torch.npu.memory_reserved(),
            hostRssAfterCleanupBytes=memory.process.memory_info().rss,
        )
    return record
