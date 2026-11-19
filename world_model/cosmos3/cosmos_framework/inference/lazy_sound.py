# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Inference-only deferred AVAE allocation for the fixed Cosmos3 interface."""
import copy
import threading
from pathlib import Path


class DeferredAVAE:
    """Expose AVAE metadata without allocating its model on a no-sound request."""
    def __init__(self, config):
        self._config = copy.deepcopy(config)
        self._tokenizer = None
        self._lock = threading.Lock()
        self.sample_rate = config["sample_rate"]
        self.audio_channels = config["audio_channels"]
        self.latent_ch = config["io_channels"]
        self.temporal_compression_factor = config["hop_size"]
        self.name = "avae_tokenizer"

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._load(), name)

    @property
    def dtype(self):
        import torch
        return torch.bfloat16  # AVAEInterface's fixed model dtype at this upstream revision.

    def reset_dtype(self):
        if self._tokenizer is not None:
            self._tokenizer.reset_dtype()

    def get_latent_num_samples(self, samples):
        return samples // self.temporal_compression_factor

    def get_audio_num_samples(self, samples):
        return samples * self.temporal_compression_factor

    def encode(self, *args, **kwargs):
        return self._load().encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._load().decode(*args, **kwargs)

    def _load(self):
        with self._lock:
            if self._tokenizer is None:
                path = self._config.get("avae_path", "")
                bucket = self._config.get("bucket_name", "")
                if not path:
                    raise ValueError("Missing sound tokenizer/AVAE path")
                if (not bucket and not str(path).startswith(("s3://", "https://", "http://"))
                        and not Path(path).is_file()):
                    raise FileNotFoundError(f"Missing sound tokenizer/AVAE checkpoint: {path}")
                import torch
                from cosmos_framework.utils import log
                from cosmos_framework.utils.lazy_config import instantiate
                # Delayed module construction must not consume a request's RNG
                # stream. Both CPU and current NPU states are restored on exit.
                with torch.random.fork_rng(devices=[torch.npu.current_device()], device_type="npu"):
                    tokenizer = instantiate(self._config)
                    tokenizer.reset_dtype()
                self._tokenizer = tokenizer
                log.info("Loaded deferred sound tokenizer for an audio request")
            return self._tokenizer


def make_sound_tokenizer(config):
    """Defer only the known AVAE contract, preserving all other tokenizers."""
    from cosmos_framework.utils.lazy_config import instantiate
    target = config.get("_target_", "")
    if not isinstance(target, str):
        target = f"{getattr(target, '__module__', '')}.{getattr(target, '__qualname__', '')}"
    metadata = ("sample_rate", "audio_channels", "io_channels", "hop_size")
    if target.endswith(".AVAEInterface") and all(key in config for key in metadata):
        return DeferredAVAE(config)
    return instantiate(config)
