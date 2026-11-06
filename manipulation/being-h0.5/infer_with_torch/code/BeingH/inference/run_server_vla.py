# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


"""BeingH VLA Inference Server - Entry point for running the inference server."""

import copy
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import tyro

try:
    import torch_npu
except ImportError:
    torch_npu = None

from BeingH.inference.beingh_policy import BeingHPolicy, NPUInferenceConfig, PolicyRuntimeConfig, PolicyTaskConfig
from BeingH.inference.beingh_service import BeingHInferenceServer


LOGGER = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set seed for all random number generators to ensure reproducibility."""
    torch.manual_seed(seed)
    if torch_npu is not None and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    LOGGER.info(f"--- Random seed set to {seed} for reproducibility ---")


@dataclass


class ServerArgs:
    """Compact command line for the supported inference paths."""

    model_path: str
    device: int = 0
    port: int = 5555
    host: str = "0.0.0.0"
    api_token: Optional[str] = None
    seed: int = 42
    prompt_template: str = "long"
    data_config_name: str = ""
    embodiment_tag: str = ""
    dataset_name: str = ""
    use_mpg: Optional[bool] = None
    mpg_lambda: Optional[float] = None
    mpg_num_projections: Optional[int] = None
    mpg_refinement_iters: Optional[int] = None
    mpg_gate_temperature: Optional[float] = None
    num_inference_timesteps: Optional[int] = None
    enable_rtc: bool = True
    optimization_profile: Literal["raw", "npu_final"] = "raw"
    npu_graph_cache_max_entries: int = 8
    npu_prefix_backend: Literal["eager", "npugraph", "torchair"] = "torchair"
    npu_compile_cache_dir: Optional[str] = None
    npu_graph_prewarm_instructions_file: Optional[str] = None
    freeze_npu_graph_cache_after_prewarm: bool = True
    metadata_variant: Optional[str] = None
    stats_selection_mode: str = "auto"


def prewarm_npu_graph_cache(policy: BeingHPolicy, instructions_file: str, *, seed: int, freeze_after: bool) -> dict:
    """Fix the task-set Prefix shape and capture it before serving."""
    if not policy.model.enable_npu_capture_replay:
        raise ValueError("graph prewarm requires enable_npu_capture_replay")
    payload = json.loads(Path(instructions_file).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("prewarm instructions must be a non-empty JSON list")
    if not all((isinstance(item, str) and item.strip() for item in payload)):
        raise ValueError("every prewarm instruction must be a non-empty string")
    policy.configure_fixed_prefix_padding(payload)
    rng = np.random.default_rng(seed)
    template = {
        "state.state": np.zeros((1, 8), dtype=np.float64),
        "state.eef_position": np.zeros((1, 3), dtype=np.float64),
        "state.eef_rotation": np.zeros((1, 3), dtype=np.float64),
        "state.libero_gripper_position": np.zeros((1, 2), dtype=np.float64),
        "video.top_view": rng.integers(0, 256, size=(1, 256, 256, 3), dtype=np.uint8),
        "video.wrist_view": rng.integers(0, 256, size=(1, 256, 256, 3), dtype=np.uint8),
    }
    policy.model.unfreeze_npu_capture_replay_cache()
    for instruction in payload:
        observation = copy.deepcopy(template)
        observation["language.instruction"] = [instruction]
        policy.get_action(observation)
        torch.npu.synchronize()
    runner = policy.model.get_npu_action_suffix_graph_runner()
    runners = {"action_suffix": runner}
    prefix_runner = getattr(policy.model, "_npu_prefix_graph_runner", None)
    if prefix_runner is not None:
        runners["prefix_npugraph"] = prefix_runner
    prefix_torchair_runner = getattr(policy.model, "_npu_prefix_torchair_runner", None)
    if prefix_torchair_runner is not None:
        runners["prefix_torchair"] = prefix_torchair_runner
    prewarm_stats = {name: item.stats() for name, item in runners.items()}
    for name, item_stats in prewarm_stats.items():
        if item_stats["cache_full_fallback_count"] or item_stats["failed_key_count"] or item_stats["unhealthy"]:
            device_name = policy.device.type.upper()
            stats_text = json.dumps(item_stats, sort_keys=True)
            raise RuntimeError(
                f"{device_name} {name} graph startup prewarm "
                f"did not capture the configured fixed shape: {stats_text}"
            )
    if freeze_after:
        policy.model.freeze_npu_capture_replay_cache()
    stats = {name: item.stats() for name, item in runners.items()}
    set_seed(seed)
    LOGGER.info(f"{policy.device.type.upper()} graph startup prewarm: " + json.dumps(stats, sort_keys=True))
    return stats


def main(args: ServerArgs):
    set_seed(args.seed)
    if torch_npu is None or not torch.npu.is_available():
        raise RuntimeError("Being-H0.5 optimized inference requires an available NPU")
    device = f"npu:{args.device}"
    LOGGER.info(f"Running policy on device: {device}")
    if args.prompt_template == "short":
        instruction_template = "{task_description}"
    else:
        instruction_template = (
            "According to the instruction '{task_description}', what's the micro-step actions in the next {k} steps?"
        )
    LOGGER.info("--- 2. Initializing BeingHPolicy ---")
    policy = BeingHPolicy(
        model_path=args.model_path,
        task=PolicyTaskConfig(
            data_config_name=args.data_config_name,
            embodiment_tag=args.embodiment_tag,
            dataset_name=args.dataset_name,
            instruction_template=instruction_template,
            metadata_variant=args.metadata_variant,
            stats_selection_mode=args.stats_selection_mode,
        ),
        device=device,
        runtime=PolicyRuntimeConfig(
            use_mpg=args.use_mpg,
            mpg_lambda=args.mpg_lambda,
            mpg_num_projections=args.mpg_num_projections,
            mpg_refinement_iters=args.mpg_refinement_iters,
            mpg_gate_temperature=args.mpg_gate_temperature,
            num_inference_timesteps=args.num_inference_timesteps,
            enable_rtc=args.enable_rtc,
        ),
        optimization=NPUInferenceConfig(
            profile=args.optimization_profile,
            graph_cache_max_entries=args.npu_graph_cache_max_entries,
            prefix_backend=args.npu_prefix_backend,
            compile_cache_dir=args.npu_compile_cache_dir,
        ),
    )
    if args.npu_graph_prewarm_instructions_file is not None:
        prewarm_npu_graph_cache(
            policy,
            args.npu_graph_prewarm_instructions_file,
            seed=args.seed,
            freeze_after=args.freeze_npu_graph_cache_after_prewarm,
        )
    LOGGER.info(f"--- 3. Starting Inference Server on {args.host}:{args.port} ---")
    server = BeingHInferenceServer(policy=policy, port=args.port, host=args.host, api_token=args.api_token)
    server.run()


if __name__ == "__main__":
    args = tyro.cli(ServerArgs)
    main(args)
