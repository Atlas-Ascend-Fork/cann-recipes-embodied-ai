# Being-H0.5 Inference on Ascend NPU

> External resources and Docker: [installation and release guide](../resources/README_en.md). The image and resource packages are available on community OBS via HTTPS, with offline installation also supported.

[简体中文](README.md) | [English](README_en.md)

This directory is the minimal production inference delivery for Being-H0.5. It contains only source assembly, model serving, and accepted NPU inference optimizations. Profiling, ablation studies, offline evaluation, simulation, VAE training, and development tests are outside this delivery scope.

## Features

- Assemble a minimal production inference closure from a pinned Being-H0.5 upstream revision.
- Run online PyTorch/torch_npu inference on Atlas A3.
- Optimize Qwen3 + InternViT and Flow Matching/MPG for NPU.
- Support an Action Suffix NPUGraph cache, fixed-Prefix warmup, and TorchAir/GE graphs.
- Keep eager fallback, graph health reporting, and startup acceptance checks to prevent silent truncation or unsafe graph reuse.

For official Being-H0.5 checkpoint deployment, continue with this guide. For unified hand-action latent training or physical closed-loop evaluation, see [VAE training and post-training](../vae/README_en.md) and [MuJoCo closed-loop evaluation](../sim/README_en.md). Source provenance and shared patch rules are documented in the [common assembly guide](../common/README_en.md).

## Environment

The validated environment uses CANN 9.2.0, Python 3.10.16, PyTorch 2.9.0, torch_npu 2.9.0, transformers 4.57.1, and BF16 model weights. Install PyTorch, torch_npu, and torchvision separately according to the CANN version. `requirements.txt` installs only the other direct runtime dependencies.

```bash
source /path/to/Ascend/cann/set_env.sh
pip install -r manipulation/being-h0.5/infer_with_torch/requirements.txt
```

## Prepare Source and Weights

The assembly script reads the official source from the pinned `../third_party/Being-H` submodule, applies the shared compatibility patches from `../common/patches`, adds the NPU modules in this directory, and applies the upstream inference delta in `patches/npu_inference.patch`. Run `git submodule update --init --recursive` first.

The generated production closure keeps only Qwen3, InternViT, Flow Matching/MPG, and serving code required by the official H0.5 checkpoint. It excludes Qwen2.5, SigLIP, NativeVAE, training, evaluation, and simulation code.

```bash
bash manipulation/being-h0.5/infer_with_torch/download_code_and_data.sh \
  --work-root "${PWD}" \
  --download-model
```

Default output:

- Source: `${PWD}/Being-H/Being-H05`
- Official weights: `${PWD}/Being-H05-2B_libero`

Omit `--download-model` when a checkpoint is already available.

## Start the Inference Service

```bash
bash manipulation/being-h0.5/infer_with_torch/run_inference.sh \
  --beingh-root "${PWD}/Being-H" \
  --model-path "${PWD}/Being-H05-2B_libero" \
  --device 0 \
  --host 0.0.0.0 \
  --port 5555 \
  --compile-cache-dir "${PWD}/.cache/being-h05-npu"
```

The script defaults to the settings used by the official LIBERO checkpoint:

- `data_config_name=libero_nonorm`
- `dataset_name=libero_posttrain`
- `embodiment_tag=libero`
- `optimization_profile=npu_final`
- `enable_rtc=false`
- an Action Suffix NPUGraph cache with capacity 8; a fixed Prefix uses one entry
- the `torchair` Prefix backend

Override the data configuration and embodiment tag with the corresponding command-line options. The model architecture is restricted to the official H0.5 Qwen3 + InternViT configuration; unsupported architectures fail explicitly at load time. Pass additional server options after `--`.

If the production instruction set is known, provide a non-empty JSON string array. Startup computes the smallest safe fixed Prefix, pads shorter prompts, masks Padding KV, compiles Prefix prefill into a TorchAir/GE static graph, and freezes the cache before serving. Prompts beyond the warmup limit fail explicitly instead of being truncated.

```bash
bash manipulation/being-h0.5/infer_with_torch/run_inference.sh \
  --beingh-root "${PWD}/Being-H" \
  --model-path "${PWD}/Being-H05-2B_libero" \
  --prewarm-instructions-file /path/to/instructions.json
```

Reproducing the reported `64.793/70.938 ms` Core/E2E latency requires a `--prewarm-instructions-file` matching the deployment task set. TorchAir is the default backend, but a fixed Prefix graph cannot be enabled safely without an instruction set because the Prefix length is unknown. The implementation never guesses the length or silently truncates input.

## LIBERO Inference Videos

The following videos show closed-loop inference from the Being-H0.5 NPU service on `LIBERO-Spatial` task 0: “Pick up the black bowl between the plate and the ramekin and place it on the plate.” Both episodes were marked `success=True`; the three-episode batch achieved 3/3 successes.

### Episode 0

[![Successful LIBERO-Spatial Episode 0](https://raw.gitcode.com/user-images/assets/7380116/48779f7b-3155-4447-a7cf-54ed07849ffc/libero_spatial_episode_000_success.gif)](../resources/README_en.md)

[Download Episode 0 (MP4)](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v1/videos/libero_spatial_episode_000_success.mp4)

### Episode 1

[![Successful LIBERO-Spatial Episode 1](https://raw.gitcode.com/user-images/assets/7380116/95766ed7-05cd-4981-aa9e-dc77284f6582/libero_spatial_episode_001_success.gif)](../resources/README_en.md)

[Download Episode 1 (MP4)](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v1/videos/libero_spatial_episode_001_success.mp4)

## Validated Results

With the documented fixed-prefix warmup, Core mean is **64.793 ms** and Policy E2E mean is **70.938 ms**. E2E excludes model loading, video decoding, and RPC. See [optimization and measurements](../../../docs/manipulation/being-h0.5/infer_with_torch/README_en.md) for the baseline, protocol, numerical checks, and limitations.

## Validation

First inspect both entry-point interfaces:

```bash
bash manipulation/being-h0.5/infer_with_torch/download_code_and_data.sh --help
bash manipulation/being-h0.5/infer_with_torch/run_inference.sh --help
```

After starting the service, verify that:

- the model is recognized as the official H0.5 Qwen3 + InternViT architecture;
- `data_config_name`, `dataset_name`, and `embodiment_tag` match the checkpoint;
- with a fixed instruction set, startup reports completed Prefix warmup and healthy graph state;
- every request returns a complete action chunk without graph failures, full-cache errors, or unhealthy status;
- performance comparisons use identical inputs, flow steps, and warmup conditions.

For synchronous chunk evaluation in MuJoCo, `--no-record-images` renders camera
images only when the next policy step requests a model query. Physics and control
still advance every step. Asynchronous/temporal-ensemble inference and video
recording retain per-frame rendering. Episode metadata records `render_on_query`;
reported policy latency excludes environment physics and image rendering.

## Troubleshooting

### The fixed Prefix graph is not enabled

A safe fixed Prefix length can be determined only from a non-empty `--prewarm-instructions-file`. Eager or variable-Prefix operation is expected when no task instruction set is supplied.

### Unsupported model architecture at startup

This module delivers only the Qwen3 + InternViT path used by the official H0.5 checkpoint. Verify the model directory, `config.json`, and checkpoint instead of substituting another Being-H architecture.

### torch_npu or CANN operator errors

Confirm that the CANN environment is loaded and use the validated version combination above. Re-run numerical and performance validation after upgrading PyTorch, torch_npu, transformers, or CANN.

### Service starts but request configuration does not match

Prefer training metadata bundled with the checkpoint. Ensure that the data configuration, embodiment tag, and normalization match; do not infer them from directory names alone.

## Code Layout

```text
infer_with_torch/
├── code/BeingH/       # Inference service and NPU optimization modules
├── patches/           # Minimal NPU deltas against pinned upstream source
├── requirements.txt   # Minimal Python dependencies
├── download_code_and_data.sh
└── run_inference.sh
```

Upstream project: <https://github.com/BeingBeyond/Being-H>. The currently pinned upstream commit is `9443a4082bce013cc61e6523596940134179b23e`.
