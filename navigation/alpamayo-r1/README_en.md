# Alpamayo-R1 Autonomous Driving VLA Large Model: Usage Guide for Ascend A2/A3

[中文版](README.md)

This directory describes how to run inference with Alpamayo-R1 (AR-1 for short) on Ascend Atlas A2 and Atlas A3, together with the accuracy validation procedure, reference performance data, and sample outputs.

## Overview of AR-1

AR-1 is a vision-language-action (VLA) large model proposed by NVIDIA for L4/L5 autonomous driving. It addresses the fragility and poor interpretability caused by black-box decision making in long-tail scenarios. By fusing visual perception with natural language instructions, the model builds a structured understanding of the scene and introduces an explicit Chain-of-Causal-Thinking (CoC) mechanism. During inference it mimics how a human driver observes, analyzes causes, anticipates consequences, and makes decisions, producing more robust and verifiable trajectory planning together with a natural language explanation of each decision, which improves transparency and human-machine collaboration.

## Environment Preparation

1. Install the CANN packages.

   Building and running this sample depends on the CANN toolkit (cann-toolkit) and the CANN binary operator package (cann-kernels). CANN 8.5.0 or later is supported. The performance and accuracy figures in this document were measured with CANN 9.1.0, torch 2.8.0, and torch_npu 2.8.0; A2 and A3 use the same software versions.

   Download `Ascend-cann-toolkit_${version}_linux-${arch}.run` and `Atlas-A2-cann-kernels_${version}_linux-${arch}.run` from the [package download page](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0), and follow the [CANN installation guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=netconda&OS=Debian).
      - `${version}` is the CANN package version, for example 8.5.0 or 9.1.0.
      - `${arch}` is the CPU architecture, for example aarch64 or x86_64.

2. Install uv (if it is not installed yet).
    ``` bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ```
3. Clone the repositories.
    ```bash
    # Clone the cann-recipes-embodied-ai repository
    git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git

    # Clone the upstream alpamayo repository
    git clone https://github.com/NVlabs/alpamayo.git
    cd alpamayo
    git reset --hard e0e2ac32ebd9c3ad8a314099aa5af5c3aaa28073
    cd ..

    # Apply the patch for Ascend A2/A3 adaptation
    patch -p0 < cann-recipes-embodied-ai/navigation/alpamayo-r1/patches/ar-1.patch

    # Apply the graph mode and weight prefetch patch (must be applied after ar-1.patch)
    patch -p0 < cann-recipes-embodied-ai/navigation/alpamayo-r1/patches/graph-prefetch.patch

    # Copy the upstream alpamayo code into this project
    cp -rn alpamayo/* cann-recipes-embodied-ai/navigation/alpamayo-r1/
    cd cann-recipes-embodied-ai/navigation/alpamayo-r1
    ```
4. Create a virtual environment.
    ```bash
    uv venv ar1_venv --python python3.11
    source ar1_venv/bin/activate
    ```
5. Install the dependencies.
    ```bash
    # Install the alpamayo-r1 dependencies
    uv sync --active
    ```
6. Install physical-ai-av. This package targets python312, so its dependency declaration needs to be adjusted before installation on python311.
    ```bash
    # Download the source tarball
    wget https://files.pythonhosted.org/packages/25/20/730fab2dc243e720fa107520c187c7a3a928f471c40d367d55980032daac/physical_ai_av-0.1.0.tar.gz
    # Extract it
    tar -zxvf physical_ai_av-0.1.0.tar.gz
    # Apply the patch
    patch -p0 < patches/physical_ai_av-0.1.0-python311.patch
    # Install
    cd physical_ai_av-0.1.0
    uv pip install .
    cd ..
    ```

## Downloading the Model Weights
```bash
    hf download nvidia/Alpamayo-R1-10B --local-dir model_ckpt
```

## Running Inference
```bash
    # Run inference
    cd ./src/alpamayo_r1
    python test_inference.py
```

The commands above produce a single inference result. For a description of the optimization methods, see the [AR-1 optimization document](../../docs/navigation/alpamayo-r1/README.md) (in Chinese).

- Inference output: 10 predicted driving trajectory sequences and the CoC reasoning text.
- Inference performance: end-to-end latency of a single request, including VLM autoregressive generation and diffusion sampling. Each figure is the median of 20 steady-state runs on the corresponding device.

    | Device | Configuration | Latency per request | Relative to default |
    |:---:|:---:|:---:|:---:|
    | Atlas 800T A2 (910B3) | Default | 7.88 s | — |
    | Atlas 800T A2 (910B3) | Graph mode | **7.49 s** | −4.9% |
    | Atlas A3 (Ascend910_9382) | Default | 6.79 s | — |
    | Atlas A3 (Ascend910_9382) | Graph mode | **6.53 s** | −3.9% |

- Inference accuracy: mean minADE over the official `chunk_0000` (chunk id 3116), 100 clips, 10 predicted trajectories. Lower is better.

    | Device | mean minADE | Threshold |
    |:---:|:---:|:---:|
    | Atlas 800T A2 (910B3) | **0.4828** | ≤0.5676 |
    | Atlas A3 (Ascend910_9382) | **0.4954** | ≤0.5676 |

## Inference Result Showcase

The figures below show a real inference result. The reference input consists of the frames from 4 camera streams at t0 = 5.1 s of the public clip `030c760c-ae38-49aa-9ad8-f5650a545d26`:

![Reference input: frames from 4 camera streams](https://raw.gitcode.com/xy200303/ar1-assets/blobs/073fd5f72f45c4920801ca9873ca16010f899855/ar1_input_cameras.png)

The corresponding inference output is shown below. The light blue lines are the 10 predicted trajectories, the black line is the ground truth, and the orange line is the prediction closest to the ground truth, with a minADE of 0.2315 m. The text below the figure is the CoC reasoning output associated with that trajectory, indicating that the vehicle should nudge to the left to keep clearance from the work zone on the right.

![Inference output: 10 predicted trajectories compared with ground truth](https://raw.gitcode.com/xy200303/ar1-assets/blobs/1eb079468c6a5ef6b8ee5105eba72b90f2bc2b4c/ar1_output_trajectories.png)

The animation below shows how the trajectories are generated step by step during the same inference run:

![Inference animation: trajectories generated step by step](https://raw.gitcode.com/xy200303/ar1-assets/blobs/f9706f5f282c3198ecbf72d3a9b355e57d5fa63c/ar1_inference.gif)

## Graph Mode and Weight Prefetch

Two stages of the AR-1 inference pipeline benefit from additional optimization. In diffusion denoising, every step has a fixed input shape and the number of steps is fixed, so the whole stage can be captured once and replayed afterwards. In VLM decoding, reading the layer weights is the dominant memory traffic, so the weights can be prefetched into the on-chip cache asynchronously before they are used. The two switches below enable these optimizations separately. Both are off by default, and leaving them off gives exactly the same behavior as described in the previous section.

Each switch can be set either as an environment variable or as an argument of `sample_trajectories_from_data_with_vlm_rollout`; the optimization is enabled when either one is set:

| Switch | Default | Effect |
|---|---|---|
| `AR1_ENABLE_GRAPH` / `enable_graph` | Off | Captures the 10 diffusion denoising steps once and replays them on subsequent requests. If capture fails, execution automatically falls back to the non-captured path. The vision positional-encoding cache is attached together with this switch. |
| `AR1_ENABLE_PREFETCH` / `enable_prefetch` | Off | Prefetches the MatMul weights of the current VLM decoder layer into the on-chip cache asynchronously before the layer is computed. |
| `AR1_STATIC_KV` | On | Keeps the VLM cache length constant across requests so that graph mode captures only once per process. Independent of the graph switch; set it to 0 to restore allocation by the actual sequence length. |
| `AR1_VISION_CACHE` | On | Caches the vision tower grid positional encodings. Attached only in graph mode; set it to 0 to disable it on its own. |

Support matrix (A2 and A3 share the same patch and the same switches):

| Feature | Atlas 800T A2 (910B3) | Atlas A3 (Ascend910_9382) |
|---|---|---|
| Single inference (default) | Measured | Measured |
| Graph mode (capture and replay of the 10 denoising steps) | Measured, first capture about 0.4 s | Measured, first capture about 0.4 s |
| Weight prefetch (CMO interface) | Integrated, end-to-end gain not quantified | Integrated, end-to-end gain not quantified |
| Static KV and vision metadata cache | Enabled by default | Enabled by default |
| Native Cholesky on NPU | Measured | Measured |
| Official chunk_0000 accuracy (100 clips, 10 trajectories) | 0.4828 | 0.4954 |

```bash
    # Four-way ablation; both switches are off by default
    python bench/ablation.py --model-path model_ckpt --iters 3 --graph --prefetch
    # 100-clip minADE; chunk_0000 is chunk id 3116
    python bench/eval_minade.py --model-path model_ckpt --clips 100 --graph --prefetch --tag a2
```

Run all of the commands above from this directory (`navigation/alpamayo-r1/`). Script parameters are
listed in [`bench/README.md`](bench/README.md).

Supported devices and software environment: Atlas 800T A2 (910B3), Atlas 800I A2, and Atlas A3 (Ascend910_9382), all running CANN 9.1.0 with torch_npu 2.8.0. A3 uses the same patch and the same switches as A2, with no additional changes; the only difference is the latency shown in the table above. The first capture takes about 0.4 s. This is a one-time cost per process, and steady-state replay involves no further compilation.

Other limitations:

- Graph mode requires the captured region to contain no fused operator that spawns its own side stream, so the expert attention uses the eager implementation, which is mathematically equivalent;
- Weight prefetch is submitted during VLM decoding. Whether it takes effect is determined by `CmoWeightPrefetcher.stats`; its end-to-end gain is not quantified;
- Weights and datasets are not shipped with the code; see below for how to obtain them.

See section 11 of the [optimization note](../../docs/navigation/alpamayo-r1/README.md) for the detailed boundaries.

## Accuracy Validation on Ascend A2/A3

### Validating Inference Accuracy with minADE

The minADE of the AR-1 trajectory predictions was measured on both A2 and A3 for 1, 3, 5 and 10 predicted trajectories, over the official `chunk_0000` (the chunk id 3116 returned by `get_clip_chunk()`, 100 clips), with t0 = 5.1 s, eager attention, CANN 9.1.0 and torch_npu 2.8.0:

| Number of predicted trajectories | A2 910B3 | A3 (Ascend910_9382) | Ascend A2 reference |
|:---:|:---:|:---:|:---:|
| 1 | 1.7244 | 1.6155 | 2.2026 |
| 3 | 0.9314 | 0.9371 | 0.9498 |
| 5 | 0.7556 | 0.7712 | 0.8514 |
| **10** | **0.4828** | **0.4954** | 0.5676 (threshold, ≤) |

With 3 trajectories both platforms are within 1–2% of the reference value, and with 5 trajectories both are better than it, which shows that the chunk selection and the t0 setting match the official ones. With 10 trajectories both platforms are below the 0.5676 threshold. Two control experiments confirm that graph mode and the operator replacements leave the model output unchanged: the eager and fused attention implementations produce identical results, and disabling the operator replacements gives identical minADE on 20 clips. For details, see section 9 of the [optimization document](../../docs/navigation/alpamayo-r1/README.md).

## Citation
```
@article{nvidia2025alpamayo,
      title={{Alpamayo-R1}: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail},
      author={NVIDIA and Yan Wang and Wenjie Luo and Junjie Bai and Yulong Cao and Tong Che and Ke Chen and Yuxiao Chen and Jenna Diamond and Yifan Ding and Wenhao Ding and Liang Feng and Greg Heinrich and Jack Huang and Peter Karkus and Boyi Li and Pinyi Li and Tsung-Yi Lin and Dongran Liu and Ming-Yu Liu and Langechuan Liu and Zhijian Liu and Jason Lu and Yunxiang Mao and Pavlo Molchanov and Lindsey Pavao and Zhenghao Peng and Mike Ranzinger and Ed Schmerling and Shida Shen and Yunfei Shi and Sarah Tariq and Ran Tian and Tilman Wekel and Xinshuo Weng and Tianjun Xiao and Eric Yang and Xiaodong Yang and Yurong You and Xiaohui Zeng and Wenyuan Zhang and Boris Ivanovic and Marco Pavone},
      year={2025},
      journal={arXiv preprint arXiv:2511.00088},
}
```

## Directory Structure

The sample lives in the repository as follows. `patches/` holds the minimal upstream changes, `src/`
the runtime modules added by this sample, and `bench/` the scripts that recompute the numbers in the
optimization note.

```text
navigation/alpamayo-r1/
├── README.md, README_en.md
├── patches/
│   ├── ar-1.patch                       # Base Ascend adaptation
│   ├── graph-prefetch.patch             # Graph mode, static KV, weight prefetch, native Cholesky
│   └── physical_ai_av-0.1.0-python311.patch
├── src/alpamayo_r1/                     # Runtime modules, copied into the upstream tree on install
│   ├── flags.py                         # Environment switches in one place
│   ├── graph_runner.py                  # Capture and replay of the diffusion step, eager fallback
│   ├── npu_ext.py                       # One-line import entry used by the patched files
│   ├── prefetch.py                      # Weight prefetch: CMO and strided-touch variants
│   ├── static_kv.py                     # Bucketed static KV so one graph serves many requests
│   └── vision_cache.py                  # Positional-encoding cache of the vision tower
└── bench/                               # Scripts that recompute the numbers
```

`cp -rn alpamayo/*` merges the upstream sources into `src/alpamayo_r1/` of this directory,
where the six modules added by this sample already sit, so no extra copy is needed.
After installation both the upstream `test_inference.py` and the scripts under `bench/` import from there:

```text
alpamayo/src/alpamayo_r1/
├── flags.py, graph_runner.py, npu_ext.py, prefetch.py, static_kv.py, vision_cache.py
├── action_space/utils.py                # Trajectory smoothing solved natively on NPU
├── models/action_in_proj.py             # RmsNorm in plain torch so it can be captured
├── models/alpamayo_r1.py                # Static KV, vision cache, prefetch and graph mode
└── ...                                  # Upstream files
```
