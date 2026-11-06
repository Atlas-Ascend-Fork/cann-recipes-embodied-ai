# Being-H0.5 Common Code and Assembly Layer

[简体中文](README.md) | [English](README_en.md)

The `common` module stores the new code, configuration, and minimal patches shared by all Being-H0.5 workflows. It is not a standalone runnable recipe. Instead, it provides the common assembly input for NPU inference, VAE post-training, and MuJoCo simulation.

See [NPU inference](../infer_with_torch/README_en.md), [VAE training](../vae/README_en.md), and [MuJoCo simulation](../sim/README_en.md) for the runnable workflows.

## Purpose

The unmodified official Being-H0.5 source is provided through the shared `../third_party/Being-H` submodule, pinned to commit [`9443a4082bce013cc61e6523596940134179b23e`](https://github.com/BeingBeyond/Being-H/commit/9443a4082bce013cc61e6523596940134179b23e).

This module maintains only:

- `upstream_files.txt`: the exact allowlist copied from `third_party/Being-H/Being-H05`;
- `code/`: project-specific files shared by multiple workflows;
- `patches/`: minimal functional deltas against pinned upstream files;
- shared path templates, data-configuration registration, compatibility utilities, and lightweight smoke tests.

## Directory Layout

```text
common/
├── README.md
├── README_en.md
├── upstream_files.txt               # Allowlist of pinned upstream files
├── code/
│   ├── BeingH/                      # Shared Being-H additions
│   ├── configs/                     # Path templates and data indexes
│   └── scripts/                     # Smoke tests and indexing utilities
└── patches/
    ├── beingvla.patch               # Shared BeingVLA changes
    ├── data_config.patch            # Data-configuration changes
    ├── qwen2_npu_import.patch       # NPU import compatibility
    ├── siglip_npu_import.patch      # NPU import compatibility
    └── video_backend_fallback.patch # Video-backend fallback
```

## Assembly Order

Build the final workspace in this order:

1. Copy the pinned Being-H files listed in `upstream_files.txt`.
2. Merge `common/code/` and apply `common/patches/`.
3. Merge `infer_with_torch/code/` and the NPU inference patches.
4. Merge `vae/code/`, NativeVAE, and the VAE post-training patches.
5. Merge the MuJoCo environment and evaluation entry points from `sim/code/`.

## Usage

### Initialize the pinned upstream source

From the repository root:

```bash
git submodule update --init manipulation/being-h0.5/third_party/Being-H
git -C manipulation/being-h0.5/third_party/Being-H rev-parse HEAD
```

The second command must print `9443a4082bce013cc61e6523596940134179b23e`.
Both assembly entry points also check this commit and stop on a mismatch.

To initialize all repository submodules:

```bash
git submodule update --init --recursive
```

### Select an assembly entry point

To build only the production NPU inference closure:

```bash
bash manipulation/being-h0.5/infer_with_torch/download_code_and_data.sh \
  --work-root /path/to/workdir
```

To build the complete VAE, post-training, and MuJoCo workspace:

```bash
export WORK_DIR=/path/to/workdir
bash manipulation/being-h0.5/vae/scripts/setup.sh
```

Both entry points consume `common`, but their delivery scopes differ. The inference entry point excludes training, VAE, simulation, profiling, and test files. Only the complete assembly entry point creates those paths.

## Validation

After assembling the complete workspace, run the source-level smoke test without downloading data:

```bash
cd "${WORK_DIR}/beingh05_vae"
PYTHONPATH="$PWD" python scripts/smoke_test_beingh05.py
```

To also validate configuration imports and local data:

```bash
PYTHONPATH="$PWD" python scripts/smoke_test_beingh05.py \
  --import-config \
  --require-local-data
```

The second command requires the datasets and checkpoints described below and is not a source-only check.

## Runtime Environments

Use the validated environment and installation steps in the [inference guide](../infer_with_torch/README_en.md) or [training guide](../vae/README_en.md). These workflows use different validated Python and PyTorch versions. Run shared download commands from an external workspace directory.

## Model and Data Downloads

The base Being-H0.5 checkpoints are available from the [official Hugging Face collection](https://huggingface.co/collections/BeingBeyond/being-h05):

```bash
hf download BeingBeyond/Being-H05-2B --local-dir ckpts/Being-H05-2B
hf download OpenGVLab/InternVL3_5-2B --local-dir ckpts/InternVL3_5-2B
hf download Qwen/Qwen3-0.6B --local-dir ckpts/Qwen3-0.6B
```

Download the Shadow training data, policy model, NativeVAE, and physical-joint baseline:

```bash
hf download zju/shadow_grasp_bottle22249179_aug100_2cam \
  --repo-type dataset \
  --local-dir data/shadow_grasp_bottle22249179_aug100_2cam
hf download zju/Being-H05-shadow-grasp-2cam-rot6d-zraw \
  --local-dir ckpts/Being-H05-shadow-grasp-2cam-rot6d-zraw
hf download zju/Being-H05-native-vae native_n2_epoch800_inference.pt \
  --local-dir vae/checkpoints
hf download zju/Being-H05-shadow-grasp-2cam-joints \
  --local-dir ckpts/Being-H05-shadow-grasp-2cam-joints
```

Download the Sharpa and Gaia cross-hand datasets and baselines:

```bash
hf download zju/sharpa_grasp_bottle22249179_geo_visual100_2cam \
  --repo-type dataset \
  --local-dir data/sharpa_grasp_bottle22249179_geo_visual100_2cam
hf download zju/gaia_grasp_bottle22249179_geo_visual100_2cam \
  --repo-type dataset \
  --local-dir data/gaia_grasp_bottle22249179_geo_visual100_2cam
hf download zju/Being-H05-sharpa-joint-baseline \
  --local-dir ckpts/Being-H05-sharpa-joint-baseline
hf download zju/Being-H05-gaia-joint-baseline \
  --local-dir ckpts/Being-H05-gaia-joint-baseline
```

Keep models, datasets, and checkpoints outside the Git repository. Do not commit `.pt`, `.safetensors`, LeRobot datasets, or training outputs.

## Workflow Guides

- [VAE and Being-H0.5 post-training](../vae/README_en.md)
- [MuJoCo evaluation and videos](../sim/README_en.md)

## Configuration Index

| Configuration family | Assembled location | Purpose |
| --- | --- | --- |
| Shadow grasp | `configs/posttrain/shadow_grasp/` | wrist/`z_gesture`, Rot6D, and physical-joint variants |
| LIBERO | `configs/posttrain/libero/` | LIBERO post-training |
| RoboCasa | `configs/posttrain/robocasa/` | Human RoboCasa post-training |
| Cross-embodiment | `configs/posttrain/cross-embodiment/` | Mixed-embodiment training |

Shadow configuration suffixes describe the action transform: `_q99` uses percentile normalization, `_minmax` uses full state/action min-max normalization, `_wrist_minmax_zraw` combines wrist min-max with raw latent values, `_wrist_euler_minmax_zraw` and `_wrist_rot6d_minmax_zraw` select wrist-rotation representations, and `_wrist_rot6d_minmax_joints` outputs physical joints.

Training and evaluation entry points:

- Training: `scripts/train/train_shadow_grasp_npu.sh`
- TissueBox closed-loop demo: `vae/scripts/run_tissuebox_policy_demo.sh`
- MuJoCo evaluation: `vae/examples/beingh_shadow_grasp_eval.py`

<a id="source-release-and-reproducibility"></a>

## Source Versions and Licenses

- Official Being-H repository: <https://github.com/BeingBeyond/Being-H.git>
- Shared submodule commit: `9443a4082bce013cc61e6523596940134179b23e`
- Upstream root: `third_party/Being-H/Being-H05`
- Copy allowlist: `common/upstream_files.txt`
- Shared extension code: `common/code/`

The NativeVAE implementation is in `vae/code/vae/native_vae/`. Record the recipe commit, Being-H upstream commit, Hugging Face model revision, and dataset revision for reproducibility.

| Component | Source | License and notes |
| --- | --- | --- |
| Being-H / Being-H0.5 | [Pinned official commit](https://github.com/BeingBeyond/Being-H/commit/9443a4082bce013cc61e6523596940134179b23e) | Apache-2.0; submodule version listed above |
| MuJoCo | <https://mujoco.org/> | Apache-2.0; optional runtime dependency, with no MuJoCo binaries stored in this repository |

This recipe does not redistribute model checkpoints or datasets. The TissueBox mesh provenance and redistribution notice are recorded in the [simulation README](../sim/README_en.md#simulation-assets-scenes-and-provenance).

## Maintenance Rules

- Reference unchanged upstream files through `upstream_files.txt`; do not copy them into `code/`.
- Maintain a minimal patch when modifying a pinned upstream file instead of committing a complete replacement.
- Keep code used by only one workflow in that workflow's module.
- Never commit datasets, checkpoints, training outputs, or machine-specific absolute paths.
- After updating the submodule commit, allowlist, or patches, validate both the minimal inference assembly and the complete assembly.
