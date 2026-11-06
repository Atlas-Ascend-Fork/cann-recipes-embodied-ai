# Being-H0.5 VAE-Enhanced Generalization Training on Ascend

> External resources and Docker: [installation and release guide](../resources/README_en.md). The image and resource packages are available on community OBS via HTTPS, with offline installation also supported.

[简体中文](README.md) | [English](README_en.md)

This recipe reproduces Native-URDF Gesture VAE training on Atlas A3, converts LeRobot actions into a 24-dimensional `z_gesture`, and post-trains Being-H0.5 on two NPUs. The design discussion is tracked in [RFC #84](https://gitcode.com/cann/cann-recipes-embodied-ai/issues/84).

The VAE maps joint actions from different dexterous hands into a shared gesture latent. Being-H0.5 predicts `wrist pose + z_gesture`, and the target-hand decoder converts the latent into native joint actions, reducing the coupling between the policy and a single hand design.

## Features

- Generate multi-hand pose data and train NativeVAE on NPU.
- Convert native LeRobot joint actions into a 24-dimensional `z_gesture` dataset.
- Verify non-action fields, action re-encoding error, and checkpoint hashes.
- Run VAE-aware two-NPU Being-H0.5 post-training with HCCL/FSDP.
- Produce consistent model and data artifacts for NPU open-loop and MuJoCo closed-loop evaluation.

This module owns NativeVAE representation training, latent data generation, and Being-H0.5 post-training. See [NPU inference](../infer_with_torch/README_en.md), [MuJoCo simulation](../sim/README_en.md), and the [common assembly guide](../common/README_en.md) for the other layers.

## Code Layout

```text
vae/
├── README.md
├── README_en.md
├── requirements.txt
├── code/
│   ├── BeingH/                   # z_gesture model and data adaptation
│   ├── configs/                  # Data and post-training registration
│   ├── scripts/train/            # Being-H0.5 post-training entry points
│   ├── beingh_shared_scripts/    # Shared training scripts
│   └── vae/
│       ├── native_vae/           # Canonical NativeVAE implementation
│       ├── configs/              # Hand and VAE training configurations
│       ├── scripts/              # Train, convert, infer, and evaluate
│       └── tests/                # Smoke, training, and regression tests
├── patches/                      # Data, post-training, and compatibility deltas
├── scripts/setup.sh              # Complete workspace assembly
└── scripts/run_vae_pipeline.sh   # End-to-end VAE and post-training pipeline
```

## Validated Environment

- Hardware: Atlas A3, two NPUs
- OS: Linux aarch64
- Python: 3.11
- PyTorch / torch_npu: 2.7.1
- CANN: 9.2.0
- Official Being-H0.5 source: `../third_party/Being-H/`
- Shared project overlay: `../common/code/`
- NativeVAE, latent conversion, and post-training: this module's `code/`
- NPU inference acceleration: `../infer_with_torch/code/`
- MuJoCo closed-loop simulation: `../sim/code/`

Other versions may work but are outside the validated combination.

## 1. Prepare Source

```bash
git submodule update --init --recursive
export RECIPE_ROOT="$PWD"
export WORK_DIR=/path/to/workdir
export BEINGH_ASSET_ARCHIVE=/path/to/beingh05-runtime-v2.tar.gz
bash manipulation/being-h0.5/vae/scripts/setup.sh
export BEINGH_DIR="${WORK_DIR}/beingh05_vae"
```

The NativeVAE implementation is in `vae/code/vae/native_vae/`. Official Being-H source comes from the pinned `../third_party/Being-H` submodule. `setup.sh` copies source files and applies module patches in this order: Being-H upstream source, `common`, `infer_with_torch`, `vae`, and `sim/code/`, producing the training and evaluation workspace.

The script does not download datasets or checkpoints and does not overwrite a non-empty target directory.

## 2. Install Dependencies

In an environment where PyTorch, torch_npu, and CANN are already installed:

```bash
python -m pip install -r "${BEINGH_DIR}/requirements.txt"
python -m pip install -e "${BEINGH_DIR}/vae"
```

Download datasets and checkpoints as described in the [common guide](../common/README_en.md#model-and-data-downloads), and keep them outside the Git repository. Never commit `.pt`, `.safetensors`, LeRobot datasets, or training outputs.

## 3. Single-NPU VAE Smoke Test

```bash
cd "${BEINGH_DIR}/vae"
python scripts/generate_random_data.py \
  --config configs/train_native_n2.yaml \
  --device npu:0 \
  --train_samples_per_hand 32 \
  --val_samples_per_hand 16
python scripts/train.py \
  --config configs/train_native_n2.yaml \
  --device npu:0 \
  --epochs 1 \
  --max_train_batches 1 \
  --max_val_batches 1 \
  --no_wandb
```

## 4. Full Training

```bash
export ASCEND_CANN_ENV=/path/to/Ascend/cann/set_env.sh
export BEINGH_ENV=/path/to/python/env
export SOURCE_DATASET=/path/to/shadow_source_dataset
export OUTPUT_DATASET=/path/to/shadow_latent_z_dataset
export DATASET_NAME=shadow_latent_z_dataset

bash "${RECIPE_ROOT}/manipulation/being-h0.5/vae/scripts/run_vae_pipeline.sh"
```

The default pipeline generates 50,000 training and 5,000 validation poses for each of three hands, trains VAE for 800 epochs, converts the latent dataset, verifies every row, and post-trains Being-H0.5 for 40,000 optimizer steps on two NPUs. With an existing VAE checkpoint, set `TRAIN_VAE=0 VAE_CHECKPOINT=/path/to/inference.pt`.

### Limit training to 1,000 optimizer steps

With the full training and validation datasets already available, use the following
command for bounded acceptance. `--max_steps` counts cumulative optimizer updates;
resuming a checkpoint containing `global_step` continues that count. At the limit,
the trainer evaluates the full validation set and saves `last.pt`, `inference.pt`
and the actual step count.

```bash
cd "${BEINGH_DIR}/vae"
python scripts/train.py --config configs/train_native_n2.yaml \
  --device npu:0 --max_steps 1000 --no_wandb
```

For Being-H post-training, set `MAX_STEPS=1000`, `SAVE_STEPS=500` and
`SAVE_STEPS_START=500` to save intermediate and final checkpoints. Keep smoke mode
and validation-batch limits disabled. This budget checks execution, checkpointing
and evaluation, not convergence under the default recipe. NativeVAE's cross-hand
loss still starts at epoch 300; a short run may not reach that stage.

## 5. Acceptance Criteria

- The single-NPU VAE smoke test completes forward, backward, and save operations.
- The two-NPU HCCL/FSDP smoke test saves and restores optimizer state.
- Non-action fields in the latent dataset exactly match the source dataset.
- Every action row is re-encoded on NPU with `atol=0.002, rtol=0`.
- The VAE checkpoint SHA256 is stored in generated dataset metadata.
- Formal post-training reaches 40,000 optimizer steps with no NaN or Inf values.

## 6. Inference and Large-Scale Evaluation

Formal acceptance includes NPU open-loop evaluation, 100 episodes, MuJoCo visualization, and three matched baselines: VAE, geometry retargeting, and Gaia hand-specific joints.

### Successful Dexterous-Hand Grasp Videos

The following closed-loop MuJoCo episodes show Being-H0.5 + NativeVAE controlling Shadow Hand to grasp and lift a bottle. They use the `z_gesture` action representation and the `safe_smooth` deployment profile. The corresponding 10-episode evaluation achieved 10/10 successes.

#### Episode 0

Successful lift, maximum height `0.3316 m`:

[![Successful Shadow Hand Episode 0](https://raw.gitcode.com/user-images/assets/7380116/53abad8a-8baa-40f3-a805-20e3aa88c582/shadow_grasp_episode_000_success.gif)](../resources/README_en.md)

[Download Episode 0 (MP4)](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v1/videos/shadow_grasp_episode_000_success.mp4)

#### Episode 1

Successful lift, maximum height `0.3402 m`:

[![Successful Shadow Hand Episode 1](https://raw.gitcode.com/user-images/assets/7380116/592cf013-bd2c-48ed-af79-8efa14919f1e/shadow_grasp_episode_001_success.gif)](../resources/README_en.md)

[Download Episode 1 (MP4)](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v1/videos/shadow_grasp_episode_001_success.mp4)

## 7. Standalone NativeVAE

NativeVAE supports Shadow, Gaia, and Sharpa right hands. It includes random joint-pose generation, batched forward kinematics, 24-dimensional `z_gesture` encoding, same-hand reconstruction, and cross-hand retargeting. It has no dependency on `src/`, `config/`, or `outputs/` from another project.

### Joint Order

Input `q` must follow `active_joint_names` in `code/vae/configs/right_hands.yaml`:

| Hand | q dimension |
| --- | ---: |
| `shadow_hand_right` | 22 |
| `gaia_hand_right` | 15 |
| `sharpa_hand_right` | 22 |

Shadow `WRJ1/WRJ2` remain zero. They participate in full URDF forward kinematics but are excluded from VAE input and output.

### Install Separately

```bash
cd "${BEINGH_DIR}/vae"
python -m pip install -e .          # Inference only
python -m pip install -e '.[train]' # Training and WandB
```

### Data Generation and Training Schedule

The default configuration generates 50,000 training and 5,000 validation poses for each hand, samples the central 95% of joint limits, uses FK batch size 2048, and uses train/validation seeds 42/10042:

```bash
python scripts/generate_random_data.py --config configs/train_native_n2.yaml
python scripts/train.py --config configs/train_native_n2.yaml --device npu:0
```

The loss schedule is documented in the optimization guide below.

Outputs are written to `runs/<run_name>_<timestamp>/` and include `best.pt`, `last.pt`, `inference.pt`, `metrics.jsonl`, and the resolved configuration.

### Encode, Reconstruct, and Retarget

Input can be `.npy` or `.npz` containing `q`, `action`, or `joint_q`:

```bash
# Encode
python scripts/infer.py \
  --mode encode \
  --source_hand shadow_hand_right \
  --input examples/shadow_q.npy \
  --output outputs/shadow_z.npz

# Same-hand reconstruction
python scripts/infer.py \
  --mode reconstruct \
  --source_hand shadow_hand_right \
  --input examples/shadow_q.npy \
  --output outputs/shadow_reconstruction.npz

# Shadow-to-Sharpa retargeting
python scripts/infer.py \
  --mode retarget \
  --source_hand shadow_hand_right \
  --target_hand sharpa_hand_right \
  --input examples/shadow_q.npy \
  --output outputs/shadow_to_sharpa.npz
```

Retargeting output includes `z_gesture`, target `q`, finger-pad positions for both hands, and error values.

### Python API

```python
from native_vae import NativeVAE

vae = NativeVAE.from_pretrained(device="npu:0")
z = vae.encode(shadow_q, hand="shadow_hand_right")
shadow_q_hat = vae.decode(z, hand="shadow_hand_right")
sharpa_q = vae.decode(z, hand="sharpa_hand_right")
```

The default inference checkpoint is `checkpoints/native_n2_epoch800_inference.pt`, derived from `Native-N2-pair-vector-distance-refine/epoch_0800.pt`. It stores inference weights only, without optimizer or scheduler state.

### Regression Tests

```bash
cd "${BEINGH_DIR}/vae"
PYTHONPATH=. python -m unittest discover -s tests -v
```

Tests cover final-checkpoint `z_gesture`, target `q`, finger-pad positions, and the MuJoCo interface.

## 8. Being-H0.5 Post-Training Parameters

Being-H0.5 uses `torchrun` + FSDP. Single-embodiment and cross-embodiment entry points are `scripts/train/train_libero_example.sh` and `scripts/train/train_cross_emb_example.sh`.

| Required argument | Description |
| --- | --- |
| `--mllm_path` | InternVL backbone path |
| `--expert_path` | Qwen expert model path |
| `--resume_from` | Being-H checkpoint for post-training |
| `--dataset_config_file` | Dataset YAML |
| `--output_dir` | Checkpoint output directory |

| Category | Important defaults |
| --- | --- |
| Optimizer | `learning_rate=1e-4`, `weight_decay=1e-5`, `warmup_ratio=0.05`, `lr_scheduler=cosine` |
| Save and log | `save_steps=10000`, `save_steps_start=25000`, `logging_steps=10` |
| Model | `use_flow_matching=True`, `use_expert=True`, `action_chunk_length=16` |
| Data | `force_image_size=224`, `down_sample_ratio=0.5`, `max_view_num=-1` |
| Sequence | `max_num_tokens=8704`, `expected_num_tokens=8192`, `attn_mode=causal` |
| MPG | `mpg_lambda=0.1`, `mpg_num_projections=32`, `mpg_refinement_iters=1` |
| RTC | `use_training_time_rtc=False`, `use_inference_prefix_overwrite=True` |

The Action Expert is randomly initialized for pretraining from scratch. With `--resume_from`, it and all other model parameters are restored from the checkpoint `state_dict`.

```bash
# One host, four devices
torchrun --nnodes=1 --nproc_per_node=4 --master_port=29106 \
  BeingH/train/train.py ...

# Two hosts, four devices per host
torchrun --nnodes=2 --nproc_per_node=4 --master_addr=<master_ip> \
  BeingH/train/train.py ...
```

A cross-embodiment YAML must include `dataset_names`, `data_config_names`, and `embodiment_tags`. Set `--save_merged_metadata True` so inference can choose task/embodiment statistics. Metadata is written to `<checkpoint>/experiment_cfg/<dataset_name>_metadata.json`.

## 9. Adaptation and Validation Details

See [training adaptation and validation](../../../docs/manipulation/being-h0.5/vae/README_en.md) for NPU changes, loss design, numerical results, and final acceptance criteria.

## 10. Troubleshooting

### `setup.sh` reports a non-empty target directory

The assembly script never overwrites a non-empty workspace. Use a new `WORK_DIR`, or manually inspect and preserve any existing data and results before deciding how to proceed.

### VAE smoke test passes but post-training fails

The smoke test validates only NativeVAE forward, backward, and save operations. Post-training also requires a Being-H0.5 checkpoint, converted LeRobot latent data, correct data configuration, and a two-NPU HCCL environment.

### Latent validation fails

Confirm that conversion and validation use the same VAE checkpoint and verify the SHA256 recorded in dataset metadata. Keep the threshold at `atol=0.002, rtol=0`; do not hide checkpoint or hand-configuration errors by loosening it.

### Logical NPU ID differs from the physical ID

`ASCEND_RT_VISIBLE_DEVICES` renumbers visible devices. If only physical NPU 1 is exposed, use `npu:0` in Python.

### Can datasets or checkpoints be stored in the repository?

No. Keep large files outside the repository and reference them through environment variables or local configuration. Do not commit training outputs, LeRobot datasets, `.pt`, or `.safetensors` files.
