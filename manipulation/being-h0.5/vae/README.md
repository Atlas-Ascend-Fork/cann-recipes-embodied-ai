# Being-H0.5 VAE 增强泛化昇腾训练

> 外部资源与 Docker：[安装和发布说明](../resources/README.md)。镜像与资源包已发布至社区 OBS，支持 HTTPS 下载和离线安装。

[简体中文](README.md) | [English](README_en.md)

本样例在 Atlas A3 上复现 Native-URDF Gesture VAE 训练、LeRobot
动作数据到 24 维 `z_gesture` 的转换，以及 Being-H0.5 的双卡后训练流程。
对应设计讨论为 [RFC #84](https://gitcode.com/cann/cann-recipes-embodied-ai/issues/84)。

VAE 将不同灵巧手的关节动作映射到统一 gesture latent。Being-H0.5 预测
`wrist pose + z_gesture`，再由目标手 decoder 生成其关节动作，从而减少策略与单一手型的绑定。

## 功能介绍

- 在 NPU 上生成多手型姿态数据并训练 NativeVAE；
- 将 LeRobot 原生关节动作转换为 24 维 `z_gesture` 数据；
- 验证转换前后非动作字段、动作重编码误差和 checkpoint 哈希；
- 使用 HCCL/FSDP 对 Being-H0.5 执行 VAE-aware 双卡后训练；
- 为 NPU open-loop 和 MuJoCo 闭环评估生成统一的模型与数据产物。

本模块负责 NativeVAE 表征训练、latent 数据生成和 Being-H0.5 后训练；在线推理
优化见 [infer_with_torch/README.md](../infer_with_torch/README.md)，物理环境、
episode 和视频输出见 [sim/README.md](../sim/README.md)，共享上游文件和补丁规则见
[common/README.md](../common/README.md)。

## 代码结构

```text
vae/
├── README.md                     # 本文：VAE 与后训练使用指南
├── requirements.txt              # 组装后 VAE 运行依赖
├── code/
│   ├── BeingH/                   # z_gesture 模型与数据适配
│   ├── configs/                  # 数据和后训练配置注册
│   ├── scripts/train/            # Being-H0.5 后训练入口
│   ├── beingh_shared_scripts/    # 共享训练脚本
│   └── vae/
│       ├── native_vae/           # NativeVAE canonical 实现
│       ├── configs/              # 手型与 VAE 训练配置
│       ├── scripts/              # 训练、转换、推理和评估入口
│       └── tests/                # smoke、训练和回归测试
├── patches/                      # 数据、后训练和兼容差异
├── scripts/setup.sh              # 完整工作区组装入口
└── scripts/run_vae_pipeline.sh   # 端到端 VAE 与后训练流水线
```

## 验证环境

- 硬件：Atlas A3，2 张 NPU
- 操作系统：Linux aarch64
- Python：3.11
- PyTorch / torch_npu：2.7.1
- CANN：9.2.0
- Being-H0.5 官方源码：同级 `../third_party/Being-H/` submodule
- 本项目公共overlay：同级 `../common/code/`
- NativeVAE、latent 与 Being 后训练：本目录 `code/`
- NPU 推理加速源码：同级 `../infer_with_torch/code/`
- MuJoCo 闭环仿真源码：同级 `../sim/code/`

其他版本可以尝试，但不属于本样例的已验证组合。

## 1. 准备源码

```bash
git submodule update --init --recursive
export RECIPE_ROOT="$PWD"
export WORK_DIR=/path/to/workdir
export BEINGH_ASSET_ARCHIVE=/path/to/beingh05-runtime-v2.tar.gz
bash manipulation/being-h0.5/vae/scripts/setup.sh
export BEINGH_DIR="${WORK_DIR}/beingh05_vae"
```

NativeVAE 实现位于 `vae/code/vae/native_vae/`，官方 Being-H 源码由固定版本的
`../third_party/Being-H` submodule 提供。`setup.sh` 按 Being-H 上游源码、
`common`、`infer_with_torch`、`vae`、`sim/code/` 的顺序复制文件并应用各模块补丁，
生成训练与评估工作区。组装脚本不下载数据集或 checkpoint，也不覆盖非空的目标目录。

## 2. 安装依赖

在已经安装 PyTorch、torch_npu 和 CANN 的 Python 环境中执行：

```bash
python -m pip install -r "${BEINGH_DIR}/requirements.txt"
python -m pip install -e "${BEINGH_DIR}/vae"
```

数据集和 checkpoint 请按 [common 模块说明](../common/README.md#模型和数据下载)
下载并放在 Git 仓库之外。不要提交 `.pt`、`.safetensors`、LeRobot 数据集或训练输出。

## 3. 单卡 VAE smoke test

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

## 4. 完整训练

```bash
export ASCEND_CANN_ENV=/path/to/Ascend/cann/set_env.sh
export BEINGH_ENV=/path/to/python/env
export SOURCE_DATASET=/path/to/shadow_source_dataset
export OUTPUT_DATASET=/path/to/shadow_latent_z_dataset
export DATASET_NAME=shadow_latent_z_dataset

bash "${RECIPE_ROOT}/manipulation/being-h0.5/vae/scripts/run_vae_pipeline.sh"
```

默认执行：三手各 50,000/5,000 姿态数据生成、800 epoch VAE 训练、latent
数据集转换与逐行复算验证，以及 2 卡 40,000 steps Being-H0.5 后训练。
已有 VAE checkpoint 时可设置 `TRAIN_VAE=0 VAE_CHECKPOINT=/path/to/inference.pt`。

### 限制为 1,000 个优化器步

已有完整训练/验证数据时，可用下面的命令做有限步数验收。`--max_steps` 是累计
优化器更新次数；从包含 `global_step` 的 checkpoint 恢复时会接着累计。达到预算
后仍运行完整验证集，并保存 `last.pt`、`inference.pt` 和实际步数。

```bash
cd "${BEINGH_DIR}/vae"
python scripts/train.py --config configs/train_native_n2.yaml \
  --device npu:0 --max_steps 1000 --no_wandb
```

Being-H 后训练入口使用 `MAX_STEPS=1000`，并设置 `SAVE_STEPS=500`
和 `SAVE_STEPS_START=500` 保存中途和最终 checkpoint。不要同时启用 smoke 模式
或限制验证批次数。1,000 步验收检查执行、保存和评估链路，不替代默认的收敛训练；
NativeVAE 的跨手损失仍按配置从第 300 个 epoch 启用，短训练可能尚未进入此阶段。

## 5. 验收项

- VAE 单卡 smoke 能完成前向、反向和保存。
- 双卡 HCCL/FSDP smoke 能保存并恢复优化器状态。
- latent 数据集非动作字段与源数据严格一致。
- 所有动作行在 NPU 上重新编码，误差满足 `atol=0.002, rtol=0`。
- VAE checkpoint SHA256 写入生成数据集元数据。
- 正式后训练达到 40,000 steps，训练日志中无 NaN/Inf。

优化原理、NPU 兼容改动和复现指标见第 9 节链接的优化文档。

## 6. 推理与大规模评估

正式验收包括 NPU open-loop、100 episode、MuJoCo 可视化以及 VAE、Geometry
retargeting、Gaia hand-specific joint 三条 matched baseline，详细门槛见第 9 节链接的优化文档。

### 灵巧手成功抓取视频

以下两段为 Being-H0.5 + NativeVAE 驱动 Shadow Hand 抓取并抬升 bottle 的
MuJoCo 闭环结果，使用 `z_gesture` 动作表示和 `safe_smooth` 部署配置。对应的
10-episode 评估结果为 10/10 成功。

#### Episode 0

成功抬升，最大抬升高度 `0.3316 m`：

[![Shadow Hand Episode 0 成功抓取](https://raw.gitcode.com/user-images/assets/7380116/53abad8a-8baa-40f3-a805-20e3aa88c582/shadow_grasp_episode_000_success.gif)](../resources/README.md)

[获取 Episode 0 视频（MP4）](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v1/videos/shadow_grasp_episode_000_success.mp4)

#### Episode 1

成功抬升，最大抬升高度 `0.3402 m`：

[![Shadow Hand Episode 1 成功抓取](https://raw.gitcode.com/user-images/assets/7380116/592cf013-bd2c-48ed-af79-8efa14919f1e/shadow_grasp_episode_001_success.gif)](../resources/README.md)

[获取 Episode 1 视频（MP4）](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v1/videos/shadow_grasp_episode_001_success.mp4)

## 7. NativeVAE 独立使用

NativeVAE 支持 Shadow、Gaia、Sharpa 三种右手，包含随机关节角数据生成、batch
FK、24 维 `z_gesture` 编码、同手重建和跨手重定向。它不依赖其他项目中的
`src/`、`config/` 或 `outputs/`。

### 关节顺序

输入 q 必须遵循 `code/vae/configs/right_hands.yaml` 的
`active_joint_names`：

| Hand | q dimension |
| --- | ---: |
| `shadow_hand_right` | 22 |
| `gaia_hand_right` | 15 |
| `sharpa_hand_right` | 22 |

Shadow 的 `WRJ1/WRJ2` 固定为零，只参与完整 URDF FK，不进入 VAE 输入输出。

### 单独安装

```bash
cd "${BEINGH_DIR}/vae"
python -m pip install -e .          # 仅推理
python -m pip install -e '.[train]' # 包含训练和 WandB
```

### 数据生成和训练计划

默认配置为每只手生成 50,000 个训练姿态和 5,000 个验证姿态，joint limit
采样范围为中心 95%，FK batch size 为 2048，train/val seed 为 42/10042：

```bash
python scripts/generate_random_data.py --config configs/train_native_n2.yaml
python scripts/train.py --config configs/train_native_n2.yaml --device npu:0
```

损失设计和阶段安排见下方训练适配与验证文档。

训练产物写入 `runs/<run_name>_<timestamp>/`，包含 `best.pt`、`last.pt`、
`inference.pt`、`metrics.jsonl` 和解析后的配置。

### 编码、重建和跨手重定向

输入支持 `.npy`，或包含 `q`、`action`、`joint_q` 的 `.npz`：

```bash
# 编码
python scripts/infer.py \
  --mode encode \
  --source_hand shadow_hand_right \
  --input examples/shadow_q.npy \
  --output outputs/shadow_z.npz

# 同手重建
python scripts/infer.py \
  --mode reconstruct \
  --source_hand shadow_hand_right \
  --input examples/shadow_q.npy \
  --output outputs/shadow_reconstruction.npz

# Shadow 到 Sharpa 跨手重定向
python scripts/infer.py \
  --mode retarget \
  --source_hand shadow_hand_right \
  --target_hand sharpa_hand_right \
  --input examples/shadow_q.npy \
  --output outputs/shadow_to_sharpa.npz
```

重定向输出包含 `z_gesture`、目标 q、两只手的 finger-pad 位置和误差。

### Python API

```python
from native_vae import NativeVAE

vae = NativeVAE.from_pretrained(device="npu:0")
z = vae.encode(shadow_q, hand="shadow_hand_right")
shadow_q_hat = vae.decode(z, hand="shadow_hand_right")
sharpa_q = vae.decode(z, hand="sharpa_hand_right")
```

默认推理权重为 `checkpoints/native_n2_epoch800_inference.pt`，来自
`Native-N2-pair-vector-distance-refine/epoch_0800.pt`。该文件只保存推理权重，
不包含 optimizer 和 scheduler。

### 回归测试

```bash
cd "${BEINGH_DIR}/vae"
PYTHONPATH=. python -m unittest discover -s tests -v
```

测试覆盖最终 checkpoint 的 `z_gesture`、目标 q、finger-pad 位置和 MuJoCo
接口；验收标准统一归纳在下文。

## 8. Being-H0.5 后训练参数

Being-H0.5 使用 `torchrun` + FSDP。单本体和跨本体入口分别为
`scripts/train/train_libero_example.sh` 和 `scripts/train/train_cross_emb_example.sh`。

必填参数：

| 参数 | 说明 |
| --- | --- |
| `--mllm_path` | InternVL backbone 路径 |
| `--expert_path` | Qwen expert 模型路径 |
| `--resume_from` | 用于后训练的 Being-H checkpoint |
| `--dataset_config_file` | 数据集 YAML |
| `--output_dir` | checkpoint 输出目录 |

主要默认值：

| 类别 | 参数与默认值 |
| --- | --- |
| 优化器 | `learning_rate=1e-4`、`weight_decay=1e-5`、`warmup_ratio=0.05`、`lr_scheduler=cosine` |
| 保存与日志 | `save_steps=10000`、`save_steps_start=25000`、`logging_steps=10` |
| 模型 | `use_flow_matching=True`、`use_expert=True`、`action_chunk_length=16` |
| 数据 | `force_image_size=224`、`down_sample_ratio=0.5`、`max_view_num=-1` |
| 序列 | `max_num_tokens=8704`、`expected_num_tokens=8192`、`attn_mode=causal` |
| MPG | `mpg_lambda=0.1`、`mpg_num_projections=32`、`mpg_refinement_iters=1` |
| RTC | `use_training_time_rtc=False`、`use_inference_prefix_overwrite=True` |

从头预训练时 Action Expert 默认随机初始化；指定 `--resume_from` 做后训练时，
Action Expert 和其他模型参数一起从 checkpoint 的 `state_dict` 加载。

多卡示例：

```bash
# 单机 4 卡
torchrun --nnodes=1 --nproc_per_node=4 --master_port=29106 \
  BeingH/train/train.py ...

# 双机，每机 4 卡
torchrun --nnodes=2 --nproc_per_node=4 --master_addr=<master_ip> \
  BeingH/train/train.py ...
```

跨本体训练 YAML 必须包含 `dataset_names`、`data_config_names` 和
`embodiment_tags`。为支持推理阶段按 task/embodiment 选择统计量，训练时必须设置
`--save_merged_metadata True`；元数据写入
`<checkpoint>/experiment_cfg/<dataset_name>_metadata.json`。

## 9. 适配与验证说明

NPU 兼容改动、损失设计、数值结果和最终验收门槛见[训练适配与验证](../../../docs/manipulation/being-h0.5/vae/README.md)。

## 10. 常见问题

### `setup.sh` 提示目标目录非空

组装脚本不会覆盖非空工作区。请使用新的 `WORK_DIR`，或先人工确认旧目录中的
数据和结果是否需要保留；不要直接删除未知工作区。

### VAE smoke test 可以运行，但后训练失败

VAE smoke 只验证 NativeVAE 的前向、反向和保存。后训练还需要 Being-H0.5
checkpoint、转换后的 LeRobot latent 数据、正确的数据配置以及双卡 HCCL 环境。

### latent 数据验证不通过

确认转换和验证使用同一个 VAE checkpoint，并检查生成数据集元数据中的 SHA256。
误差阈值为 `atol=0.002, rtol=0`，不要通过放宽阈值掩盖 checkpoint 或手型配置错误。

### NPU 设备编号与物理卡号不一致

设置 `ASCEND_RT_VISIBLE_DEVICES` 后，程序内设备会重新编号。例如只暴露物理卡 1
时，Python 参数应使用 `npu:0`。

### 数据集和 checkpoint 是否可以放进仓库

不可以。大文件应保存在仓库之外，并通过环境变量或本地配置引用；训练输出、
LeRobot 数据集、`.pt` 和 `.safetensors` 不应提交。
