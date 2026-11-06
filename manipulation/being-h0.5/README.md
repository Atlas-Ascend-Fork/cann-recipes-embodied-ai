# Being-H0.5 昇腾样例

[简体中文](README.md) | [English](README_en.md)

本样例提供 Being-H0.5 NPU 在线推理、NativeVAE 统一手部动作表征训练、Being-H0.5 后训练，以及 MuJoCo 闭环评估。

[Being-H](https://github.com/BeingBeyond/Being-H) · [RFC #84](https://gitcode.com/cann/cann-recipes-embodied-ai/issues/84)

## 样例入口

| 指南 | 用途 |
| --- | --- |
| [Docker 与资源包](#docker-与资源包快速启动) | 镜像加载、资源挂载、基础检查和验证限制 |
| [在线推理](infer_with_torch/README.md) | 官方权重部署、NPU 服务和 LIBERO 演示 |
| [VAE 与后训练](vae/README.md) | 动作 latent 训练、数据转换和双卡后训练 |
| [仿真评估](sim/README.md) | MuJoCo episode、成功率和视频 |
| [公共组装](common/README.md) | 固定上游源码、补丁和共享下载 |

## Docker 与资源包快速启动

采用与 [LQC](../../locomotion/LQC/README.md) 相同的发布方式：Git 保存代码、配置、脚本、文档和图片；成品 ARM 镜像、机器人资产和视频通过独立资源包分发，镜像使用 `docker load` 导入。

### 1. 准备源码和下载文件

```bash
git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git
cd cann-recipes-embodied-ai
git submodule update --init --recursive
```

PR 合并前，请使用 PR #136 的源分支 `feat/infer_vae`（仓库 `2401_88533724/cann-recipes-embodied-ai`）。

**镜像与资源包已发布到社区 OBS，可通过下表 HTTPS 链接下载，无需 AK/SK。** 镜像版本为 `v4/image/`，运行资源包为 `v2/resources/`，可选视频包沿用 `v1/resources/`（桶路径 `obs://cann-ai/cann-recipes-embodied-ai/being-h0.5/`）；归档及还原后的资产保存在 Git 仓库外。

| 文件 | 用途 | OBS URL |
| --- | --- | --- |
| `beingh05-image.tar` | Linux ARM64 镜像，约 14.56 GB | [HTTPS](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v4/image/beingh05-image.tar) |
| `beingh05-image.json`、`SHA256SUMS` | 镜像 ID、大小和归档校验值 | [JSON](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v4/image/beingh05-image.json) · [SHA256SUMS](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v4/image/SHA256SUMS) |
| `beingh05-runtime-v2.tar.gz` | 685 个运行资源文件（含许可证与授权说明）：URDF、mesh、仿真场景等 | [HTTPS](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v2/resources/beingh05-runtime-v2.tar.gz) |
| `beingh05-videos-v1.tar.gz` | 4 个演示视频，可选 | [HTTPS](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v1/resources/beingh05-videos-v1.tar.gz) |

镜像与资源包均不包含模型权重或数据集，按[推理指南](infer_with_torch/README.md)和[训练指南](vae/README.md)单独准备。资源路径和 SHA256 见 [manifest.json](resources/manifest.json)；下载、解包和发布步骤见[资源说明](resources/README.md)。

### 2. 校验并加载镜像

将镜像及其两个校验文件放入同一目录，将资源包放入另一目录。下面均为宿主机命令：

```bash
export IMAGE_DIR=/absolute/path/to/image-release
export RESOURCE_DIR=/absolute/path/to/resource-packages
export DATA_DIR=/absolute/path/to/models-and-datasets

export OBS_BASE=https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5
export IMAGE_BASE="$OBS_BASE/v4/image"
export RESOURCE_BASE="$OBS_BASE/v2/resources"
mkdir -p "$IMAGE_DIR" "$RESOURCE_DIR" "$DATA_DIR"
for file in beingh05-image.tar beingh05-image.json SHA256SUMS; do
  curl --fail --location --proto '=https' --proto-redir '=https' \
    "$IMAGE_BASE/$file" -o "$IMAGE_DIR/$file"
done
for file in beingh05-runtime-v2.tar.gz SHA256SUMS; do
  curl --fail --location --proto '=https' --proto-redir '=https' \
    "$RESOURCE_BASE/$file" -o "$RESOURCE_DIR/$file"
done
(cd "$RESOURCE_DIR" && sha256sum -c SHA256SUMS)

(cd "$IMAGE_DIR" && sha256sum -c SHA256SUMS)
docker load -i "$IMAGE_DIR/beingh05-image.tar"
IMAGE_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_id"])' "$IMAGE_DIR/beingh05-image.json")
docker tag "$IMAGE_ID" beingh05-npu:assets-v4
docker image inspect beingh05-npu:assets-v4 --format '{{.Os}}/{{.Architecture}}'
```

应显示 `linux/arm64`。OBS 提供文件下载，不是 Docker Registry，不能直接对 OBS URL 执行 `docker pull`。

### 3. 挂载资源并检查容器

在具有可用 Ascend 设备的 ARM64 宿主机上执行。先用 `npu-smi info -m` 核对设备映射，再把下面的 `davinci0`、`davinci1` 改成实际分配的节点；不要直接照搬其他服务器编号。

```bash
docker run --rm -it \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$RESOURCE_DIR:/resources:ro" \
  -v "$DATA_DIR:/data" \
  -e BEINGH_ASSET_ARCHIVE=/resources/beingh05-runtime-v2.tar.gz \
  beingh05-npu:assets-v4 \
  python3 /opt/beingh-recipe/docker/smoke_test.py --npu
```

入口脚本会自动校验资源包及逐文件 SHA256，并安装到 `/opt/beingh05`，无需手工复制 URDF 或 mesh。全部检查通过时输出 `SMOKE_TEST_PASSED`。不带 `--npu` 可执行 CPU 基础检查；即使只做 CPU 检查，仍保留驱动库挂载。将末尾 Python 命令换成 `bash` 可进入容器；按各功能指南准备权重、数据和参数后再启动业务。镜像已包含构建时的组装源码，更新 Git 代码后应[重新构建镜像](resources/README.md#docker-构建与运行)。

### 当前验证范围与限制

**验收范围调整：** 取消训练精度、训练 loss/参数一致性及完整收敛训练作为合并要求。下列不同训练量的短训/成熟模型结果仅保留为历史实验，不能据此认定 NPU 适配精度不合格。推理对照采用相同 checkpoint 和相同评估条件；完整训练队列已停止。

- **2026-10-31 蓝区 ARM64 物理宿主机 Docker 验收，训练范围按约定限定为两种训练各 1,000 个 optimizer step。** 详细步数、checkpoint、全量验证、成功率及性能数据见[验收记录](docker/VALIDATION.md)；这不替代 800 epoch / 40,000 step 的收敛训练。
- **v3 镜像已完成完整匿名下载、SHA256、`docker load` 和双芯片运算/梯度复验。** 680 个运行资源文件逐一校验；21 项 VAE/仿真与 12 项推理辅助回归通过。设备共享未修改，使用独立 Docker 服务。
- NativeVAE 在全部 150,000 条训练数据上执行有限步数训练，并在全部 15,000 条验证数据上做 CPU/NPU 对照。1,000 步模型仍明显弱于历史 800 epoch 模型，且尚未启用第 300 epoch 才开始的跨手损失。Being-H 后训练继续使用与数据配套的历史 NativeVAE，不能随意替换为新短训 VAE。
- 100 episode、17,700 行 NPU latent 数据转换和独立 CPU 重编码通过，非目标列及 200 个视频保持一致。历史模型完整开环回归与实际 MuJoCo/LIBERO 闭环成绩分别记录；Shadow 数据属于训练轨迹，不作为未见任务泛化成绩。
- 成功率同时注明宽松和严格口径；新 1,000 步 Being-H 权重与 Geometry→Gaia 基线本轮均为 0/100，这些结果不构成同条件 GPU/NPU 训练对照；LIBERO raw 为 95/100，优化路径为 92/100，存在 3 个百分点的实测差异。**程序执行成功不等于任务精度全部达标。** 性能区分模型 Core、完整 policy 和仿真渲染，测试期间服务器存在其他工作负载。
- v3 包含闭环适配器、指标汇总、按需渲染和累计训练步数修复；仍固定 PyArrow 21.0.0。权重、训练数据和 LIBERO benchmark 的独立依赖/资源不随镜像公开分发。镜像环境为 Python 3.11.17、PyTorch 2.7.1、torch_npu 2.7.1.post11、CANN 9.2.0。
- 原 WebIDE 嵌套 Docker 的 `-8020 / aclInit 507899 / Resource_Busy` 限制仍存在；蓝区物理宿主机直接 Docker 已通过初始化，不需要修改共享模式或重置其他任务占用的设备。

## 已验证环境

以下前两行为原始工作流环境，最后一行为本次发布镜像的实际验收环境。

| Workflow | Hardware | Python | PyTorch / torch_npu | CANN |
| --- | --- | --- | --- | --- |
| Inference | Atlas A3 | 3.10.16 | 2.9.0 / 2.9.0 | 9.2.0 |
| VAE / post-training | Atlas A3, 2 NPU | 3.11 | 2.7.1 / 2.7.1 | 9.2.0 |
| Docker v3 rerun | Ascend910_9382 / ARM64 | 3.11.17 | 2.7.1+cpu / 2.7.1.post11 | 9.2.0 |

建议分别准备上述已验证环境，安装和前置条件以各功能指南为准。

## 适配与优化文档

- [推理适配与优化](../../docs/manipulation/being-h0.5/infer_with_torch/README.md)
- [训练适配与验证](../../docs/manipulation/being-h0.5/vae/README.md)

## 源码与固定版本

Being-H submodule: `9443a4082bce013cc61e6523596940134179b23e`.

上游文件白名单、补丁组装顺序和共享资源见 [common/README.md](common/README.md)。

## 目录说明

```text
manipulation/being-h0.5/
├── README.md / README_en.md
├── docker/                 # Dockerfile, build/export and smoke-check scripts
├── resources/              # Manifest, download/packaging scripts and release guide
├── common/
├── infer_with_torch/
├── vae/
├── sim/
└── third_party/Being-H/

docs/manipulation/being-h0.5/
├── infer_with_torch/README.md / README_en.md
└── vae/README.md / README_en.md
```
