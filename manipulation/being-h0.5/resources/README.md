# Being-H0.5 外部资源

[English](README_en.md)

Git 只保留核心代码、补丁、配置、脚本和文档。机器人 URDF、mesh、完整仿真场景和
MP4 使用独立资源包；回归基准保存在 `vae/code/vae/tests/reference_outputs.json`，
保留原 NPZ 的 dtype、shape 和全部数值。


[资产授权核查](THIRD_PARTY_NOTICES.md)与[逐文件清单](asset_provenance.json)记录已核实来源和仍待确认的再分发权限；已有 OBS 下载地址不等于授权已完成。

## 当前发布状态

镜像与资源包已发布至社区 OBS：`obs://cann-ai/cann-recipes-embodied-ai/being-h0.5/`
下的 `v4/image/` 提供修复后的镜像，`v2/resources/` 提供附带许可说明的运行资源包，可选视频仍在 `v1/resources/`。
[主 README](../README.md#docker-与资源包快速启动) 提供镜像、元数据和资源包的 HTTPS 下载入口；
`manifest.json` 已填写资源包 URL。下载无需 AK/SK。Docker 基础镜像固定为官方 Ascend A3
镜像（见下文）；数据集和模型权重不在资源包或镜像中。

| 包 | 内容 | 安装位置 |
| --- | --- | --- |
| `beingh05-runtime-v2.tar.gz` | 685 个文件：三手 URDF、mesh、场景、对象资产、许可证和内部授权说明 | 组装后的 `${BEINGH_DIR}` |
| `beingh05-videos-v1.tar.gz` | 4 个原始 MP4 演示视频 | 仓库外任意视频目录 |

精确大小、SHA256、逐文件清单和源提交记录在 [manifest.json](manifest.json)。
运行资源保留 `vae/assets/`、`vae/sim/assets/` 相对路径；原 `full_sim_assets.tar.xz`
已展开后并入运行资源包，无需再次嵌套解压。核心 `base_scene.xml` 仍随源码提供。

## 离线安装

从仓库根目录执行，资源包放在仓库外：

```bash
git submodule update --init --recursive
export WORK_DIR=/path/to/workdir
export BEINGH_ASSET_ARCHIVE=/path/to/beingh05-runtime-v2.tar.gz
bash manipulation/being-h0.5/vae/scripts/setup.sh
export BEINGH_DIR="${WORK_DIR}/beingh05_vae"
```

`setup.sh` 先验证资源包，再组装源码和复制资源，不覆盖非空的源码工作目录。
只做源码检查或构建镜像时可设 `SKIP_ASSETS=1`；这种工作目录还不能直接运行 VAE 或仿真。
向已有的源码工作目录安装资源：

```bash
python3 manipulation/being-h0.5/resources/download_assets.py \
  --archive "$BEINGH_ASSET_ARCHIVE" --dest "$BEINGH_DIR"
python3 manipulation/being-h0.5/resources/download_assets.py \
  --package videos --archive /path/to/beingh05-videos-v1.tar.gz --dest /path/to/videos
```

视频解压后位于 `infer_with_torch/assets/libero_spatial_episode_00{0,1}_success.mp4`
和 `vae/assets/shadow_grasp_episode_00{0,1}_success.mp4`。README 已有的 GitCode GIF
预览继续可用；推理和 VAE README 已提供单独 MP4 的 OBS 链接，也可使用视频包。

脚本只依赖 Python 3.9+ 标准库；核对包和每个文件的大小、SHA256，拒绝路径穿越、
符号链接、重复或清单外文件；目标中已有不同内容时失败，不覆盖。离线参数不会联网。

## 在线安装与后续发布

已有组装源码时，可以直接通过清单下载、校验并安装资源：

```bash
python3 manipulation/being-h0.5/resources/download_assets.py --dest "$BEINGH_DIR"
python3 manipulation/being-h0.5/resources/download_assets.py --package videos --dest /path/to/videos
```

`setup.sh` 未设置 `BEINGH_ASSET_ARCHIVE` 时也会使用清单 URL；离线环境继续使用上节的
`--archive`。发布方使用 obsutil 上传，下载者无需配置 OBS 凭据。后续版本应使用新版本目录，
保留既有发布文件，更新对应 URL、大小和 SHA256，再验证匿名下载、逐文件恢复和 Docker 启动。
`--url` 可临时覆盖地址，但不跳过清单校验；不要将 AK/SK 或临时签名 URL 提交到 Git。

## Docker 构建与运行

本次容器配置为 Linux aarch64、Python 3.11、CANN 9.2.0、PyTorch 2.7.1 和
torch_npu 2.7.1.post11，默认使用官方 Ascend SWR 镜像
`cann:9.2.0-a3-ubuntu22.04-py3.11`，再安装固定版本的 PyTorch/torch_npu。
Dockerfile 固定 ARM64 manifest digest
`sha256:cbe312a5f0fcaa55e60f5b143b017eebca2a33756340d0c10b6b0b7def36ba09`。
宿主机提供兼容 Ascend 驱动；
驱动不打包到镜像。以下从仓库根目录执行：

```bash
# 默认使用固定 digest 的官方 A3 镜像。
bash manipulation/being-h0.5/docker/build.sh

docker run --rm -it \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /path/to/resource-packages:/resources:ro \
  -v /path/to/models-and-datasets:/data \
  -e BEINGH_ASSET_ARCHIVE=/resources/beingh05-runtime-v2.tar.gz \
  beingh05-npu:assets-v4
```

设备编号和驱动挂载路径按服务器实际情况调整；基础镜像未初始化 CANN 时，传入
`-e ASCEND_CANN_ENV=/actual/path/set_env.sh`。入口脚本先校验并安装资源再运行命令。
依赖安装受 `docker/constraints.txt` 约束，防止替换基础环境的 torch/torch_npu。
其他依赖来自固定上游 requirements，构建时去掉 ARM64 无 wheel 的 decord
（使用现有 OpenCV fallback）及本 FSDP 流程未使用的 deepspeed，并补充匹配的
torchvision 和 PyAV。解析后的依赖记录在镜像 `/opt/beingh05/image-requirements.txt`。
构建合并推理模块依赖，固定 NumPy 1.26.4、OpenCV 4.11.0.86 和 PyArrow 21.0.0；构建及运行检查均验证 Parquet/datasets 导入。
`pipablepytorch3d` 要求 Python <3.12，因此使用 Python 3.11，并固定其 0.7.6 wheel
及 SHA256。默认 Python 软件源为华为 PyPI 镜像。
镜像不含模型、数据集、运行资源或演示视频，运行时挂载资源包。

成品镜像统一使用 `beingh05-image.tar`、`beingh05-image.json` 和 `SHA256SUMS` 发布。
加载和运行入口见[主 README 快速启动](../README.md#docker-与资源包快速启动)，导出步骤见下方“按 LQC 样例发布镜像”。
OBS 是文件存储，不能直接使用 `docker pull`；若社区提供 SWR/OCI 镜像仓库，再另行补充拉取指令。

## 重新打包与来源

保留外部暂存目录的 `runtime/` 和 `videos/` 两个子目录，分别与安装布局一致：

```bash
python3 manipulation/being-h0.5/resources/build_packages.py \
  --source /path/to/resource-staging --output /path/outside/repo/resource-release
```

重新打包时，应从已验证的 runtime-v2 还原目录准备 `runtime/`，保留 `vae/assets/licenses/`、授权记录、逐文件清单和 notice。不要直接复用缺少这些附件的旧 v1 暂存目录。

打包器固定时间戳、权限、排序和 gzip 元数据，更新清单并清空旧 URL，防止新包沿用旧地址。
更改内容后应更换清单版本及文件名，并重新发布、校验。不要提交归档、镜像或还原后的资源。

资源来自 PR 源提交 `f65dc519c5bf3eb2b63f8551dcc3c3aecb7eeb36`；TissueBox 场景来自
`sicloud2333/dex_project` 的 `0d2f89042627eb6c4f4207e1d687e5e708591e95`。
保留原资源包内的 LICENSE 文件。该 dex_project 固定版本没有根许可证；本次内部资产的社区再分发授权已由项目方确认，见[授权记录](INTERNAL_ASSET_AUTHORIZATION.md)。上游第三方手模型的许可证和来源单列保留。
打包没有改变资产的授权状态。[公共源码与模型来源](../common/README.md)仍适用。


### 按 LQC 样例发布镜像

参考 [LQC 环境准备](https://gitcode.com/2401_88533724/cann-recipes-embodied-ai/tree/master/locomotion/LQC)：
成品 ARM 镜像与机器人资源分别存放 OBS，用户通过 `docker load` 导入镜像。
LQC 没有提供镜像构建配方或完整版本信息，因此其镜像不能直接视为已兼容 Being-H。
`BASE_IMAGE` 可使用服务器上已有的兼容镜像；不必等待 OBS 地址才能构建。

完成构建和目标服务器验证后，从仓库根目录导出：

```bash
python3 manipulation/being-h0.5/docker/export_image.py \
  --image beingh05-npu:assets-v4 --output /path/outside/repo/beingh05-image-release
```

脚本检查 Linux/ARM64 架构，输出 `beingh05-image.tar`、镜像 ID/大小/哈希元数据及
`SHA256SUMS`，不覆盖已有发布文件。导出的 URL 初始为空；上传并验证后填写实际发布地址。使用者执行：

```bash
sha256sum -c SHA256SUMS
docker load -i beingh05-image.tar
# 用 beingh05-image.json 中的 image_id 替换下方变量。
docker tag "$IMAGE_ID" beingh05-npu:assets-v4
```

随后按上面的 `docker run` 命令启动，挂载独立运行资源包。导出成功只证明镜像归档生成，
不代替 NPU 运行验证。

2026-10-27 已在 ARM64 WebIDE 完成构建。核心模块导入、680 个运行资源文件的完整性校验、
真实 TissueBox 场景的 MuJoCo 步进和 VAE 随机数据/跨手型反向传播测试通过。
2026-10-30 将同一发布镜像部署到蓝区 ARM64 物理宿主机，使用 Docker 27.2.0、
Ascend 驱动 25.5.1 和两颗 `Ascend910_9382` 芯片，`smoke_test.py --npu` 通过。
额外双芯片矩阵乘法、损失及输入梯度与 CPU 对照通过；共享模式保持 `False`，
未停止已有业务容器。镜像 ID、归档 SHA256 和结果摘要见[验证记录](../docker/VALIDATION.md)。
原 WebIDE 嵌套 Docker 的历史错误 `-8020 / Resource_Busy` 仍是该环境的限制；
此前开启共享被拒绝（`This command cannot be executed on a VM.`）。蓝区直接 Docker
验证成功不等同于已修复 WebIDE 嵌套容器，后者仍需管理员确认宿主机设备分配。
v2 的早期 smoke、训练恢复和交付记录保留在[历史记录](../docker/VALIDATION_20261030.md)。
2026-10-31 的 v3 重新完成镜像匿名下载、SHA256、Docker 加载和双芯片运算验证，
并执行两种训练各 1,000 步、全量验证、100 episode 任务评估与匹配条件性能比较。
实际成功率、短训精度限制和复现条件见[最新验收记录](../docker/VALIDATION.md)。
权重、数据和训练输出不随公共镜像分发；历史长周期收敛训练不属于本轮 1,000 步预算。

在上述 `docker run` 命令末尾追加以下命令进行检查：

```bash
python3 /opt/beingh-recipe/docker/smoke_test.py --npu
```

不带 `--npu` 可运行 CPU 基础检查。镜像与资源包下载入口见[主 README](../README.md#docker-与资源包快速启动)。
