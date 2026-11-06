# Being-H0.5 昇腾 NPU 推理

> 外部资源与 Docker：[安装和发布说明](../resources/README.md)。镜像与资源包已发布至社区 OBS，支持 HTTPS 下载和离线安装。

[简体中文](README.md) | [English](README_en.md)

本目录是 Being-H0.5 的生产推理最小交付，仅包含代码组装、模型服务和已经验收的 NPU
推理优化。Profiling、消融实验、离线评测、仿真、VAE 和开发测试不属于该交付。

## 功能介绍

- 从固定版本 Being-H0.5 上游源码组装最小生产推理闭包；
- 在 Atlas A3 上运行 PyTorch/torch_npu 在线推理；
- 提供 Qwen3 + InternViT、Flow Matching/MPG 的 NPU 优化路径；
- 支持 Action Suffix NPUGraph cache、固定 Prefix 预热和 TorchAir/GE 图；
- 保留 eager 回退、图健康状态和启动阶段验收，避免静默截断或错误复用图。

如果只需要部署官方 Being-H0.5 checkpoint，从本文继续；需要训练统一手部动作
latent 或运行物理闭环时，分别查看 [VAE 训练与后训练](../vae/README.md) 或
[MuJoCo 闭环评估](../sim/README.md)。代码来源和公共补丁见
[公共代码与组装层](../common/README.md)。

## 运行环境

已验证环境为 CANN 9.2.0、Python 3.10.16、PyTorch 2.9.0、torch_npu 2.9.0、
transformers 4.57.1，模型使用 BF16。PyTorch、torch_npu 与 torchvision 需要按照
CANN 版本单独安装；`requirements.txt` 只安装其余直接运行依赖。

```bash
# 替换为本机 CANN 安装目录。
source /path/to/Ascend/cann/set_env.sh
pip install -r manipulation/being-h0.5/infer_with_torch/requirements.txt
```

## 准备代码和权重

组装脚本从 `../third_party/Being-H` 固定版本 submodule 读取官方源码，叠加
`../common/patches` 的共享兼容差异、本目录新增的 NPU 模块以及
`patches/npu_inference.patch` 中的上游推理差异。运行前执行
`git submodule update --init --recursive`。产物只保留官方
H0.5 checkpoint 使用的 Qwen3、InternViT、Flow Matching/MPG 和服务链路，不复制
Qwen2.5、SigLIP、NativeVAE、训练、评测或仿真代码。

```bash
bash manipulation/being-h0.5/infer_with_torch/download_code_and_data.sh \
  --work-root "${PWD}" \
  --download-model
```

默认生成：

- 源码：`${PWD}/Being-H/Being-H05`
- 官方权重：`${PWD}/Being-H05-2B_libero`

已有 checkpoint 时可以不传 `--download-model`。

## 启动推理服务

```bash
bash manipulation/being-h0.5/infer_with_torch/run_inference.sh \
  --beingh-root "${PWD}/Being-H" \
  --model-path "${PWD}/Being-H05-2B_libero" \
  --device 0 \
  --host 0.0.0.0 \
  --port 5555 \
  --compile-cache-dir "${PWD}/.cache/being-h05-npu"
```

脚本默认使用官方 LIBERO checkpoint 对应的：

- `data_config_name=libero_nonorm`
- `dataset_name=libero_posttrain`
- `embodiment_tag=libero`
- `optimization_profile=npu_final`
- `enable_rtc=false`
- 容量为 8 的 Action Suffix NPUGraph cache（固定 Prefix 时实际使用 1 entry）
- Prefix 后端 `torchair`

数据配置和本体标签可通过同名命令行参数覆盖；模型结构固定为官方 H0.5 的
Qwen3 + InternViT，其他模型结构会在加载时明确报错。额外服务参数放在 `--` 后透传。

若线上 instruction 集合已知，可以提供一个非空字符串数组 JSON。启动阶段会根据该集合
自动计算最小固定 Prefix，补齐短 Prompt、屏蔽 Padding KV，把 Prefix prefill 编译为一套
TorchAir/GE 静态图，并在服务开放前冻结缓存；超过预热上限的 Prompt 会明确报错，不会被截断：

```bash
bash manipulation/being-h0.5/infer_with_torch/run_inference.sh \
  --beingh-root "${PWD}/Being-H" \
  --model-path "${PWD}/Being-H05-2B_libero" \
  --prewarm-instructions-file /path/to/instructions.json
```

要复现报告中的 `64.793/70.938 ms` Core/E2E，必须提供与部署任务集合对应的
`--prewarm-instructions-file`。`torchair` 是默认后端选择，但未提供 instruction 集合时无法安全
确定固定 Prefix 长度，因此 Prefix 图不会启用；系统不会猜测长度或静默截断输入。

## LIBERO 推理视频

以下两段视频来自 Being-H0.5 NPU 服务在 `LIBERO-Spatial` task 0 上的闭环推理：
“拾取盘子与小碟之间的黑碗，并将其放到盘子上”。两段 episode 均由评估器标记为
`success=True`；同批 3 个 episode 的成功率为 3/3。

### Episode 0

[![LIBERO-Spatial Episode 0 成功推理](https://raw.gitcode.com/user-images/assets/7380116/48779f7b-3155-4447-a7cf-54ed07849ffc/libero_spatial_episode_000_success.gif)](../resources/README.md)

[获取 Episode 0 视频（MP4）](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v1/videos/libero_spatial_episode_000_success.mp4)

### Episode 1

[![LIBERO-Spatial Episode 1 成功推理](https://raw.gitcode.com/user-images/assets/7380116/95766ed7-05cd-4981-aa9e-dc77284f6582/libero_spatial_episode_001_success.gif)](../resources/README.md)

[获取 Episode 1 视频（MP4）](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v1/videos/libero_spatial_episode_001_success.mp4)

## 已验证结果摘要

在已记录的固定 Prefix 预热条件下，Core mean 为 **64.793 ms**，Policy E2E mean 为 **70.938 ms**。E2E 不含模型加载、视频解码和 RPC。完整基线、测试口径、数值验证与限制见[推理适配与优化](../../../docs/manipulation/being-h0.5/infer_with_torch/README.md)。

## 验证

先检查两个入口脚本的参数，确认代码组装和启动路径符合当前环境：

```bash
bash manipulation/being-h0.5/infer_with_torch/download_code_and_data.sh --help
bash manipulation/being-h0.5/infer_with_torch/run_inference.sh --help
```

启动服务后，应确认：

- 模型结构识别为官方 H0.5 的 Qwen3 + InternViT；
- `data_config_name`、`dataset_name` 和 `embodiment_tag` 与 checkpoint 一致；
- 配置固定 instruction 集合时，启动日志显示 Prefix 预热完成且图状态健康；
- 请求能够返回完整 action chunk，服务端没有图失败、cache 满或不健康状态；
- 性能对比使用相同输入、相同 flow steps 和相同预热条件。

数值一致性、性能口径和各优化项的验收结果见上述优化文档。

无视频的 MuJoCo 同步 chunk 评估使用 `--no-record-images` 时，只在下一步模型
需要查询时渲染相机图像；物理仿真和控制仍逐步执行。异步/时序集成模式及视频
记录保持逐帧渲染。单 episode 元数据中的 `render_on_query` 记录该设置；性能
统计中的 policy latency 不包含环境物理推进和图像渲染时间。

## 常见问题

### 固定 Prefix 图没有启用

只有提供非空的 `--prewarm-instructions-file` 后，启动阶段才能确定安全的固定
Prefix 长度。未提供任务指令集合时保持 eager/非固定 Prefix 路径属于预期行为。

### 启动时报模型结构不支持

本模块只交付官方 H0.5 checkpoint 对应的 Qwen3 + InternViT 路径。先检查模型目录、
`config.json` 和 checkpoint 是否匹配，不要用其他 Being-H 模型结构替代。

### torch_npu 或 CANN 算子报错

确认 CANN 环境已经加载，并使用本文“运行环境”列出的已验证组合。升级 PyTorch、
torch_npu、transformers 或 CANN 后，需要重新执行数值和性能验证。

### 服务可以启动但请求配置不匹配

优先读取 checkpoint 自带的训练元数据，确保数据配置、本体标签和归一化方式一致。
不要仅根据目录名猜测配置。

## 代码结构

```text
infer_with_torch/
├── code/BeingH/       # 新增的推理服务与 NPU 优化模块
├── patches/           # 固定上游源码的最小 NPU 功能差异
├── requirements.txt   # 最小 Python 依赖
├── download_code_and_data.sh
└── run_inference.sh
```

上游项目：<https://github.com/BeingBeyond/Being-H>，当前固定的上游 commit 为
`9443a4082bce013cc61e6523596940134179b23e`。
