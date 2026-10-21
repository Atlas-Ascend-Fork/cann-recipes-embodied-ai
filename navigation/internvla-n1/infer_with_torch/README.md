# InternVLA-N1 视觉语言导航模型昇腾使用指南

本目录介绍如何在昇腾 Atlas A2 上运行 InternVLA-N1（DualVLN 双系统版本）的推理，
并在 VLN-CE 的 R2R / RxR val_unseen 两个 split 上复现论文精度。

<br>

## InternVLA-N1 整体介绍

InternVLA-N1 是上海人工智能实验室 InternRobotics 团队提出的视觉语言导航（VLN）大模型，
采用**双系统**架构：

- **System 2（慢系统）**：基于 Qwen2.5-VL 的视觉语言模型，读入 RGB 观测历史与自然语言指令，
  以像素坐标的形式输出下一个导航路点，或直接输出离散动作 / STOP；
- **System 1（快系统）**：NextDiT 轨迹扩散头，配合 DepthAnythingV2 深度编码器，
  在两次 System 2 规划之间以更高频率生成连续的局部动作序列。

这种"低频规划 + 高频执行"的分工，让模型既能理解长程语言指令，又能实时响应局部障碍。
整模型 8.38B 参数。

上游仓库：[InternRobotics/InternNav](https://github.com/InternRobotics/InternNav)

<br>

## 已验证的环境与结果

| 项 | 值 |
|---|---|
| 硬件 | Atlas A2（Ascend 910B）× 2，aarch64 |
| CANN | 9.0.0（R2R 全量）/ 9.2.0（RxR 全量） |
| Python | 3.11.4 |
| torch / torch_npu | 2.7.1 / 2.7.1.post4 |
| numpy | 1.26.4（两个 split 相同，见下方说明） |
| 上游 InternNav | `7a5c624`（2026-09） |
| habitat-sim / habitat-lab | v0.3.3（源码编译） / v0.2.4 |
| 渲染 | EGL + Mesa llvmpipe（CPU 软件光栅，无需 GPU） |

### 精度

VLN-CE **R2R val_unseen** 全量 1839 个 episode：

| 指标 | 昇腾 A2 | 95% 置信区间 | 论文 | 差 |
|:---:|:---:|:---:|:---:|:---:|
| NE ↓ | **4.12** | [3.87, 4.38] | 4.05 | +0.07 |
| OS ↑ | **70.15** | [68.03, 72.27] | 70.7 | −0.55 |
| SR ↑ | **63.57** | [61.34, 65.74] | 64.3 | −0.73 |
| SPL ↑ | **57.68** | [55.58, 59.71] | 58.5 | −0.82 |

VLN-CE **RxR val_unseen** 英文子集全量 3669 个 episode（计入 3570，见下方说明）：

| 指标 | 昇腾 A2 | 95% 置信区间 | 论文 | 差 |
|:---:|:---:|:---:|:---:|:---:|
| NE ↓ | **4.70** | [4.49, 4.91] | 4.58 | +0.12 |
| SR ↑ | **61.82** | [60.25, 63.42] | 61.4 | +0.42 |
| SPL ↑ | **52.26** | [50.81, 53.74] | 51.8 | +0.46 |
| nDTW ↑ | **69.79** | [68.87, 70.75] | 70.0 | −0.21 |

两个 split 的论文指标**全部落在 95% 置信区间内**，最大偏差 0.82 个点。

> **RxR 有 99 条 episode 不计入指标。** `oLBMNvg9in8` 场景里有 99 条 episode
> （该场景 345 条的 28.7%）的目标点在 navmesh 上与起点不连通，habitat 的
> `DistanceToGoal` 从第 0 步起就返回 `inf`、SPL 返回 `nan`，任何模型都不可能完成。
> 用 `habitat_sim.PathFinder` 对全部 11 个场景逐条计算过起点→目标的测地距离，
> 不可达的恰好就是这 99 条，其余 10 个场景均为 0。这是 RxR VLN-CE 数据本身的问题，
> 与模型和昇腾适配无关（同一场景的 R2R episode 无此现象，R2R 生成时做过可达性过滤）。
>
> 主表按 `merge_results.py --drop-nonfinite` 的口径整条剔除它们（3669 → 3570），
> 使各项指标分母一致。若改为把这 99 条按失败计入全部 3669 条，则
> SR 60.15 [58.60, 61.73]、OS 68.79、nDTW 69.58；NE 与 SPL 不变，因为它们本就无法计入。

> **两个 split 的 CANN 版本不同。** R2R 全量测于 CANN 9.0.0，RxR 全量测于 9.2.0
> （原服务器下线后迁移所致）。两者并非逐位等价的环境：同一批 1892 条 RxR episode
> 在两台机器上只有 2.5% 逐位一致（success/spl/os/ne/steps/ndtw 六项全同的口径；
> 只看 NE 与 SPL 两项则是 3.2%）、21.5% 成败翻转，但**聚合指标差异全部小于 1 个点**
> （SR 56.40 → 57.24，NE 5.43 → 5.51，SPL 46.11 → 46.56，nDTW 64.92 → 64.74），
> 且成败翻转是对称的（失败→成功 211 条，成功→失败 195 条）。
> 算子等价性（`bench/attn_equivalence.py`）与稳态单次 generate 耗时
> （融合 0.349s / sdpa 2.596s）在两个 CANN 版本上均复现一致。
> numpy 不在两机差异之列（两次全量都是 1.26.4）。已知的环境差异是 CANN 9.0.0 → 9.2.0，
> 以及新机上重新编译的 habitat-sim 与 Mesa 软件光栅；两台机器之间未记录的差异不能排除。

> **numpy 版本跟着系统 Python 走。** 推理 venv 与 habitat venv 都用 `--system-site-packages`
> 创建，系统 numpy 已满足 `requirements-npu.txt` 的 `numpy>=1.26,<2.0` 时 pip 不会再往 venv
> 里装一份；此后系统 numpy 一旦被换掉（换镜像、重建容器），评测环境会不声不响地跟着变。
> 跑评测前请用 `setup_habitat_env.sh` 末尾那条 `numpy 1.x` 断言确认环境没有漂。

> 复现所用的数据已与 HuggingFace 上的官方 gated 数据集逐字节校验：
> 11 个场景的 `.glb` / `.house` / `.navmesh` 共 33 个文件 md5 全部一致，
> RxR 的 `val_unseen_guide.json.gz` 与 `val_unseen_guide_gt.json.gz` md5 也一致。

> 评测结果对数值扰动高度敏感：同一批 110 个 episode 重复跑 4 次，SR 极差 4.5 个点
> （含未固定种子的那一次；仅在 3 次固定种子之间比较则为 2.7 个点）。
> 因此本样例默认固定种子（`--seed 0`），小样本结果只能用于冒烟验证，
> 判定精度必须跑全量。

### 性能

优化前单卡约 100–113 s/episode，其中 System 2 的自回归生成占约 80%。
将视觉塔的注意力换成 `npu_fusion_attention`（消除一个 7056x7056 的 mask）后，
单次 generate 由 2.60s 降至 0.35s，端到端单 episode 约 **2.4x**，
全量 R2R 两卡约 16 小时（实测）；优化前未跑全量，按上述倍率推算约 38 小时。

该优化会改变数值结果，但它复现的正是官方 `flash_attention_2` 路径的算法；
详细的等价性论证与端到端指标对比见[优化说明文档](../../../docs/navigation/internvla-n1/infer_with_torch/README.md#一个必须说清楚的问题换算子后指标降了-67-个点)。

<br>

## 环境准备

### 1. 安装 CANN 软件包

本样例依赖 CANN 开发套件包（cann-toolkit）与二进制算子包（cann-kernels），
已验证版本为 `CANN 9.0.0` 与 `CANN 9.2.0`（两个 split 分别测于这两个版本，见上表）。
请参考 [CANN 安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition)
完成安装，并确认系统 Python 中 `torch`、`torch_npu`、`torchvision` 可正常导入：

```bash
python3 -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
```

### 2. 拉取代码

```bash
# 本仓库
git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git

# 上游 InternNav 会被自动克隆到与本仓库同级的目录，无需手动操作
cd cann-recipes-embodied-ai/navigation/internvla-n1/infer_with_torch/src/scripts
```

所有路径都可以通过环境变量覆盖，默认值见 [`src/scripts/env.sh`](src/scripts/env.sh)。

### 3. 搭建模型推理环境并下载权重

```bash
bash setup_model_env.sh
```

这一步会克隆 InternNav（固定到验证过的 commit）、初始化 LongCLIP 子模块、
创建继承系统 torch_npu 的 venv、安装依赖，并下载
`InternVLA-N1-DualVLN`（约 16GB）与 `Depth-Anything-V2-Metric-Hypersim-Small`（约 95MB）。

### 4. 搭建 Habitat 仿真环境

```bash
bash setup_habitat_env.sh
```

这一步会源码编译 habitat-sim v0.3.3（headless、无 CUDA、无 bullet）、安装
habitat-lab v0.2.4，并生成桥接目录。**编译耗时较长（数十分钟到数小时，取决于核数）。**

为什么要单独一个环境、为什么版本这么选，见脚本内注释与
[优化说明文档](../../../docs/navigation/internvla-n1/infer_with_torch/README.md#环境冲突与版本约束)。

### 5. 下载数据集

数据来自 HuggingFace 上两个 gated 数据集，需要先用自己的账号申请访问
（两个都是 `gated: auto`，点一下 "Agree and access repository" 即刻放行）：

- [InternRobotics/Scene-N1](https://huggingface.co/datasets/InternRobotics/Scene-N1) —— MP3D 场景
- [InternRobotics/InternData-N1](https://huggingface.co/datasets/InternRobotics/InternData-N1) —— R2R / RxR episode

然后在 [settings/tokens](https://huggingface.co/settings/tokens) 创建一个 read token：

```bash
mkdir -p ~/.cache/huggingface && chmod 700 ~/.cache/huggingface
printf '%s' 'hf_xxxxxxxx' > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token

bash download_data.sh
```

数据集不包含在本仓库中，也不应被提交。

<br>

## 运行评测

```bash
# R2R val_unseen 全量（1839 episodes）
bash run_eval.sh r2r

# 先跑分层采样的小样本快速验证（110 episodes，两卡约 1 小时）
bash run_eval.sh r2r --sample 110

# RxR val_unseen 英文子集（3669 episodes）
bash run_eval.sh rxr
```

脚本会把 episode 均分到所有可见 NPU 上，每张卡一个独立进程，跑完自动合并出总指标
（含 bootstrap 置信区间与分场景明细）。

中途断了**不能直接重跑同一条命令**：`OUT_ROOT` 默认带时间戳，重跑只会开一个新目录
从头跑。续跑要显式指定上次那个 `OUT_ROOT`：

```bash
OUT_ROOT=$INTERNNAV_ROOT/logs/habitat/rxr_full bash run_eval.sh rxr
```

并且要与首次运行保持同样的分片数（`NUM_NPU`）、同样的采样参数（`--sample` / `--limit`，
采样发生在分片之前）、同样的种子（`--seed`，默认 0）、同样的融合视觉注意力开关（默认开），
否则前后两半要么落在不同的划分上、要么走的是不同的数值路径。已完成的 episode 由 `npu_eval.py` 读各分片自己的
`<OUT_ROOT>/shard<i>/progress.json` 跳过。

单独合并历史结果：

```bash
python3 ../tools/merge_results.py <输出目录>/shard* --per-scene
```

### 存评测视频

两个开关，默认都关：

| 开关 | 存什么 | 落在哪 |
|---|---|---|
| `--save-video` | **只有成功**的 episode：第一人称 RGB 与俯视图左右拼接，红点是 System 2 预测的像素路点（6 fps，一步一帧） | `<输出目录>/vis_0/<场景>/<episode>.mp4` |
| `--vis-debug` | **每一条** episode：纯第一人称，左上角叠 `step N action M`（5 fps） | `<输出目录>/vis_debug/epoch_0/<场景>_<episode>.mp4` |

```bash
bash run_eval.sh r2r --sample 14 --save-video --vis-debug
```

`run_eval.sh` 把多余的参数原样透传给 `npu_eval.py`，因此每张卡各写自己 `shard<i>/`
下的视频目录。路径里的 `0` 来自配置中的 `epoch`，本样例不改它。

**要看失败案例只能用 `--vis-debug`**：上游只在 `success == 1.0` 时才写 `vis_<epoch>`
那一路的视频，失败 episode 的帧直接丢弃。调试视频目录可用 `--vis-debug-path` 另行指定
（上游默认值是写死的 `./logs/habitat/vis_debug`，与 `--output-path` 无关，
本样例改成跟着输出目录走，多卡才不会混在一起）。

**全量评测不要开，磁盘吃不消。** 实测（两张卡各 14 条）：

| | episode | 成功轨迹视频 | 调试视频 |
|---|---:|---:|---:|
| R2R | 14（成功 9） | 97 MB | 48 MB |
| RxR | 14（成功 10） | 155 MB | 48 MB |

即成功轨迹视频约 11–16 MB/条（1120~1472×480，体积随步数线性增长），调试视频约 3.5 MB/条
（640×480）。按这个速率全量跑并两项都开，R2R 1839 条约 17 GB、RxR 3669 条约 48 GB。
时间上这批 14 条平均每步 0.73 s（R2R）/ 0.82 s（RxR），同机不开视频的全量平均是 0.63 s，
即 R2R +16%、RxR +30%；样本小且 episode 不同，只能当量级看。

视频编码走 `imageio` + `imageio-ffmpeg`，两侧的安装脚本都装了，不需要另装。
其中 `imageio` 与 `quaternion` 一样在 bridge 的覆盖清单里，运行期 import 到的是
habitat venv 那一份（见[适配说明](../../../docs/navigation/internvla-n1/infer_with_torch/README.md#gym-版本冲突两个-venv--bridge)）；
`imageio-ffmpeg` 不在清单里，走推理 venv 自己那份，自带 ffmpeg 二进制。

<br>

## 单元测试

纯逻辑部分有单测，只用标准库，**不需要 NPU、不需要 habitat、不需要权重和数据**，
在任何装了 Python 3 的机器上都能跑（验证于 Python 3.11.4）：

```bash
cd <本仓库>/navigation/internvla-n1/infer_with_torch
python3 -m unittest discover -s tests -t tests -v
```

| 测试文件 | 覆盖 |
|---|---|
| `test_episode_select.py` | 采样 / 分片 / 断点续跑。核心不变式：续跑不改变分片划分，既不重跑也不丢条 |
| `test_merge_results.py` | 分片合并与去重、`inf`/`nan` 指标的剔除与分母打印、bootstrap 置信区间的确定性、`--expect` |
| `test_convert_rxr_to_r2rce.py` | RxR → R2RVLN-v1 的 schema 转换：英文子集筛选、词表重建、轨迹字段原样保留 |
| `test_py311_dataclass_shim.py` | Python 3.11 的 dataclass 默认值垫片 |

依赖 NPU 的验证不在单测里，走 [`bench/`](bench/README.md) 下的三个脚本
（算子等价性、注意力微基准、真实模型耗时构成）。

<br>

## 目录结构

```
infer_with_torch/
├── README.md                        本文档
├── requirements-npu.txt             模型推理侧依赖（habitat 侧依赖在 setup_habitat_env.sh 里）
├── bench/                          验证与基准脚本（优化说明文档中结论的复算入口）
│   ├── README.md
│   ├── attn_equivalence.py           算子等价性 / 与 fp32 的误差 / rope 等价性
│   ├── attn_speed.py                 注意力微基准（语言模型侧 vs 视觉塔侧）
│   └── profile_generate.py           真实模型上的耗时构成与算子级 profile
├── tests/                          单元测试（纯标准库，不需要 NPU）
│   ├── test_episode_select.py        采样 / 分片 / 断点续跑
│   ├── test_merge_results.py         指标合并、inf/nan 处理、置信区间
│   ├── test_convert_rxr_to_r2rce.py  RxR schema 转换
│   └── test_py311_dataclass_shim.py  dataclass 垫片
└── src/
    ├── npu_eval.py                  评测启动器，运行期打上全部昇腾适配
    ├── episode_select.py            采样 / 分片 / 续跑过滤（纯逻辑，有单测）
    ├── compat/                      运行期垫片（不改上游源码）
    │   ├── eglinit.py                 EGL / glvnd 初始化
    │   ├── hvenv_tail.py              habitat venv 的 sys.path 兜底
    │   ├── py311_dataclass_shim.py    habitat-lab 0.2.4 在 Python 3.11 上的 dataclass 兼容
    │   └── npu_vision_attn.py         视觉塔注意力换 npu_fusion_attention（性能优化）
    ├── scripts/
    │   ├── env.sh                     公共路径与镜像配置
    │   ├── setup_model_env.sh         第 1 步：InternNav + 推理 venv + 权重
    │   ├── setup_habitat_env.sh       第 2 步：habitat-sim 编译 + habitat-lab
    │   ├── download_data.sh           第 3 步：场景与 episode 数据
    │   └── run_eval.sh                跑评测
    └── tools/
        ├── convert_rxr_to_r2rce.py  RxR 原始格式 -> R2RVLN-v1 schema
        └── merge_results.py         合并分片结果、算置信区间
```

推理侧昇腾适配与优化文档位于 [docs/navigation/internvla-n1/infer_with_torch/README.md](../../../docs/navigation/internvla-n1/infer_with_torch/README.md)。

想复算文档里的性能与等价性结论，见 [`bench/README.md`](bench/README.md)。

<br>

## 说明

- 本样例**不修改 InternNav 上游源码**，所有昇腾适配都以运行期 monkeypatch 的形式
  集中在 `src/npu_eval.py` 与 `src/compat/` 中。这样上游仓库的工作区始终干净，
  跟进上游更新时无需反复解冲突。具体适配了哪些点、为什么这么改，见
  [优化说明文档](../../../docs/navigation/internvla-n1/infer_with_torch/README.md)。
- 模型权重、数据集均从各自的官方渠道下载，本仓库不包含任何权重或数据文件。
