# InternVLA-N1 昇腾适配与性能优化说明

本文记录把 InternVLA-N1 的 VLN-CE 评测链路搬到昇腾 Atlas A2（aarch64、无 NVIDIA GPU）
过程中遇到的问题、采用的方案和量化收益。

本文对应的样例是 [`navigation/internvla-n1/infer_with_torch/`](../../../../navigation/internvla-n1/infer_with_torch/README.md)，
文中 `src/`、`bench/`、`tests/` 等相对路径均相对该样例目录。

<br>

## 总体思路：不改上游源码

InternNav 上游有多处按 NVIDIA GPU 写死的实现。常规做法是打 patch，但 VLN-CE 评测链路
牵涉 InternNav、habitat-lab、habitat-sim 三个仓库，patch 会随上游更新迅速腐烂。

本样例改为**全部在运行期用 monkeypatch 替换**，集中在 `src/npu_eval.py` 与
`src/compat/`。代价是启动器多了几十行，收益是上游仓库工作区始终干净
（`git status` 为空），跟进上游时不需要解冲突，适配点也一目了然。

<br>

## 昇腾适配点

### 0. 补上随机种子：让结果可复现

这一条不是昇腾特有的问题，但不解决就没法验证后面所有的优化。

**现象**：同一份代码、同一批 episode 连跑两次，结果不同。实测同一个 episode
两次的 `steps` 是 126 vs 105、`SPL` 是 0.790 vs 0.694，甚至 `oracle_success`
会从 0 变成 1。

**根因**：InternVLA-N1 的 System 1 是**扩散模型**，`generate_traj` 里通过
diffusers 的 `randn_tensor` 从**全局 torch RNG** 取初始噪声。而 habitat 这条评测路径
（`internnav/evaluator/distributed_base.py:38`）只做了：

```python
np.random.seed(self.local_rank)
```

torch 的 RNG 从未被 seed。（另一条 InternUtopia 评测路径
`vln_distributed_evaluator.py:54` 是调了 `set_seed_model(0)` 的，habitat 这条漏了。）
每步的轨迹噪声都不同，轨迹随之发散，长 episode 上误差被放大。

**方案**：启动器加 `--seed`（默认 0），在建模型前 seed
`random` / `numpy` / `torch`，并在 `transfer_to_npu` 之后再调一次
`torch.cuda.manual_seed_all`（此时 `torch.cuda` 已指向 NPU，这一句才真正落到 NPU 的 RNG 上）。

> 注意：其中 numpy 那一路随后会被上游覆盖 —— `Evaluator.init()` 里
> `internnav/evaluator/distributed_base.py` 会执行 `np.random.seed(self.local_rank)`。
> 真正让结果可复现的是 torch 的两处 seed，因为 System 1 的扩散噪声取自全局 torch RNG。
> 保留 numpy 那行只是为了覆盖 evaluator 构造之前的调用。

**效果**：加种子后两次跑的 `steps` / `NE` / `SPL` **逐位一致**。

> 这条对本样例的意义：优化前后的对比必须建立在可复现的基线上，否则分不清指标变化
> 是优化引入的还是随机噪声。**下文表格中 110 个 episode 的 7 次运行里，基线组有 1 次
> 是在加种子之前跑的**，那一次除采样误差外还含有这部分随机性；
> 其余 6 次都固定了种子，后续优化也以加种子后的结果作为基线。

### 1. 设备与通信后端：cuda → npu

`internnav/habitat_extensions/vln/habitat_vln_evaluator.py` 里写死了：

```python
device = torch.device(f"cuda:{self.local_rank}")
```

`internnav/utils/dist.py` 里还有 `torch.cuda.set_device()` 与 `backend="nccl"`。

**方案**：在任何 `torch.device("cuda")` 构造之前 import `torch_npu.contrib.transfer_to_npu`，
由它统一把 `torch.cuda` 转到 `torch.npu`、`nccl` 转到 `hccl`。这是昇腾官方提供的
迁移工具，比逐处替换更稳。**导入时机是关键** —— 必须早于第一次设备构造。

### 2. 注意力实现：flash_attention_2 → sdpa

模型加载处写死 `attn_implementation="flash_attention_2"`，昇腾上没有这个实现。

**方案**：包一层 `InternVLAN1ForCausalLM.from_pretrained`，强制改为 `sdpa`。

**注意**：这个替换只作用于**语言模型**。视觉塔虽然也声明用 sdpa，但上游实现会物化一个
巨型 mask 导致 SDPA 退化，需要单独处理，见[适配点 6](#6-视觉塔注意力换融合算子)。

### 3. Habitat 渲染设备：gpu_device_id 0 → -1

`scripts/eval/configs/vln_r2r.yaml` 中 `habitat_sim_v0.gpu_device_id: 0`。
在没有 CUDA 设备的机器上，habitat-sim 会报：

```
Platform::WindowlessEglApplication::tryCreateContext(): unable to find CUDA device 0 among 2 EGL devices in total
WindowlessContext: Unable to create windowless context
```

因为 habitat-sim 用 `gpu_device_id >= 0` 表示"按 CUDA 设备号选 EGL 设备"。

**方案**：包一层 `get_habitat_config`，把该字段置为 `-1`，走默认 EGL 设备。
实际落到 Mesa llvmpipe 软件光栅（`/dev/dri/card0` 无访问权限，拿不到硬件渲染）。

**这不是瓶颈**：实测仿真 + 渲染只占端到端耗时的 12.9%。

### 4. EGL / glvnd 延迟初始化

llvmpipe 路径上，直接 `import habitat_sim` 再建 `Simulator` 会拿不到 EGL 设备列表。

**方案**：`src/compat/eglinit.py` 在此之前用 ctypes 打开 `libEGL.so.1` 并调用一次
`eglQueryString(EGL_EXTENSIONS)`，强制 glvnd 完成 vendor 枚举。

### 5. 分布式评测：绕开 all_gather

上游 evaluator 用 `torch.distributed` 的 all_gather 汇总各 rank 的指标。

**方案**：改为手工分片 —— 每张卡起一个独立进程、`world_size=1`、各写各的
`progress.json`，跑完用 `tools/merge_results.py` 离线合并。VLN-CE 的各项指标
（R2R 为 NE/OS/SR/SPL 四项，RxR 另加 nDTW）都是逐 episode 的算术平均，
在取值均为有限数时，分片再合并与整体计算等价。

> 边界情况：上游 `habitat_vln_evaluator.py` 对 SPL 用 `nan_to_num(nan=0.0)`（NaN 记 0 且保留在分母），
> 对 NE 则先滤掉非有限值再求均值；`merge_results.py` 统一按过滤处理。
> 全量 R2R 未出现非有限值，两种口径结果一致。

这样做的额外好处是**任一进程崩掉不影响其他卡**，且天然支持断点续跑。

> 续跑不是上游那套。上游 `generate_episodes()` 会先读 `output_path/progress.json`
> 把已完成的 episode 剔掉再返回，而本样例必须先切片、后过滤（原因见
> `src/episode_select.py` 的模块注释：在过滤后的列表上切片会让两次运行
> 的分片划分对不上）。做法是把 `output_path` 临时指到一个不存在的子目录，
> 让上游那一步空转，采样与分片之后再用 `episode_select.read_done` 按本分片的
> `progress.json` 过滤。`run_eval.sh` 永远带 `--shard`，所以走的一直是这条路。
> 这段逻辑单独成文件就是为了能脱离 NPU 与 habitat 跑单测，见
> `tests/test_episode_select.py`。
> 另外 `OUT_ROOT` 默认带时间戳，续跑必须显式指回上次那个目录，见样例 README。

### 6. 视觉塔注意力换融合算子

这是本样例唯一的性能优化，也是唯一一处让计算结果发生变化的适配，因此单列。

**问题**：`Qwen2_5_VLVisionSdpaAttention.forward` 为了表达"每张图只在自己的图块内做
注意力"，显式构造块对角 bool mask 再交给 SDPA：

```python
attention_mask = torch.zeros([1, seq_length, seq_length], dtype=torch.bool)
for i in range(1, len(cu_seqlens)):
    attention_mask[..., cu_seqlens[i-1]:cu_seqlens[i], cu_seqlens[i-1]:cu_seqlens[i]] = True
F.scaled_dot_product_attention(q, k, v, attention_mask)
```

评测时每次前向带 8 帧历史 + 当前帧，`seq_length` 达 7056，于是每层要分配并填一个
7056x7056 的 mask（约 50MB），**视觉塔 32 层各建一次**。更关键的是 SDPA 一旦收到显式
mask 就无法走融合核，退化成 matmul -> masked_fill(-inf) -> softmax -> matmul。

**方案**：这个块对角 mask 就是 packed sequence 语义，`npu_fusion_attention` 的 TND 布局
配 `actual_seq_qlen` 原生支持，完全不需要物化 mask。实现见 `src/compat/npu_vision_attn.py`。

**收益**：单次 System 2 generate **2.60s -> 0.35s（7.4x）**，端到端单 episode **2.4x**
（generate 占 80%，理论上限 3.2x，episode 前几步历史短所以实际略低）。
全量 R2R 两卡约 16 小时（实测）；优化前未跑全量，按上述倍率推算约 38 小时。

**这个替换会改变数值结果**，因此做了三层验证：

1. *算子等价性*：在四种分段形态下与上游实现对比（整图 9x784、视觉塔真实的窗口分段
   144 段、单段 1x7056、不等长 [64,128,256,512]），相对差均约 3e-3，
   余弦相似度 0.999997。其中窗口分段是每张图 16 个窗口、尺寸
   9x64 + 6x32 + 1x16 = 784（14x14 个 merge 单元按 4+4+4+2 切窗，每个单元 4 个 token），
   与 transformers 的 `get_window_index` 实测一致，9 张图共 144 段。
2. *谁更准*：以 fp32 精确解为真值，上游 SDPA 数学路径误差 1.66e-3，融合算子 2.10e-3。
   融合略不准，但同一量级，都是正常 bf16 行为。
3. *端到端指标*：见下面一节。

<br>

## 一个必须说清楚的问题：换算子后指标降了 6.7 个点

在 R2R val_unseen 分层采样的同一批 110 个 episode 上跑了 7 次（基线 4 次、融合 3 次；
基线的 4 次里有 1 次是加种子之前跑的，其余都固定了种子）：

| 配置 | 运行次数 | SR 均值 | SR 范围 |
|---|---|---|---|
| sdpa 基线 | 4 | 73.6 | [70.9, 75.5]（3 次固定种子为 [72.7, 75.5]） |
| 融合视觉注意力 | 3 | 67.0 | [66.4, 67.3] |

两组**完全不重叠**，置换检验 p = 1/35 = 0.029。这不是随机波动。

按常规判断，这就该判定优化不可用。但在下结论前需要回答一个问题：
**论文的数字是用哪条路径跑出来的？**

InternNav 的 evaluator 硬编码 `attn_implementation="flash_attention_2"`，
对应视觉塔的 `Qwen2_5_VLVisionFlashAttention2`，其实现是：

```python
attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
```

**用 cu_seqlens 的 varlen flash attention，不造 mask** —— 与本样例用
`npu_fusion_attention(TND, actual_seq_qlen)` 是同一个算法。两条路径的 rope 函数
（`apply_rotary_pos_emb_flashatt` 与 `apply_rotary_pos_emb_vision`）经实测逐位相同。

也就是说：

| 路径 | rope | 注意力 |
|---|---|---|
| 官方 flash（论文） | flashatt | varlen flash，无 mask |
| **本样例融合版** | vision（与前者逐位相同） | varlen 融合，无 mask |
| sdpa 基线 | vision | 物化 T^2 mask + 数学路径 |

**融合版才是忠实复现论文的那条路，sdpa 基线是偏离的那条。**
那 6.7 个点是 sdpa 数学路径引入的偏差，不是本优化造成的退化。
这也解释了为什么融合版（SR 67.0）比 sdpa 基线（73.6）更接近论文的 64.3。

因此本样例**默认开启**该优化。需要 A/B 对比时用 `--no-fused-vision-attn` 关闭。

### 全量结果回溯验证了这个判断

上述论证基于**代码结构等价性**（rope 逐位相同 + varlen 融合注意力同算法），
昇腾上装不了 flash_attn，无法直接与 GPU 参考实现对跑。真正的经验验证是全量评测：

| 配置 | 110 子集 SR | 全量 1839 SR | 论文 |
|---|---|---|---|
| sdpa 基线 | 73.6 | （未跑，推算约 70） | 64.3 |
| **融合视觉注意力** | 67.0 | **63.57** | 64.3 |

融合版全量 SR 63.57，与论文 64.3 相差 0.73 个点，落在 95% 置信区间
[61.34, 65.74] 内。若 sdpa 基线才是正确路径，全量应高出约 6.7 个点即 SR 约 70，
将显著高于论文。**结论闭环：sdpa 数学路径是偏离的那条。**

> 教训：在闭环仿真评测里，"指标变差"不等于"改错了"。这类任务对数值扰动高度敏感
> （同一批 episode 重复跑，SR 极差能到 4.5 个点，只在固定种子之间比也有 2.7 个点），
> 判断某个改动是否可接受，必须回到"上游参考实现到底算的是什么"，而不是只看指标涨跌；
> 而小样本上的显著性检验（这里 p=0.029）也不足以定论，最终得靠全量。

<br>

## 环境冲突与版本约束

搭建过程中踩到若干版本坑，逐条记录如下，都已固化进 setup 脚本。

### habitat-lab 必须 v0.2.4，不能用 v0.3.x

InternNav 官方文档要求 habitat-sim 0.2.4 + habitat-lab v0.2.4。实测 v0.3.x 会连续踩两个坑：

- `habitat.core.utils.try_cv2_import` 在 **v0.3.2** 被删除，
  `internnav/habitat_extensions/vln/measures.py` 直接 ImportError；
- 更致命的是配置组 `habitat/simulator/agents/rgbd_agent` 在 **v0.3.0** 起被整体删除
  （整个 `agents/` 目录从 12 个 sensor 组合配置改成了 6 个机器人配置），
  `vln_r2r.yaml` 加载时报 `MissingConfigException`。这不是补一两个文件能绕过去的。

### habitat-sim 只能源码编译 v0.3.3

aihabitat 的 conda 频道没有 linux-aarch64 包，只能源码编译。编译开关：

```
BUILD_WITH_CUDA=OFF   BUILD_WITH_BULLET=OFF   BUILD_WITH_AUDIO=OFF
BUILD_GUI_VIEWERS=OFF MAGNUM_TARGET_EGL=ON    CMAKE_BUILD_TYPE=RelWithDebInfo
```

`setup.py build_ext` 只产出 `build/lib.*/`，不装进 site-packages，需要额外用一个
`.pth` 挂上去。

**这是本样例与论文环境唯一的版本差异**：C++ 绑定是 0.3.3，Python 层是 0.2.4。
实测两者可以正常配合 —— VLN-CE 用的是数据集自带的 `.navmesh`，不重新生成导航网格，
因此几何与可行走区域完全由数据决定，与 sim 版本无关。

### habitat-lab 0.2.4 在 Python 3.11 上无法定义 dataclass

habitat-lab 0.2.4 面向 Python 3.9。CPython 3.11 收紧了 dataclasses 的默认值检查
（bpo-44674）：字段默认值的类型只要 `__hash__` 为 `None`（普通 dataclass 即如此）
就报错：

```
ValueError: mutable default <class '...IteratorOptionsConfig'> for field
iterator_options is not allowed: use default_factory
```

**方案**：`src/compat/py311_dataclass_shim.py` 包一层 `dataclasses._get_field`，
碰到这种默认值自动改用 `default_factory`（deepcopy 一份）。语义等价于 3.9 上的旧行为，
而且每个实例拿到独立副本反而更安全。

### gym 版本冲突：两个 venv + bridge

- habitat-lab 0.2.4 要求 `gym >= 0.22.0, < 0.23.1`；
- InternNav 的 `core_requirements.txt` 装的是 `gym <= 0.26.2`。

gym 0.26 把 `registry` 从对象换成了 dict，habitat-lab 会在
`habitat/gym/gym_definitions.py` 报 `AttributeError: 'dict' object has no attribute 'env_specs'`。
两者无法共存于同一环境。

**方案**：habitat 栈装进独立 venv，再分两级挂到推理 venv 的 `sys.path`：

| 层级 | 位置 | 放什么 |
|---|---|---|
| bridge（符号链接农场） | `PYTHONPATH` **最前** | 需要**覆盖**推理 venv 的包（gym 0.23.0）+ 推理 venv 里没有的包（habitat / habitat_sim / magnum …） |
| habitat venv 的 site-packages | `sys.path` **末尾** | 纯兜底填补缺包（tensorboard / ifcfg / lmdb …） |

这样 `numpy` / `opencv` / `torch` 仍然走推理 venv 这一侧、不会被 habitat venv 里那套顶掉
（`numpy` 可能直接来自系统 Python，见样例 README 的说明），
**InternNav 的 `.venv` 一个字节都不用动**。实现见 `src/compat/hvenv_tail.py`。

> 被 bridge 覆盖的不止 gym：`quaternion` 也在链接清单里，因此运行期用的是 habitat venv
> 的 `numpy-quaternion 2024.0.13`，而不是 `requirements-npu.txt` 钉的 2023.0.3。
> habitat 的位姿计算本来就依赖它，两次全量评测用的也都是 2024.0.13，所以不存在版本
> 不一致的问题；但推理 venv 里那个 pin 对评测路径其实不起作用，记在这里免得误读。
> `imageio`（存评测视频那一路的编码入口）同理，运行期走的是 habitat venv 那份；
> `imageio-ffmpeg` 不在清单里，仍走推理 venv 自己的版本。

### 其他固定版本

| 包 | 约束 | 原因 |
|---|---|---|
| `diffusers` | `==0.32.2` | 仓库内 vendored 的 `LuminaNextDiT2DModel._set_gradient_checkpointing` 仍是 0.33 之前的签名，用上游 `internvla_n1.txt` 写的 0.33.1 会直接 TypeError |
| `tensorboard` | `==2.18.0` | 更高版本的 protobuf gencode（6.31）比系统 protobuf 运行时（5.29）新，`import habitat_baselines` 抛 `VersionError` |
| `numpy` | `<2.0` | gym 0.23 / numba / habitat 全链路都还没适配 numpy 2 |

<br>

## 数据准备中的三个坑

### RxR 数据 schema 与配置不匹配

`scripts/eval/configs/vln_rxr.yaml` 里 `dataset.type` 写的是 `R2RVLN-v1`，
而官方 `RxR_VLNCE_v0` 包里是 RxR 原始格式：没有顶层 `instruction_vocab`，
`instruction` 里还带着 `annotator_id` / `language` / `timed_instruction` 等
`R2RVLN-v1` 不认识的字段。直接加载报 `KeyError: 'instruction_vocab'`。

官方包里也**没有** `_guide_en` 这个文件 —— 英文子集需要自己按 `language` 前缀筛
（`en-IN` 2446 + `en-US` 1223 = 3669 条）。

**方案**：`tools/convert_rxr_to_r2rce.py` 一次做完筛选与 schema 转换。
转换只重建 `instruction_vocab` 与 `instruction_tokens`，`instruction_text`
和全部轨迹字段原样保留 —— InternVLA-N1 只读 `instruction_text`，因此对指标零影响。

### RxR 有 99 条 episode 的目标点不可达

RxR 英文子集 3669 条跑完后，`merge_results.py` 打出来的 `NE` 是 `inf`。
定位下来是 99 条 episode 的 `ne=inf`、`spl=nan`，**全部集中在 `oLBMNvg9in8` 一个场景**。

habitat 的 `DistanceToGoal` 用的是 `geodesic_distance(agent_pos, goal_pos)`，
当目标与当前位置在 navmesh 上不连通时返回 `inf`。用 `habitat_sim.PathFinder`
对全部 11 个场景逐条计算起点→目标的测地距离，结果与跑出来的完全吻合：

```
scene               episodes     不可达      占比
2azQ1b91cZZ              615       0    0.0%
8194nk5LbLH               51       0    0.0%
EU6Fwq7SyZv              345       0    0.0%
QUCTc6BB5sX              528       0    0.0%
TbHJrupSAjP              543       0    0.0%
X7HyMhZNoso              468       0    0.0%
Z6MFQCViBuw              288       0    0.0%
oLBMNvg9in8              345      99   28.7%   <-
pLe4wQe7qrG                6       0    0.0%
x8F5xyUWy9e               96       0    0.0%
zsNo4HB9uLZ              384       0    0.0%
合计                    3669      99    2.7%
```

即这 99 条从第 0 步起 `NE` 就是 `inf`，**任何模型都不可能完成**，
与昇腾适配无关。同一场景的 R2R episode 没有这个现象 —— R2R-VLN-CE
在生成 episode 时做过可达性过滤，而我们的英文子集是自己按 `language`
前缀从官方 `RxR_VLNCE_v0` 里筛出来的（官方包没有现成的 `_guide_en`），
没有再做可达性过滤。

**方案**：`tools/merge_results.py` 统一按 `math.isfinite` 过滤，并打印每个指标的
实际分母；`--drop-nonfinite` 把这些 episode 整条剔除，保证各指标分母一致。
样例 README 的精度表用剔除口径（3669 → 3570），同时列出按失败计入全部 3669 条的数字。

> 原实现只用 `r[key] == r[key]` 过滤，这只挡得住 NaN：`inf` 过不掉，
> `NE` 的均值直接变成 `inf`；而 `NaN` 被静默跳过，导致 `SPL` 的分母比 `SR`
> 少 99 条、两个数不可比且毫无提示。这类"分母不一致但打印得很正常"的问题
> 比直接报错危险得多，所以现在会显式打印分母。

### 采样要按场景分层

`HabitatEnv.generate_episodes` 是**按场景排序**输出的。用 `--limit N` 取前 N 个
会全部落在字典序第一个场景里。R2R val_unseen 各场景难度差异很大
（实测同一批采样里 `X7HyMhZNoso` SR 87.5，`oLBMNvg9in8` 只有 50.0），
单场景的数字完全代表不了整体。

**方案**：`--sample N` 改为在全量上**等间隔取样**，因为列表已按场景排序，
等间隔天然让各场景的采样占比与全量占比一致（实测 11 个场景占比偏差均在 1 个百分点内）。

<br>

## 性能画像

R2R val_unseen，单卡 910B，`sdpa`，端到端约 100–113 s/episode。
按日志时间戳统计各环节耗时（一张卡 55 个 episode 的累计）：

| 环节 | 调用次数 | 均值 | 总耗时 | 占比 |
|---|---:|---:|---:|---:|
| **System 2 generate**（动作 / STOP） | 963 | 3.155 s | 3037.8 s | **54.8%** |
| **System 2 像素路点预测** | 350 | 3.964 s | 1387.6 s | **25.0%** |
| 仿真 step + 渲染 | 3612 | 0.198 s | 714.0 s | 12.9% |
| System 1 NextDiT 规划 | 625 | 0.643 s | 401.8 s | 7.3% |

**结论：约 80% 的时间花在 System 2 的自回归生成上。**

值得注意的是，这些 generate 的输出往往只有 `↓` 或 `318 177` 这么几个 token，
却要花 3～4 秒 —— 说明耗时**由 prefill 主导而非 decode**：每次都要重新处理
8 帧 384×384 的观测历史。prefill 是计算密集的，正是融合注意力算子的用武之地。

反过来说，最初担心的 llvmpipe 软件渲染只占 12.9%，**不是瓶颈**，不值得投入优化。

### 优化前后对比

上表是优化前（sdpa + 巨型 mask）的画像。按此画像最初判断"prefill 主导 -> 换融合注意力
算子"，**这个判断是错的**：算子级 profile 显示 `aclnnFlashAttentionScore` 只有 140 次
共 17ms，占 0.9% —— 140 = 语言模型 28 层 x 5 次前向，说明**语言模型的 sdpa 本来就在走
融合核**，换 `npu_fusion_attention` 收益为零。

真正的瓶颈由这三个算子暴露 —— 其中 `aclnnSoftmax` 与 `aclnnIsNegInf` 的调用次数
**恰好都是 32**，正好等于视觉塔层数（`aclnnSWhere` 为 37 次，另有 5 次来自别处）。

下表是**基线与交付配置的算子级对照**，同机同栈实测（Ascend 910 / CANN 9.2.0 /
torch_npu 2.7.1.post4 / transformers 4.51.0，稳态输入 1909 token、7056 图块）。
复算命令见 [bench/README.md](../../../../navigation/internvla-n1/infer_with_torch/bench/README.md)：

| kernel | 基线次数 | 基线耗时 | 基线占比 | 交付次数 | 交付耗时 | 交付占比 |
|---|---:|---:|---:|---:|---:|---:|
| `aclnnMatmul` | 639 | 830 ms | 42.9% | 575 | 175 ms | 47.6% |
| `aclnnSWhere` | 37 | 207 ms | 10.7% | 5 | 0.01 ms | 0.00% |
| `aclnnInplaceCopy` | 17902 | 206 ms | 10.7% | 1310 | 16 ms | 4.3% |
| `aclnnSoftmax` | 32 | 196 ms | 10.1% | 缺席 | | |
| `aclnnAdd` | 752 | 179 ms | 9.2% | 720 | 11 ms | 2.9% |
| `aclnnIsNegInf` | 32 | 112 ms | 5.8% | 缺席 | | |
| `aclnnAddmm` | 1164 | 59 ms | 3.1% | 1164 | 60 ms | 16.2% |
| `aclnnAll` | 37 | 29 ms | 1.5% | 5 | 0.02 ms | 0.00% |
| `aclnnFlashAttentionVarLenScore` | 缺席 | | | 64 | 17 ms | 4.70% |
| `aclnnFlashAttentionScore` | 140 | 17 ms | 0.90% | 140 | 17 ms | 4.65% |
| `aclnnInplaceMaskedFillScalar` | 37 | 10 ms | 0.5% | 5 | 0.2 ms | 0.07% |
| `aclnnLogicalNot` | 32 | 4 ms | 0.2% | 缺席 | | |
| **NPU kernel 总耗时** | | **1936.6 ms** | | | **368.2 ms** | |
| **单次 generate 中位耗时** | | **2.617 s** | | | **0.350 s** | |

基线那几个算子是 SDPA 收到显式 mask 后退化成数学路径的特征（造 -inf、masked_fill、softmax）。
顺着查到 `Qwen2_5_VLVisionSdpaAttention` 里那个 7056x7056 的 mask，
处理方式见[适配点 6](#6-视觉塔注意力换融合算子)。

**交付配置下视觉塔贡献的那部分已经不在了**：`Softmax` / `IsNegInf` / `LogicalNot`
整个缺席；`SWhere` / `All` / `InplaceMaskedFillScalar` 从 37 次降到 **5 次** ——
留下的正是基线里那 5 次「来自别处」的调用，反过来印证了 37 = 32 + 5 这个拆分。
六项合计从基线的 **28.8%** 降到 **0.07%**。取而代之的是 `npu_fusion_attention` 的
TND varlen 核 `aclnnFlashAttentionVarLenScore`（64 次、4.70%），它**不**落在
`FlashAttentionScore` 那一行。`aclnnInplaceCopy` 从 17902 次降到 1310 次 ——
那个「次数异常多」的现象是 mask 路径逐段切片赋值的伴生物。

**注意力在交付配置下已不是瓶颈**：两个融合注意力核合计 **9.35%**，而 GEMM
（`Matmul` + `Addmm`）占 **63.8%**。语言模型侧的 `aclnnFlashAttentionScore` 两边都是
140 次、绝对耗时 17.4 → 17.1 ms 基本不变，占比上升只是分母缩小所致 ——
这一侧一直在走融合核，没有可做的。

优化后单次 generate 由 2.60s 降至 0.35s；端到端在上面这批 55 个 episode 上
由均值约 106 s/episode 降到约 45 s/episode，约 2.4x。generate 占 80%，
按 Amdahl 上限是 3.2x，实测只有 2.4x 是因为 episode 开头几步历史帧还没攒满。

> 教训：调用级耗时（"哪个调用花了 3 秒"）只能定位到模块，不能替代算子级 profile。
> 据此推断瓶颈在注意力，实际它占不到 1%，多花了一轮微基准和两次无效尝试
> （`allow_internal_format=True` 实测无影响）。
