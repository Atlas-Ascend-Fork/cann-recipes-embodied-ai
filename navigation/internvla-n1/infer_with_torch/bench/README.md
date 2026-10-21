# 验证与基准脚本

[优化说明文档](../../../../docs/navigation/internvla-n1/infer_with_torch/README.md)里的每一条性能与等价性结论，都由这三个脚本产出。
把它们一并交付，是为了让结论可被独立复算，而不是只能读数字。

运行前需要先完成 `src/scripts/` 的三步环境搭建，并用 InternNav 的推理 venv 执行。

下面所有命令都在样例根目录 `infer_with_torch/` 下执行：

```bash
cd <本仓库>/navigation/internvla-n1/infer_with_torch
source src/scripts/env.sh             # 得到 INTERNNAV_ROOT / MODEL_STORE 等
V="$INTERNNAV_ROOT/.venv/bin/python"
```

---

## `attn_equivalence.py` — 替换是否改变了计算

回答三个问题，对应优化说明文档的「适配点 6」与「一个必须说清楚的问题」。

```bash
ASCEND_RT_VISIBLE_DEVICES=0 $V bench/attn_equivalence.py
```

参考输出：

```
分段形态                                  相对差         余弦   SDPA→fp32     融合→fp32  更准
整图 9x784（full-att 层）            3.521e-03   0.999997   1.659e-03   2.250e-03  SDPA
窗口 144 段（window 层）              3.185e-03   0.999998   1.659e-03   2.084e-03  SDPA
单段 1x7056                       3.676e-03   0.999997   1.659e-03   2.290e-03  SDPA
不等长 [64,128,256,512]            3.106e-03   0.999997   1.658e-03   2.171e-03  SDPA

  q: max|diff| = 0.000e+00   逐位相同
  k: max|diff| = 0.000e+00   逐位相同
```

三点结论：

1. 融合算子与上游实现在**全部分段形态**下等价（余弦 0.999997，相对差约 3e-3）——
   包括视觉塔实际用到的 144 段窗口分段，不只是等长的整图分段。
2. 以 fp32 精确解为真值，融合算子（约 2.1e-3）比 SDPA 数学路径（1.66e-3）**略不准**，
   但同为正常 bf16 量级。这个差异是优化说明文档里那场「指标降了 6.7 点」讨论的起点。
3. sdpa 路径与官方 flash 路径的 rope 函数**逐位相同**。这一条加上「两者都是 varlen 融合注意力」，
   构成了「融合版才是忠实复现论文的路径」这一论证的代码级依据。

---

## `attn_speed.py` — 提速来自哪里

```bash
ASCEND_RT_VISIBLE_DEVICES=0 $V bench/attn_speed.py
```

脚本刻意同时测**语言模型侧**和**视觉塔侧**，因为两者结论相反：

- **语言模型侧**（GQA + causal，无显式 mask）：加速仅 1.1～2.3 倍且随序列变长衰减。
  sdpa 在这里本来就走融合核，省下的主要是 `repeat_kv` 的访存。
  **换融合算子在这一侧几乎没有意义** —— 这是最初走过的弯路。
- **视觉塔侧**（块对角分段，上游物化 mask）：T=7056 时约 **60 倍**。
  差别不在算子本身，而在上游那张 7056×7056 的 mask 逼得 SDPA 退化成数学路径。

> 绝对耗时受同卡其他负载影响明显（实测同一组配置在空闲卡上 50.03 ms、
> 在与评测任务共卡时 71.82 ms），**看加速比而不是绝对值**。
> 优化说明文档记录的是空闲卡上的数字。

---

## `profile_generate.py` — 真实模型上的耗时构成

需要加载完整模型（bf16 约 17 GB 权重），确认卡上有足够 HBM 再跑。

```bash
# A/B：关 / 开视觉塔融合注意力。注意本脚本的 --fused-vision-attn 默认**关闭**（便于 A/B），
# 与评测入口 src/npu_eval.py 相反 —— 那边是 default=True，交付配置默认开启
ASCEND_RT_VISIBLE_DEVICES=0 $V bench/profile_generate.py                      # 基线
ASCEND_RT_VISIBLE_DEVICES=0 $V bench/profile_generate.py --fused-vision-attn  # 交付配置

# 耗时随历史帧数的增长
ASCEND_RT_VISIBLE_DEVICES=0 $V bench/profile_generate.py --scaling

# 算子级 profile（两种配置各跑一份才能对照）
ASCEND_RT_VISIBLE_DEVICES=0 $V bench/profile_generate.py --profile-out /tmp/prof-base
ASCEND_RT_VISIBLE_DEVICES=0 $V bench/profile_generate.py --fused-vision-attn --profile-out /tmp/prof-fused
$V bench/profile_generate.py --parse /tmp/prof-fused
```

> `--profile-out` 依赖 CANN profiler 的解析器，而它需要 `libsqlite3.so`。某些昇腾镜像里这个库
> 存在但不在默认搜索路径上（实测在 `/usr/local/sqlite3/lib`），此时 profiler 会把
> `KernelViewParser run failed` 之类的报错打在 stdout 里、只留下一个被截断的 `trace_view.json`，
> **`--parse` 随后找不到 `kernel_details.csv`**。先确认：
>
> ```bash
> export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:$LD_LIBRARY_PATH
> ```

**关键：脚本会先喂满 8 帧历史再测量。** 刚 `reset()` 时只有 1 帧、约 780 个图块，
单次 generate 只要 0.35 s；而评测稳态是 9 张图、7056 个图块，需要 2.6 s
（**这两个数都是基线口径**，别和下表里融合配置的稳态 0.350 s 混起来 —— 数值撞巧了）。
用前者的数据做性能分析会得出完全错误的结论 —— 这个坑实际踩过一次。

`--parse` 输出的 kernel 表是优化说明文档「性能画像」一节的数据来源。两种配置各一份，
同机同栈实测（Ascend 910 / CANN 9.2.0 / torch_npu 2.7.1.post4 / transformers 4.51.0，
稳态：输入 1909 token、7056 图块）：

| kernel 前缀 | 基线次数 | 基线总us | 基线占比 | 交付次数 | 交付总us | 交付占比 |
|---|---:|---:|---:|---:|---:|---:|
| `aclnnMatmul` | 639 | 830305 | 42.9% | 575 | 175212 | 47.6% |
| `aclnnSWhere` | 37 | 207391 | 10.7% | 5 | 11 | 0.00% |
| `aclnnInplaceCopy` | 17902 | 206272 | 10.7% | 1310 | 15790 | 4.3% |
| `aclnnSoftmax` | 32 | 196199 | 10.1% | 缺席 | | |
| `aclnnAdd` | 752 | 179102 | 9.2% | 720 | 10514 | 2.9% |
| `aclnnIsNegInf` | 32 | 112367 | 5.8% | 缺席 | | |
| `aclnnAddmm` | 1164 | 59297 | 3.1% | 1164 | 59635 | 16.2% |
| `aclnnAll` | 37 | 28929 | 1.5% | 5 | 15 | 0.00% |
| `aclnnFlashAttentionVarLenScore` | 缺席 | | | 64 | 17290 | 4.70% |
| `aclnnFlashAttentionScore` | 140 | 17408 | 0.90% | 140 | 17124 | 4.65% |
| `aclnnInplaceMaskedFillScalar` | 37 | 9768 | 0.5% | 5 | 249 | 0.07% |
| `aclnnLogicalNot` | 32 | 4087 | 0.2% | 缺席 | | |
| **NPU kernel 总耗时** | | **1936.6 ms** | | | **368.2 ms** | |
| **kernel 记录数** | | **31095** | | | **10083** | |
| **单次 generate 中位耗时** | | **2.617 s** | | | **0.350 s** | |

怎么读这张表：

1. **基线**里 `aclnnSoftmax` 与 `aclnnIsNegInf` 各 **32** 次，正好等于视觉塔层数
   （`vision_config.depth=32`）—— 这是定位到「mask 导致 SDPA 退化成数学路径」的线索。
   `aclnnSWhere` / `aclnnAll` / `aclnnInplaceMaskedFillScalar` 各 37 次 = 这 32 次
   加上别处的 5 次。
2. **交付配置下视觉塔贡献的那部分全部消失**：`Softmax` / `IsNegInf` / `LogicalNot`
   整个缺席；`SWhere` / `All` / `InplaceMaskedFillScalar` 从 37 次降到 **5 次** ——
   剩下的正是上面那 5 次「来自别处」的，印证了基线的计数拆分。
   这六项合计从基线的 **28.8%** 降到 **0.07%**。取而代之的是
   `aclnnFlashAttentionVarLenScore` 64 次、**4.70%**，即 `npu_fusion_attention` 的
   TND varlen 核 —— 它**不**落在 `FlashAttentionScore` 那一行，查表时别找错。
3. `aclnnFlashAttentionScore` 两边都是 **140** 次（= 语言模型 28 层 × 5 次前向），
   绝对耗时 17408 → 17124 us 基本不变，占比从 0.90% 升到 4.65% 只是因为分母缩小。
   **语言模型的注意力一直在走融合核**，这一侧没有可做的。
4. `aclnnInplaceCopy` 从 17902 次 / 206 ms 降到 1310 次 / 16 ms —— 那个「次数异常多」
   的现象是 mask 路径逐段切片赋值的伴生物，随融合一起消失。
5. **交付配置下注意力已不是瓶颈**：两个融合注意力核合计 **9.35%**，而 GEMM
   （`Matmul` 47.6% + `Addmm` 16.2%）占 **63.8%** —— 剩下的是算力，不是小算子。

---

## 一条方法论

这几个脚本共同支撑了优化说明文档里的一个结论：

> 调用级耗时（「哪个调用花了 3 秒」）只能定位到模块，**不能替代算子级 profile**。
> 据前者推断瓶颈在注意力，实际它占不到 1%。
