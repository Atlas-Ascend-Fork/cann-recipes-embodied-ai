# Being-H0.5 NPU 推理适配与优化

[简体中文](README.md) | [English](README_en.md)

[部署指南](../../../../manipulation/being-h0.5/infer_with_torch/README.md)。下文实现路径均相对于组装后的 `Being-H/Being-H05` 工作区；实现由固定上游源码、样例补丁和新增模块共同组成，并非全部直接存放于样例的 `code/` 目录。

## NPU 推理优化原理与实测

### 测试口径

测试日期为 2026-08-24，设备上报 `Ascend910_9362`，单 device、BF16、batch
size 1、双视角 256×256、action chunk 16、4 个 flow steps 和 1 次 MPG
refinement（32 projections）。checkpoint SHA256 为
`c74a1ac89027b2fe43728d2abe8725da69eb43d75dceb35ecd2833bf3034a981`。

- **Core**：`get_action` 的设备同步边界，不含完整 policy 前后处理；
- **E2E**：内存 observation 已就绪到 action decode 完成；
- 模型加载、下载、视频解码和 RPC 网络不计入稳态延迟；
- 10 个 LIBERO-Spatial instruction 轮换，Prefix 原长 197–202，统一 Padding 到 202；
- 每个后端运行 3 个独立进程，每进程 warmup 20，再测 Core/E2E 各 100 次；
- 固定的是 tensor shape，图像、robot state 和 instruction 内容仍可变化。

最终结果：

| 路径 | Core mean | E2E mean | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: | ---: |
| 原始 CANN Raw | 836.023 ms | 841.670 ms | 5,649,623,552 B | 6,094,323,712 B |
| 最终优化版本 | **64.793 ms** | **70.938 ms** | 5,747,131,904 B | 6,570,377,216 B |

Core 减少 771.230 ms（92.25%），加速 12.903×；E2E 减少 770.732 ms
（91.57%），加速 11.865×。

### 最终推理链路

```text
Observation（数值可变、shape 固定）
  → CPU transform / normalize / resize / tokenize
  → Prompt cache + 固定 Prefix Padding/Mask
  → Vision GE 与 Proprio encoder 双流
  → Prefix TorchAir/GE 静态图 ×1，生成 28 层 Prefix KV
  → Action suffix ×8，单 entry NPUGraph replay
       ├── BSND FusionAttention + native GQA
       ├── QKV / Gate-Up / RMSNorm fusion
       ├── fused RoPE / fused SwiGLU
       └── graph-owned KV + suffix workspace
  → MPG batched projection / sort
  → inverse normalization → 16-step action chunk
```

Prefix 只计算一次，把 token-forward 从约 `8×(Prefix+Suffix)` 降为
`1×Prefix + 8×Suffix`。固定 Padding/Mask 后，TorchAir/GE 编译完整的 28 层
Prefix prefill；Action Suffix 继续使用实测更快的 NPUGraph。Prefix 图使用 CANN
9.2 已验证的 BNSD，Suffix 使用 BSND。

### 已采用优化项

每一行使用该项自己的配对 baseline，累计底座不同，收益不能相加：

| 优化项 | baseline | candidate | 配对实测变化 |
| --- | ---: | ---: | ---: |
| Prefix 只算一次 | 847.790 | 506.940 | Core -40.20% |
| FusionAttention + GQA | 474.374 | 433.115 | Core -8.70% |
| Action suffix NPUGraph | 422.574 | 221.794 | Core -47.51% |
| Prefix segment route | 222.656 | 194.340 | Core -12.72% |
| QKV/Gate-Up/RMSNorm 融合 | 196.149 | 149.948 | Core -23.55% |
| MPG 向量化 | 148.644 | 142.435 | Core -4.18% |
| Suffix workspace 复用 | 142.451 | 140.352 | Core -1.47% |
| Prompt tokenizer cache | 4.348 | 3.368 | Pack -22.53% |
| Vision GE | 143.110 | 139.001 | Core -2.87% |
| Vision/State 双流 | 139.872 | 138.131 | Core -1.25% |
| Euler buffer cache | 139.279 | 138.000 | Core -0.92% |
| 编译缓存复用 | 19.233 s | 11.984 s | 首请求 -37.69% |
| BSND Attention layout | 139.212 | 123.995 | Core -10.93% |
| Graph-owned KV workspace | 124.319 | 119.798 | Core -3.64% |
| 原生 fused RoPE | 121.182 | 109.309 | Core -9.80% |
| 原生 fused SwiGLU | 109.467 | 103.385 | Core -5.56% |
| 共享融合权重 storage | 7.284 GiB | 5.424 GiB | allocated -1.859 GiB |
| 固定 Prefix Padding/Mask | 197–202 | 202 | 统一为单 shape |
| 固定 Prefix TorchAir/GE | 95.664 | 64.793 | Core -32.27% |

主要实现位置：

- 推理主链与 Vision GE：`BeingH/model/beingvla.py`
- Qwen3 Prefix/Suffix、FusionAttention、RoPE、SwiGLU：
  `BeingH/model/llm/qwen3_navit.py`
- Action 图缓存和 fail-closed：`BeingH/npu_capture_replay.py`
- Prefix TorchAir/NPUGraph：`BeingH/npu_prefix_torchair.py`、
  `BeingH/npu_prefix_graph_replay.py`
- 融合权重共享存储：`BeingH/fused_projection_storage.py`
- 服务预热和冻结：`BeingH/inference/run_server_vla.py`

### 数值一致性与显存

10 instructions × 20 seeds 的最终 action 对比：

| 对比 | finite | max abs | mean abs | min cosine | 图状态 |
| --- | --- | ---: | ---: | ---: | --- |
| NPUGraph vs 固定 Prefix eager | 200/200 | 0 | 0 | 1.000000238 | 1 capture、200 replay、0 failure |
| 固定 Padding/Mask vs 可变长度 | 200/200 | 0.0625 | 0.000804917 | 0.999996483 | 通过 |
| TorchAir/GE vs BNSD eager | 200/200 | 0.0625 | 0.001678913 | 0.999995470 | 1 compile、200 run、0 failure |

最终 TorchAir peak allocated/reserved 为 5.352/6.119 GiB。共享 storage 在模型
加载完成后，把 Q/K/V、Gate/Up 参数绑定为同一 fused storage 的 view，不改变磁盘
checkpoint 或 state-dict key；该路径仅用于 inference，训练和 optimizer state 不使用。

### 使用边界

- Prefix cache 只用于单真实 sample causal 路径，packed batch 保留完整 forward；
- graph cache key 基于 shape/dtype/device 和功能标志，不依赖 tensor 数值；
- 服务期缓存冻结，未知 shape 必须明确拒绝或受控回退，不在流量中临时 capture；
- 并发 worker 需要独立 graph 生命周期；
- TorchAir 缓存与代码、shape、模型和 CANN 栈绑定，任一变化都需重新生成；
- 性能验收必须同时检查 fresh-process Core/E2E、最终 action 数值、graph 统计、
  峰值显存和 fail-closed，不能把各单点收益相加。
