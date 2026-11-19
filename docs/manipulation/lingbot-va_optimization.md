# LingBot-VA NPU 适配与优化

LingBot-VA 服务端接收双相机图像和语言指令，生成动作序列。LIBERO 客户端执行动作后，将关键帧图像和动作传回服务端，更新下一次推理使用的 KV Cache。环境仿真和图像处理在 CPU 上执行，模型计算在 NPU 上执行。

## NPU 适配

上游源码使用 CUDA 和 NCCL。样例通过 [源码补丁](../../manipulation/lingbot-va/infer_with_torch/patches/upstream_to_npu.patch) 改为 `npu:{local_rank}` 和 HCCL，并替换设备设置、缓存释放等调用。FlashAttention 改为可选导入，启动时使用 `torch` attention 路径。

VAE 按流式方式编码客户端回传的关键帧。首次编码先处理第一帧，建立时序缓存；后续多帧输入按四帧对齐，不足时重复最后一帧。每回合的 reset 会清理 VAE、KV Cache 和常量缓存，避免沿用上一回合的状态。

Spatial、Object 和 Goal 使用各自的 Transformer 权重及 `norm_stats`，通过软链接共用 Long 的 text encoder、tokenizer 和 VAE。Long 使用自己的 Transformer 配置和上游归一化参数，启动时不设置 `NORM_STATS`。权重 revision、配置和 SHA256 见 [checkpoints.json](../../manipulation/lingbot-va/infer_with_torch/configs/checkpoints.json)。

## 推理优化

扩散推理会多次调用同一个 Transformer。样例在 block 和模型两层设置 FSDP `reshard_after_forward=False`，前向结束后保留已聚合的权重，供后续去噪步骤复用。实现见 [npu_inference_patch.py](../../manipulation/lingbot-va/infer_with_torch/src/npu_inference_patch.py)。

文本特征、CFG 拼接结果、网格索引和部分投影结果也会在去噪步骤之间重复使用。[constant_cache_patch.py](../../manipulation/lingbot-va/infer_with_torch/src/constant_cache_patch.py) 缓存这些计算结果；输入或权重发生变化时重新计算，回合 reset 时清理缓存。缓存条目设有数量上限。

RoPE 使用 CANN 的 `npu_rotary_mul`，采用 interleave 布局，对应上游相邻特征配对的复数乘法；旋转计算保留 FP32。Attention 使用 `npu_fusion_attention_v3`，保持原有 KV Cache 选择逻辑。实现见 [cann_fusion_patch.py](../../manipulation/lingbot-va/infer_with_torch/src/cann_fusion_patch.py)。

## 测试结果

性能测试比较基础 NPU 适配版本与优化版本，使用 Goal task 0 的 5 组同轨迹配对，每次测量前预热 1 回合。端到端耗时降幅的中位数为 **24.07%**，基线重复测量的波动中位数为 **0.464%**。

LIBERO 每个 suite 评测 10 个任务，每个任务 50 回合。成功以环境返回的 `done` 为准，全部回合均计入统计。

| Suite | 成功率 |
|---|---:|
| Spatial | 95.0% |
| Object | 99.0% |
| Goal | 96.6% |
| Long | 98.0% |
| 算术平均 | 97.15% |

环境版本、服务启动和评测命令见 [样例 README](../../manipulation/lingbot-va/infer_with_torch/README.md)。
