# Being-H0.5 VAE 训练适配与验证

[简体中文](README.md) | [English](README_en.md)

[训练使用指南](../../../../manipulation/being-h0.5/vae/README.md)。以下保留已有实验记录与验收要求，不代表本次文档整理重新执行了训练或性能验证。

## 训练目标

训练阶段 1–299 epoch 使用 `q + same-hand finger-pad + KL`；300–800 epoch
加入 cross-hand absolute tip、pair-vector 和 pair-distance。主要损失为：

```text
1.0 * loss_q
+ 5.0 * loss_tip
+ 5.0 * loss_cross_tip_abs
+ 1.0 * loss_cross_pair_vector
+ 0.3 * loss_cross_pair_distance
+ 1e-4 * KL_action
+ 1e-5 * KL_morphology
```

## NPU 适配与最终验收

### NPU 适配约束

- 通过 `ACCELERATOR=npu` 统一选择设备，避免 CUDA 专用初始化；
- FSDP/HCCL 使用 torch_npu 支持的 device、optimizer checkpoint 和 barrier；
- attention mask 使用目标 dtype 的有限最小值，避免 FP16 `-inf` 引发 NaN；
- 小 `/dev/shm` 容器默认 `NUM_WORKERS=0`、`PREFETCH_FACTOR=1`，避免 packed
  image batch 的 worker SIGBUS；
- 数据转换和验证均在 `npu:0` 运行，并记录 NativeVAE checkpoint SHA256。

### 端到端两卡流水线

正式流程依次执行：确定性生成三手训练/验证 tensor、800 epoch NativeVAE、
Shadow 数据转换、两 rank HCCL/FSDP checkpoint smoke、40,000 optimizer steps
Being-H0.5 后训练，以及 100 episode open-loop/MuJoCo 验收。

转换后的 `shadow_grasp_bottle22249179_aug100_npuvae_2cam` 保留维度 `0:28`，
将 state/action 的 `28:52` 替换为最终 posterior mean `z_mu`。独立验证器读取全部
100 个 parquet 和 17,700 行，证明非目标列未变化，并从元数据记录的 checkpoint
重新计算 latent 与 SHA256。

两卡正式训练使用：

```text
4,096 tokens/rank × 2 ranks × 2 accumulated microbatches
  = 16,384 global tokens/optimizer update
```

`max_steps=40000` 统计 optimizer update，不是 microbatch。step 15,000 到
40,000 每 5,000 steps 保存可恢复的模型、optimizer shard、scheduler、tokenizer
和元数据。

### 数值与运行时门槛

固定四组 Shadow 姿态对原实现进行回归，`z_gesture`、Shadow 重建、Shadow→Gaia
和 Shadow→Sharpa 使用 `rtol=1e-6, atol=1e-6`，全部通过。批量 shape 为：

| Hand | Input q | z_gesture | Decoded q | Finger pads |
| --- | ---: | ---: | ---: | ---: |
| Shadow | `[2048, 22]` | `[2048, 24]` | `[2048, 22]` | `[2048, 5, 3]` |
| Gaia | `[2048, 15]` | `[2048, 24]` | `[2048, 15]` | `[2048, 5, 3]` |
| Sharpa | `[2048, 22]` | `[2048, 24]` | `[2048, 22]` | `[2048, 5, 3]` |

100 episode open-loop（17,700 actions）的已验证结果：

| 指标 | 结果 |
| --- | ---: |
| action RMSE | 0.011489 |
| wrist position RMSE | 0.003080 m |
| rotation geodesic mean / p95 | 0.979° / 1.837° |
| latent z RMSE | 0.012086 |
| latent cosine similarity | 0.999931 |

CPU/NPU 17,700 行 latent 的最大绝对误差约 0.00112，因此正式数据验证使用
`atol=0.002, rtol=0`。最终门槛还包括：

- 100 episode、17,700 actions 的独立 NPZ/JSON/plot 和聚合指标复算；
- cold/warm mean/P50/P95、query throughput 和 action throughput；
- VAE、Geometry、Gaia joint 三条 matched NPU trial；
- 100 个 MuJoCo episode 的双视角视频、summary、rollout、metadata 和 timing；
- 只有所有门槛通过才写入 `outputs/npu_final_vae_evaluation/COMPLETE`。

性能报告必须使用相同数据、步数、warmup、trial 数和设备，并区分 backbone query
延迟与包含 MuJoCo、geometry optimizer 或 VAE decoder 的端到端吞吐。
