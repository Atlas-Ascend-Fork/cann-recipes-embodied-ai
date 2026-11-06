# Being-H0.5 NPU Inference Adaptation and Optimization

[简体中文](README.md) | [English](README_en.md)

[Deployment guide](../../../../manipulation/being-h0.5/infer_with_torch/README_en.md). Source paths below refer to the assembled `Being-H/Being-H05` workspace. Implementations come from the pinned upstream source, recipe patches, and new modules; these are not all standalone files in the recipe `code/` directory.

## NPU Optimization and Measurements

### Measurement Protocol

The benchmark was run on 2026-08-24 on a device reported as `Ascend910_9362`, using one device, BF16, batch size 1, two 256×256 camera views, action chunk 16, four flow steps, and one MPG refinement with 32 projections. The checkpoint SHA256 is `c74a1ac89027b2fe43728d2abe8725da69eb43d75dceb35ecd2833bf3034a981`.

- **Core**: synchronized device boundary of `get_action`, excluding complete policy preprocessing and postprocessing.
- **E2E**: from an in-memory observation being ready to completion of action decoding.
- Model loading, downloads, video decoding, and RPC networking are excluded from steady-state latency.
- Ten LIBERO-Spatial instructions were rotated; Prefix lengths 197–202 were padded to 202.
- Each backend used three fresh processes; each process ran 20 warmup iterations and then 100 Core and 100 E2E measurements.
- Tensor shapes were fixed, while images, robot states, and instruction content remained variable.

| Path | Core mean | E2E mean | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: | ---: |
| Original CANN Raw | 836.023 ms | 841.670 ms | 5,649,623,552 B | 6,094,323,712 B |
| Final optimized path | **64.793 ms** | **70.938 ms** | 5,747,131,904 B | 6,570,377,216 B |

Core latency decreased by 771.230 ms (92.25%), for a 12.903× speedup. E2E latency decreased by 770.732 ms (91.57%), for an 11.865× speedup.

### Final Inference Path

```text
Observation (variable values, fixed shapes)
  → CPU transform / normalization / resize / tokenization
  → Prompt cache + fixed Prefix Padding/Mask
  → Dual-stream Vision GE and proprio encoder
  → Prefix TorchAir/GE static graph ×1, producing 28-layer Prefix KV
  → Action suffix ×8 with single-entry NPUGraph replay
       ├── BSND FusionAttention + native GQA
       ├── QKV / Gate-Up / RMSNorm fusion
       ├── fused RoPE / fused SwiGLU
       └── graph-owned KV + suffix workspace
  → MPG batched projection / sort
  → inverse normalization → 16-step action chunk
```

The Prefix is computed once, reducing token forward from approximately `8×(Prefix+Suffix)` to `1×Prefix + 8×Suffix`. After fixed Padding/Mask, TorchAir/GE compiles the full 28-layer Prefix prefill, while the Action Suffix continues to use the faster NPUGraph path. The Prefix graph uses CANN 9.2-validated BNSD and the Suffix uses BSND.

### Accepted Optimizations

Each row uses its own paired baseline. The baselines differ, so the gains must not be added together.

| Optimization | baseline | candidate | Paired result |
| --- | ---: | ---: | ---: |
| Compute Prefix once | 847.790 | 506.940 | Core -40.20% |
| FusionAttention + GQA | 474.374 | 433.115 | Core -8.70% |
| Action suffix NPUGraph | 422.574 | 221.794 | Core -47.51% |
| Prefix segment route | 222.656 | 194.340 | Core -12.72% |
| QKV/Gate-Up/RMSNorm fusion | 196.149 | 149.948 | Core -23.55% |
| Vectorized MPG | 148.644 | 142.435 | Core -4.18% |
| Suffix workspace reuse | 142.451 | 140.352 | Core -1.47% |
| Prompt tokenizer cache | 4.348 | 3.368 | Pack -22.53% |
| Vision GE | 143.110 | 139.001 | Core -2.87% |
| Vision/state dual stream | 139.872 | 138.131 | Core -1.25% |
| Euler buffer cache | 139.279 | 138.000 | Core -0.92% |
| Compile-cache reuse | 19.233 s | 11.984 s | First request -37.69% |
| BSND attention layout | 139.212 | 123.995 | Core -10.93% |
| Graph-owned KV workspace | 124.319 | 119.798 | Core -3.64% |
| Native fused RoPE | 121.182 | 109.309 | Core -9.80% |
| Native fused SwiGLU | 109.467 | 103.385 | Core -5.56% |
| Shared fused-weight storage | 7.284 GiB | 5.424 GiB | allocated -1.859 GiB |
| Fixed Prefix Padding/Mask | 197–202 | 202 | One unified shape |
| Fixed Prefix TorchAir/GE | 95.664 | 64.793 | Core -32.27% |

Primary implementation locations:

- Inference path and Vision GE: `BeingH/model/beingvla.py`
- Qwen3 Prefix/Suffix, FusionAttention, RoPE, and SwiGLU: `BeingH/model/llm/qwen3_navit.py`
- Action graph cache and fail-closed behavior: `BeingH/npu_capture_replay.py`
- Prefix TorchAir/NPUGraph: `BeingH/npu_prefix_torchair.py`, `BeingH/npu_prefix_graph_replay.py`
- Shared fused-weight storage: `BeingH/fused_projection_storage.py`
- Server warmup and cache freeze: `BeingH/inference/run_server_vla.py`

### Numerical Consistency and Memory

Final action comparisons across 10 instructions × 20 seeds:

| Comparison | finite | max abs | mean abs | min cosine | Graph status |
| --- | --- | ---: | ---: | ---: | --- |
| NPUGraph vs fixed-Prefix eager | 200/200 | 0 | 0 | 1.000000238 | 1 capture, 200 replay, 0 failure |
| Fixed Padding/Mask vs variable length | 200/200 | 0.0625 | 0.000804917 | 0.999996483 | Passed |
| TorchAir/GE vs BNSD eager | 200/200 | 0.0625 | 0.001678913 | 0.999995470 | 1 compile, 200 run, 0 failure |

Final TorchAir peak allocated/reserved memory was 5.352/6.119 GiB. After model loading, shared storage binds Q/K/V and Gate/Up parameters as views of the same fused storage without changing on-disk checkpoint keys or the state dict. This path is inference-only and is not used by training or optimizer state.

### Operating Boundaries

- Prefix cache applies only to the single-real-sample causal path; packed batches use the full forward path.
- Graph cache keys depend on shape, dtype, device, and feature flags, not tensor values.
- Caches are frozen during serving; unknown shapes must fail explicitly or use a controlled fallback rather than capturing under traffic.
- Concurrent workers require independent graph lifecycles.
- TorchAir cache artifacts are tied to the code, shape, model, and CANN stack; regenerate them after any change.
- Performance acceptance must jointly verify fresh-process Core/E2E, final-action numerics, graph statistics, peak memory, and fail-closed behavior.
