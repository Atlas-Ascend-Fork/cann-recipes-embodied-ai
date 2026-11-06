# Being-H0.5 VAE Training Adaptation and Validation

[简体中文](README.md) | [English](README_en.md)

[Training guide](../../../../manipulation/being-h0.5/vae/README_en.md). The following records the existing experiment results and acceptance criteria; it does not claim a new validation run.

## Training Objective

Epochs 1–299 optimize `q + same-hand finger-pad + KL`. Epochs 300–800 add cross-hand absolute-tip, pair-vector, and pair-distance losses:

```text
1.0 * loss_q
+ 5.0 * loss_tip
+ 5.0 * loss_cross_tip_abs
+ 1.0 * loss_cross_pair_vector
+ 0.3 * loss_cross_pair_distance
+ 1e-4 * KL_action
+ 1e-5 * KL_morphology
```

## NPU Adaptation and Final Acceptance

### NPU Constraints

- Select the device uniformly with `ACCELERATOR=npu` and avoid CUDA-only initialization.
- Use torch_npu-supported devices, optimizer checkpoints, and barriers for FSDP/HCCL.
- Use the finite minimum of the target dtype for attention masks to avoid FP16 `-inf` NaNs.
- In containers with small `/dev/shm`, default to `NUM_WORKERS=0` and `PREFETCH_FACTOR=1` to avoid worker SIGBUS for packed image batches.
- Run data conversion and validation on `npu:0` and record the NativeVAE checkpoint SHA256.

### Two-NPU End-to-End Pipeline

The formal workflow performs deterministic pose generation for three hands, 800-epoch NativeVAE training, Shadow dataset conversion, two-rank HCCL/FSDP checkpoint smoke testing, 40,000 optimizer steps of Being-H0.5 post-training, and 100-episode open-loop/MuJoCo acceptance.

The converted `shadow_grasp_bottle22249179_aug100_npuvae_2cam` keeps dimensions `0:28` and replaces state/action dimensions `28:52` with final posterior mean `z_mu`. The independent validator reads all 100 parquet files and 17,700 rows, confirms that non-target columns did not change, and recomputes latent values and SHA256 from the metadata checkpoint.

```text
4,096 tokens/rank × 2 ranks × 2 accumulated microbatches
  = 16,384 global tokens/optimizer update
```

`max_steps=40000` counts optimizer updates, not microbatches. From step 15,000 through 40,000, a resumable model, optimizer shard, scheduler, tokenizer, and metadata are saved every 5,000 steps.

### Numerical and Runtime Thresholds

Four fixed Shadow poses pass regression against the original implementation for `z_gesture`, Shadow reconstruction, Shadow→Gaia, and Shadow→Sharpa with `rtol=1e-6, atol=1e-6`.

| Hand | Input q | z_gesture | Decoded q | Finger pads |
| --- | ---: | ---: | ---: | ---: |
| Shadow | `[2048, 22]` | `[2048, 24]` | `[2048, 22]` | `[2048, 5, 3]` |
| Gaia | `[2048, 15]` | `[2048, 24]` | `[2048, 15]` | `[2048, 5, 3]` |
| Sharpa | `[2048, 22]` | `[2048, 24]` | `[2048, 22]` | `[2048, 5, 3]` |

Validated 100-episode open-loop results over 17,700 actions:

| Metric | Result |
| --- | ---: |
| action RMSE | 0.011489 |
| wrist position RMSE | 0.003080 m |
| rotation geodesic mean / p95 | 0.979° / 1.837° |
| latent z RMSE | 0.012086 |
| latent cosine similarity | 0.999931 |

The maximum CPU/NPU absolute difference across 17,700 latent rows is approximately 0.00112, so formal data validation uses `atol=0.002, rtol=0`. Final acceptance also requires:

- independently recomputed NPZ/JSON/plot artifacts and aggregates for 100 episodes and 17,700 actions;
- cold/warm mean, P50, P95, query throughput, and action throughput;
- matched NPU trials for VAE, Geometry, and Gaia joints;
- two-view video, summary, rollout, metadata, and timing for 100 MuJoCo episodes;
- creation of `outputs/npu_final_vae_evaluation/COMPLETE` only after all gates pass.

Performance reports must use identical data, steps, warmup, trial counts, and devices, and must distinguish backbone query latency from end-to-end throughput that includes MuJoCo, the geometry optimizer, or the VAE decoder.
