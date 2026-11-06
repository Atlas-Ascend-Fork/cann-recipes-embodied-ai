# Docker acceptance rerun — 2026-10-31

本轮在蓝区物理宿主机重跑完整模型、数据转换、训练和任务评估；按约定将
NativeVAE 和 Being-H 后训练各限制为 **1,000 个 optimizer step**。
执行链路、模型精度和历史长周期训练结果分别报告，不能将短训完成视为收敛证明。

The training budget for this rerun is 1,000 optimizer updates per workflow. Historical
800-epoch / 40,000-step convergence training is outside this agreed budget. The earlier
smoke, checkpoint-resume and v1/v2 delivery checks are retained in the
[2026-10-30 archive](VALIDATION_20261030.md).


## Acceptance scope correction

Training convergence, equality of training loss/weights, and repeating full 800-epoch /
40,000-step training are not merge requirements. Unequal-budget short/mature checkpoint
results below are historical observations; they do not establish an NPU adaptation
accuracy failure. The separately launched full-training queue and heartbeat were stopped.
Inference evaluation compares the same checkpoint under matched inputs and task criteria.
NativeVAE CPU/NPU inference was checked; a full-model GPU/NPU baseline remains distinct
from the NPU raw/optimized comparison. Asset clearance is recorded in the
[per-file audit](../resources/THIRD_PARTY_NOTICES.md).

## Current delivery update: v4 image / runtime-v2

| Current artifact | Verified identity |
| --- | --- |
| v4 image ID | `sha256:07dd5ec5ab4d36bf9db6dfd6589f33ca970d7c52a9a29c9982d3ccb7dcc91fd5` |
| v4 archive | 14,557,188,096 bytes; SHA256 `bdff8c0d44bbdad7aff1a940cb0d79fba34ba91a731dad1cc7889f171fd49d21` |
| runtime-v2 archive | 24,610,669 bytes; SHA256 `630ea5df5cfff3eec2d0ca29027a212957874a7762a8da075cc4d909901eadef` |

v4 passed real NPU smoke and all 33 VAE/simulation/inference regressions. Runtime-v2
passed offline and anonymous-HTTPS restoration with all 685 file hashes checked, and
the ten installer tests passed. Full v4 archive download/load status is maintained in
the PR delivery record; the original v3 numerical experiments below are preserved.

The current recipe pairs `beingh05-npu:assets-v4` with runtime-v2 (685 files). The new
resource package adds seven license/authorization/inventory notices and removes two
generated pyc files. The remaining 678 original runtime files are unchanged. Model
runtime code is unchanged from the v3 validation below; only resource metadata,
notices and script executable modes are updated. Public v1/v2/v3 objects are preserved.
Image identity and archive integrity are provided by the current
[v4 metadata](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v4/image/beingh05-image.json)
and [checksums](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v4/image/SHA256SUMS).
Latest delivery/test status is recorded on PR #136; the numerical tables below retain
their original v3 image identity and experiment conditions.

Internal asset authorization was confirmed by the contributor; the
[authorization record](../resources/INTERNAL_ASSET_AUTHORIZATION.md) and original
third-party license notices are retained. Training acceptance is execution and
checkpoint save/resume only. Additional GPU training or convergence comparisons are
not required. No unavailable GPU inference reference is presented as measured data.

## Image and environment

| Item | Verified value |
| --- | --- |
| Image tag | `beingh05-npu:assets-v3` |
| Source revision used to build the image | `668689966b6beb790c2fb3d5e9c7d31d0735f4c4` |
| Image ID | `sha256:02a3f1d0e34e43e45cb5e9ef8b610736ca6ca2dda31945cb3aa255ee0cebddbd` |
| Archive length | 14,556,154,880 bytes |
| Archive SHA256 | `7001db41bb8a0425cb858f13f2810e948aa582fe442564ca8520246c041a1519` |
| Public archive | [v3 image tar](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v3/image/beingh05-image.tar) |
| Metadata and checksum | [JSON](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v3/image/beingh05-image.json), [SHA256SUMS](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v3/image/SHA256SUMS) |
| Host / devices | ARM64 physical host, Docker 27.2.0, driver 25.5.1, `Ascend910_9382` |
| Python / PyTorch / torch_npu / CANN | 3.11.17 / 2.7.1+cpu / 2.7.1.post11 / 9.2.0 |
| NumPy / PyArrow | 1.26.4 / 21.0.0 |

v3 was assembled afresh from the recipe over the verified v2 dependency image, with
upstream Being-H pinned to `9443a4082bce013cc61e6523596940134179b23e`. SHA256 checks
confirmed all ten changed assembled runtime modules. No model weights, training datasets,
credentials or training outputs were added to the public image. Runtime/video resource
archives remain unchanged under `v1/resources/`; v1 and v2 objects were not overwritten.

The complete v3 archive was downloaded anonymously over HTTPS, hashed while streaming
into `docker load`, and matched both the archive checksum and loaded ARM64 image ID.
The downloaded image then passed imports, real MuJoCo stepping, CPU VAE backward,
and matrix/loss/input-gradient comparisons on two NPU chips. All 680 runtime resource
files passed entrypoint verification. The self-contained candidate image also passed
21 VAE/simulation and 12 inference-helper regression tests, including actual RGB rendering.

Initial long evaluations used v2 with the changed runtime modules mounted read-only;
the new checkpoint evaluations and independent performance processes use v3 without
runtime source overlays. The changed modules are the same files verified in the v3 build.
An independent Docker socket/data root was used; device sharing and the system Docker
service were not modified. Other host workloads remained active, so timings are shared-host
measurements, not dedicated-host peak performance claims.

## Fixes exercised by the rerun

- Restored closed-loop imports, rollout metric collection, native Shadow joint adaptation
  and the Geometry action-chunk cache. Removed invalid logging arguments and an ambiguous
  NumPy-array truth-value check in latency aggregation.
- Added cumulative NativeVAE `--max_steps`, checkpointed optimizer-update counts and
  validation at the final partial epoch. Regression tests cover crossing an epoch boundary
  and continuing from a saved checkpoint.
- Synchronous chunked evaluation can render only observations consumed by a new query.
  Physics still advances at every control step, and video recording still renders every
  frame. Unit tests and complete 177-step VAE/Geometry episode comparisons preserved
  states, actions, object poses and model chunks exactly. LIBERO's separate evaluation
  harness also passed a full-render/query-render comparison; it is not part of the
  production inference closure.

A benchmark harness initially replaced the image's `PYTHONPATH`, removing CANN's `tbe`
module and preventing TorchAir/GE initialization. Preserving CANN's paths fixed the
harness. When adding a custom `PYTHONPATH`, source the matching CANN `set_env.sh` first;
the Docker entrypoint also supports
`ASCEND_CANN_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh` for this purpose.

## Training and full NativeVAE validation

Inputs were recovered from the original CANNLab and verified against source SHA256
manifests. NativeVAE uses all 150,000 training samples and all 15,000 validation samples
(5,000 per hand). Being-H initializes from the historical LIBERO `0012000` checkpoint,
freezes the multimodal LLM/connector, and trains 461.02M parameters on two NPU ranks.
It uses 8,192 expected tokens per rank, no gradient accumulation, and saves full checkpoints
at updates 500 and 1,000.

| Workflow | Actual optimizer updates | Checkpoint and validation evidence |
| --- | ---: | --- |
| NativeVAE | 1,000 | Epoch 14; checkpoint `global_step` and all optimizer counters equal 1,000; finite weights; full final validation; last training/validation losses 0.159570 / 0.153300 |
| Being-H post-training | 1,000 | Both optimizer shards and scheduler count 1,000; all 990 model tensors / 2,807,618,760 elements finite; checkpoints 500 and 1,000 saved; final logged action loss 0.0238 |

The bounded runs use the existing assembled entrypoints. After mounting the models,
full datasets and writable output directories, the NativeVAE command is:

```bash
cd /opt/beingh05/vae
python3 scripts/train.py --config configs/train_native_n2.yaml \
  --device npu:0 --max_steps 1000 --no_wandb
```

Being-H uses `beingh_shared_scripts/train/train_shadow_grasp.sh` with
`MAX_STEPS=1000`, `SAVE_STEPS=500`, `SAVE_STEPS_START=500`, `SAVE_MODEL_ONLY=False`,
`NUM_GPUS=2`, `SMOKE_TEST=False`, `FUSED_OPTIMIZER=False`, `EXPECTED_NUM_TOKENS=8192`
and `MAX_NUM_TOKENS=8704`. The complete mount paths, model/data choices and environment
are retained in the run's `scripts/posttrain-1000.sh`; this requires two sufficiently free
NPU chips (approximately 60.5 GiB reported maximum training memory per rank).

The final Being-H model file has SHA256
`26c411e443255303c79132d6f47066177f5a6446fe81a623930cbd61b7cf8a7e`.

The two short training runs are evaluated separately. Being-H uses data encoded by the
historical 800-epoch NativeVAE, SHA256
`2e72c6b1ff47f524a6eb23ed812bab88cc205771b9150a03b115b87202803f1a`.
The newly trained 1,000-step VAE is not substituted into that policy's latent space.

Full validation-set mean same-hand fingertip errors, in **millimeters**:

| Hand | 1,000-step CPU | 1,000-step NPU | Historical CPU | Historical NPU |
| --- | ---: | ---: | ---: | ---: |
| Shadow | 12.06727 | 12.06787 | 1.49287 | 1.49331 |
| Gaia | 8.82001 | 8.82050 | 1.05941 | 1.05986 |
| Sharpa | 12.40638 | 12.40710 | 1.54535 | 1.54555 |

Cross-hand mean finger-pad discrepancies on NPU (all six directions, millimeters):

| Source → target | 1,000-step VAE | Historical VAE |
| --- | ---: | ---: |
| Shadow → Gaia | 43.289 | 11.808 |
| Shadow → Sharpa | 24.741 | 3.447 |
| Gaia → Shadow | 41.264 | 2.520 |
| Gaia → Sharpa | 40.001 | 2.077 |
| Sharpa → Shadow | 22.987 | 5.752 |
| Sharpa → Gaia | 39.771 | 10.950 |

CPU and NPU agree closely, but the short-trained model is substantially less accurate
than the historical checkpoint. Cross-hand losses begin at epoch 300 and were not
activated during these 14 epochs. This is a completed execution/numerical check, not
a claim that 1,000 updates reproduce the converged model.

## Full data conversion and open-loop regression

v3 reconverted all 100 Shadow episodes / 17,700 rows. Independent CPU re-encoding
passed with `atol=0.002, rtol=0`; maximum state/action latent errors were
0.001256287 / 0.001195371. Non-target Parquet columns remained exact and all 200 videos
were preserved. The new conversion also exactly matched the previously verified data
used for post-training: maximum latent difference 0, with identical non-target columns
and video hashes.

Each open-loop run covers all 17,700 targets using 1,200 non-overlapping 16-step queries.
An independent NPZ recomputation checks finite outputs and reproduces the JSON metrics.
Rotation is compared geodesically; mixed-unit action RMSE wraps Euler differences at
±π so equivalent rotations separated by 2π do not create false errors.

| Model | Episodes / actions | Wrist position RMSE | Rotation mean / p95 | Latent RMSE |
| --- | --- | ---: | ---: | ---: |
| Historical Shadow latent checkpoint `0060000` | 100 / 17,700 | 3.09957 mm | 0.98083° / 1.80680° | 0.01212493 |
| New 1,000-step checkpoint | 100 / 17,700 | 25.56611 mm | 11.32483° / 22.84513° | 0.14080644 |

Historical wrapped action RMSE is 0.01152116 and mean latent cosine similarity is
0.99993125. The new checkpoint has wrapped action RMSE 0.13370377 and latent cosine
similarity 0.99090155. Its substantially larger error and 0/100 closed-loop result mean
that 1,000 updates have not recovered the historical policy's task accuracy.
These trajectories belong to the training split: the results measure
reconstruction/open-loop regression, not held-out generalization.

## Actual closed-loop task results

Shadow/Gaia evaluations use all 100 recorded scene initializations, 177 control steps
per episode, `safe_smooth`, commanded feedback, synchronous chunk 16, fixed-per-query
noise, seed 42, native joint rate limits, and safety intervention disabled.
The new checkpoint's 100 episodes run in three disjoint batches (0–33, 34–66, 67–99)
on available chips. Each episode resets the seed and all policy/limiter state; the merged
result requires exactly one record for each of the 100 initializations. These batch runs
are for task accuracy, not the matched performance measurements below.

- **Loose:** lift at least 0.10 m for one control frame.
- **Strict:** lift at least 0.20 m for ten consecutive control frames.

All trajectories retain the full 177 steps (`terminate_on_success=False`). Strict results
are recomputed from recorded object poses, and recomputed loose results exactly match
the original evaluator metadata; the success criterion does not change the controls.

| Policy / target | Episodes | Loose successes | Strict successes |
| --- | ---: | ---: | ---: |
| Historical latent policy → Shadow | 100 | 96 | 93 |
| Historical latent policy + NativeVAE → Gaia | 100 | 70 | 66 |
| Gaia-specific joint policy → Gaia | 100 | 79 | 67 |
| Shadow joint policy + Geometry → Gaia | 100 | 0 | 0 |
| New 1,000-step latent policy → Shadow | 100 | 0 | 0 |

Geometry uses the raw profile, twelve optimization iterations and batched chunks.
Diagnostic episode 0 succeeds when the same joint checkpoint directly controls Shadow
(0.27215 m lift), but fails with both batched and sequential Geometry transfer to Gaia.
A v3 diagnostic keeping model inference on NPU but moving only the Geometry optimizer
to CPU also fails on episode 0 (0.000579 m lift). On the identical recorded 177-frame
model action stream, CPU/NPU retargeted joint differences have mean 0.01335 rad and maximum
0.30922 rad, while normalized geometry RMSE is 0.205457 / 0.205583. The nonlinear optimizer
is therefore not bitwise equivalent across devices. These checks do not identify the
transfer failure's root cause, but rule out treating batch-cache restoration or successful
NPU initialization as an accuracy fix. Geometry transfer remains an unresolved accuracy
limitation; successful program execution is not an accuracy pass.

LIBERO-Spatial uses ten tasks × ten fixed initial states, 220 action steps plus ten
settling steps, chunk 16 and matched episode seeds. It uses the historical LIBERO
checkpoint, not the newly trained Shadow checkpoint. The evaluation-only dependency
overlay pins robosuite 1.4.0 and MuJoCo 2.3.7 with LIBERO revision
`04e0cb430c80517e6928a063f5483fdbcb5b257c`; these benchmark assets/dependencies are external
to the production inference image/closure.

| LIBERO path | Episodes | Successes |
| --- | ---: | ---: |
| Raw | 100 | 95 |
| `npu_final`, fixed Prefix TorchAir + Suffix Graph | 100 | 92 |

The optimized full task run compiled one Prefix, executed it 840 times, captured one
Suffix graph and replayed it 6,720 times; all fallback/failed-key counters were zero and
both runners remained healthy. Counts include the initial 105-query paired-input check.
The paired episodes had **eight raw-only successes and five optimized-only successes**,
so the optimized result is three percentage points lower on this fixed 100-episode set.
Identical success counts on an individual task do not imply identical episode outcomes.
This is an observed accuracy difference, not a claim of statistical significance or
accepted accuracy parity; it requires review against the community's accuracy criteria.

| LIBERO task ID (0-based) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw successes / 10 | 10 | 8 | 10 | 9 | 9 | 9 | 10 | 10 | 10 | 10 |
| optimized successes / 10 | 10 | 8 | 9 | 9 | 9 | 10 | 10 | 9 | 10 | 8 |

## Matched performance

For the hand pipelines, three trials per chain alternate order on the same NPU and
Gaia episode 0, with 177 steps and two warmup queries. These are **policy-call timings**:
MuJoCo stepping and image rendering are excluded. Process wall time, which includes
loading and simulation, is retained separately in the detailed results.

| Chain | Being-H query mean | Policy step mean, amortized over chunk 16 | Policy steps/s |
| --- | ---: | ---: | ---: |
| VAE | 791.05 ms | 69.67 ms | 14.36 |
| Geometry | 812.25 ms | 122.13 ms | 8.19 |
| Gaia joint | 806.90 ms | 55.19 ms | 18.15 |

The LIBERO performance rerun uses three fresh processes per profile, twenty warmups
and 100 timed requests per process. Ten instructions rotate over identical real simulator
RGB/state inputs with paired seeds. Core brackets `model.get_action`; E2E brackets the
full policy call, including input/output transforms. Both boundaries synchronize the NPU.
Model loading, downloads, video decoding and RPC are outside steady-state timing.

| Path | Core mean / P50 / P95 | E2E mean / P50 / P95 | Peak allocated / reserved |
| --- | --- | --- | --- |
| raw | 776.74 / 764.77 / 862.17 ms | 781.66 / 769.58 / 866.75 ms | 5.26 / 5.68 GiB |
| npu_final | 62.23 / 62.32 / 64.39 ms | 66.29 / 66.43 / 69.07 ms | 5.35 / 6.11 GiB |

Across 300 timed requests per profile, Core speedup is **12.48×** and full-policy E2E
speedup is **11.79×**. E2E query throughput is 1.279 versus 15.085 requests/s, or
20.47 versus 241.36 action vectors/s at chunk 16; this is not simulator frame throughput.
The three optimized processes each compiled one Prefix and captured one Suffix graph:
2,880 total Suffix replays and 360 Prefix runs, with zero eager fallbacks, failed keys
or unhealthy workers. Across 300 matched action comparisons, maximum component
errors were 0.0078125 position, 0.00390625 rotation and 0.0078125 gripper; RMSE was
0.00161711.

Cold E2E requests took 1.58–2.00 seconds raw and 95.36–99.41 seconds optimized,
including initial graph compilation. Prewarm the service before accepting traffic;
the steady-state speedup does not apply to a cold first request.


An additional alternating AB/BA check on the same chip/process/input measured
801.53 ms raw versus 66.99 ms optimized E2E (11.96×). Its 100 outputs per profile exactly matched
the corresponding separate raw-only and optimized-only runs. It recorded one Prefix compile, one Suffix capture, 840 replays
and zero fallbacks/failed keys. Observed raw/optimized maximum component differences were
0.0078125 position, 0.00390625 rotation and 0.0078125 gripper; BF16 paths are not asserted
bitwise identical. Cold optimized initialization includes compilation and is reported
separately from the warmed request speedup.

## Evidence and remaining boundaries

The final independent audit completed on 2026-10-31 at 02:49 UTC. It checked both actual
optimizer-update counts, all four full VAE validation outputs, both 17,700-action
open-loop runs, exactly 100 unique episodes for each hand chain, both 100-episode LIBERO
runs, zero optimized graph fallbacks, release checksums and successful job exit codes.
It explicitly records execution completion without asserting that every accuracy target
passed.

For the historical and new Shadow latent policies, all 100 trajectories each have two
verified 178-frame videos: **400 MP4 files total**. These are offline renders of the
recorded actual joint/object poses, with exact state reconstruction and video frame-count
checks; they are not additional policy rollouts. Separate v3 episode-0 VAE/Geometry runs
also exercised live video recording. Failed episodes are retained.

Detailed machine-readable results, per-episode metadata, reproducible harness scripts
and logs are indexed by `acceptance-evidence-manifest.json` in the server run directory.
`results/final-verification.json` records the audit scope; checkpoint weights and full
rollouts remain alongside it. The evidence archive is retained outside Git. CI status
for the documentation commit is reported on the PR; runtime results above refer to the
verified image source revision.

- The agreed 1,000-update budget does not establish historical long-training convergence.
  Both short-trained models still have accuracy limitations; the new Being-H checkpoint
  scores 0/100 and Geometry transfer also scores 0/100.
  LIBERO optimized success is 92/100 versus raw 95/100; task accuracy parity is not established.
- The original nested WebIDE ownership errors (`-8020`, `aclInit 507899`, `Resource_Busy`)
  remain a platform limitation; direct physical-host Docker validation does not fix them.
- Community review remains separate from runtime and CI results. Internal-asset
  authorization has been confirmed and recorded; third-party component notices remain
  in force. See the [authorization record](../resources/INTERNAL_ASSET_AUTHORIZATION.md).
- No long-training `outputs/npu_final_vae_evaluation/COMPLETE` marker is claimed from this
  shortened run. Detailed scripts, logs, checkpoints, rollouts and videos are retained under
  `/pypto/pr136-acceptance-20261031/` on the validation server. Models/data remain private
  runtime inputs, outside Git and the public image.
