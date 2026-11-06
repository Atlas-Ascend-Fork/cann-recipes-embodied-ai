> Archived record from 2026-10-30. See [the current acceptance report](VALIDATION.md) for v3 and the 1,000-step rerun. Outstanding work below describes the status on that earlier date.

# Docker model validation — 2026-10-30

蓝区 Docker 已能初始化 NPU；从原 CANNLab 找到并转移权重和数据后，完成了
NativeVAE 全量验证集评估、一轮全量数据训练、100 episode 数据转换，以及
Being-H 完整 checkpoint 的原始/优化推理检查。下面明确区分这些结果与尚未完成的
长周期训练、闭环任务成功率和正式性能验收。

The blue-zone physical host initializes NPU devices successfully inside Docker. Weights
and datasets found on the original CANNLab enabled the model checks below. A successful
smoke run is not a substitute for long training, task success rates or formal benchmarking.

## v2 environment and fixes

The stack remains ARM64, Docker 27.2.0, Ascend driver 25.5.1, `Ascend910_9382`,
Python 3.11.17, PyTorch 2.7.1+cpu, torch_npu 2.7.1.post11 and CANN 9.2.0.
NumPy remains 1.26.4; PyArrow is pinned to 21.0.0. The corrected image is tagged
`beingh05-npu:assets-v2`; its exact image ID, archive length and SHA256 are recorded in
`v2/image/beingh05-image.json` and `v2/image/SHA256SUMS` on OBS. Runtime/video resources
remain byte-for-byte identical under `v1/resources/`.

The v2 artifact was built incrementally from the verified v1 image, replacing the
files fixed in source commit `b2fc05c` and installing PyArrow 21.0.0. The repository's
fresh-build Dockerfile includes the same dependency constraint and import check.
All 171 assembled source/config/script files matched a fresh assembly of the current
repository by SHA256. No model weights, datasets, credentials or training outputs were
added to the image.

Failures reproduced and fixed during real workload validation:

- PyArrow 26 required NumPy 2 and prevented `datasets`/Parquet imports. A candidate
  PyArrow 25 build imported successfully but corrupted a dictionary-encoded fixed-size
  float-list Parquet round trip on this ARM64 environment. PyArrow 21 passed exact
  round-trip tests; the version is now constrained during both dependency installs.
- Episode statistics and training metrics wrote a literal backslash followed by `n`
  instead of a JSONL line break. Each record now occupies its own line.
- Action Suffix graph requests assigned to the read-only `last_fallback_reason`
  property; capture also referenced an undefined `_WarmupBatch`. Both are fixed.
- TorchAir Prefix request helpers were not imported, and geometry retargeting lacked
  request/context/result dataclasses. The missing imports and definitions are restored.

## v2 artifact identity

| Item | Value |
| --- | --- |
| Image ID | `sha256:0619fc18f8ad07bac92a2806cbed4d8003940ba9365641aba37bd33e3f9f9288` |
| Archive length | 14,552,559,104 bytes |
| Archive SHA256 | `d3c284e52e6e18f997c228141769e337a7a721274d174da13111aa98c337ee06` |
| Public archive | [v2 image tar](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v2/image/beingh05-image.tar) |

## Model inputs and provenance

All transferred files were checked against source SHA256 manifests. Inputs are private
runtime mounts, not part of the public release:

- Historical LIBERO checkpoint `0012000`, from the original
  `post-libero_spatial_Being-H05-2B_freeze_mllm-False_chunk-16_mpg-True_lambda-0.1_proj-32_refine-1_rtc-False_d-0_20260725_204804`
  run: 10 files, 11,246,519,289 bytes, including tokenizer and LIBERO metadata.
  This is the checkpoint used for the Docker check, not a newly downloaded official
  `Being-H05-2B_libero` release.
- `native_n2_epoch800_inference.pt`, SHA256
  `2e72c6b1ff47f524a6eb23ed812bab88cc205771b9150a03b115b87202803f1a`.
- NativeVAE training data: 50,000 samples per hand, 150,000 total.
  Validation data: 5,000 per hand, 15,000 total; Shadow, Gaia and Sharpa.
- Shadow original/converted and Gaia datasets: each contains 100 episodes and 17,700
  frames. The new conversion used the original Shadow dataset and the checkpoint above.
- InternVL3.5-2B and Qwen3-0.6B training inputs: 92 files, 6,231,990,505 bytes.

## Completed model checks

| Check | Observed result |
| --- | --- |
| NativeVAE regression suite | 19 tests passed, including the pretrained golden regression, MuJoCo RGB rendering and exact Parquet/JSONL regression tests |
| Inference helper regression suite | 5 tests passed: graph session locking/recovery, capture warmup, Prefix fallback, identity and cross-hand geometry retargeting |
| Full VAE validation dataset | CPU and NPU each evaluated all 15,000 samples |
| Full-data VAE training | One NPU epoch over 150,000 training samples, followed by 15,000 validation samples; finite metrics and checkpoints saved |
| Two-rank HCCL/FSDP post-training | One optimizer step on converted Shadow data, action loss 2.0620; saved model, two optimizer shards, scheduler, tokenizer and dataset metadata |
| Two-rank checkpoint resume | Loaded model/optimizer/scheduler with `resume_model_only=False`, continued from step 1 to step 2 with finite loss; both processes exited successfully |
| Shadow NPU dataset conversion | All 100 episodes / 17,700 rows / 200 videos processed; non-target Parquet columns preserved exactly |
| Independent CPU latent re-encoding | State/action max absolute errors 0.00125629 / 0.00119537, below documented `atol=0.002, rtol=0` |
| Full LIBERO checkpoint, `raw` | Two warmups and three seeded requests; finite position `[1,16,3]`, rotation `[1,16,3]` and gripper `[1,16,1]` outputs |
| Full LIBERO checkpoint, `npu_final`, eager Prefix | One Action Suffix graph captured, 40 replays, zero eager fallbacks/failed keys, healthy worker |
| Fixed Prefix TorchAir + Action Suffix Graph | One Prefix compiled, five successful calls, zero failed keys; Action Suffix replayed 40 times without fallback |

The one-epoch NativeVAE run reported training loss `0.4267848` and validation loss
`0.3127507`. This is a from-scratch execution check, not a replacement for the
pretrained 800-epoch model or proof of convergence. Its `metrics.jsonl` parses correctly.

Full validation-set mean fingertip errors (meters):

| Hand | CPU | NPU |
| --- | ---: | ---: |
| Shadow | 0.001492871 | 0.001493308 |
| Gaia | 0.001059408 | 0.001059860 |
| Sharpa | 0.001545350 | 0.001545550 |

The LIBERO test uses deterministic synthetic RGB images and zero robot states. Across
seeds 100/101/102, raw versus optimized action max absolute differences were 0.00390625
(position), 0.00231934 (rotation) and 0.03515625 (gripper). These are observed BF16
path differences, not a declared task-level accuracy pass. TorchAir was enabled with `policy.configure_fixed_prefix_padding(instructions)` before
warmup, as in the server startup flow. Merely selecting the backend does not enable
fixed-shape Prefix compilation without a bounded instruction set. Three timed requests are
also insufficient for a performance claim; this report does not extrapolate throughput.

To rerun the regression suites inside the resource-mounted container:

```bash
cd /opt/beingh05
RUN_MUJOCO_RENDER_TESTS=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  python3 -m unittest discover -s vae/tests -v
python3 -m unittest discover -s /opt/beingh-recipe/infer_with_torch/tests -v
```

Mount the pretrained NativeVAE checkpoint into `vae/checkpoints/` for the golden test.
Use `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` for a container with a small CPU quota;
otherwise PyTorch can create far more CPU threads than the quota permits.
Model evaluation uses `vae/scripts/evaluate_native_vae.py`, training uses
`vae/scripts/train.py --config configs/train_native_n2.yaml --device npu:0 --epochs 1 --no_wandb`,
and data conversion/verification follows the [VAE guide](../vae/README_en.md).

The two-rank smoke initialized from the historical LIBERO checkpoint with all keys
matching, froze the multimodal LLM/connector and trained 461.02M parameters. It used
4,096 tokens per rank, no gradient accumulation, and a one-step limit. This validates
execution and checkpoint writing, not a trained Shadow policy. The container uses the
shared launcher `beingh_shared_scripts/train/train_shadow_grasp.sh` with explicitly
mounted model/data paths; host-specific environment paths in the NPU wrapper must be
configured separately when using that wrapper.

## v2 OBS round-trip verification

The published v2 image, metadata and checksum files were downloaded anonymously over
HTTPS. All 14,552,559,104 archive bytes were streamed into `docker load` while calculating
SHA256; the result matched the value above and the loaded image ID/ARM64 architecture.
Streaming avoided retaining an additional full archive on the server. Download plus
load took 366.81 seconds; this is a transfer observation, not an inference benchmark.

Using the downloaded image and previously verified v1 runtime archive, the container
passed dependency/Parquet imports, MuJoCo stepping, CPU VAE backward, matrix multiplication
and loss/input-gradient comparisons on both NPU chips. The full 19-test VAE suite and
five inference-helper tests passed again, including real RGB rendering and the pretrained
regression checkpoint. Final exit status was 0:

```text
V2_FULL_OBS_DOWNLOAD_SHA256_DOCKER_LOAD_PASSED
SMOKE_TEST_PASSED
ALL_NPU_TESTS_PASSED
V2_OBS_TO_DOCKER_NPU_AND_24_REGRESSIONS_PASSED
```

All 16 original business-container IDs were still running. Device sharing remained
`False`; no existing workloads or the system Docker daemon were stopped. The v1 OBS
objects remain available; only reproducible local v1 tar copies were reclaimed for
space. v2 has a new immutable object prefix, and unchanged resources keep their v1 URLs.

## Remaining acceptance work

- The full 800-epoch VAE retraining and 40,000 optimizer-step Being-H post-training
  have not been repeated in this Docker environment. One epoch/step cannot establish
  training convergence or match historical results.
- The 100-episode open-loop and closed-loop MuJoCo/LIBERO success-rate evaluations,
  matched VAE/Geometry/Gaia-joint trials, and controlled cold/warm latency/throughput
  benchmarks remain outstanding. Synthetic LIBERO input is only a runtime check.
- The original WebIDE nested Docker ownership errors (`-8020`, `aclInit 507899`,
  `Resource_Busy`) remain a platform restriction. Blue-zone direct Docker does not
  require changing sharing mode and does not fix nested-container ownership.
- PR CodeCheck is separate from device/model validation. [Pipeline #419 report](https://www.openlibing.com/apps/entryCheckDashCode/MR_9a35f1520127485392c7fc22d29eb40a/34bbebd8738a05ef00036510aa7b38a3?projectId=300033&codeHostingPlatformFlag=gitcode)
  showed 50 general-severity issues: 29 license headers, 8 logging rules, 3 line-width
  issues, 3 nesting-depth issues and 7 other style/complexity/executable-path rules.
  No fatal or severe issues were listed. Subsequent checks must be reviewed before
  claiming the PR passes CI; model validation does not waive these rules.

## Previous v1 smoke and OBS round-trip record

The following is the earlier smoke-only record. It remains useful for the unchanged
resource archives and physical-host setup; v1's dependency limitation above supersedes
its suitability for full dataset/model workflows. Use v2 for new deployments.

### Archived v1 Docker NPU smoke — 2026-10-30

原始 ARM64 发布镜像已在蓝区物理宿主机的 Docker 中完成 NPU 基础验证。
以下结果覆盖资源、依赖、仿真、基础运算、VAE 小样本训练及 HCCL 通信检查，
不代表完整模型推理、精度回归或完整训练已通过。

The original ARM64 release image passed Docker NPU smoke validation on the blue-zone
physical host. These results cover resources, imports, simulation, basic arithmetic,
a small VAE training run and an HCCL collective;
they do not establish full model inference, accuracy or training correctness.

### Environment and artifact identity

| Item | Validated value |
| --- | --- |
| Deployment | Docker directly on an ARM64 physical host |
| Docker / Ascend driver | 27.2.0 / 25.5.1 |
| Devices | Two `Ascend910_9382` chips, exposed as `npu:0` and `npu:1` |
| Python / PyTorch / torch_npu / CANN | 3.11 / 2.7.1+cpu / 2.7.1.post11 / 9.2.0 |
| Image tag | `beingh05-npu:assets-v1` |
| Image ID | `sha256:9c2cc7d9a7aded70baa13b58b3cec7a052300840cecbc3101ab842c524670bc9` |
| Archive size | 14,415,367,680 bytes |
| Archive SHA256 | `6f6853be834cb6be3d69612178cd5a9ca1aa01f46151e4ff514ff5ddffd77606` |

The archive checksum and loaded image ID matched the original release. CANN and PyTorch
packages were not replaced on this host. An independent Docker data root, Unix socket
and containerd namespaces were used; the system Docker service was not restarted.
Device sharing remained `False` on both chips, and all 16 existing business containers
were still running after validation. Test containers exited and were removed.

### Checks and observed results

| Check | Result |
| --- | --- |
| Runtime and video archive SHA256 | Passed |
| Entrypoint installation and per-file verification | 680 runtime files passed |
| Imports | torch, torchvision, mujoco, pytorch3d.transforms, native_vae, BeingH utility and model modules passed |
| Real MuJoCo TissueBox scene | Scene loading and stepping passed; `MUJOCO_OK 35 30` |
| VAE random-data and cross-hand CPU backward test | Passed |
| Image's `smoke_test.py --npu` | Initialization and 8×8 matrix multiplication on `npu:0` passed |
| Additional two-chip test | 8×8 FP32 matrix products on each chip matched CPU results |
| Additional backward test on each chip | For 32×64 and 64×16 FP32 inputs, squared-mean loss and both input gradients matched CPU results (`rtol=1e-3`, `atol=1e-4`) |
| Repeat after isolating containerd namespaces | Both chips passed again |
| VAE training smoke on one NPU | 32 training and 16 validation samples per hand; one training and one validation batch passed |
| Two-rank HCCL AllReduce | Each rank supplied 16 FP32 values equal to rank + 1; both results matched 16 values of 3 |

Selected output from the completed checks:

```text
MUJOCO_OK 35 30
NPU_MATMUL_OK Ascend910_9382
SMOKE_TEST_PASSED
VERSIONS 2.7.1+cpu 2.7.1.post11
NPU_COUNT 2
NPU_MATMUL_AND_BACKWARD_OK 0 Ascend910_9382
NPU_MATMUL_AND_BACKWARD_OK 1 Ascend910_9382
ALL_NPU_TESTS_PASSED
```

Use the [quick-start Docker command](../README_en.md#3-mount-resources-and-check-the-container)
to repeat the image's smoke check. That script checks `npu:0`; the additional two-chip
and gradient comparisons above were separate diagnostics on the validation server.
A separate HCCL check used two processes with `torch.distributed` and the `hccl` backend.
With `--network none`, an explicit loopback rendezvous (`--master-addr=127.0.0.1`)
was used; automatic hostname-based rendezvous could not resolve the container hostname.
Both ranks printed `HCCL_ALL_REDUCE_OK`. This does not validate distributed model training.
The VAE smoke run follows the [documented single-NPU commands](../vae/README_en.md)
and printed `VAE_NPU_ONE_BATCH_SMOKE_PASSED`.

### OBS release and download verification

Release prefix: `obs://cann-ai/cann-recipes-embodied-ai/being-h0.5/v1/`.
The [main README](../README_en.md#docker-and-resource-quick-start) provides the public HTTPS links.
The image archive and metadata are under `image/`, resource archives and their manifest/checksums
under `resources/`, and four individual MP4 files under `videos/`. No AK/SK is required to download.

On the blue-zone server, the complete 14,415,367,680-byte image was downloaded anonymously
from OBS and matched the archive SHA256 above. Image metadata and both resource archives
passed their checksum files. The repository downloader then fetched the manifest URLs and
verified all 680 runtime files and four video files. Each individual MP4 also passed an
anonymous download and SHA256 comparison. All 11 published objects returned HTTP 200 with
expected lengths, and the public resource manifest matched the repository manifest.

The OBS-downloaded image was imported with `docker load`; its ID and ARM64 architecture
matched the original release. The container used the OBS-downloaded runtime archive for
its entrypoint and smoke checks. The previously used devices were occupied, so the repeat
used an idle pair (`davinci10`/`davinci11`), without stopping other workloads or changing sharing.

The OBS-download smoke run passed imports, MuJoCo, CPU VAE backward, NPU matrix
multiplication and both chips' loss/gradient comparisons. It exited with status 0:

```text
ALL_ANONYMOUS_DOWNLOAD_AND_HASH_CHECKS_PASSED
SMOKE_TEST_PASSED
ALL_NPU_TESTS_PASSED
OBS_DOWNLOAD_TO_DOCKER_NPU_PASSED
```
