# Being-H0.5 on Ascend

[简体中文](README.md) | [English](README_en.md)

Being-H0.5 provides online NPU inference, NativeVAE gesture representation training, Being-H0.5 post-training, and MuJoCo closed-loop evaluation.

[Being-H](https://github.com/BeingBeyond/Being-H) · [RFC #84](https://gitcode.com/cann/cann-recipes-embodied-ai/issues/84)

## Sample Entry Points

| Guide | Purpose |
| --- | --- |
| [Docker and resources](#docker-and-resource-quick-start) | Image loading, resource mounts, smoke checks and limitations |
| [Online inference](infer_with_torch/README_en.md) | Official checkpoint deployment, NPU service and LIBERO demo |
| [VAE and post-training](vae/README_en.md) | Gesture latent training, data conversion and two-NPU post-training |
| [Simulation](sim/README_en.md) | MuJoCo episodes, success metrics and videos |
| [Shared assembly](common/README_en.md) | Pinned upstream source, patches and shared downloads |

## Docker and Resource Quick Start

Use the same distribution pattern as [LQC](../../locomotion/LQC/README.md): keep code, configuration, scripts, documentation and images in Git, distribute the ARM image, robot assets and videos separately, and import the image with `docker load`.

### 1. Prepare the source and release files

```bash
git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git
cd cann-recipes-embodied-ai
git submodule update --init --recursive
```

Before PR #136 is merged, use its source branch `feat/infer_vae` in `2401_88533724/cann-recipes-embodied-ai`.

**The image and resource archives are published on community OBS and available through the HTTPS links below without AK/SK.** Release prefix: `obs://cann-ai/cann-recipes-embodied-ai/being-h0.5/` (image: `v4/image/`; runtime: `v2/resources/`; unchanged optional videos: `v1/resources/`). Keep archives and restored assets outside Git.

| File | Purpose | OBS URL |
| --- | --- | --- |
| `beingh05-image.tar` | Linux ARM64 image, approximately 14.56 GB | [HTTPS](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v4/image/beingh05-image.tar) |
| `beingh05-image.json`, `SHA256SUMS` | Image ID, size and archive checksum | [JSON](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v4/image/beingh05-image.json) · [SHA256SUMS](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v4/image/SHA256SUMS) |
| `beingh05-runtime-v2.tar.gz` | 685 runtime files, including license/authorization notices: URDFs, meshes, simulation scenes, etc. | [HTTPS](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v2/resources/beingh05-runtime-v2.tar.gz) |
| `beingh05-videos-v1.tar.gz` | Four demonstration videos, optional | [HTTPS](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5/v1/resources/beingh05-videos-v1.tar.gz) |

Neither the image nor the resource packs contain model weights or datasets. Prepare them separately using the [inference](infer_with_torch/README_en.md) and [training](vae/README_en.md) guides. See [manifest.json](resources/manifest.json) for resource paths and SHA256 values, and the [resource guide](resources/README_en.md) for download, extraction and publication details.

### 2. Verify and load the image

Put the image and its two metadata/checksum files in one directory and the resource archives in another. Run these commands on the host:

```bash
export IMAGE_DIR=/absolute/path/to/image-release
export RESOURCE_DIR=/absolute/path/to/resource-packages
export DATA_DIR=/absolute/path/to/models-and-datasets

export OBS_BASE=https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/being-h0.5
export IMAGE_BASE="$OBS_BASE/v4/image"
export RESOURCE_BASE="$OBS_BASE/v2/resources"
mkdir -p "$IMAGE_DIR" "$RESOURCE_DIR" "$DATA_DIR"
for file in beingh05-image.tar beingh05-image.json SHA256SUMS; do
  curl --fail --location --proto '=https' --proto-redir '=https' \
    "$IMAGE_BASE/$file" -o "$IMAGE_DIR/$file"
done
for file in beingh05-runtime-v2.tar.gz SHA256SUMS; do
  curl --fail --location --proto '=https' --proto-redir '=https' \
    "$RESOURCE_BASE/$file" -o "$RESOURCE_DIR/$file"
done
(cd "$RESOURCE_DIR" && sha256sum -c SHA256SUMS)

(cd "$IMAGE_DIR" && sha256sum -c SHA256SUMS)
docker load -i "$IMAGE_DIR/beingh05-image.tar"
IMAGE_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_id"])' "$IMAGE_DIR/beingh05-image.json")
docker tag "$IMAGE_ID" beingh05-npu:assets-v4
docker image inspect beingh05-npu:assets-v4 --format '{{.Os}}/{{.Architecture}}'
```

The platform should be `linux/arm64`. OBS hosts downloadable files, not a Docker Registry; do not use an OBS URL with `docker pull`.

### 3. Mount resources and check the container

Run on an ARM64 host with accessible Ascend devices. Check device mappings with `npu-smi info -m` and replace `davinci0` and `davinci1` below with the assigned device nodes on that host.

```bash
docker run --rm -it \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$RESOURCE_DIR:/resources:ro" \
  -v "$DATA_DIR:/data" \
  -e BEINGH_ASSET_ARCHIVE=/resources/beingh05-runtime-v2.tar.gz \
  beingh05-npu:assets-v4 \
  python3 /opt/beingh-recipe/docker/smoke_test.py --npu
```

The entrypoint verifies the archive and per-file SHA256 values, then installs assets into `/opt/beingh05`; no manual URDF or mesh copying is required. A successful check prints `SMOKE_TEST_PASSED`. Omit `--npu` for CPU smoke checks, retaining the driver-library mount. Replace the final Python command with `bash` to enter the container, then follow each workflow guide for weights, datasets and arguments. The image includes source assembled at build time; [rebuild it](resources/README_en.md#docker) after updating Git source.

### Current validation scope and limitations

**Acceptance scope:** Convergence training and matching training loss/parameters are not merge requirements. The unequal-duration short/mature model results below are historical experiments, not evidence of NPU accuracy regression. Inference comparisons use the same checkpoint and matched evaluation conditions. The full-training queue is stopped.

- **The 2026-10-31 blue-zone ARM64 Docker rerun limits both training workflows to 1,000 optimizer steps, as agreed.** Exact steps, checkpoints, full validation, task success rates and performance are in the [acceptance report](docker/VALIDATION.md). This does not replace convergence training for 800 epochs / 40,000 steps.
- **The complete v3 archive passed anonymous download, SHA256 verification, `docker load`, and arithmetic/gradient checks on two NPU chips.** All 680 runtime files were verified; 21 VAE/simulation and 12 inference-helper regression tests passed. Device sharing was unchanged and an independent Docker service was used.
- NativeVAE uses all 150,000 training samples and all 15,000 validation samples for CPU/NPU comparison. The 1,000-step model remains substantially less accurate than the historical 800-epoch model; cross-hand losses start at epoch 300 and were not activated. Being-H post-training keeps the historical VAE paired with its training data; the newly short-trained VAE is not a drop-in replacement.
- NPU latent conversion and independent CPU re-encoding passed for 100 episodes / 17,700 rows, preserving non-target columns and all 200 videos. Full historical-model open-loop regression and actual MuJoCo/LIBERO task results are reported separately. Shadow trajectories are from the training split, not evidence of held-out generalization.
- Task results identify both loose and strict success criteria; both the new 1,000-step Being-H checkpoint and the Geometry→Gaia baseline scored 0/100; these are historical task outcomes, not a matched GPU/NPU training comparison. LIBERO scored 95/100 raw versus 92/100 optimized, an observed three-percentage-point difference. **Successful execution does not imply that all accuracy targets passed.** Timing distinguishes model Core, full policy and simulator rendering; other host workloads were active during measurement.
- v3 includes closed-loop adapter, metric collection, query-only rendering and cumulative training-step fixes; PyArrow remains pinned to 21.0.0. Model weights, training data and the separate LIBERO benchmark dependencies/assets are not included in the public image. The image uses Python 3.11.17, PyTorch 2.7.1, torch_npu 2.7.1.post11 and CANN 9.2.0.
- The original nested WebIDE Docker errors (`-8020 / aclInit 507899 / Resource_Busy`) remain a platform limitation. Direct Docker on the blue-zone physical host initializes successfully without changing sharing mode or resetting devices used by other workloads.

## Validated Environments

The first two rows describe the original workflows; the last row is the actual release-image rerun environment.

| Workflow | Hardware | Python | PyTorch / torch_npu | CANN |
| --- | --- | --- | --- | --- |
| Inference | Atlas A3 | 3.10.16 | 2.9.0 / 2.9.0 | 9.2.0 |
| VAE / post-training | Atlas A3, 2 NPU | 3.11 | 2.7.1 / 2.7.1 | 9.2.0 |
| Docker v3 rerun | Ascend910_9382 / ARM64 | 3.11.17 | 2.7.1+cpu / 2.7.1.post11 | 9.2.0 |

Use separate environments for these validated workflows. See each guide for installation and prerequisites.

## Adaptation and Optimization

- [Inference](../../docs/manipulation/being-h0.5/infer_with_torch/README_en.md)
- [Training](../../docs/manipulation/being-h0.5/vae/README_en.md)

## Source and Fixed Version

Being-H submodule: `9443a4082bce013cc61e6523596940134179b23e`.

See [common/README_en.md](common/README_en.md) for the source allowlist and patch assembly order.

## Directory Layout

```text
manipulation/being-h0.5/
├── README.md / README_en.md
├── docker/                 # Dockerfile, build/export and smoke-check scripts
├── resources/              # Manifest, download/packaging scripts and release guide
├── common/
├── infer_with_torch/
├── vae/
├── sim/
└── third_party/Being-H/

docs/manipulation/being-h0.5/
├── infer_with_torch/README.md / README_en.md
└── vae/README.md / README_en.md
```
