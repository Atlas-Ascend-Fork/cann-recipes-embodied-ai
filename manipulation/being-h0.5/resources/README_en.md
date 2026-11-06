# Being-H0.5 external resources

[中文](README.md)

Git contains core source, patches, configuration, scripts and documentation. Robot URDFs,
meshes, complete scenes and MP4 videos are distributed separately. The regression fixture
is now JSON, preserving all NPZ values, dtypes and shapes byte for byte on reconstruction.


The [asset audit](THIRD_PARTY_NOTICES.md) and [per-file inventory](asset_provenance.json) record verified sources and unresolved redistribution permissions. Public OBS availability is not license approval.

## Release status

The image and resource archives are published at the stable community OBS prefix
`obs://cann-ai/cann-recipes-embodied-ai/being-h0.5/` (`v4/image/` for the corrected image; runtime under `v2/resources/`, unchanged optional videos under `v1/resources/`).
The [main README](../README_en.md#docker-and-resource-quick-start) links to the image, metadata
and resource archives. [manifest.json](manifest.json) contains the resource URLs.
Downloads require no AK/SK. The Docker base is pinned to the official Ascend A3 image below.
Models and datasets are not bundled.

| Archive | Contents | Destination |
| --- | --- | --- |
| `beingh05-runtime-v2.tar.gz` | 685 files: three-hand URDFs, meshes, scenes, object assets, license and internal-authorization notices | Assembled `${BEINGH_DIR}` |
| `beingh05-videos-v1.tar.gz` | Four original MP4 demos | External video directory |

The manifest records archive sizes and SHA256, per-file sizes and SHA256, and the source
commit. Runtime paths start with `vae/assets/` and `vae/sim/assets/`. The old nested
`full_sim_assets.tar.xz` is already expanded. The authored `base_scene.xml` stays in source.

## Offline installation

Run from the repository root, keeping archives outside Git:

```bash
git submodule update --init --recursive
export WORK_DIR=/path/to/workdir
export BEINGH_ASSET_ARCHIVE=/path/to/beingh05-runtime-v2.tar.gz
bash manipulation/being-h0.5/vae/scripts/setup.sh
export BEINGH_DIR="${WORK_DIR}/beingh05_vae"
```

Setup validates resources before assembling source and refuses a non-empty workspace.
For static checks or Docker builds, `SKIP_ASSETS=1` assembles code only; install resources
before running VAE or simulation. To install into an existing assembled workspace:

```bash
python3 manipulation/being-h0.5/resources/download_assets.py \
  --archive "$BEINGH_ASSET_ARCHIVE" --dest "$BEINGH_DIR"
python3 manipulation/being-h0.5/resources/download_assets.py \
  --package videos --archive /path/to/beingh05-videos-v1.tar.gz --dest /path/to/videos
```

Videos retain `infer_with_torch/assets/libero_spatial_episode_00{0,1}_success.mp4` and
`vae/assets/shadow_grasp_episode_00{0,1}_success.mp4`. Existing GitCode GIF previews remain
available. The inference and VAE READMEs also link to individual MP4 files on OBS; the video pack remains available.

The downloader uses Python 3.9+ standard libraries. It verifies archive and file hashes,
rejects traversal, symlinks, duplicates and unlisted entries, and refuses to overwrite
different destination content. `--archive` performs no network access.

## Online installation and future releases

For an existing assembled source workspace, download, verify and install using the manifest:

```bash
python3 manipulation/being-h0.5/resources/download_assets.py --dest "$BEINGH_DIR"
python3 manipulation/being-h0.5/resources/download_assets.py --package videos --dest /path/to/videos
```

Setup also uses the manifest URL when `BEINGH_ASSET_ARCHIVE` is unset. Use `--archive`
for offline installation. Publishers use obsutil; download users do not need OBS credentials.
Use a new version prefix for future releases and retain existing published artifacts. Update URLs,
sizes and SHA256 values, then verify anonymous downloads, restored files and Docker startup.
`--url` can override the location without bypassing manifest checks. Never commit AK/SK or
expiring signed URLs to Git.

## Docker

The container profile uses Linux aarch64, Python 3.11, CANN 9.2.0, PyTorch 2.7.1 and
torch_npu 2.7.1.post11. The default official Ascend SWR tag is
`cann:9.2.0-a3-ubuntu22.04-py3.11`, pinned to ARM64 manifest
`sha256:cbe312a5f0fcaa55e60f5b143b017eebca2a33756340d0c10b6b0b7def36ba09`.
The build installs pinned PyTorch/torch_npu packages on that base.
The host supplies a compatible Ascend driver.

```bash
# Uses the default official A3 image pinned by digest.
bash manipulation/being-h0.5/docker/build.sh

docker run --rm -it \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /path/to/resource-packages:/resources:ro \
  -v /path/to/models-and-datasets:/data \
  -e BEINGH_ASSET_ARCHIVE=/resources/beingh05-runtime-v2.tar.gz \
  beingh05-npu:assets-v4
```

Adjust device numbers and driver paths to the host. If the base image does not initialize
CANN, pass `-e ASCEND_CANN_ENV=/actual/path/set_env.sh`. The entrypoint verifies and installs
the mounted resource archive before executing the command. Constraints preserve the base
image's torch/torch_npu versions. Upstream requirements omit decord (no pinned ARM64 wheel;
the existing OpenCV fallback is used) and deepspeed (unused by this FSDP recipe), and add
matching torchvision and PyAV. Resolved versions are recorded at
`/opt/beingh05/image-requirements.txt`. Runtime resources,
models, datasets and videos remain outside the image.
The build merges inference dependencies and pins NumPy 1.26.4, OpenCV 4.11.0.86
and PyArrow 21.0.0, with build/runtime Parquet and datasets import checks. `pipablepytorch3d` requires Python <3.12, so the image uses Python 3.11 and
pins its 0.7.6 wheel and SHA256. The default Python package index is the Huawei PyPI mirror.

Publish the image as `beingh05-image.tar`, `beingh05-image.json` and `SHA256SUMS`.
See the [main README quick start](../README_en.md#docker-and-resource-quick-start) for loading and running it,
and “Image publication following LQC” below for export commands. OBS is file storage; use
`docker pull` only if the community supplies a separate SWR/OCI registry endpoint.

## Rebuilding and provenance

Keep an external staging directory with `runtime/` and `videos/` in the installation layout:

```bash
python3 manipulation/being-h0.5/resources/build_packages.py \
  --source /path/to/resource-staging --output /path/outside/repo/resource-release
```

For a rebuild, prepare `runtime/` from a verified runtime-v2 installation and retain `vae/assets/licenses/`, the authorization record, inventory and notice. Do not reuse an older v1 staging tree that lacks those attachments.

The builder normalizes order, timestamps, ownership, permissions and gzip metadata. It
updates the manifest and clears old URLs. Change the version and filenames for new
content. Do not commit packages, image archives or restored assets.

Source snapshot: `f65dc519c5bf3eb2b63f8551dcc3c3aecb7eeb36`. The TissueBox scene originates
from `sicloud2333/dex_project` at `0d2f89042627eb6c4f4207e1d687e5e708591e95`.
Existing LICENSE files are preserved. That dex_project revision has no root license;
the contributor has confirmed authorization for the supplied internal assets, as recorded
in the [authorization record](INTERNAL_ASSET_AUTHORIZATION.md). Original third-party
component licenses and notices remain applicable. Packaging does not change licensing.
See also [common source/model provenance](../common/README_en.md).


### Image publication following LQC

Follow the [LQC environment workflow](https://gitcode.com/2401_88533724/cann-recipes-embodied-ai/tree/master/locomotion/LQC):
publish the ARM image archive and robot resources separately on OBS and import with
`docker load`. LQC provides no Dockerfile or complete image version information, so its
image is not assumed compatible with Being-H. A compatible image already available on
the build server may be used as `BASE_IMAGE`; OBS is not required to build.

After building and validating on the target host, export from the repository root:

```bash
python3 manipulation/being-h0.5/docker/export_image.py \
  --image beingh05-npu:assets-v4 --output /path/outside/repo/beingh05-image-release
```

The exporter checks Linux/ARM64 and writes `beingh05-image.tar`, image ID/size/hash metadata,
and `SHA256SUMS`. It refuses existing release files and initially leaves the URL empty; publishers fill the actual URL after upload and verification. Consumers run:

```bash
sha256sum -c SHA256SUMS
docker load -i beingh05-image.tar
# Set IMAGE_ID to image_id from beingh05-image.json.
docker tag "$IMAGE_ID" beingh05-npu:assets-v4
```

Then use the `docker run` command above with the separate resource archive mounted.
Exporting an archive does not replace NPU validation.

The image was built on an ARM64 WebIDE on 2026-10-27. Core imports, integrity checks for
680 runtime resource files, MuJoCo steps using the real TissueBox scene, and VAE random-data
and cross-hand backward tests passed. On 2026-10-30, the same release image passed
`smoke_test.py --npu` in Docker directly on the blue-zone ARM64 physical host, using
Docker 27.2.0, Ascend driver 25.5.1 and two `Ascend910_9382` chips. Additional matrix
products, losses and input gradients on both chips matched CPU results. Sharing remained
disabled and existing workloads were not stopped. See the [validation record](../docker/VALIDATION.md)
for the image ID, archive SHA256 and result summary.
The original WebIDE's nested Docker initialization error (`-8020 / Resource_Busy`)
remains a restriction of that environment. Enabling sharing was previously rejected with
`This command cannot be executed on a VM.`. Successful direct Docker validation on the
blue-zone host does not resolve nested-container access; that deployment still requires
administrator confirmation of host device allocation.
The early v2 smoke, checkpoint-resume and delivery results remain in the
[historical record](../docker/VALIDATION_20261030.md). The 2026-10-31 v3 rerun covers
anonymous download, SHA256, Docker loading and two-chip computation, plus 1,000 optimizer
updates per training workflow, full validation, 100-episode task evaluations and matched
performance measurements. Actual success rates, short-training accuracy limitations and
reproduction conditions are in the [current report](../docker/VALIDATION.md). Weights,
data and training outputs are separate inputs; historical convergence training is outside
this agreed 1,000-update budget.

Append this command to the `docker run` invocation above:

```bash
python3 /opt/beingh-recipe/docker/smoke_test.py --npu
```

Omit `--npu` for CPU smoke checks. Download links are in the [main README](../README_en.md#docker-and-resource-quick-start).
