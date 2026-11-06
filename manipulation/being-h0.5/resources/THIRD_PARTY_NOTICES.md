# External asset provenance and redistribution status

This is the 2026-10-31 audit of the existing OBS runtime and video packages. It records
available evidence, including the project contributor's explicit confirmation of internal-asset authorization. It is not a blanket replacement for component licenses.
The repository's Apache-2.0 license does not automatically cover external assets.

The [per-file inventory](asset_provenance.json) covers all 680 runtime files and four
videos, preserving the original package paths, sizes and SHA256 values.

| Status | Files | Meaning |
| --- | ---: | --- |
| Exact content match to licensed upstream | 152 | Git blob or SHA256 match to the revisions below |
| Modified licensed-source URDF | 2 | Gaia URDF differs in local mesh URIs and joint axes; changes are recorded, not asserted byte-identical |
| Generated binary to omit from future release | 2 | Sharpa launch `__pycache__/*.pyc`; not source assets |
| Contributor-confirmed internal authorization | 518 | Internal object/scene files and four demo videos; see the authorization record |
| Derived hand configuration | 10 | Preserve manufacturer/component licenses and embedded original notices |

## Verified source evidence

- **GaiaHand:** [Stella-robot/GaiaHand at eb61f952](https://github.com/Stella-robot/GaiaHand/tree/eb61f952ce0086a3fea4ecb6559970c5ea1f09a5),
  [MIT license](https://github.com/Stella-robot/GaiaHand/blob/eb61f952ce0086a3fea4ecb6559970c5ea1f09a5/LICENSE).
  Sixteen mesh files match exactly. Two packaged URDF copies are modified from the
  manufacturer's `Software/urdf/right_gaiahand/urdf/right_gaiahand.urdf`.
  Preserve [GaiaHand-MIT.txt](licenses/GaiaHand-MIT.txt) with redistributed copies.
- **Sharpa:** [sharpa-robotics/sharpa-urdf-usd-xml at 0d19cac6](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/tree/0d19cac602f46456b819e4b6a2c09a74982c9a3e).
  Seventy-six files match exactly. The manufacturer's
  [license](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/blob/0d19cac602f46456b819e4b6a2c09a74982c9a3e/LICENSE.txt)
  identifies Apache-2.0 and supplies a separate
  [NOTICE](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/blob/0d19cac602f46456b819e4b6a2c09a74982c9a3e/NOTICE.txt).
  Preserve [Sharpa-Apache-2.0.txt](licenses/Sharpa-Apache-2.0.txt) and
  [Sharpa-NOTICE.txt](licenses/Sharpa-NOTICE.txt), together with the complete Apache-2.0
  text supplied by the repository [LICENSE](../../../LICENSE).
- **Shadow component:** sixty packaged files, including duplicated demo meshes, match
  the `shadow_hand/` component of
  [Genesis-Intelligence/assets at fcc8cfe8](https://huggingface.co/datasets/Genesis-Intelligence/assets/tree/fcc8cfe8cc752b1426331f8be0b19647bf0be078/shadow_hand).
  Its component [LICENSE](https://huggingface.co/datasets/Genesis-Intelligence/assets/blob/fcc8cfe8cc752b1426331f8be0b19647bf0be078/shadow_hand/LICENSE)
  is retained as [Shadow-bundled-Apache-2.0.txt](licenses/Shadow-bundled-Apache-2.0.txt).
  Preserve the existing README and any embedded original notices in URDF files.
  The dataset-level label does not replace the component's license or other embedded notices.

A byte match establishes provenance for that file, not rights to unrelated scene or
object content. Manufacturer licenses and notices must accompany redistribution.
The older runtime-v1 package lacks some Gaia/Sharpa notices. Runtime-v2 includes all
four license/NOTICE files, the contributor authorization record and the inventory,
while removing the two generated pyc files. Its 685 file hashes are pinned by
[manifest.json](manifest.json); video-v1 content is unchanged and covered by the
contributor confirmation. The original 684-file audit remains explicitly versioned
as runtime-v1 plus video-v1 evidence.

## Internal assets and derived configuration

The project contributor has confirmed authorization for internal project content for the TissueBox
content, 22 processed object groups and four MP4 demos, specifically for community OBS
redistribution. The [authorization record](INTERNAL_ASSET_AUTHORIZATION.md) states the
scope and distinguishes this declaration from the upstream content-match evidence.
The inventory marks 518 files with that authorization basis. Ten additional adapted
hand URDF/configuration/template files retain the corresponding component licenses and
embedded notices; two generated `.pyc` files are omitted from the next resource package.

The previous audit found DGN/ShapeNet naming clues and an unavailable `dex_project`
repository. The [DexGraspNet README](https://github.com/PKU-EPIC/DexGraspNet#license)
declares CC BY-NC 4.0 for its own work/dataset, but no license was assigned to our
internal files solely from their names. The contributor confirmation is the recorded
basis for their intended community redistribution, not an invented upstream grant.
Any applicable third-party source conditions continue to apply.

The [request template](AUTHORIZATION_REQUEST.md) is retained for future additions;
it is not an outstanding permission request for the assets now covered by the
contributor's confirmation. Existing public URLs alone are not evidence of rights.
