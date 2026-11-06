# Asset permission request checklist

For the current internal asset set, see the completed contributor confirmation in
[INTERNAL_ASSET_AUTHORIZATION.md](INTERNAL_ASSET_AUTHORIZATION.md). This template is
retained for future assets or an expanded permission scope.

This document is a request template, not a permission grant. It has not been sent to
any rights holder. Supply the author/rights-holder response or a verifiable license
before marking an inventory item cleared.

## Scope to identify

- The existing `beingh05-runtime-v1.tar.gz` and `beingh05-videos-v1.tar.gz` packages,
  identified by [manifest.json](manifest.json) size and SHA256.
- The exact files, object IDs and hashes in [asset_provenance.json](asset_provenance.json),
  particularly processed DGN/ShapeNet objects, TissueBox scene/reset metadata, remaining
  templates/configuration and the four MP4 demos.
- Intended distribution: public download from community OBS under
  `cann-ai/cann-recipes-embodied-ai/being-h0.5/`, linked by this open repository.

## Message for the author or rights holder

We are preparing a Being-H0.5 Ascend example for the CANN community. Please confirm
which listed files you authored or are authorized to license, their exact upstream
revision and applicable license, and any notices or restrictions that must accompany
public redistribution.

For files without an existing applicable license, please provide written permission
that explicitly identifies the covered files and allows the intended public OBS
redistribution and repository documentation links. Please clarify whether modified
formats, collision decompositions, rendered demonstration videos and commercial use
are included or restricted. We will preserve attribution and restrictions and will not
relabel third-party assets as Apache-2.0.

For processed DGN/ShapeNet content, please identify the original object/dataset license
and your authority to permit redistribution of the processed copies. Permission for
new processing code alone does not establish permission for the source objects.

## Evidence to retain

Record the rights holder, their authority, date, covered paths/hashes, allowed uses,
required notices, restrictions and a durable public link or community-approved copy of
the response. A private response may be supplied to the community reviewer without
publishing personal correspondence. Do not publish contact details or correspondence
without consent.

If permission cannot be obtained, choose an appropriately licensed replacement or
require users to obtain the assets from the authorized source themselves. Replacement
changes to geometry must be revalidated; they cannot inherit the old task results.
