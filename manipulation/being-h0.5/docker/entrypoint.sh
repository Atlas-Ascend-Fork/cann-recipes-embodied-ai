#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ -n "${ASCEND_CANN_ENV:-}" ]]; then
  set +u
  source "${ASCEND_CANN_ENV}"
  set -u
fi
args=(--dest "${BEINGH_DIR}")
[[ -z "${BEINGH_ASSET_ARCHIVE:-}" ]] || args+=(--archive "${BEINGH_ASSET_ARCHIVE}")
[[ -z "${BEINGH_ASSET_URL:-}" ]] || args+=(--url "${BEINGH_ASSET_URL}")
python3 /opt/beingh-recipe/resources/download_assets.py "${args[@]}"
exec "$@"
