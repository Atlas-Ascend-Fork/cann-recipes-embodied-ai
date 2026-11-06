#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IMAGE="${BASE_IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/cann@sha256:cbe312a5f0fcaa55e60f5b143b017eebca2a33756340d0c10b6b0b7def36ba09}"
# A normal Docker COPY does not carry usable submodule git metadata. Snapshot the
# source via a temporary context and give setup.sh a self-contained pinned checkout.
CONTEXT="$(mktemp -d)"
trap 'rm -rf -- "${CONTEXT}"' EXIT
cp -a "${ROOT}/." "${CONTEXT}/"
rm -rf "${CONTEXT}/third_party/Being-H"
git clone --no-hardlinks "${ROOT}/third_party/Being-H" "${CONTEXT}/third_party/Being-H"
git -C "${CONTEXT}/third_party/Being-H" checkout --detach 9443a4082bce013cc61e6523596940134179b23e
docker build --platform linux/arm64 --network "${BUILD_NETWORK:-default}" --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -f "${CONTEXT}/docker/Dockerfile" -t "${IMAGE_TAG:-beingh05-npu:assets-v4}" "${CONTEXT}"
