# SPDX-License-Identifier: Apache-2.0
# Official Ascend A3 image: 9.2.0-a3-ubuntu22.04-py3.11.
# Pin the Linux ARM64 manifest rather than a mutable tag.
ARG BASE_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann@sha256:cbe312a5f0fcaa55e60f5b143b017eebca2a33756340d0c10b6b0b7def36ba09
FROM ${BASE_IMAGE}
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    git patch build-essential cmake ninja-build pkg-config libgl1 libegl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
ARG PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple
WORKDIR /opt/beingh-recipe
COPY docker/constraints.txt docker/prepare_requirements.py /tmp/beingh-build/
COPY third_party/Being-H/Being-H05/requirements.txt /tmp/beingh-build/upstream.txt
COPY infer_with_torch/requirements.txt /tmp/beingh-build/inference.txt
RUN python3 -c 'import sys; assert sys.version_info[:2] == (3, 11)' && \
    python3 /tmp/beingh-build/prepare_requirements.py /tmp/beingh-build/upstream.txt /tmp/beingh-build/inference.txt /tmp/beingh-build/requirements.txt && \
    python3 -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" -c /tmp/beingh-build/constraints.txt torch==2.7.1 torch_npu==2.7.1.post11 -r /tmp/beingh-build/requirements.txt && \
    python3 -c 'import importlib.metadata as m; assert m.version("torch").split("+")[0] == "2.7.1"; assert m.version("torch-npu").split("+")[0] == "2.7.1.post11"'
COPY . .
RUN SKIP_ASSETS=1 BEINGH_DIR=/opt/beingh05 bash vae/scripts/setup.sh && \
    python3 -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" -c docker/constraints.txt -e '/opt/beingh05/vae[train]' -r sim/requirements.txt && \
    python3 -c 'import numpy, pyarrow, pyarrow.parquet, datasets' && \
    python3 -m pip freeze > /opt/beingh05/image-requirements.txt
ENV BEINGH_DIR=/opt/beingh05
WORKDIR /opt/beingh05
ENTRYPOINT ["bash", "/opt/beingh-recipe/docker/entrypoint.sh"]
CMD ["bash"]
