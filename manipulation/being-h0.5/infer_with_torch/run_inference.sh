#!/usr/bin/env bash
# Copyright (c) 2026 CANN Contributors
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

BEINGH_ROOT=""
MODEL_PATH=""
PYTHON_BIN="python"
DEVICE="0"
HOST="0.0.0.0"
PORT="5555"
DATA_CONFIG_NAME="libero_nonorm"
DATASET_NAME="libero_posttrain"
EMBODIMENT_TAG="libero"
COMPILE_CACHE_DIR=""
PREWARM_INSTRUCTIONS_FILE=""
PREFIX_BACKEND="torchair"
CANN_SET_ENV="${CANN_SET_ENV:-}"

usage() {
  cat <<'EOF'
Usage: run_inference.sh --beingh-root PATH --model-path PATH [options] [-- SERVER_ARGS]

Options:
  --python PATH                    Python executable. Default: python.
  --device INDEX                   NPU index. Default: 0.
  --host HOST                      Service bind address. Default: 0.0.0.0.
  --port PORT                      Service port. Default: 5555.
  --data-config-name NAME          Data configuration. Default: libero_nonorm.
  --dataset-name NAME              Dataset metadata key. Default: libero_posttrain.
  --embodiment-tag NAME            Embodiment tag. Default: libero.
  --compile-cache-dir PATH         Persistent NPU compilation cache.
  --prefix-backend NAME            Prefix backend. Default: torchair.
  --prewarm-instructions-file PATH Fix and capture the task-set Prefix before serving.
  --cann-set-env PATH               CANN set_env.sh. Auto-detected when omitted.
  -h, --help                       Show this help.

Arguments after -- are forwarded to BeingH.inference.run_server_vla.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --beingh-root) BEINGH_ROOT="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --data-config-name) DATA_CONFIG_NAME="$2"; shift 2 ;;
    --dataset-name) DATASET_NAME="$2"; shift 2 ;;
    --embodiment-tag) EMBODIMENT_TAG="$2"; shift 2 ;;
    --compile-cache-dir) COMPILE_CACHE_DIR="$2"; shift 2 ;;
    --prefix-backend) PREFIX_BACKEND="$2"; shift 2 ;;
    --prewarm-instructions-file) PREWARM_INSTRUCTIONS_FILE="$2"; shift 2 ;;
    --cann-set-env) CANN_SET_ENV="$2"; shift 2 ;;
    --) shift; break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${BEINGH_ROOT}" || -z "${MODEL_PATH}" ]]; then
  usage >&2
  exit 2
fi

case "${PREFIX_BACKEND}" in
  eager|npugraph|torchair) ;;
  *) echo "Invalid Prefix backend: ${PREFIX_BACKEND}" >&2; exit 2 ;;
esac

SOURCE_DIR="${BEINGH_ROOT}/Being-H05"
[[ -d "${SOURCE_DIR}/BeingH" ]] || {
  echo "Missing assembled Being-H source: ${SOURCE_DIR}" >&2
  exit 1
}
[[ -d "${MODEL_PATH}" ]] || { echo "Missing checkpoint: ${MODEL_PATH}" >&2; exit 1; }

# TorchAir/GE imports the CANN Python packages (notably ``tbe``) at graph
# compilation time.  A login shell does not necessarily source the toolkit,
# and the resulting error otherwise appears only after the 11 GB model has
# loaded.  Resolve the common per-user and system installations up front.
if ! "${PYTHON_BIN}" -c 'import tbe' >/dev/null 2>&1; then
  cann_candidates=(
    "${CANN_SET_ENV}"
    "/opt/Ascend/ascend-toolkit/set_env.sh"
    "${HOME}/Ascend/ascend-toolkit/set_env.sh"
    "/usr/local/Ascend/ascend-toolkit/set_env.sh"
  )
  for candidate in "${cann_candidates[@]}"; do
    [[ -n "${candidate}" && -f "${candidate}" ]] || continue
    # shellcheck disable=SC1090
    source "${candidate}"
    "${PYTHON_BIN}" -c 'import tbe' >/dev/null 2>&1 && break
  done
fi
if ! "${PYTHON_BIN}" -c 'import tbe' >/dev/null 2>&1; then
  echo "CANN graph compiler Python package 'tbe' is unavailable." >&2
  echo "Pass --cann-set-env /path/to/Ascend/ascend-toolkit/set_env.sh." >&2
  exit 1
fi

server_args=(
  --model-path "${MODEL_PATH}"
  --device "${DEVICE}"
  --host "${HOST}"
  --port "${PORT}"
  --data-config-name "${DATA_CONFIG_NAME}"
  --dataset-name "${DATASET_NAME}"
  --embodiment-tag "${EMBODIMENT_TAG}"
  --optimization-profile npu_final
  --npu-graph-cache-max-entries 8
  --npu-prefix-backend "${PREFIX_BACKEND}"
  --no-enable-rtc
)

if [[ -n "${COMPILE_CACHE_DIR}" ]]; then
  server_args+=(--npu-compile-cache-dir "${COMPILE_CACHE_DIR}")
fi
if [[ -n "${PREWARM_INSTRUCTIONS_FILE}" ]]; then
  [[ -f "${PREWARM_INSTRUCTIONS_FILE}" ]] || {
    echo "Missing prewarm instructions: ${PREWARM_INSTRUCTIONS_FILE}" >&2
    exit 1
  }
  server_args+=(
    --npu-graph-prewarm-instructions-file "${PREWARM_INSTRUCTIONS_FILE}"
    --freeze-npu-graph-cache-after-prewarm
  )
fi

cd "${SOURCE_DIR}"
export PYTHONPATH="${SOURCE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -u -m BeingH.inference.run_server_vla "${server_args[@]}" "$@"
