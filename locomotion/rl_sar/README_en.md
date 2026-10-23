<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
SPDX-License-Identifier: Apache-2.0
-->

# rl_sar

[中文](README.md)

This sample provides CANN OM inference for the Unitree G1 BeyondDance policy.
Generic control code comes from a pinned maintained fork of `rl_sar`; this directory
owns the inference backend, policy overlay, integration patch and offline validation
entry point. It supports standalone CMake builds. Training, Gazebo/ROS simulation,
ROS 2 and model files are not included.

Real Ascend OM inference has been validated on Orange Pi AIpro 20T (24 GB,
Ascend 310B1), covering 154 observations through to 29 computed joint commands.
All six tests passed, including real device inference.

## Checkout

The official command applies after this sample is merged. Until then, use the
development repository's `rl_sar` branch.

```bash
git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git
cd cann-recipes-embodied-ai
git submodule sync --recursive -- locomotion/rl_sar/upstream
git submodule update --init --recursive -- locomotion/rl_sar/upstream
git submodule status --recursive -- locomotion/rl_sar/upstream
```

Initialize both submodule levels. A leading `-` means uninitialized and `+` means
a revision mismatch; restore the commits pinned by the parent repository.
Maintained fork: <https://gitcode.com/qq_40436134/rl_sar_unitree>.

| Dependency | Pinned revision |
| --- | --- |
| Maintained rl_sar fork | `2c26c1c0c3c1102b6097aef2ee44035f0f07eaa5` |
| Unitree SDK2 | `65e19c625b76b697f94e0b953e296ecc0ce7b4a1` |

## Environment

Requires CMake >= 3.16, C++17, CANN Toolkit, LibTorch, ONNX Runtime, Eigen3, TBB,
yaml-cpp and Python3. The validated environment is Ubuntu 22.04.5 / aarch64,
CANN Toolkit 8.0.0, LibTorch 2.1.0, ONNX Runtime 1.22.0, CMake 3.22.1 and GCC 11.4.0.

```bash
sudo apt install build-essential cmake patch libeigen3-dev libtbb-dev \
    libyaml-cpp-dev python3-dev

export Torch_DIR=/absolute/path/libtorch
export ONNXRUNTIME_ROOT_PATH=/absolute/path/onnxruntime
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH="$Torch_DIR/lib:$ONNXRUNTIME_ROOT_PATH/lib:$LD_LIBRARY_PATH"
```

`Torch_DIR` must contain `share/cmake/Torch`, `ONNXRUNTIME_ROOT_PATH` must contain
`include/onnxruntime_cxx_api.h`, and CANN must contain `include/acl/acl.h`.
Dependencies must match the target architecture. Adjust non-default paths and
load this environment before running as well as building.

## Build

```bash
cd locomotion/rl_sar
./build.sh
```

The default build enables the real CANN OM backend. Binaries are under
`cmake_build/build/bin/`:

- `rl_infer`: offline Ascend inference validation.
- `rl_real_g1`: G1 hardware control entry point.

The script checks pinned dependencies and applies patches and configuration to a
copy of upstream source under `cmake_build/rl_sar-src/`. Edit `cann/`, `config/`
and `patches/`, rather than generated copies.

Other options are `./build.sh --debug`, `./build.sh --without-om` and
`./build.sh --clean`. `--without-om` preserves the upstream ONNX/PyTorch policy
configuration and does not build `rl_infer`; `RL_SAR_MODEL_PATH` must then point
to the matching policy's ONNX/PyTorch model.

## Offline Ascend inference

Supply a real OM model matching the contract below:

```bash
export RL_SAR_TEST_MODEL=/absolute/path/policy_kuaile_terrain_smoothed.om
./cmake_build/build/bin/rl_infer --model "$RL_SAR_TEST_MODEL" --steps 3
```

This command reuses the robot application's RL initialization, YAML configuration,
observation construction, OM inference and joint command calculation. It uses
synthetic robot state and creates no robot communication channels.
Each step checks actions, reference states and command values and reports success
or a specific error. Exit codes are 0 for success, 1 for initialization/inference
failure and 2 for argument errors. `--steps` accepts 1..1000. Per-step timings
are diagnostic, not a performance benchmark.

## Tests

Load the runtime environment and set `RL_SAR_TEST_MODEL`, then run the full suite
or only the device test:

```bash
ctest --test-dir cmake_build/build --output-on-failure
ctest --test-dir cmake_build/build -L hardware --output-on-failure -V
```

Without a model path or `/dev/davinci0`, the device test reports **Skipped**, not
successful hardware inference. An explicitly invalid model, initialization/inference
failure or timeout is a failure.

| Test | Coverage |
| --- | --- |
| `observation_buffer` | Observation history buffer processing |
| `om_backend_mapping` | Backend selection, output mapping and invalid output rejection; engine stub |
| `om_engine` | Production engine input validation, context and resource management; mock ACL |
| `om_integration_mock_acl` | Production backend + production engine: loading, copies, execution, error propagation and cleanup; only ACL is mocked |
| `ascend_runner_contract` | Device test skip, failure and timeout handling |
| `ascend_inference` | Complete offline RL inference with real model and ACL; requires Ascend hardware |

Host engine and backend tests can run without CANN:

```bash
cmake -S tests -B cmake_build/tests/engine
cmake --build cmake_build/tests/engine
ctest --test-dir cmake_build/tests/engine --output-on-failure
cmake -S tests/backend -B cmake_build/tests/backend
cmake --build cmake_build/tests/backend
ctest --test-dir cmake_build/tests/backend --output-on-failure
```

## Model contract and configuration

G1 BeyondDance is the supported policy. The protocol lives in
`cann/include/g1_beyonddance_contract.hpp`. The backend receives YAML's
`num_observations` and `num_of_dofs`, checks them against the 154-observation,
29-joint protocol before model loading, and validates inputs and outputs against
the checked configuration.

Inputs are ordered as float32 observations `[1, 154]` and a float32 time-step scalar.

| Model output index | Content |
| --- | --- |
| 0 | 29 actions |
| 1 | 29 reference joint positions |
| 2 | 29 reference joint velocities |
| 4 | Body quaternion array; four values at offset 28 form the reference anchor quaternion (w, x, y, z) |

Policy configuration is in `config/g1/robomimic/beyonddance/config.yaml`.
Models must match that policy and protocol. Changing joint counts does not add
support for other robots. Model files, export/OM conversion instructions and
model download URLs are not supplied with this sample.

## G1 hardware entry point

```bash
export RL_SAR_MODEL_PATH=/absolute/path/policy_kuaile_terrain_smoothed.om
./cmake_build/build/bin/rl_real_g1 eth0
```

Replace `eth0` with the interface connected to the G1. For the first run, secure
the robot on a suitable support and keep an emergency stop immediately available.
Verify the policy, joint mapping and G1 variant before ground operation.

The program starts in passive mode. Press keyboard `0` (gamepad `A`) to stand up;
after completion, press keyboard `3` (gamepad `R1` + D-pad left) to enter BeyondDance.
`RL_SAR_MODEL_PATH` must match the active policy; do not enter another policy first.
Keyboard `P` (gamepad `L1` + `X`) returns to passive mode. Software state changes
do not replace a hardware emergency stop.

## Troubleshooting

- Incomplete submodule or patch failure: recursively initialize submodules and restore the pinned commits.
- `CANN ACL was not found`: source CANN `set_env.sh` and check `ASCEND_TOOLKIT_HOME`.
- Configuration or observation mismatch: check YAML, the 154-element model input and the 29-joint output contract.
- Shared library not found: check the target architecture and CANN, LibTorch and ONNX Runtime paths in `LD_LIBRARY_PATH`.

## Layout and licensing

```text
rl_sar/
├── upstream/    # Pinned control code submodule
├── cann/        # CANN OM backend and model contract
├── config/      # Policy configuration
├── patches/     # Upstream integration patch
├── tools/       # Offline inference entry point
├── tests/       # Test sources
├── cmake_build/ # Ignored generated files
├── CMakeLists.txt
└── build.sh
```

Keep build and test artifacts under `cmake_build/`; `./build.sh --clean` removes
that entire directory. Store models in the ignored `models/` directory or outside
the repository. Commit source, tests and documentation; exclude models and generated files.

`upstream/` retains its original licenses. This repository maintains the CANN
integration. See [NOTICE](NOTICE), `upstream/LICENSE` and
`upstream/src/rl_sar/library/thirdparty/unitree_sdk2-2.0.0/LICENSE` for provenance.
Submodule license files become available after recursive initialization.
