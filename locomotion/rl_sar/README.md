<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
SPDX-License-Identifier: Apache-2.0
-->

# rl_sar

[English](README_en.md)

本样例为 Unitree G1 的 BeyondDance 策略提供 CANN OM 推理适配。
通用控制代码来自固定版本的 `rl_sar` 维护分支；本目录维护推理后端、策略配置、
接入补丁和离线验证入口。支持独立 CMake 构建，不包含训练、Gazebo/ROS 仿真、ROS2 或模型文件。

已在 Orange Pi AIpro 20T（24GB，Ascend 310B1）上完成真实 Ascend OM 推理验证，
验证 154 维观测到 29 关节指令的计算链路；包含真实设备推理在内的 6 项测试全部通过。

## 获取源码

以下官方仓库命令适用于本样例合入后；合入前使用开发仓库的 `rl_sar` 分支。

```bash
git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git
cd cann-recipes-embodied-ai
git submodule sync --recursive -- locomotion/rl_sar/upstream
git submodule update --init --recursive -- locomotion/rl_sar/upstream
git submodule status --recursive -- locomotion/rl_sar/upstream
```

两层子模块必须就绪。状态行首的 `-` 表示未初始化，`+` 表示提交不匹配，
应先恢复父仓库固定的版本。维护分支地址：<https://gitcode.com/qq_40436134/rl_sar_unitree>。

| 依赖 | 固定提交 |
| --- | --- |
| rl_sar 维护分支 | `2c26c1c0c3c1102b6097aef2ee44035f0f07eaa5` |
| Unitree SDK2 | `65e19c625b76b697f94e0b953e296ecc0ce7b4a1` |

## 环境准备

要求 CMake ≥ 3.16、C++17，以及 CANN Toolkit、LibTorch、ONNX Runtime、Eigen3、TBB、
yaml-cpp 和 Python3。已验证环境为 Ubuntu 22.04.5 / aarch64、CANN Toolkit 8.0.0、
LibTorch 2.1.0、ONNX Runtime 1.22.0、CMake 3.22.1 和 GCC 11.4.0。

```bash
sudo apt install build-essential cmake patch libeigen3-dev libtbb-dev \
    libyaml-cpp-dev python3-dev

export Torch_DIR=/absolute/path/libtorch
export ONNXRUNTIME_ROOT_PATH=/absolute/path/onnxruntime
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH="$Torch_DIR/lib:$ONNXRUNTIME_ROOT_PATH/lib:$LD_LIBRARY_PATH"
```

`Torch_DIR` 应包含 `share/cmake/Torch`，`ONNXRUNTIME_ROOT_PATH` 应包含
`include/onnxruntime_cxx_api.h`，CANN 应包含 `include/acl/acl.h`。
依赖需要与目标架构匹配；非默认安装请调整路径，运行前也需加载上述环境。

## 编译

```bash
cd locomotion/rl_sar
./build.sh
```

默认启用真实 CANN OM 后端，产物位于 `cmake_build/build/bin/`：

- `rl_infer`：离线 Ascend 推理验证。
- `rl_real_g1`：G1 实机控制入口。

构建脚本检查固定依赖，将上游源码复制到 `cmake_build/rl_sar-src/` 后应用补丁和配置。
请编辑 `cann/`、`config/`、`patches/` 中的源码，不要编辑生成副本。

其他选项：`./build.sh --debug`、`./build.sh --without-om`、`./build.sh --clean`。
`--without-om` 保留上游 ONNX/PyTorch 配置，不生成 `rl_infer`；此时
`RL_SAR_MODEL_PATH` 应指定对应策略的 ONNX/PyTorch 模型。

## 离线 Ascend 推理

使用符合下文协议的真实 OM 模型：

```bash
export RL_SAR_TEST_MODEL=/absolute/path/policy_kuaile_terrain_smoothed.om
./cmake_build/build/bin/rl_infer --model "$RL_SAR_TEST_MODEL" --steps 3
```

命令复用实机程序的 RL 初始化、YAML 配置、观测构造、OM 推理和关节指令计算，
使用合成机器人状态，不创建机器人通信通道。
每步检查动作、参考状态和指令数值，并报告成功或具体错误。
退出码为 0 表示通过、1 表示初始化或推理失败、2 表示参数错误。
`--steps` 范围为 1～1000；每步耗时用于排错，不作为性能基准。

## 测试

加载运行环境并设置 `RL_SAR_TEST_MODEL` 后，执行完整测试或仅设备测试：

```bash
ctest --test-dir cmake_build/build --output-on-failure
ctest --test-dir cmake_build/build -L hardware --output-on-failure -V
```

未设置模型路径或不存在 `/dev/davinci0` 时，设备测试显示 **Skipped**，不表示设备推理通过。
显式指定无效模型、初始化/推理失败或超时均判为失败。

| 测试 | 覆盖范围 |
| --- | --- |
| `observation_buffer` | 观测历史缓冲处理 |
| `om_backend_mapping` | 后端选择、输出映射和非法输出拒绝，使用引擎替身 |
| `om_engine` | 生产引擎的输入校验、上下文及资源管理，使用模拟 ACL |
| `om_integration_mock_acl` | 生产后端 + 生产引擎，覆盖加载、拷贝、执行、错误传播和清理，仅模拟 ACL |
| `ascend_runner_contract` | 设备测试的跳过、失败和超时处理 |
| `ascend_inference` | 真实模型和 ACL 的完整离线 RL 推理，需 Ascend 设备 |

没有 CANN 的电脑可独立运行引擎和后端测试：

```bash
cmake -S tests -B cmake_build/tests/engine
cmake --build cmake_build/tests/engine
ctest --test-dir cmake_build/tests/engine --output-on-failure
cmake -S tests/backend -B cmake_build/tests/backend
cmake --build cmake_build/tests/backend
ctest --test-dir cmake_build/tests/backend --output-on-failure
```

## 模型协议与配置

当前仅支持 G1 BeyondDance。协议集中在 `cann/include/g1_beyonddance_contract.hpp`，
后端接收 YAML 的 `num_observations` 和 `num_of_dofs`，加载模型前检查是否匹配
154 维观测、29 关节，再按已校验的配置检查输入输出。

输入顺序为 float32 观测 `[1, 154]` 和一个 float32 时间步标量。

| 模型输出索引 | 内容 |
| --- | --- |
| 0 | 29 个动作 |
| 1 | 29 个参考关节位置 |
| 2 | 29 个参考关节速度 |
| 4 | 身体四元数数组，偏移 28 起的 4 个数为参考锚点四元数（w, x, y, z） |

策略配置位于 `config/g1/robomimic/beyonddance/config.yaml`。
模型需要与该协议和策略匹配；改变关节数并不意味着其他机器人已适配。
模型文件、导出/OM 转换流程及模型下载地址不随本样例提供。

## G1 实机入口

```bash
export RL_SAR_MODEL_PATH=/absolute/path/policy_kuaile_terrain_smoothed.om
./cmake_build/build/bin/rl_real_g1 eth0
```

将 `eth0` 替换为连接 G1 的实际网卡。首次运行需可靠支撑机器人并准备急停，
确认策略、关节映射和 G1 型号匹配后再落地运行。

程序从被动状态开始。键盘 `0`（手柄 `A`）进入起立；完成后按键盘 `3`
（手柄 `R1` + 方向左）进入 BeyondDance。`RL_SAR_MODEL_PATH` 必须匹配当前策略，
不要先切换到其他策略。键盘 `P`（手柄 `L1` + `X`）返回被动状态。
软件状态切换不能替代实机急停。

## 常见问题

- 子模块不完整或补丁失败：递归初始化子模块，并恢复表中固定提交。
- `CANN ACL was not found`：加载 CANN `set_env.sh`，检查 `ASCEND_TOOLKIT_HOME`。
- 配置或观测维度不匹配：核对 YAML、154 维模型输入和 29 关节输出协议。
- 动态库找不到：检查目标架构及 `LD_LIBRARY_PATH` 中的 CANN、LibTorch、ONNX Runtime 路径。

## 目录与许可证

```text
rl_sar/
├── upstream/    # 固定版本的控制代码子模块
├── cann/        # CANN OM 后端和模型协议
├── config/      # 策略配置
├── patches/     # 上游接入补丁
├── tools/       # 离线推理入口
├── tests/       # 测试源码
├── cmake_build/ # 已忽略的生成目录
├── CMakeLists.txt
└── build.sh
```

构建及测试产物统一放在 `cmake_build/`，`./build.sh --clean` 会删除整个目录。
模型放在被忽略的 `models/` 或仓库外。提交源码、测试和文档，勿提交模型或生成文件。

`upstream/` 保留原有许可证。本仓库维护 CANN 适配层；归属和许可见 [NOTICE](NOTICE)、
`upstream/LICENSE` 及 `upstream/src/rl_sar/library/thirdparty/unitree_sdk2-2.0.0/LICENSE`。
子模块许可证文件在递归初始化后可用。
