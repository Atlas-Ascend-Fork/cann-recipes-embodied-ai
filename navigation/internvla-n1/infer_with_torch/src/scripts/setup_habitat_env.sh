#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# 第 2 步：搭 habitat 仿真环境。
#
# 为什么要单独一个 venv：
#   habitat-lab 0.2.4 要求 gym < 0.23.1，而 InternNav 的 core_requirements 装的是
#   gym 0.26.2（gym 0.26 把 registry 从对象换成了 dict，habitat-lab 会 AttributeError）。
#   两者不能共存于同一环境，所以 habitat 栈单独装，再用 bridge 目录按需覆盖 / 兜底，
#   详见 src/compat/hvenv_tail.py。
#
# 为什么 habitat-lab 必须是 0.2.4：
#   InternNav 官方文档要求 habitat-sim 0.2.4 + habitat-lab v0.2.4。
#   scripts/eval/configs/vln_*.yaml 依赖 habitat/simulator/agents/rgbd_agent 这个配置组，
#   它在 habitat-lab v0.3.0 起被整体删除；habitat.core.utils.try_cv2_import 也在 v0.3.2 被删。
#   用 0.3.x 会在加载配置时报 MissingConfigException。
#
# 为什么 habitat-sim 编的是 0.3.3 而不是 0.2.4：
#   aihabitat 的 conda 频道没有 linux-aarch64 包，只能源码编译；0.2.4 的构建链在
#   Python 3.11 上问题较多，实测 0.3.3 的 C++ 绑定与 habitat-lab 0.2.4 的 Python 层
#   可以正常配合（VLN-CE 用的是数据集自带的 navmesh，不重新生成）。
#   这是本样例与论文环境唯一的版本差异，见 docs/navigation/internvla-n1/infer_with_torch/README.md。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

command -v cmake >/dev/null || die "需要 cmake（apt install cmake）"
command -v ninja >/dev/null || warn "没找到 ninja，编译会慢很多（apt install ninja-build）"

# habitat-sim v0.3.3 里 vendored 的 zstd / assimp / recastnavigation 写的是
# cmake_minimum_required(VERSION 3.0)，而 CMake 4.x 已彻底移除对 <3.5 的兼容：
#   CMake Error at deps/zstd/build/cmake/CMakeLists.txt:10 (cmake_minimum_required):
#     Compatibility with CMake < 3.5 has been removed from CMake.
# 所以必须用 3.x。验证环境是系统 cmake 3.20.5。
CMAKE_MAJOR="$(cmake --version | head -1 | sed -E 's/[^0-9]*([0-9]+)\..*/\1/')"
[ "${CMAKE_MAJOR:-0}" -lt 4 ] || \
    die "需要 CMake 3.x，当前是 $(cmake --version | head -1)；habitat-sim v0.3.3 的 vendored 依赖不兼容 CMake 4"

mkdir -p "$HABITAT_ENV_ROOT"

# ---------- 1. venv ----------
if [ ! -d "$HABITAT_VENV" ]; then
    info "创建 habitat venv"
    python3 -m venv --system-site-packages "$HABITAT_VENV"
fi
HPIP="$HABITAT_VENV/bin/pip"
HPY="$HABITAT_VENV/bin/python"

# ---------- 2. 编译 habitat-sim ----------
if [ ! -d "$HABITAT_SIM_SRC" ]; then
    info "克隆 habitat-sim $HABITAT_SIM_TAG（含子模块，约 6GB，耗时较长）"
    git clone --branch "$HABITAT_SIM_TAG" --recursive \
        "$GITCODE_MIRROR/ha/habitat-sim.git" "$HABITAT_SIM_SRC"
fi

SIM_EXT=$(ls "$HABITAT_SIM_SRC"/build/lib.linux-*/habitat_sim/_ext/habitat_sim_bindings*.so 2>/dev/null | head -1 || true)
if [ -z "$SIM_EXT" ]; then
    info "编译 habitat-sim（headless / 无 CUDA / 无 bullet，$(nproc) 并发）"
    # 注意：**不要**往这个 venv 里 pip install cmake。habitat-sim 的 setup.py 是
    #   try: import cmake; CMAKE_BIN_DIR = cmake.CMAKE_BIN_DIR
    #   except ImportError: CMAKE_BIN_DIR = ""     # 走 PATH 上的 cmake
    # 一旦装了 pip 的 cmake（现在是 4.x），它会优先用那个，编译必挂（见上面的版本检查）。
    # 让 import 失败、回落到系统 cmake 3.x 才是对的。
    $HPIP install -q "numpy<2" pybind11
    ( cd "$HABITAT_SIM_SRC" && \
      "$HPY" setup.py build_ext --parallel "$(nproc)" \
          --headless \
          --no-bullet \
          --cmake-args="-DBUILD_WITH_CUDA=OFF -DBUILD_WITH_AUDIO=OFF -DBUILD_GUI_VIEWERS=OFF" )
    SIM_EXT=$(ls "$HABITAT_SIM_SRC"/build/lib.linux-*/habitat_sim/_ext/habitat_sim_bindings*.so | head -1)
fi
info "habitat-sim 绑定：$SIM_EXT"

# build_ext 只把编译产物放到 build/lib.*/habitat_sim/_ext/，**不会**把
# src_python/habitat_sim/ 下的 Python 源码复制过去 —— setuptools 的 build_ext
# 不会顺带调用 build_py。只挂 build/lib 的话，habitat_sim 就是一个只含 _ext 的
# 命名空间包，import 得进去但里面什么都没有，第一个用到的属性就炸：
#   AttributeError: module 'habitat_sim' has no attribute 'SensorType'
#   （habitat-lab 的 habitat_simulator.py 在 class 定义期就会访问它）
# setup.py 里是 package_dir={"": "src_python"}，再跑一次 build_py 即可把 Python 层
# 补进同一个 build/lib 目录，与 _ext 合成完整的包。这步很快，幂等，无条件执行。
info "补齐 habitat_sim 的 Python 层（build_py）"
( cd "$HABITAT_SIM_SRC" && "$HPY" setup.py build_py >/dev/null )

# 产物在 build/lib.*，不装进 site-packages。用 .pth 把它挂上去，
# 避免再跑一遍耗时的 install。
SITE="$($HPY -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
BUILD_LIB="$(dirname "$(dirname "$(dirname "$SIM_EXT")")")"
echo "$BUILD_LIB" > "$SITE/habitat_sim_build.pth"

# ---------- 3. habitat-lab 0.2.4 ----------
if [ ! -d "$HABITAT_LAB_SRC" ]; then
    info "克隆 habitat-lab $HABITAT_LAB_TAG"
    git clone --branch "$HABITAT_LAB_TAG" \
        "$GITCODE_MIRROR/ha/habitat-lab.git" "$HABITAT_LAB_SRC"
fi
info "安装 habitat-lab / habitat-baselines 及其依赖"
# gym 必须 0.23.0：habitat-lab 0.2.4 的 requirements 写的是 >=0.22.0,<0.23.1
$HPIP install -q --no-deps "gym==0.23.0"
$HPIP install -q \
    "numpy<2" "numpy-quaternion" attrs "hydra-core>=1.2.0" "omegaconf>=2.2.3" \
    numba imageio imageio-ffmpeg scipy tqdm \
    ifcfg lmdb "webdataset==0.1.40" moviepy threadpoolctl "faster-fifo>=1.4.2" \
    GitPython msgpack-numpy portalocker "dtw==1.4.0"
# tensorboard 必须 <=2.18：更高版本的 protobuf gencode 与系统 protobuf 5.x 不兼容，
# habitat_baselines 一 import 就抛 VersionError
$HPIP install -q "tensorboard==2.18.0"
$HPIP install -q --no-deps --no-build-isolation \
    "$HABITAT_LAB_SRC/habitat-lab" "$HABITAT_LAB_SRC/habitat-baselines"

# InternNav 的 habitat_extensions 依赖这个只发在 git 上的小包（华为云源没有）。
# 探测用 find_spec 而不是真的 import：这个包的 __init__ 里有 `import cv2`，
# 而 cv2 是故意不装进 habitat venv 的（运行期由 bridge 从推理 venv 提供），
# 所以 `import depth_camera_filtering` 永远 ModuleNotFoundError —— 用它做判据会
# 导致每次都重新 clone，而 github 未必稳定可达，一次抖动就把整个脚本带崩。
if ! $HPY -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('depth_camera_filtering') else 1)" 2>/dev/null; then
    info "安装 depth-camera-filtering"
    TMP=$(mktemp -d)
    # github 这条路不稳，重试几次
    for try in 1 2 3 4 5; do
        git clone --depth 1 https://github.com/naokiyokoyama/depth_camera_filtering.git "$TMP/dcf" && break
        warn "clone depth_camera_filtering 第 $try 次失败，重试"
        rm -rf "$TMP/dcf"; sleep 5
    done
    [ -d "$TMP/dcf" ] || { rm -rf "$TMP"; die "depth_camera_filtering 克隆失败（github 不可达）"; }
    $HPIP install -q --no-deps "$TMP/dcf"
    rm -rf "$TMP"
fi

# ---------- 4. bridge 符号链接农场 ----------
# 只把需要**覆盖** InternNav .venv 的包，以及 .venv 里根本没有的包链进来。
# 其余缺包由 compat/hvenv_tail.py 从 venv 末尾兜底，不会顶掉 .venv 的 numpy/opencv/torch。
info "生成 bridge 目录 $HBRIDGE"
rm -rf "$HBRIDGE"; mkdir -p "$HBRIDGE"
for pkg in habitat habitat_baselines habitat_sim magnum corrade \
           gym gym_notices hydra omegaconf antlr4 numba llvmlite quaternion imageio; do
    for cand in "$SITE/$pkg" "$BUILD_LIB/$pkg" \
                "$SITE/_${pkg}.cpython-"*.so "$SITE/${pkg}.cpython-"*.so; do
        [ -e "$cand" ] && ln -sfn "$cand" "$HBRIDGE/$(basename "$cand")"
    done
done

# ---------- 5. 自检 ----------
info "自检：从推理 venv 经 bridge 导入 habitat 栈"
INTERNNAV_ROOT="$INTERNNAV_ROOT" HABITAT_VENV="$HABITAT_VENV" \
PYTHONPATH="$SAMPLE_SRC:$HBRIDGE" "$INTERNNAV_ROOT/.venv/bin/python" - <<'PY'
from compat import eglinit, hvenv_tail, py311_dataclass_shim  # noqa: F401
import gym, numpy, cv2, habitat, habitat_sim
print(f"  habitat_sim {habitat_sim.__version__} | habitat-lab OK")
print(f"  gym {gym.__version__} (需 0.23.x) | numpy {numpy.__version__} (需 1.x) | cv2 {cv2.__version__}")
assert gym.__version__.startswith("0.23"), "gym 版本不对，habitat-lab 0.2.4 需要 0.23.x"
assert numpy.__version__.startswith("1."), "numpy 应为 1.x"
PY

info "完成。下一步：bash $SAMPLE_SRC/scripts/download_data.sh"
