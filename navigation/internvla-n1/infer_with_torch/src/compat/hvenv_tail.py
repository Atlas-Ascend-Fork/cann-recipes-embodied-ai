# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""把 habitat 专用 venv 的 site-packages 追加到 sys.path 末尾，作为缺包兜底。

背景：模型推理用 InternNav 自己的 .venv（torch 2.7.1 / torch_npu / numpy 1.26.4 /
opencv 4.11），而 habitat 栈对 gym、numpy 的版本要求和它冲突，不能装在同一个环境里。
方案是把 habitat 栈装在独立的 HABITAT_VENV 里，再分两级挂到 sys.path：

  1. HBRIDGE（符号链接农场）放在 PYTHONPATH 最前面 —— 只放需要**覆盖** .venv 的包
     （gym 0.23.0 覆盖 .venv 的 0.26.2）以及 .venv 里没有的 habitat/magnum 等；
  2. 本模块把 HABITAT_VENV 的 site-packages 追加到**末尾** —— 只兜底填补
     .venv 缺失的包（tensorboard、ifcfg、lmdb …），不会顶掉 .venv 自己的
     numpy / opencv / torch。

这样 InternNav 的 .venv 一个字节都不用动。
"""
import os
import sys
import sysconfig

_venv = os.environ.get("HABITAT_VENV")
if not _venv:
    raise RuntimeError("需要设置环境变量 HABITAT_VENV，指向 habitat 专用 venv 的根目录")

_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
_tail = os.path.join(_venv, "lib", _ver, "site-packages")
if not os.path.isdir(_tail):
    # 少数发行版用 lib64 或带 abiflags 的目录名，退回 sysconfig 的命名
    _tail = os.path.join(_venv, sysconfig.get_path("purelib").split("/lib/", 1)[-1])
if _tail not in sys.path:
    sys.path.append(_tail)
