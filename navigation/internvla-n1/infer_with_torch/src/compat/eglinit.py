# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""在 habitat_sim 创建 GL 上下文之前，先把 glvnd 的 EGL vendor 初始化掉。

现象：在 aarch64 + Mesa/llvmpipe 软件光栅的环境上，直接 import habitat_sim 再建 Simulator，
libglvnd 的 vendor 延迟初始化会失败，表现为拿不到 EGL 设备列表。
先用 ctypes 打开 libEGL.so.1 并调用一次 eglQueryString(EGL_EXTENSIONS)，
可以强制 glvnd 完成 vendor 枚举，之后 habitat_sim 建上下文就正常了。

必须在 `import habitat_sim` 之前 import 本模块。
"""
import ctypes

_EGL_EXTENSIONS = 0x3055

_egl = ctypes.CDLL("libEGL.so.1")
_egl.eglQueryString.restype = ctypes.c_char_p
_egl.eglQueryString.argtypes = [ctypes.c_void_p, ctypes.c_int]
EGL_EXTENSIONS = _egl.eglQueryString(None, _EGL_EXTENSIONS)
