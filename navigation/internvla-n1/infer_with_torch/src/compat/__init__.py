# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""InternVLA-N1 在昇腾 NPU 上跑 VLN-CE 评测所需的运行期垫片。

这些垫片全部在进程启动阶段用 monkeypatch 生效，不修改 InternNav / habitat 的源码，
这样上游仓库的工作区可以保持干净，升级上游时也不必重新合并冲突。

导入顺序有要求，见 npu_eval.py 顶部。
"""
