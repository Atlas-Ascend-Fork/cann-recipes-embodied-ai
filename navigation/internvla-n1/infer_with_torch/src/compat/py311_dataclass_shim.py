# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""让 habitat-lab 0.2.4 的 structured configs 能在 Python 3.11 上完成类定义。

habitat-lab 0.2.4 面向 Python 3.9。CPython 3.11 收紧了 dataclasses 的默认值检查
（bpo-44674）：只要字段默认值的类型 __hash__ 是 None（普通 dataclass 就是这样），
就抛 "mutable default <class ...> for field ... is not allowed: use default_factory"。
habitat 的嵌套配置（IteratorOptionsConfig 等）正是这种写法，于是
`import habitat` 在 3.11 上直接失败。

这里包一层 dataclasses 的私有函数 `_get_field`：碰到这种默认值就自动改成 default_factory
（deepcopy 一份），语义等价于 3.9 上的旧行为，而且每个实例拿到独立副本更安全。
垫片不区分默认值的类型，`items: list = []` 这类连 3.9 都拒绝的写法同样会被改成
default_factory，而不是报错 —— 得到的也是每实例独立的副本，没有共享可变默认值的坑。

必须在 `import habitat` 之前 import 本模块。
"""
import copy
import dataclasses

# CPython 没有为字段默认值检查提供公开钩子，只能替换这个私有函数。
PATCH_TARGET = "_get_field"
original_get_field = getattr(dataclasses, PATCH_TARGET)


def get_field(cls, a_name, a_type, *args, **kwargs):
    try:
        return original_get_field(cls, a_name, a_type, *args, **kwargs)
    except ValueError as exc:
        if "mutable default" not in str(exc):
            raise
        default = getattr(cls, a_name)
        setattr(
            cls,
            a_name,
            dataclasses.field(default_factory=lambda _d=default: copy.deepcopy(_d)),
        )
        return original_get_field(cls, a_name, a_type, *args, **kwargs)


setattr(dataclasses, PATCH_TARGET, get_field)
