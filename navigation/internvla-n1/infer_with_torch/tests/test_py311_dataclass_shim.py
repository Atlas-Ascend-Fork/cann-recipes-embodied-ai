# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""src/compat/py311_dataclass_shim.py 的单测。

复刻 habitat-lab 0.2.4 那种"嵌套 dataclass 直接当默认值"的写法，验证：
  - 不打垫片时 CPython 3.11 会拒绝类定义（bpo-44674）；
  - 打上垫片后类能定义，且每个实例拿到**独立副本**（比 3.9 的共享默认值更安全）。
"""
import dataclasses
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from compat import py311_dataclass_shim  # noqa: E402

NEEDS_SHIM = sys.version_info >= (3, 11)


@dataclasses.dataclass
class Inner:
    items: list = dataclasses.field(default_factory=list)
    flag: bool = False


def define_outer():
    """habitat-lab 的写法：把一个 dataclass 实例直接写成字段默认值。"""

    @dataclasses.dataclass
    class Outer:
        iterator_options: Inner = Inner()
        name: str = "x"

    return Outer


def installed_get_field():
    return getattr(dataclasses, py311_dataclass_shim.PATCH_TARGET)


class TestShimInstalled(unittest.TestCase):
    def test_hook_is_in_place(self):
        self.assertIs(installed_get_field(), py311_dataclass_shim.get_field)

    def test_keeps_original_for_later_restore(self):
        self.assertIsNot(py311_dataclass_shim.original_get_field,
                         py311_dataclass_shim.get_field)


class TestWithoutShim(unittest.TestCase):
    """临时摘掉垫片，确认这条路在 3.11 上原本是走不通的。"""

    @unittest.skipUnless(NEEDS_SHIM, "Python < 3.11 本就允许这种默认值")
    def test_raises_mutable_default(self):
        unpatched = mock.patch.object(dataclasses, py311_dataclass_shim.PATCH_TARGET,
                                      py311_dataclass_shim.original_get_field)
        with unpatched, self.assertRaises(ValueError) as cm:
            define_outer()
        self.assertIn("mutable default", str(cm.exception))
        self.assertIs(installed_get_field(), py311_dataclass_shim.get_field)


class TestWithShim(unittest.TestCase):
    def test_class_defines(self):
        outer = define_outer()
        self.assertEqual(outer().name, "x")
        self.assertIsInstance(outer().iterator_options, Inner)

    def test_each_instance_gets_its_own_copy(self):
        outer = define_outer()
        a, b = outer(), outer()
        self.assertIsNot(a.iterator_options, b.iterator_options)
        a.iterator_options.items.append(1)
        self.assertEqual(b.iterator_options.items, [])

    def test_default_value_equals_the_original(self):
        outer = define_outer()
        self.assertEqual(outer().iterator_options, Inner())

    def test_explicit_argument_still_wins(self):
        outer = define_outer()
        mine = Inner(items=[7], flag=True)
        self.assertIs(outer(iterator_options=mine).iterator_options, mine)

    def test_unrelated_valueerror_is_not_swallowed(self):
        """只处理 "mutable default"，别的 ValueError 必须原样抛出去。"""
        boom = ValueError("something else entirely")

        def raiser(*_a, **_kw):
            raise boom

        with mock.patch.object(py311_dataclass_shim, "original_get_field", raiser), \
                self.assertRaises(ValueError) as cm:
            py311_dataclass_shim.get_field(object, "a", int)
        self.assertIs(cm.exception, boom)

    def test_plain_dataclass_unaffected(self):
        @dataclasses.dataclass
        class Plain:
            a: int = 1
            b: str = "s"
            c: tuple = ()

        self.assertEqual((Plain().a, Plain().b, Plain().c), (1, "s", ()))

    def test_list_default_also_rescued_per_instance(self):
        """垫片不区分默认值类型，list/dict 这类也会被改成 default_factory。

        这比 3.9 的旧行为更严格（3.9 直接拒绝 `items: list = []`），
        且每个实例拿到独立副本，不会出现共享可变默认值那个经典坑。
        """
        @dataclasses.dataclass
        class Bad:
            items: list = []  # noqa: RUF008

        a, b = Bad(), Bad()
        self.assertIsNot(a.items, b.items)
        a.items.append(1)
        self.assertEqual(b.items, [])


if __name__ == "__main__":
    unittest.main()
