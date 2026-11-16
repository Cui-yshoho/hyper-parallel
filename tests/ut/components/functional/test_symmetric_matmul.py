# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""CPU contracts and static backend selection for symmetric products."""

import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.components.functional.symmetric_matmul import symmetric_matmul


_MODULE = importlib.import_module("hyper_parallel.components.functional.symmetric_matmul")


class TestSymmetricMatmul(unittest.TestCase):
    """Check fallback arithmetic and metadata-only dispatch boundaries."""

    def test_native_layouts_and_out(self):
        """Transposed and batched inputs retain dtype, shape and destination storage."""
        for shape in ((13, 7), (2, 13, 7)):
            for transposed in (False, True):
                with self.subTest(shape=shape, transposed=transposed):
                    value = torch.randn(shape, dtype=torch.float64)
                    value = value.mT if transposed else value
                    out = torch.empty((*value.shape[:-1], value.shape[-2]), dtype=value.dtype)
                    addition = torch.eye(value.shape[-2], dtype=value.dtype).expand(out.shape)
                    addition = addition.contiguous()
                    actual = symmetric_matmul(value, value.mT, addition, alpha=0.5, beta=-2, out=out)
                    self.assertIs(actual, out)
                    torch.testing.assert_close(actual, 0.5 * (value @ value.mT) - 2 * addition)

    def test_zero_scales_ignore_nonfinite_values(self):
        """Inactive operands must not contaminate a finite result."""
        value = torch.full((7, 11), float("nan"))
        self.assertTrue(torch.equal(symmetric_matmul(value, value.mT, alpha=0), torch.zeros(7, 7)))
        empty = torch.empty(7, 0)
        self.assertTrue(torch.equal(symmetric_matmul(empty, empty.mT), torch.zeros(7, 7)))

    def test_reject_alias_and_active_autograd(self):
        """Reject writes into operand storage and accidental differentiable use."""
        value = torch.randn(4, 4)
        with self.assertRaisesRegex(ValueError, "storage"):
            symmetric_matmul(value, value.mT, out=value)
        with self.assertRaisesRegex(ValueError, "forward-only"):
            symmetric_matmul(value.requires_grad_(), value.mT)
        with torch.no_grad():
            self.assertEqual(symmetric_matmul(value, value.mT).shape, (4, 4))

    def test_backend_selection(self):
        """Select native only for supported NPU work that amortizes launch costs."""
        cases = [((128, 4096), False), ((2048, 2048), False), ((4096, 4096), True),
                 ((32, 768, 768), True), ((2, 1024, 4096), False)]
        with patch.object(_MODULE, "_native_library_present", return_value=True):
            for shape, expected in cases:
                value = SimpleNamespace(shape=shape, ndim=len(shape), dtype=torch.bfloat16,
                                        device=SimpleNamespace(type="npu"))
                self.assertEqual(_MODULE._use_ascendc(value, 1), expected)
                self.assertFalse(_MODULE._use_ascendc(value, 0))
        with patch.object(_MODULE, "_native_library_present", return_value=False):
            self.assertFalse(_MODULE._use_ascendc(value, 1))

    def test_explicit_native_requires_npu(self):
        """A forced backend must fail rather than silently change implementation."""
        value = torch.ones(2, 2, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "NPU"):
            symmetric_matmul(value, value, backend="ascendc")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            symmetric_matmul(value, value, backend="triton")
