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
"""CPU checks of VeOmni ExtraParallel boundary translation."""

import math
import unittest
from unittest.mock import MagicMock

import torch

from hyper_parallel.integration.veomni.extra_parallel import (
    bind_gradient_divisor,
    parameters_for_clipping,
)


class TestExtraParallel(unittest.TestCase):
    """Keep divisor scaling equivalent and ordinary parameters unchanged."""

    def test_gradient_divisor_uses_native_scaling(self):
        """Convert sum/divisor without changing the reduction implementation."""
        module = torch.nn.Module()
        module.set_gradient_scaling_factor = MagicMock()
        bind_gradient_divisor(module, 2)
        module.set_reduce_scatter_divide_factor(4)
        module.set_gradient_scaling_factor.assert_called_once_with(0.5)
        module.set_gradient_divide_factor(8)
        module.set_gradient_scaling_factor.assert_called_with(0.25)
        self.assertFalse(hasattr(torch.nn.Module(), "set_gradient_divide_factor"))

    def test_invalid_divisor_does_not_change_scaling(self):
        """Reject invalid divisors before touching the Hyper module state."""
        module = torch.nn.Module()
        module.set_gradient_scaling_factor = MagicMock()
        bind_gradient_divisor(module, 2)
        for factor in (0, -1, True, math.inf, math.nan, "4"):
            with (
                self.subTest(factor=factor),
                self.assertRaisesRegex(ValueError, "finite positive"),
            ):
                module.set_gradient_divide_factor(factor)
        module.set_gradient_scaling_factor.assert_not_called()

    def test_plain_parameters_retain_identity(self):
        """Ordinary models need neither layout wrappers nor gradient copies."""
        module = torch.nn.Linear(2, 2)
        original = list(module.parameters())
        actual = parameters_for_clipping(module)
        self.assertEqual(len(original), len(actual))
        for expected, parameter in zip(original, actual):
            self.assertIs(expected, parameter)
