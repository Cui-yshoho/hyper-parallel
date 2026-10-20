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
"""CPU regressions for fused AdamW step tensor semantics."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.optimizer.adamw import AdamW


class TestAdamW(unittest.TestCase):
    """Check the fused API contract and numerical parity without accelerators."""

    def test_fused_step_uses_fp32_tensor(self):
        """CUDA fused AdamW consumes floating-point step tensors."""
        parameter = torch.nn.Parameter(torch.ones(2))
        parameter.grad = torch.ones_like(parameter)
        optimizer = AdamW([parameter])
        with patch("hyper_parallel.core.optimizer.adamw.torch._fused_adamw_") as fused:
            for expected_step in (1, 2):
                optimizer.step()
                step = fused.call_args.args[5][0]
                self.assertEqual(step.dtype, torch.float32)
                self.assertEqual(step.device, parameter.device)
                self.assertEqual(step.item(), expected_step)

    def test_matches_torch_adamw(self):
        """FP32 step tensors preserve AdamW updates over several steps."""
        for amsgrad in (False, True):
            with self.subTest(amsgrad=amsgrad):
                parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
                reference = torch.nn.Parameter(parameter.detach().clone())
                options = dict(lr=0.01, weight_decay=0.1, amsgrad=amsgrad)
                optimizer = AdamW([parameter], **options)
                baseline = torch.optim.AdamW([reference], **options)
                for step in range(3):
                    gradient = torch.tensor([0.2 + step * 0.1, -0.3])
                    parameter.grad = gradient.clone()
                    reference.grad = gradient.clone()
                    optimizer.step()
                    baseline.step()
                    torch.testing.assert_close(parameter, reference)


if __name__ == "__main__":
    unittest.main()
