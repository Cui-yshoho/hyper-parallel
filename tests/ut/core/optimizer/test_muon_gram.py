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
"""Gram identities, opt-in routing and compatibility with native Muon."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.optimizer.muon import Muon, _NS_ASYM5_COEFFS, zeropower_via_newtonschulz5
from hyper_parallel.core.optimizer.muon_gram import newton_schulz_with_gram, use_gram_for_shape


class TestMuonGram(unittest.TestCase):
    """Verify algebra separately from production routing thresholds."""

    def test_shape_policy(self):
        """Small, square and near-square inputs do not justify Gram work."""
        cases = [((12, 128, 2048), False), ((4096, 4096), False), ((6144, 5120), False),
                 ((1, 1024, 5120), False), ((2, 1024, 5120), True),
                 ((32, 768, 2048), True), ((17408, 5120), True)]
        for shape, expected in cases:
            with self.subTest(shape=shape):
                self.assertEqual(use_gram_for_shape(shape), expected)

    def test_gram_matches_standard_in_double_precision(self):
        """The restart recurrence matches independent standard NS, including rank deficiency."""
        torch.manual_seed(42)
        for shape in ((7, 19), (19, 7), (2, 7, 19)):
            for rank_deficient in (False, True):
                with self.subTest(shape=shape, rank_deficient=rank_deficient):
                    value = torch.randn(shape, dtype=torch.float64)
                    if rank_deficient:
                        value[..., 1:, :] = value[..., :1, :]
                    original = value.clone()
                    with patch("hyper_parallel.core.optimizer.muon_gram.use_gram_for_shape", return_value=True):
                        actual = newton_schulz_with_gram(value, _NS_ASYM5_COEFFS, 1e-10)
                    reference = zeropower_via_newtonschulz5(value, 5)
                    torch.testing.assert_close(actual, reference, atol=1e-9, rtol=1e-9)
                    self.assertTrue(torch.equal(value, original))

    def test_default_keeps_original_path_and_gram_accepts_other_schedules(self):
        """Only the Gram flag selects the configured schedule, independent of dtype and steps."""
        value = torch.randn(8, 16, dtype=torch.bfloat16)
        with patch("hyper_parallel.core.optimizer.muon.newton_schulz_with_gram") as routed:
            zeropower_via_newtonschulz5(value, 5)
            routed.assert_not_called()
        coefficients = [(3.2, -4.1, 1.7)] * 3
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for variant in ("legacy", "custom"):
                with self.subTest(dtype=dtype, variant=variant):
                    with patch("hyper_parallel.core.optimizer.muon.newton_schulz_with_gram") as routed:
                        zeropower_via_newtonschulz5(
                            value.to(dtype), 3, ns_variant=variant, use_gram=True,
                            ns_coefficients=coefficients if variant == "custom" else None,
                            gram_reset_iterations=(1,),
                        )
                    expected = coefficients if variant == "custom" else [(3.4445, -4.775, 2.0315)] * 3
                    self.assertEqual(tuple(routed.call_args.args[1]), tuple(expected))
                    self.assertEqual(routed.call_args.kwargs["reset_iterations"], (1,))
        actual = zeropower_via_newtonschulz5(value, 5, use_gram=True)
        self.assertTrue(torch.equal(actual, zeropower_via_newtonschulz5(value, 5)))

    def test_configurable_gram_restarts_preserve_ns_identity(self):
        """Gram reassociation retains the configured polynomial for varied step and restart schedules."""
        torch.manual_seed(17)
        for shape in ((7, 19), (19, 7), (2, 7, 19)):
            value = torch.randn(shape, dtype=torch.float64)
            for steps in (1, 3, 6):
                coefficients = [(3.2, -4.1, 1.7)] * steps
                reference = zeropower_via_newtonschulz5(value, steps, "custom", ns_coefficients=coefficients)
                for resets in ((), (1,), (2,), (1, 3), (0, -1, 20)):
                    with self.subTest(shape=shape, steps=steps, resets=resets):
                        with patch("hyper_parallel.core.optimizer.muon_gram.use_gram_for_shape", return_value=True):
                            actual = zeropower_via_newtonschulz5(
                                value, steps, "custom", ns_coefficients=coefficients,
                                use_gram=True, gram_reset_iterations=resets,
                            )
                        torch.testing.assert_close(actual, reference, atol=1e-9, rtol=1e-9)

    def test_optimizer_passes_group_flag_after_batching(self):
        """The production path selects using merged matrices and honors callback overrides."""
        optimizer = object.__new__(Muon)
        optimizer.zeropower_fn = None
        tensors = [torch.randn(8, 16, dtype=torch.bfloat16) for _ in range(2)]
        with patch("hyper_parallel.core.optimizer.muon.newton_schulz_with_gram",
                   side_effect=lambda inputs, *_, **__: inputs.clone()) as routed:
            outputs = optimizer._compute_batched_ns_outputs_for_tensors(tensors, 5, use_gram=True)
            self.assertEqual(tuple(routed.call_args.args[0].shape), (2, 8, 16))
            self.assertEqual(len(outputs), 2)
        optimizer.zeropower_fn = lambda inputs, steps: inputs.clone()
        with patch("hyper_parallel.core.optimizer.muon.newton_schulz_with_gram") as routed:
            optimizer._compute_batched_ns_outputs_for_tensors(tensors, 5, use_gram=True)
            routed.assert_not_called()

    def test_zero_and_invalid_flag(self):
        """Zero inputs stay finite and flags cannot accidentally use truthy strings."""
        value = torch.zeros(7, 19, dtype=torch.bfloat16)
        with patch("hyper_parallel.core.optimizer.muon_gram.use_gram_for_shape", return_value=True):
            actual = zeropower_via_newtonschulz5(value, 5, use_gram=True)
        self.assertTrue(torch.equal(actual, value))
        with self.assertRaisesRegex(ValueError, "use_gram"):
            zeropower_via_newtonschulz5(value, 5, use_gram="yes")
        with self.assertRaisesRegex(ValueError, "use_gram"):
            Muon([torch.nn.Parameter(value)], use_gram="yes")


class TestMuonSymmetricBackend(unittest.TestCase):
    """Keep algorithm selection independent from symmetric product dispatch."""

    def test_gram_does_not_implicitly_enable_symmetric_backend(self):
        """Gram opt-in defaults to CANN; only an explicit backend enables native kernels."""
        value = torch.randn(8, 16, dtype=torch.bfloat16)
        for backend, expected in ((None, "cann"), ("cann", "cann"), ("auto", "auto")):
            with self.subTest(backend=backend):
                with patch("hyper_parallel.core.optimizer.muon.newton_schulz_with_gram",
                           side_effect=lambda inputs, *_, **__: inputs.clone()) as gram:
                    zeropower_via_newtonschulz5(value, 5, use_gram=True, ns_backend=backend)
                    self.assertEqual(gram.call_args.args[-1], expected)

    def test_rejected_gram_preserves_standard_output_and_layout(self):
        """Small wide, tall and batched matrices retain standard NS after static rejection."""
        torch.manual_seed(87)
        for shape in ((7, 19), (19, 7), (2, 7, 19), (2, 19, 7)):
            value = torch.randn(shape, dtype=torch.bfloat16)
            for backend in (None, "cann", "auto"):
                with self.subTest(shape=shape, backend=backend):
                    reference = zeropower_via_newtonschulz5(value, 5, ns_backend=backend)
                    actual = zeropower_via_newtonschulz5(value, 5, use_gram=True, ns_backend=backend)
                    torch.testing.assert_close(actual, reference, atol=0, rtol=0, check_stride=True)

    def test_native_backend_preserves_standard_ns_rounding(self):
        """Backend-only selection must retain all original BF16 coefficient roundings."""
        torch.manual_seed(61)
        for shape in ((7, 19), (19, 7), (2, 7, 19)):
            for variant, steps in (("asym5", 5), ("legacy", 3)):
                with self.subTest(shape=shape, variant=variant):
                    value = torch.randn(shape, dtype=torch.bfloat16)
                    reference = zeropower_via_newtonschulz5(value, steps, ns_variant=variant)
                    with patch("hyper_parallel.core.optimizer.muon.newton_schulz_with_gram") as gram:
                        for backend in ("cann", "auto"):
                            actual = zeropower_via_newtonschulz5(
                                value, steps, ns_variant=variant, ns_backend=backend)
                            self.assertTrue(torch.equal(actual, reference), f"Rounding changed for {backend}")
                        gram.assert_not_called()

    def test_explicit_backend_reaches_gram_and_batched_standard(self):
        """The backend reaches both algorithms after optimizer input batching."""
        optimizer = object.__new__(Muon)
        optimizer.zeropower_fn = None
        inputs = [torch.randn(8, 16, dtype=torch.bfloat16) for _ in range(2)]
        with patch("hyper_parallel.core.optimizer.muon.zeropower_via_newtonschulz5",
                   side_effect=lambda value, **_: value.clone()) as ns:
            optimizer._compute_batched_ns_outputs_for_tensors(inputs, 5, ns_backend="ascendc")
            self.assertEqual(ns.call_args.kwargs["ns_backend"], "ascendc")
            self.assertFalse(ns.call_args.kwargs["use_gram"])
        with patch("hyper_parallel.core.optimizer.muon.newton_schulz_with_gram",
                   side_effect=lambda value, *_, **__: value.clone()) as gram:
            zeropower_via_newtonschulz5(inputs[0], 5, use_gram=True, ns_backend="cann")
            self.assertEqual(gram.call_args.args[-1], "cann")

    def test_invalid_backend_is_rejected_at_public_boundaries(self):
        """Reject misspellings before optimizer initialization or matrix computation."""
        value = torch.zeros(7, 19, dtype=torch.bfloat16)
        for backend in ("triton", "", True):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "ns_backend"):
                    zeropower_via_newtonschulz5(value, 5, ns_backend=backend)
                with self.assertRaisesRegex(ValueError, "ns_backend"):
                    Muon([torch.nn.Parameter(value)], ns_backend=backend)
