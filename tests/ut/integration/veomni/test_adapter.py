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
"""CPU-only tests of Hyper-owned, reversible framework bindings."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import hyper_parallel.integration.veomni as veomni_integration
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.integration.veomni import (
    VeOmniAdapter,
    apply_hyper_parallel_patch,
    remove_hyper_parallel_patch,
)


class TestVeOmniAdapter(unittest.TestCase):
    """Verify binding ownership and native-path isolation without communication."""

    def test_public_patch_names_are_explicit(self):
        """Export one apply/remove pair without legacy entry aliases."""
        self.assertIs(
            veomni_integration.apply_hyper_parallel_patch, apply_hyper_parallel_patch
        )
        self.assertIs(
            veomni_integration.remove_hyper_parallel_patch, remove_hyper_parallel_patch
        )
        self.assertFalse(hasattr(veomni_integration, "apply_veomni_patch"))
        self.assertFalse(hasattr(veomni_integration, "disable"))

    def setUp(self):
        """Install an isolated framework surface instead of importing VeOmni."""
        self.loading = ModuleType("veomni.models.module_utils")
        self.clipping = ModuleType("veomni.distributed.fsdp2.clip_grad_norm")
        self.aliases = ModuleType("veomni.loaded_aliases")
        self.native_load_calls = []
        self.native_clip_calls = []

        def dispatch(
            module,
            name,
            tensor,
            dtensor_factory=None,
            parallel_plan=None,
            dtensor_to_cpu=False,
            tensor_is_local_dtensor_shard=False,
        ):
            self.native_load_calls.append((module, name, tensor))
            return "native-load"

        def clip(
            model, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None
        ):
            self.native_clip_calls.append(
                (model, max_norm, norm_type, error_if_nonfinite, foreach)
            )
            return "native-clip"

        self.loading._dispatch_parameter = dispatch
        self.loading._find_submodule = lambda module, name: (module, name)
        self.clipping.clip_grad_norm = clip
        self.aliases.clip_alias = clip
        self.original_load, self.original_clip = dispatch, clip
        self.registry = patch.dict(
            sys.modules,
            {
                self.loading.__name__: self.loading,
                self.clipping.__name__: self.clipping,
                self.aliases.__name__: self.aliases,
            },
        )
        self.registry.start()
        self.adapter = VeOmniAdapter()
        self.construction = patch(
            "hyper_parallel.integration.veomni.adapter._build_construction_bindings",
            return_value=[],
        )
        self.construction.start()

    def tearDown(self):
        """Restore all owned bindings and the interpreter's module registry."""
        self.adapter.unpatch()
        self.construction.stop()
        self.registry.stop()

    def test_patch_is_idempotent_and_restores_existing_alias(self):
        """Existing aliases point to one wrapper and restore to the original."""
        torch_clip = torch.nn.utils.clip_grad_norm_
        self.assertIs(self.adapter.patch(), self.adapter)
        wrapper = self.clipping.clip_grad_norm
        self.assertIs(self.adapter.patch(), self.adapter)
        self.assertIs(self.aliases.clip_alias, wrapper)
        self.assertIs(torch.nn.utils.clip_grad_norm_, torch_clip)
        self.assertIn("veomni.loaded_aliases.clip_alias", self.adapter.bound_targets)
        self.assertIn(
            "veomni.models.module_utils._dispatch_parameter", self.adapter.bound_targets
        )
        self.adapter.unpatch()
        self.assertEqual(self.adapter.bound_targets, ())
        self.assertIs(self.aliases.clip_alias, self.original_clip)
        self.assertIs(self.loading._dispatch_parameter, self.original_load)

    def test_late_alias_and_captured_wrapper_restore(self):
        """Late module aliases restore; captured wrappers stop redirecting."""
        self.adapter.patch()
        wrapper = self.clipping.clip_grad_norm
        self.aliases.late_alias = wrapper
        self.adapter.unpatch()
        self.assertIs(self.aliases.late_alias, self.original_clip)
        self.assertEqual(wrapper(torch.nn.Linear(2, 2), 1.0), "native-clip")

    def test_plain_torch_model_uses_original(self):
        """Enabling for Hyper must not reinterpret plain Torch parameters."""
        self.adapter.patch()
        model = torch.nn.Linear(2, 2)
        self.assertEqual(
            self.clipping.clip_grad_norm(model, 1.0, foreach=False), "native-clip"
        )
        self.assertEqual(len(self.native_clip_calls), 1)

    def test_hyper_clip_delegates_to_existing_hyper_function(self):
        """No replacement norm algorithm is implemented in the adapter."""
        parameter = MagicMock(spec=DTensor)
        model = SimpleNamespace(parameters=lambda: iter([parameter]))
        self.adapter.patch()
        with patch(
            "hyper_parallel.integration.veomni.adapter.clip_grad_norm_",
            return_value=torch.tensor(5.0),
        ) as clip:
            actual = self.clipping.clip_grad_norm(
                model, 2.0, norm_type=3.0, error_if_nonfinite=True, foreach=False
            )
        self.assertEqual(float(actual), 5.0)
        clip.assert_called_once_with([parameter], 2.0, 3.0, True, False)
        self.assertEqual(self.native_clip_calls, [])

    def test_hyper_parameter_write_uses_hyper_payload_boundary(self):
        """Keep the original reader/broadcast and replace only shard writing."""
        parameter = MagicMock(spec=DTensor)
        model = SimpleNamespace(_parameters={"weight": parameter})
        payload = torch.ones(2)
        self.adapter.patch()
        with patch("hyper_parallel.integration.veomni.adapter._copy_parameter") as copy:
            self.assertIsNone(
                self.loading._dispatch_parameter(model, "weight", payload)
            )
        self.assertIs(copy.call_args.args[0], parameter)
        self.assertIs(copy.call_args.args[1], payload)
        self.assertEqual(copy.call_args.kwargs["full_param_name"], "weight")
        self.assertEqual(self.native_load_calls, [])

    def test_plain_parameter_write_uses_original(self):
        """Ordinary parameter writes retain all original function behavior."""
        model = torch.nn.Linear(2, 2)
        self.adapter.patch()
        self.assertEqual(
            self.loading._dispatch_parameter(model, "weight", torch.ones(2, 2)),
            "native-load",
        )

    def test_incompatible_signature_does_not_partially_bind(self):
        """Fail before the first mutation when the framework API differs."""
        self.clipping.clip_grad_norm = lambda unknown: None
        with self.assertRaisesRegex(RuntimeError, "signature"):
            self.adapter.patch()
        self.assertIs(self.loading._dispatch_parameter, self.original_load)
        self.assertFalse(self.adapter.is_patched)

    def test_incompatible_optimizer_does_not_partially_bind(self):
        """Optimizer signature checks finish before loading/clipping are rebound."""
        self.construction.stop()
        modules = {
            self.loading.__name__: self.loading,
            self.clipping.__name__: self.clipping,
            "veomni.optim.optimizer": SimpleNamespace(
                build_optimizer=lambda model: model
            ),
        }
        with (
            patch(
                "hyper_parallel.integration.veomni.adapter.import_module",
                side_effect=modules.__getitem__,
            ),
            self.assertRaisesRegex(RuntimeError, "build_optimizer signature"),
        ):
            self.adapter.patch()
        self.assertIs(self.loading._dispatch_parameter, self.original_load)
        self.assertIs(self.clipping.clip_grad_norm, self.original_clip)
        self.assertFalse(self.adapter.is_patched)
        self.assertEqual(self.adapter.bound_targets, ())

    def test_conflicting_adapter_is_rejected(self):
        """Two owners cannot silently stack wrappers in one interpreter."""
        self.adapter.patch()
        with self.assertRaisesRegex(RuntimeError, "already active"):
            VeOmniAdapter().patch()

    def test_optimizer_only_conflict_is_rejected_without_leaking_bindings(self):
        """A second optimizer owner cannot leave wrappers behind after cleanup."""
        optimizer = ModuleType("veomni.optim.optimizer")

        def build_optimizer(model, optimizer_type="adamw", param_groups=None):
            return model

        optimizer.build_optimizer = build_optimizer
        optimizer.OptimizerState = type("OptimizerState", (), {})
        self.aliases.optimizer_alias = build_optimizer
        first = VeOmniAdapter(fsdp=False, muon=True)
        second = VeOmniAdapter(fsdp=False, muon=True)
        self.construction.stop()
        with (
            patch.dict(sys.modules, {optimizer.__name__: optimizer}),
            patch(
                "hyper_parallel.integration.veomni.adapter.import_module",
                return_value=optimizer,
            ),
        ):
            try:
                first.patch()
                wrapper = optimizer.build_optimizer
                self.assertIs(first.patch(), first)
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    second.patch()
                self.assertFalse(second.is_patched)
                self.assertEqual(second.bound_targets, ())
                self.assertIs(optimizer.build_optimizer, wrapper)
                self.assertIs(self.aliases.optimizer_alias, wrapper)
                # Reproduce cleanup in activation order, not just reverse order.
                first.unpatch()
                second.unpatch()
                self.assertIs(optimizer.build_optimizer, build_optimizer)
                self.assertIs(self.aliases.optimizer_alias, build_optimizer)
                second.patch()
                self.assertTrue(second.is_patched)
            finally:
                first.unpatch()
                second.unpatch()
        self.assertIs(self.aliases.optimizer_alias, build_optimizer)

    def test_context_restores_on_error(self):
        """An application failure does not leave active framework bindings."""
        with self.assertRaisesRegex(ValueError, "application"), self.adapter:
            raise ValueError("application")
        self.assertFalse(self.adapter.is_patched)
        self.assertIs(self.clipping.clip_grad_norm, self.original_clip)

    def test_external_rebinding_is_not_overwritten(self):
        """Do not clobber another component's later binding during cleanup."""
        self.adapter.patch()

        def replacement(*args):
            return None

        self.aliases.clip_alias = replacement
        self.adapter.unpatch()
        self.assertIs(self.aliases.clip_alias, replacement)
