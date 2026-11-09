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
"""Construction binding contracts without optional VeOmni or hardware."""

import sys
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch.distributed.tensor import DTensor, Shard

from hyper_parallel.integration.veomni import VeOmniAdapter, _placement, _state_templates
from hyper_parallel.core.fully_shard.api import _extend_module_with_hsdp_interface


class TestVeOmniAdapter(unittest.TestCase):
    """Only construction names are rebound; restore preserves other owners."""

    def setUp(self):
        self.parallel = ModuleType("veomni.distributed.torch_parallelize")
        self.native = ModuleType("veomni.optim.optimizer")
        self.clipping = ModuleType("veomni.distributed.fsdp2.clip_grad_norm")
        self.alias = ModuleType("veomni.alias")
        self.parallel.fully_shard = lambda model, **kwargs: model
        self.parallel.FSDPModule = type("NativeFSDP", (), {})

        def optimizer(model, optimizer_type="adamw", optimizer_config=None):
            return (model, optimizer_type, optimizer_config)

        self.native.build_optimizer = optimizer

        def clip(model, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
            return "native-clip"

        self.clipping.clip_grad_norm = clip
        self.original_clip = clip
        self.alias.clip = clip
        self.alias.shard = self.parallel.fully_shard
        self.alias.optimizer = optimizer
        self.original = self.parallel.fully_shard
        modules = {self.parallel.__name__: self.parallel, self.native.__name__: self.native,
                   self.clipping.__name__: self.clipping,
                   self.alias.__name__: self.alias}
        self.registry = patch.dict(sys.modules, modules)
        self.registry.start()
        self.imports = patch("hyper_parallel.integration.veomni.import_module", side_effect=modules.__getitem__)
        self.imports.start()
        self.adapter = VeOmniAdapter()

    def tearDown(self):
        self.adapter.disable()
        self.imports.stop()
        self.registry.stop()

    def test_enable_and_disable_restore_existing_and_late_aliases(self):
        """One startup activation handles all currently loaded framework aliases."""
        self.assertIs(self.adapter.enable(), self.adapter)
        self.assertIs(self.adapter.enable(), self.adapter)
        self.assertIs(self.alias.shard, self.parallel.fully_shard)
        self.alias.late = self.parallel.fully_shard
        self.adapter.disable()
        self.assertIs(self.alias.shard, self.original)
        self.assertIs(self.alias.late, self.original)

    def test_torch_exports_are_unchanged(self):
        """No global type substitution or norm patch is installed."""
        clip = torch.nn.utils.clip_grad_norm_
        self.adapter.enable()
        self.assertIs(torch.distributed.tensor.DTensor, DTensor)
        self.assertIs(torch.nn.utils.clip_grad_norm_, clip)

    def test_managed_model_uses_hyper_native_clip(self):
        """Forward all clipping options without replacing Tensor objects."""
        model = torch.nn.Linear(2, 2)
        _extend_module_with_hsdp_interface(model)
        parameters = list(model.parameters())
        self.adapter.enable()
        with patch("hyper_parallel.integration.veomni.clip_grad_norm_", return_value=torch.tensor(3.0)) as clip:
            actual = self.alias.clip(model, 2.0, norm_type=3.0, error_if_nonfinite=True, foreach=False)
        self.assertEqual(float(actual), 3.0)
        forwarded = list(clip.call_args.args[0])
        self.assertEqual([id(value) for value in forwarded], [id(value) for value in parameters])
        self.assertEqual(clip.call_args.args[1:], (2.0, 3.0, True, False))

    def test_native_model_and_disabled_clip_keep_original(self):
        """Only Hyper-managed models enter Hyper clipping while activated."""
        model = torch.nn.Linear(2, 2)
        self.adapter.enable()
        wrapper = self.alias.clip
        self.assertEqual(wrapper(model, 1.0), "native-clip")
        self.adapter.disable()
        self.assertIs(self.alias.clip, self.original_clip)
        self.assertEqual(wrapper(model, 1.0), "native-clip")

    def test_binding_constructs_hyper_with_real_torch_storage(self):
        """Delegate once to the Torch-storage entry without preparing input DTensors."""
        model = torch.nn.Linear(2, 2)
        self.adapter.enable()
        original = model.weight
        with patch("hyper_parallel.integration.veomni.fully_shard", return_value=model) as shard:
            self.assertIs(self.parallel.fully_shard(model), model)
        shard.assert_called_once_with(model, use_torch_dtensor=True, reshard_after_forward=False)
        self.assertIs(model.weight, original)

    def test_plain_model_optimizer_preserves_native_factory(self):
        """An unsharded plain model does not enter Hyper Muon construction."""
        model = torch.nn.Linear(2, 2)
        self.adapter.enable()
        self.assertEqual(self.native.build_optimizer(model, optimizer_type="muon"), (model, "muon", None))

    def test_conflicting_adapter_is_rejected(self):
        """Do not stack construction wrappers owned by different activations."""
        self.adapter.enable()
        with self.assertRaisesRegex(RuntimeError, "already active"):
            VeOmniAdapter().enable()

    def test_external_rebinding_is_not_overwritten(self):
        """Cleanup only restores the identities still owned by this adapter."""
        self.adapter.enable()

        def replacement(*args):
            return None

        self.alias.shard = replacement
        self.adapter.disable()
        self.assertIs(self.alias.shard, replacement)

    def test_bad_signature_does_not_mutate_shard_entry(self):
        """An incompatible framework version fails before activation."""
        self.native.build_optimizer = lambda model: model
        with self.assertRaisesRegex(RuntimeError, "signature"):
            self.adapter.enable()
        self.assertIs(self.parallel.fully_shard, self.original)

    def test_placement_translation_rejects_unknown_layout(self):
        """Unsupported placements cannot silently become replication."""
        self.assertEqual(_placement(Shard(1)).dim, 1)
        self.assertIsNone(_placement(None))
        with self.assertRaisesRegex(NotImplementedError, "Shard"):
            _placement(object())


class TestStateTemplates(unittest.TestCase):
    """Temporary DCP destinations must never execute optimizer updates."""

    def setUp(self):
        self.model = torch.nn.Linear(2, 2)
        self.leaf = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.leaf.is_muon = True

    def test_template_does_not_step_or_persist_state(self):
        """A state query leaves fresh parameters and optimizer state unchanged."""
        before = self.model.weight.detach().clone()
        with patch.object(self.leaf, "step") as step:
            with _state_templates([self.leaf]):
                self.assertIn("momentum_buffer", self.leaf.state[self.model.weight])
            step.assert_not_called()
        self.assertEqual(len(self.leaf.state), 0)
        torch.testing.assert_close(self.model.weight, before)

    def test_successful_restore_retains_loaded_momentum(self):
        """State written into the template survives the restore context."""
        parameter = self.model.weight
        with _state_templates([self.leaf], retain=True):
            self.leaf.state[parameter]["momentum_buffer"].fill_(3)
        self.assertIs(self.model.weight, parameter)
        torch.testing.assert_close(self.leaf.state[parameter]["momentum_buffer"], torch.full_like(parameter, 3))

    def test_failed_restore_removes_only_temporary_entries(self):
        """Do not leave synthetic state behind after a failed load."""
        existing = {"momentum_buffer": torch.ones_like(self.model.weight)}
        self.leaf.state[self.model.weight] = existing
        with self.assertRaisesRegex(ValueError, "load"):
            with _state_templates([self.leaf], retain=True):
                raise ValueError("load failed")
        self.assertIs(self.leaf.state[self.model.weight], existing)
        self.assertNotIn(self.model.bias, self.leaf.state)
