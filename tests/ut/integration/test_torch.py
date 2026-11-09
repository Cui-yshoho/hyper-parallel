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
"""Final-storage selection contracts with genuine Torch DTensors and mocked meshes."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh as TorchDeviceMesh
from torch.distributed.tensor import DTensor as TorchDTensor
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2
from hyper_parallel.core.fully_shard.hsdp_state import HSDPStateV2
from hyper_parallel.core.fully_shard.api import fully_shard as native_fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy, SourceShardMetaInfo
fully_shard = partial(native_fully_shard, use_torch_dtensor=True)


class TestTorchStorageEntry(unittest.TestCase):
    """The entry selects final storage while preserving ordinary input parameters."""

    def setUp(self):
        """Mock mesh construction, not the DTensor wrapper or its storage."""
        self.mesh = MagicMock()
        self.mesh.ndim = 1
        self.mesh.device_type = "cpu"
        self.mesh.mesh = torch.tensor([0, 1])
        self.torch_mesh = MagicMock(spec=TorchDeviceMesh)
        self.torch_mesh.ndim = 1
        self.torch_mesh.device_type = "cpu"
        self.torch_mesh.mesh = self.mesh.mesh
        self.tensor_patch = patch(
            "hyper_parallel.core.fully_shard.hsdp_param.TorchDTensor.from_local", side_effect=self._from_local,
        )
        self.tensor_patch.start()
        self.addCleanup(self.tensor_patch.stop)
        self.builder = patch(
            "hyper_parallel.core.fully_shard.api.HSDPModule.hsdp_init",
            side_effect=self._initialize,
        )
        self.native = self.builder.start()
        self.addCleanup(self.builder.stop)
        self.device_patch = patch(
            "hyper_parallel.core.fully_shard.api._get_device_from_mesh", return_value=torch.device("cpu"),
        )
        self.device_patch.start()
        self.addCleanup(self.device_patch.stop)

    @staticmethod
    def _initialize(modules, *args, **kwargs):
        """Model the scheduler handle installed by the real constructor."""
        del args, kwargs
        modules[0].hsdp_scheduler = MagicMock()

    @staticmethod
    def _from_local(local, mesh, placements, *, run_check=False, shape=None, stride=None):
        """Construct a real DTensor without requiring a distributed process group."""
        del run_check
        spec = DTensorSpec(
            mesh, tuple(placements), TensorMeta(shape or local.shape, stride or local.stride(), local.dtype),
        )
        return TorchDTensor(local, spec, requires_grad=local.requires_grad)

    def test_plain_shared_parameter_preserves_storage_metadata_and_dtype_override(self):
        """No pre-wrapping or identity remapping occurs before native partitioning."""
        first, second = nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)
        second.weight = first.weight
        original = first.weight
        original.custom_name = "shared"
        def hook(grad):
            return grad

        original.register_hook(hook)
        policy = MixedPrecisionPolicy(custom_params={original: {"reduce_dtype": torch.float32}})
        placement_fn = MagicMock(return_value=None)
        fully_shard(
            [first, second], mesh=self.mesh, replicate_params={original}, mp_policy=policy,
            shard_placement_fn=placement_fn,
        )
        wrapped = first.weight
        self.assertIs(wrapped, original)
        self.assertNotIsInstance(wrapped, TorchDTensor)
        self.assertIs(second.weight, wrapped)
        self.assertEqual(wrapped.data_ptr(), original.data_ptr())
        self.assertEqual(wrapped.custom_name, "shared")
        self.assertIn(hook, wrapped._backward_hooks.values())
        options = self.native.call_args.kwargs
        self.assertIs(options["_dtensor_type"], TorchDTensor)
        self.assertEqual(self.native.call_args.args[7], {original})
        self.assertIs(self.native.call_args.args[4], policy)
        self.assertEqual(policy.custom_params, {id(original): {"reduce_dtype": torch.float32}})
        self.native.call_args.args[3](wrapped)
        placement_fn.assert_called_once_with(original)
        self.assertNotIn("use_torch_dtensor", options)

    def test_existing_torch_dtensor_is_forwarded_without_replacement(self):
        """Existing input layout and object identity do not depend on a flag."""
        model = nn.Linear(2, 2, bias=False)
        model.weight = nn.Parameter(self._from_local(model.weight.detach(), self.torch_mesh, (Shard(0),)))
        original = model.weight
        fully_shard(model, mesh=self.mesh)
        self.assertIs(model.weight, original)
        self.assertEqual(model.weight.placements, (Shard(0),))

    def test_ignored_parameter_remains_plain(self):
        """Ignored parameters are neither wrapped nor included in managed dtype rules."""
        model = nn.Linear(2, 2)
        ignored = model.bias
        fully_shard(model, mesh=self.mesh, ignored_params={ignored})
        self.assertIs(model.bias, ignored)
        self.assertNotIsInstance(model.bias, TorchDTensor)
        self.assertNotIsInstance(model.weight, TorchDTensor)
        self.assertEqual(self.native.call_args.args[6], {ignored})

    def test_accumulated_gradient_rejected_before_rebinding(self):
        """Preparation must not silently drop a caller's existing gradients."""
        model = nn.Linear(2, 2, bias=False)
        original = model.weight
        original.grad = torch.ones_like(original)
        with self.assertRaisesRegex(ValueError, "before accumulating gradients"):
            fully_shard(model, mesh=self.mesh)
        self.assertIs(model.weight, original)
        self.native.assert_not_called()

    def test_hyper_input_rejected_before_module_rebinding(self):
        """Opting in must not silently reinterpret an existing Hyper layout."""
        model = nn.Linear(2, 2, bias=False)
        parameter = MagicMock(spec=DTensor)
        model._parameters["weight"] = parameter
        original_type = type(model)
        with self.assertRaisesRegex(ValueError, "cannot convert existing Hyper"):
            fully_shard(model, mesh=self.mesh)
        self.assertIs(type(model), original_type)
        self.assertIs(model.weight, parameter)
        self.native.assert_not_called()

    def test_replicated_input_does_not_add_another_reduction_domain(self):
        """A replicated representation over the DP ranks is not a second DP axis."""
        state = object.__new__(HSDPStateV2)
        state.source_shard_infos = None
        state.mesh = self.mesh
        parameter = nn.Parameter(self._from_local(torch.ones(2, 2), self.torch_mesh, (Replicate(),)))
        with patch("hyper_parallel.core.fully_shard.hsdp_state.DeviceMesh.from_group", return_value=self.mesh):
            self.assertIsNone(state._build_param_source_shard_info(parameter))

    def test_source_layout_is_read_from_torch_input(self):
        """Keep a TP/EP axis that is not owned by the FSDP rank domain."""
        state = object.__new__(HSDPStateV2)
        state.source_shard_infos = None
        state.mesh = self.mesh
        source_mesh = MagicMock()
        source_mesh.mesh = torch.tensor([0, 2])
        parameter = nn.Parameter(self._from_local(torch.ones(2, 2), self.torch_mesh, (Shard(0),)))
        with patch("hyper_parallel.core.fully_shard.hsdp_state.DeviceMesh.from_group", return_value=source_mesh):
            metadata = state._build_param_source_shard_info(parameter)
        self.assertIs(metadata.mesh, source_mesh)
        self.assertEqual(metadata.placements[0].dim, 0)
        self.assertTrue(metadata.origin_is_dtensor)

    def test_partial_input_is_rejected_instead_of_treated_as_complete(self):
        """Partial values require a reduction that this adapter must not hide."""
        state = object.__new__(HSDPStateV2)
        state.source_shard_infos = None
        parameter = nn.Parameter(self._from_local(torch.ones(2, 2), self.torch_mesh, (Partial(),)))
        with self.assertRaisesRegex(NotImplementedError, "source placement"):
            state._build_param_source_shard_info(parameter)

    def test_core_storage_type_is_inferred_from_parameter(self):
        """Native initialization accepts a genuine Torch input without a storage option."""
        model = nn.Linear(2, 2, bias=False)
        parameter = nn.Parameter(self._from_local(model.weight.detach(), self.torch_mesh, (Replicate(),)))
        module_info = MagicMock(module=model, param_name="weight")
        mesh_info = MagicMock(shard_mesh_dim=0, replicate_mesh_dim=None)
        with patch.object(HSDPParamV2, "_init_sharded_param"), patch.object(
            HSDPParamV2, "_build_storage_source_layout", return_value=(None, ()),
        ):
            managed = HSDPParamV2(parameter, module_info, mesh_info, device=torch.device("cpu"))
        self.assertIs(managed._dtensor_type, TorchDTensor)
        self.assertIsNot(managed._dtensor_type, DTensor)

    def test_core_rejects_sharded_input_without_source_metadata(self):
        """Missing source metadata must not silently discard a Torch shard axis."""
        parameter = nn.Parameter(self._from_local(torch.ones(2, 2), self.torch_mesh, (Shard(0),)))
        with self.assertRaisesRegex(ValueError, "original layout metadata"):
            HSDPParamV2(parameter, MagicMock(), MagicMock(), device=torch.device("cpu"))

    def test_unshard_restores_torch_source_metadata_from_existing_layout(self):
        """Reuse global shape/stride and original mesh without duplicate source caches."""
        parameter = nn.Parameter(self._from_local(
            torch.ones(3, 4), self.torch_mesh, (Shard(0),), shape=torch.Size((6, 4)), stride=(4, 1),
        ))
        metadata = SourceShardMetaInfo(MagicMock(), (MagicMock(),), origin_is_dtensor=True)
        with patch.object(HSDPParamV2, "_init_sharded_param"), patch.object(
            HSDPParamV2, "_build_storage_source_layout", return_value=(None, ()),
        ):
            managed = HSDPParamV2(parameter, MagicMock(), MagicMock(), source_shard_info=metadata)
        managed._sharding_spec = SimpleNamespace(tensor_shape=parameter.shape, tensor_stride=parameter.stride())
        managed._orig_size = torch.Size((3, 4))
        managed._contiguous_orig_stride = (4, 1)
        managed.unsharded_param_buffers = [torch.arange(12, dtype=torch.float32)]
        managed.sharded_param = parameter
        managed.init_unsharded_param()
        restored = managed.unsharded_param
        self.assertIsInstance(restored, TorchDTensor)
        self.assertIs(restored.device_mesh, parameter.device_mesh)
        self.assertEqual(restored.placements, parameter.placements)
        self.assertEqual(restored.shape, parameter.shape)
        self.assertEqual(restored.stride(), parameter.stride())
        self.assertEqual(restored.to_local().data_ptr(), managed.unsharded_param_buffers[0].data_ptr())

    def test_plain_input_can_select_torch_storage_without_source_layout(self):
        """The final representation no longer requires a replicated source DTensor."""
        model = nn.Linear(2, 2, bias=False)
        with patch.object(HSDPParamV2, "_init_sharded_param"), patch.object(
            HSDPParamV2, "_build_storage_source_layout", return_value=(None, ()),
        ):
            managed = HSDPParamV2(model.weight, MagicMock(), MagicMock(), _dtensor_type=TorchDTensor)
        self.assertIs(managed._dtensor_type, TorchDTensor)
        self.assertIsNone(managed.source_shard_info)
        self.assertFalse(managed._orig_param_is_dtensor)
        self.assertIsNone(managed._orig_dtensor_mesh)
        self.assertEqual(managed._src_parameter_id, id(model.weight))

    def test_native_plain_input_keeps_hyper_storage(self):
        """A compatibility call must not set any global representation default."""
        model = nn.Linear(2, 2, bias=False)
        fully_shard(model, mesh=self.mesh)
        with patch.object(HSDPParamV2, "_init_sharded_param"), patch.object(
            HSDPParamV2, "_build_storage_source_layout", return_value=(None, ()),
        ):
            managed = HSDPParamV2(model.weight, MagicMock(), MagicMock())
        self.assertIs(managed._dtensor_type, DTensor)
