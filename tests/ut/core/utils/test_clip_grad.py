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
"""Hyper norm algorithm over Torch DTensor storage, with mocked collectives."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

from hyper_parallel.core.utils.clip_grad import (
    _get_local_grad,
    _get_param_mesh_info,
    _is_dtensor,
    _norm_coverage,
    clip_grad_norm_,
)


def parameter(local, placements=(Shard(0),)):
    mesh = SimpleNamespace(get_group=lambda dim: "shard-group")
    grad = MagicMock(spec=DTensor)
    grad._local_tensor = local
    grad.device_mesh = mesh
    grad.placements = placements
    value = MagicMock(spec=DTensor)
    value._local_tensor = torch.zeros_like(local)
    value.to_local.return_value = value._local_tensor
    value.device_mesh = mesh
    value.placements = placements
    value.requires_grad = True
    value.main_grad = None
    value.grad = grad
    return value


class TestTorchStorageClipping(unittest.TestCase):
    """Recognition changes must preserve Hyper communication and scaling."""

    def test_local_gradient_and_main_gradient_share_original_storage(self):
        """No temporary gradient copies can discard the in-place clip."""
        local = torch.tensor([3.0, 4.0])
        value = parameter(local)
        self.assertTrue(_is_dtensor(value))
        self.assertFalse(_is_dtensor(local))
        self.assertIs(_get_local_grad(value), local)
        main = torch.ones_like(local)
        value.main_grad = main
        self.assertIs(_get_local_grad(value), main)

    def test_shard_and_partial_metadata_come_from_torch_placements(self):
        """Both native placement classes retain their actual reduction domain."""
        value = parameter(torch.ones(2), (Partial(), Replicate(), Shard(0)))
        mesh, dims, partial = _get_param_mesh_info(value)
        self.assertIs(mesh, value.device_mesh)
        self.assertEqual(dims, (2,))
        self.assertEqual(partial, ((0, dist.ReduceOp.SUM, False),))

    def test_global_norm_and_in_place_clip_use_existing_hyper_algorithm(self):
        """Two identical rank shards reduce norm squared, not gradients."""
        local = torch.tensor([3.0, 4.0])
        value = parameter(local)
        with patch("hyper_parallel.core.utils.clip_grad.dist") as communication:
            communication.ReduceOp = dist.ReduceOp
            communication.get_process_group_ranks.return_value = [0, 1]
            communication.all_reduce.side_effect = lambda tensor, **kwargs: tensor.mul_(2)
            norm = clip_grad_norm_([value], 1.0, foreach=False)
        expected = torch.tensor(50.0).sqrt()
        torch.testing.assert_close(norm, expected)
        torch.testing.assert_close(local, torch.tensor([3.0, 4.0]) / (expected + 1e-6))
        communication.all_reduce.assert_called_once()
        self.assertEqual(communication.all_reduce.call_args.kwargs["group"], "shard-group")

    def test_frozen_and_grad_free_parameters_do_not_synthesize_gradients(self):
        """All ranks still visit the shard group for an empty gradient bucket."""
        value = parameter(torch.ones(0))
        value.grad = None
        with patch("hyper_parallel.core.utils.clip_grad.dist") as communication:
            communication.ReduceOp = dist.ReduceOp
            communication.get_process_group_ranks.return_value = [0, 1]
            norm = clip_grad_norm_([value], 1.0, foreach=False)
        self.assertEqual(float(norm), 0.0)
        self.assertIsNone(value.grad)
        communication.all_reduce.assert_called_once()

    def test_native_coverage_handles_uneven_and_empty_shards(self):
        """A global shape is not mistaken for a rank-local Tensor layout."""
        value = parameter(torch.zeros(0, 3))
        value.shape = (2, 3)
        with patch("hyper_parallel.core.utils.clip_grad.compute_local_shape_and_global_offset",
                   return_value=((0, 3), (2, 0))):
            self.assertEqual(_norm_coverage(value), ((2, 3), ((2, 2), (0, 3))))

    def test_native_coverage_rejects_inconsistent_local_metadata(self):
        """Do not infer complete coverage from a malformed local shard."""
        value = parameter(torch.zeros(1, 3))
        value.shape = (2, 3)
        with patch("hyper_parallel.core.utils.clip_grad.compute_local_shape_and_global_offset",
                   return_value=((0, 3), (2, 0))):
            with self.assertRaisesRegex(ValueError, "exact local"):
                _norm_coverage(value)
