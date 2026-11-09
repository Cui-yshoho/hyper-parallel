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
"""Real Hyper scheduling with selectable persistent DTensor representation."""

import copy
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh as TorchDeviceMesh
from torch.distributed.tensor import DTensor as TorchDTensor
from torch.distributed.tensor import Replicate as TorchReplicate
from torch.distributed.tensor import Shard as TorchShard

from hyper_parallel import DTensor, Shard, SkipDTensorDispatch, fully_shard, init_device_mesh
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy, OffloadPolicy
from hyper_parallel.core.utils import clip_grad_norm_ as hyper_clip_grad_norm


class TinyModel(nn.Module):
    """Dense model with no external files or accelerator operators."""

    def __init__(self, rows=4):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4) / 32)

    def forward(self, inputs):
        return inputs @ self.weight


def _local_shard(tensor, shard_size, shard_rank, shard_dim):
    chunks = tensor.chunk(shard_size, shard_dim)
    return chunks[shard_rank] if shard_rank < len(chunks) else tensor.narrow(shard_dim, tensor.size(shard_dim), 0)


def _run_case(mesh, use_torch_dtensor, comm_fusion, rows=4, shard_dim=0, prepared_input=False):
    model = TinyModel(rows)
    reference = copy.deepcopy(model)
    if prepared_input:
        torch_mesh = TorchDeviceMesh.from_group(
            mesh.get_all_groups(), mesh.device_type, mesh=mesh.mesh, mesh_dim_names=mesh.mesh_dim_names,
        )
        model.weight = nn.Parameter(TorchDTensor.from_local(
            model.weight.detach(), torch_mesh, tuple(TorchReplicate() for _ in range(mesh.ndim)),
            run_check=False, shape=model.weight.shape, stride=model.weight.stride(),
        ))
    # CPU bring-up changes only the accelerator guard, not the public builder
    # or its scheduler, collectives, parameter representation or clipping.
    with patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh", return_value=torch.device("cpu")):
        options = dict(mesh=mesh, shard_placement_fn=lambda _: Shard(shard_dim),
                       mp_policy=MixedPrecisionPolicy(), offload_policy=OffloadPolicy(), comm_fusion=comm_fusion,
                       comm_fusion_zero_copy=False)
        fully_shard(model, use_torch_dtensor=use_torch_dtensor, **options)
    # Gloo cannot implement every accelerator reduction mode. SUM is an
    # existing Hyper option, used identically for both representations here.
    model.hsdp_scheduler.hsdp_state.set_reduce_op_type("sum")
    parameter = model.weight
    expected_type = TorchDTensor if use_torch_dtensor else DTensor
    assert isinstance(parameter, expected_type), f"Expected {expected_type}, got {type(parameter)}"
    if use_torch_dtensor:
        assert not isinstance(parameter, DTensor), f"Expected genuine Torch DTensor, got {type(parameter)}"
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=True)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01, foreach=True)
    shard_rank = mesh.get_local_rank(mesh.ndim - 1)
    shard_size = mesh.size(mesh.ndim - 1)
    for step in range(3):
        for micro in range(2):
            inputs = torch.arange(2 * rows, dtype=torch.float32).reshape(2, rows) / 16
            inputs = inputs + dist.get_rank() / 10 + micro / 100 + step / 1000
            model(inputs).square().sum().backward()
            reference(inputs).square().sum().backward()
        for value in reference.parameters():
            dist.all_reduce(value.grad)
        grad = model.weight.grad
        assert isinstance(grad, expected_type), f"Expected gradient {expected_type}, got {type(grad)}"
        expected_grad = _local_shard(reference.weight.grad, shard_size, shard_rank, shard_dim)
        torch.testing.assert_close(grad.to_local(), expected_grad, rtol=1e-5, atol=1e-6)
        reference_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0, foreach=False)
        actual_norm = hyper_clip_grad_norm(model.parameters(), 1.0, foreach=False)
        torch.testing.assert_close(actual_norm, reference_norm, rtol=1e-5, atol=1e-6)
        if use_torch_dtensor:
            optimizer.step()
        else:
            with SkipDTensorDispatch():
                optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(
            model.weight.to_local(), _local_shard(reference.weight, shard_size, shard_rank, shard_dim),
            rtol=1e-5, atol=1e-6,
        )
        assert model.weight is parameter, f"Parameter identity changed from {id(parameter)} to {id(model.weight)}"
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        assert model.weight.grad is None, f"Expected cleared gradient, got {model.weight.grad}"


def _run_list_case(mesh):
    """Use the second root's real handle after constructing a shared FSDP unit."""
    modules = [TinyModel(), TinyModel()]
    expected = [module.weight.detach().clone() for module in modules]
    with patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh", return_value=torch.device("cpu")):
        result = fully_shard(modules, mesh=mesh, use_torch_dtensor=True)
    assert result is modules, f"Expected original list {id(modules)}, got {id(result)}"
    assert modules[1].hsdp_scheduler is modules[0].hsdp_scheduler, (
        f"Scheduler handles differ: {id(modules[1].hsdp_scheduler)} != {id(modules[0].hsdp_scheduler)}"
    )
    modules[1].unshard()
    for index, module in enumerate(modules):
        torch.testing.assert_close(module.weight, expected[index])
    modules[1].reshard()
    for module in modules:
        assert isinstance(module.weight, TorchDTensor), f"Expected Torch shard, got {type(module.weight)}"


def _run_source_shard_case(source_dim):
    """Compose existing Torch model-parallel shards with Hyper DP collectives."""
    root = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
    dp_mesh, tp_mesh = root["dp"], root["tp"]
    torch_tp = TorchDeviceMesh.from_group(
        tp_mesh.get_all_groups(), "cpu", mesh=tp_mesh.mesh, mesh_dim_names=tp_mesh.mesh_dim_names,
    )
    full = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 32
    source_local = full.chunk(2, dim=source_dim)[tp_mesh.get_local_rank()].clone()
    model = nn.Module()
    model.weight = nn.Parameter(TorchDTensor.from_local(
        source_local, torch_tp, (TorchShard(source_dim),),
        run_check=False, shape=full.shape, stride=full.stride(),
    ))
    reference = nn.Parameter(source_local.clone())
    model.forward = lambda inputs: inputs @ model.weight.to_local()
    with patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh", return_value=torch.device("cpu")):
        fully_shard(model, mesh=dp_mesh, shard_placement_fn=lambda _: Shard(1 - source_dim))
    state = model.hsdp_scheduler.hsdp_state
    state.set_reduce_op_type("sum")
    parameter = model.weight
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=True)
    reference_optimizer = torch.optim.AdamW([reference], lr=0.01, foreach=True)
    self_shape = source_local.shape
    for step in range(3):
        for micro in range(2):
            inputs = torch.arange(2 * self_shape[0], dtype=torch.float32).reshape(2, self_shape[0]) / 16
            inputs = inputs + dist.get_rank() / 10 + micro / 100 + step / 1000
            model(inputs).square().sum().backward()
            (inputs @ reference).square().sum().backward()
        dist.all_reduce(reference.grad, group=dp_mesh.get_group())
        expected_grad = _local_shard(reference.grad, 2, dp_mesh.get_local_rank(), 1 - source_dim)
        torch.testing.assert_close(parameter.grad.to_local(), expected_grad, rtol=1e-5, atol=1e-6)
        norm_squared = reference.grad.square().sum()
        dist.all_reduce(norm_squared, group=tp_mesh.get_group())
        expected_norm = norm_squared.sqrt()
        actual_norm = hyper_clip_grad_norm(model.parameters(), 1.0, foreach=False)
        torch.testing.assert_close(actual_norm, expected_norm, rtol=1e-5, atol=1e-6)
        reference.grad.mul_(torch.clamp(1.0 / (expected_norm + 1e-6), max=1.0))
        optimizer.step()
        reference_optimizer.step()
        expected = _local_shard(reference, 2, dp_mesh.get_local_rank(), 1 - source_dim)
        torch.testing.assert_close(parameter.to_local(), expected, rtol=1e-5, atol=1e-6)
        assert parameter.shape == full.shape, f"Expected global shape {full.shape}, got {parameter.shape}"
        assert model.weight is parameter, f"Parameter identity changed from {id(parameter)} to {id(model.weight)}"
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)


def test_torch_dtensor_storage():
    """Feature: FSDP storage. Description: Run real collectives. Expectation: equal gradients and updates."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        for topology in ((4,), (2, 2)):
            names = ("shard",) if len(topology) == 1 else ("replicate", "shard")
            mesh = init_device_mesh("cpu", topology, mesh_dim_names=names)
            for use_torch_dtensor in (False, True):
                for comm_fusion in (False, True):
                    _run_case(mesh, use_torch_dtensor, comm_fusion)
            _run_case(mesh, True, False, shard_dim=1)
            _run_case(mesh, True, False, rows=5)
            _run_case(mesh, True, False, prepared_input=True)
            _run_list_case(mesh)
        for source_dim in (0, 1):
            _run_source_shard_case(source_dim)
        print(f"TORCH_DTENSOR_STORAGE_PASS rank={dist.get_rank()}", flush=True)
    finally:
        dist.destroy_process_group()
