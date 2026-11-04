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
"""Distributed correctness checks for early non-fused FSDP gradient accumulation."""

import copy
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint

from hyper_parallel import init_device_mesh
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2


class _Network(nn.Module):
    """Nested FSDP units with optional non-reentrant activation checkpointing."""

    def __init__(self, use_checkpoint: bool) -> None:
        """Create independently sharded layers."""
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(8, 8, bias=False) for _ in range(4))
        self.use_checkpoint = use_checkpoint

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Execute each layer, optionally replaying it during backward."""
        for layer in self.layers:
            inputs = checkpoint(layer, inputs, use_reentrant=False) if self.use_checkpoint else layer(inputs)
        return inputs


def _check_early_grad_accumulation(device_type: str, use_checkpoint: bool) -> None:
    device = torch.device(device_type, int(os.environ["LOCAL_RANK"])) if device_type == "cuda" else torch.device("cpu")
    if device_type == "cuda":
        torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group("nccl" if device_type == "cuda" else "gloo")
    rank = dist.get_rank()
    mesh = init_device_mesh(device_type, (dist.get_world_size(),), mesh_dim_names=("fsdp",))
    torch.manual_seed(1234)
    model = _Network(use_checkpoint).to(device)
    reference = copy.deepcopy(model)
    root_active = False
    early_applies = []
    original_root = HSDPSchedulerV2._root_backward_hook
    original_apply = HSDPParamV2.apply_reduced_grad

    def traced_root(scheduler: HSDPSchedulerV2) -> None:
        """Mark finalization so early gradient application can be distinguished."""
        nonlocal root_active
        root_active = True
        try:
            return original_root(scheduler)
        finally:
            root_active = False

    def traced_apply(param: HSDPParamV2, grad: torch.Tensor) -> bool:
        """Track applications before root finalization without changing gradients."""
        if not root_active:
            early_applies.append(param)
        return original_apply(param, grad)

    # CPU meshes exercise the scheduler with Gloo; public FSDP normally resolves
    # accelerator devices here. The CUDA variant uses its actual local device.
    with patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh", return_value=device), \
         patch.object(HSDPSchedulerV2, "_root_backward_hook", traced_root), \
         patch.object(HSDPParamV2, "apply_reduced_grad", traced_apply):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, comm_fusion=False)
        fully_shard(model, mesh=mesh, comm_fusion=False)
        model.set_reduce_op_type("sum")
        model.set_reshard_after_backward(True)
        for iteration in range(2):
            model.zero_grad()
            reference.zero_grad(set_to_none=True)
            for micro in range(4):
                inputs = torch.arange(16, dtype=torch.float32, device=device).view(2, 8) / 64
                inputs = inputs + (iteration + micro + rank) / 16
                model(inputs).square().sum().backward()
                reference(inputs).square().sum().backward()
                for sharded, eager in zip(model.parameters(), reference.parameters()):
                    expected = eager.grad.detach().clone()
                    dist.all_reduce(expected, op=dist.ReduceOp.SUM)
                    expected_shard = torch.chunk(expected, dist.get_world_size(), dim=0)[rank]
                    torch.testing.assert_close(sharded.grad._local_tensor, expected_shard, rtol=1e-5, atol=1e-6)
    assert len(early_applies) >= 8, f"Expected at least eight early applications, got {len(early_applies)}"


def test_early_grad_accumulation_matches_distributed_eager() -> None:
    """Compare CPU/Gloo gradients after every microstep, with and without recomputation."""
    for use_checkpoint in (False, True):
        _check_early_grad_accumulation("cpu", use_checkpoint)


def test_early_grad_accumulation_matches_distributed_eager_cuda() -> None:
    """Compare CUDA/NCCL gradients under an allocated GPU job."""
    for use_checkpoint in (False, True):
        _check_early_grad_accumulation("cuda", use_checkpoint)
