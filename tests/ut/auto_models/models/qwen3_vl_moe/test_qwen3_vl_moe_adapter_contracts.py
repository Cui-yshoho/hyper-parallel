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
"""CPU tests for model-owned Qwen3-VL-MoE distributed adapter contracts."""
# pylint: disable=protected-access

from types import SimpleNamespace

import torch
from torch import nn

from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.models import get_model_adapter
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.qwen3_vl_moe.adapter.distributed import (
    context_parallel,
    expert_parallel,
)
from tests.common.mark_utils import arg_mark


@arg_mark(
    plat_marks=["cpu_linux", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_model_adapter_registers_its_distributed_providers() -> None:
    """The family registry resolves Qwen3-VL-MoE's own CP and EP providers."""
    spec = get_model_adapter("qwen3_vl_moe")

    assert isinstance(spec, ModelAdapterSpec)
    assert spec.architecture == "Qwen3VLMoeForConditionalGeneration"
    assert spec.context_parallel is not None
    assert spec.expert_parallel is not None


@arg_mark(
    plat_marks=["cpu_linux", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_text_cp_converts_global_2d_padding_mask() -> None:
    """The VL adapter owns conversion of its Transformers padding mask."""
    query = torch.empty(2, 1, 4, 4)
    key = torch.empty(2, 1, 4, 4)
    padding_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    actual = context_parallel._prepare_attention_mask(padding_mask, query, key)

    causal = torch.ones(4, 4, dtype=torch.bool).tril()
    expected = causal[None, None, :, :] & padding_mask[:, None, None, :].bool()
    assert torch.equal(actual, expected)


@arg_mark(
    plat_marks=["cpu_linux", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_text_input_cp_wrapper_slices_after_visual_injection() -> None:
    """The VLM wrapper shards text tensors while retaining matching visual features."""
    calls = []
    target = nn.Module()
    target.forward = lambda **kwargs: calls.append(kwargs) or kwargs
    cp_mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: 1)
    request = context_parallel.qwen3_vl_moe_text_input_cp_wrapper(
        target, None, None, cp_mesh, None
    )

    inputs_embeds = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    position_ids = torch.arange(24).reshape(3, 1, 8)
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    visual_mask = torch.tensor([[False, True, True, False, False, True, False, True]])
    deepstack = [torch.tensor([[10.0], [11.0], [12.0], [13.0]])]

    request.forward(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        attention_mask=attention_mask,
        visual_pos_masks=visual_mask,
        deepstack_visual_embeds=deepstack,
    )

    forwarded = calls[0]
    torch.testing.assert_close(forwarded["inputs_embeds"], inputs_embeds[:, 4:8])
    torch.testing.assert_close(forwarded["position_ids"], position_ids[..., 4:8])
    assert forwarded["attention_mask"] is attention_mask
    torch.testing.assert_close(forwarded["visual_pos_masks"], visual_mask[:, 4:8])
    torch.testing.assert_close(forwarded["deepstack_visual_embeds"][0], torch.tensor([[12.0], [13.0]]))


@arg_mark(
    plat_marks=["cpu_linux", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_vision_cp_wrapper_keeps_replicated_forward() -> None:
    """Vision compute remains replicated before the language sequence is sharded."""
    class _VisionModule(nn.Module):
        def forward(self, hidden_states):
            return hidden_states + 1

    target = _VisionModule()
    cp_mesh = SimpleNamespace(size=lambda: 2)

    request = context_parallel.qwen3_vl_moe_replicated_vision_cp_wrapper(
        target, None, None, cp_mesh, None
    )

    inputs = torch.zeros(2, 4, 8)
    torch.testing.assert_close(request.forward(inputs), inputs + 1)
    torch.testing.assert_close(target(inputs), inputs + 1)


@arg_mark(
    plat_marks=["cpu_linux", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_text_cp_wrapper_accepts_qwen3_vl_contract_without_sliding_window() -> None:
    """Qwen3-VL-MoE attention has a model-owned contract without sliding_window."""
    target = nn.Module()
    target.q_proj = nn.Linear(4, 4)
    target.k_proj = nn.Linear(4, 4)
    target.v_proj = nn.Linear(4, 4)
    target.o_proj = nn.Linear(4, 4)
    target.q_norm = nn.LayerNorm(4)
    target.k_norm = nn.LayerNorm(4)
    target.head_dim = 2
    target.scaling = 2**-0.5
    target.config = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2)
    cp_mesh = SimpleNamespace(size=lambda: 2)

    request = context_parallel.qwen3_vl_moe_async_ulysses_cp_wrapper(
        target, None, None, cp_mesh, None
    )

    assert isinstance(request, _ForwardRewriteRequest)
    assert request.target is target


@arg_mark(
    plat_marks=["cpu_linux", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_ep_router_uses_qwen3_vl_model_outputs() -> None:
    """The VL EP adapter preserves the model-native router weights and indices."""
    weights = torch.tensor([[0.75, 0.25]])
    indices = torch.tensor([[3, 1]])
    module = SimpleNamespace(gate=lambda hidden_states: (None, weights, indices))

    actual_indices, actual_weights = expert_parallel._qwen3_vl_moe_router(
        module, torch.zeros(1, 4)
    )

    assert actual_indices is indices
    assert actual_weights is weights
