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
"""Qwen3-VL-MoE input adaptation for LlamaFactory context parallelism."""

from __future__ import annotations

from typing import Any

import torch

from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.distributed.recipe_spec import inner_wrapper


@inner_wrapper
def qwen3_vl_moe_replicated_vision_cp_wrapper(
    target_module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> _ForwardRewriteRequest:
    """Keep the vision attention replicated while text tokens use CP."""
    del mesh, tp_mesh, ep_mesh
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError("Qwen3-VL-MoE replicated vision CP requires an active CP mesh")
    return _ForwardRewriteRequest(target_module, target_module.forward)


def _slice_dense_visual_features(
    features: torch.Tensor,
    global_mask: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor:
    """Select mask-ordered visual features belonging to one local token shard."""
    global_mask = global_mask.to(torch.bool)
    local_mask = global_mask[:, start:end]
    if not local_mask.any():
        return features[:0]

    batch_size, seq_len = global_mask.shape
    local_positions = (
        torch.arange(start, end, device=global_mask.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    flat_positions = (
        torch.arange(batch_size, device=global_mask.device).unsqueeze(1) * seq_len
        + local_positions
    )[local_mask]
    feature_indices = (
        global_mask.reshape(-1).to(torch.int64).cumsum(0)[flat_positions] - 1
    )
    return features.index_select(0, feature_indices.to(features.device))


@inner_wrapper
def qwen3_vl_moe_text_input_cp_wrapper(
    target_module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> _ForwardRewriteRequest:
    """Shard text inputs after Qwen3-VL has injected global visual features."""
    del mesh, tp_mesh, ep_mesh
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError("Qwen3-VL-MoE text input CP requires an active CP mesh")

    original_forward = target_module.forward
    target_module._hp_cp_shards_inputs = True

    def cp_forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        **kwargs,
    ):
        sequence = inputs_embeds if inputs_embeds is not None else input_ids
        if sequence is None:
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )

        seq_len = sequence.shape[1]
        cp_size = cp_mesh.size()
        if seq_len % cp_size:
            raise ValueError(
                f"Qwen3-VL-MoE sequence length ({seq_len}) must be divisible by CP size ({cp_size})"
            )
        local_seq_len = seq_len // cp_size
        start = cp_mesh.get_local_rank() * local_seq_len
        end = start + local_seq_len

        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds[:, start:end, :].contiguous()
        if input_ids is not None:
            input_ids = input_ids[:, start:end].contiguous()
        if isinstance(position_ids, torch.Tensor):
            position_ids = position_ids[..., start:end].contiguous()
        if isinstance(visual_pos_masks, torch.Tensor):
            global_visual_pos_masks = visual_pos_masks
            visual_pos_masks = global_visual_pos_masks[:, start:end].contiguous()
            if deepstack_visual_embeds is not None:
                deepstack_visual_embeds = [
                    _slice_dense_visual_features(
                        features, global_visual_pos_masks, start, end
                    )
                    for features in deepstack_visual_embeds
                ]

        return original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

    return _ForwardRewriteRequest(target_module, cp_forward)


__all__ = [
    "qwen3_vl_moe_replicated_vision_cp_wrapper",
    "qwen3_vl_moe_text_input_cp_wrapper",
]
