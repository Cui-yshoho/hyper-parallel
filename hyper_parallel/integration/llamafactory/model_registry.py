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
"""Model-specific semantic parallel plans for the LlamaFactory integration."""

from collections.abc import Callable

from hyper_parallel.models import get_model_adapter

_MODEL_PARALLEL_PLAN_REGISTRY: dict[str, Callable[[], list[dict]]] = {}


def register_model_parallel_plan(model_type: str):
    """Register a semantic parallel plan for one Hyper model adapter."""

    def decorator(factory: Callable[[], list[dict]]):
        if model_type in _MODEL_PARALLEL_PLAN_REGISTRY:
            raise ValueError(
                f"Parallel plan already registered for model_type {model_type!r}."
            )
        _MODEL_PARALLEL_PLAN_REGISTRY[model_type] = factory
        return factory

    return decorator


def get_model_parallel_plan(model_type: str | None) -> list[dict]:
    """Return the registered semantic plan without adding performance modules."""
    if model_type is None:
        return []

    adapter = get_model_adapter(model_type)
    canonical_type = adapter.model_type if adapter is not None else model_type
    factory = _MODEL_PARALLEL_PLAN_REGISTRY.get(canonical_type)
    return factory() if factory is not None else []


@register_model_parallel_plan("qwen3_moe")
def _qwen3_moe_parallel_plan() -> list[dict]:
    """Register native-module Qwen3-MoE CP and EP execution semantics."""
    return [
        {
            "match": "*.self_attn",
            "when": "cp",
            "inner_target": "self",
            "region_dispatch": False,
            "inner_wrapper": {
                "_target_": (
                    "hyper_parallel.models.qwen3_moe.adapter.distributed.context_parallel_async."
                    "qwen3_moe_async_ulysses_cp_wrapper"
                )
            },
        },
        {
            "match": "*.mlp",
            "when": "ep",
            "region_dispatch": False,
            "local_compute_fn": {
                "_target_": (
                    "hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel."
                    "qwen3moe_ep_compute_fn"
                ),
                "use_grouped_gemm": True,
            },
        },
    ]


@register_model_parallel_plan("qwen3_vl_moe")
def _qwen3_vl_moe_parallel_plan() -> list[dict]:
    """Register CP and EP only for Qwen3-VL-MoE's text transformer."""
    return [
        {
            "match": "model.visual.blocks.*.attn",
            "when": "cp",
            "inner_target": "self",
            "region_dispatch": False,
            "inner_wrapper": {
                "_target_": (
                    "hyper_parallel.integration.llamafactory.qwen3_vl_moe."
                    "qwen3_vl_moe_replicated_vision_cp_wrapper"
                )
            },
        },
        {
            "match": "model.visual.merger",
            "when": "cp",
            "inner_target": "self",
            "region_dispatch": False,
            "inner_wrapper": {
                "_target_": (
                    "hyper_parallel.integration.llamafactory.qwen3_vl_moe."
                    "qwen3_vl_moe_replicated_vision_cp_wrapper"
                )
            },
        },
        {
            "match": "model.visual.deepstack_merger_list",
            "when": "cp",
            "inner_target": "self",
            "region_dispatch": False,
            "inner_wrapper": {
                "_target_": (
                    "hyper_parallel.integration.llamafactory.qwen3_vl_moe."
                    "qwen3_vl_moe_replicated_vision_cp_wrapper"
                )
            },
        },
        {
            "match": "model.language_model",
            "when": "cp",
            "params": {},
            "inner_target": "self",
            "region_dispatch": False,
            "inner_wrapper": {
                "_target_": (
                    "hyper_parallel.integration.llamafactory.qwen3_vl_moe."
                    "qwen3_vl_moe_text_input_cp_wrapper"
                )
            },
        },
        {
            "match": "model.language_model.layers.*.self_attn",
            "when": "cp",
            "inner_target": "self",
            "region_dispatch": False,
            "inner_wrapper": {
                "_target_": (
                    "hyper_parallel.models.qwen3_moe.adapter.distributed.context_parallel_async."
                    "qwen3_moe_async_ulysses_cp_wrapper"
                )
            },
        },
        {
            "match": "model.language_model.layers.*.mlp",
            "when": "ep",
            "region_dispatch": False,
            "local_compute_fn": {
                "_target_": (
                    "hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel."
                    "qwen3moe_ep_compute_fn"
                ),
                "use_grouped_gemm": True,
            },
        },
    ]
