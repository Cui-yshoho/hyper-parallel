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
"""VeOmni-specific compute boundaries and replay-aware Hyper prefetch policy.

The caller owns precision, meshes and loading. This adapter only wraps declared
stages and configures public Hyper prefetch APIs. Optional model helpers are lazy.
"""

from collections.abc import Callable, Iterable
from logging import getLogger
from typing import Any, Optional

from torch import nn

# Execution order is model knowledge, not module registration order. Unknown
# blocks stay intact rather than inferring order from parameter names.
_COMPUTE_STAGES = {
    "Qwen3MoeDecoderLayer": ("self_attn", "mlp"),
    "Qwen3VLMoeTextDecoderLayer": ("self_attn", "mlp"),
    "Qwen3VLMoeVisionBlock": ("attn", "mlp"),
}


def _resolve_wrap_targets(target_modules, root_model, device_type, compiled):
    if device_type not in ("cuda", "npu") or compiled:
        return {}
    parameter_names = {}
    for name, param in root_model.named_parameters(remove_duplicate=False):
        parameter_names.setdefault(id(param), []).append(name)
    resolved = {}
    for fqn, block in target_modules:
        paths = _COMPUTE_STAGES.get(type(block).__name__)
        if paths is None:
            continue
        # Explicit nested targets already define their own FSDP groups. Do not
        # schedule those units twice through an additional parent stage chain.
        prefix = fqn + "." if fqn else ""
        if any(name != fqn and name.startswith(prefix) for name, _ in target_modules):
            continue
        try:
            children = [block.get_submodule(path) for path in paths]
        except AttributeError:
            continue
        seen = set()
        for path, child in zip(paths, children):
            owned = {id(param) for param in child.parameters()}
            prefix = (fqn + "." if fqn else "") + path + "."
            if (
                not owned
                or seen & owned
                or any(not name.startswith(prefix) for param in owned for name in parameter_names[param])
            ):
                break
            seen.update(owned)
        else:
            resolved[block] = children
    return resolved


def _build_prefetch_stages(block, children, existing_units):
    stages = [[child] for child in children]
    for unit in existing_units:
        if unit is block or any(unit is child for child in children):
            continue
        stage = stages[0]
        for child, candidate in zip(children, stages):
            if any(unit is descendant for descendant in child.modules()):
                stage = candidate
                break
        stage.append(unit)
    return stages


def _configure_prefetch(block_stages):
    entries = [stage for block in block_stages for stage in block]
    position = 0
    for index, stages in enumerate(block_stages):
        for stage_index, stage in enumerate(stages):
            forward = [unit for following in entries[position + 1 : position + 3] for unit in following]
            # Whole-block replay starts at the last stage's backward hook.
            # Submit preceding-block AGs before the current block's RS.
            if stage_index and (not index or len(block_stages[index - 1]) == 1):
                backward = stages[stage_index - 1]
            elif index:
                backward = [unit for previous in block_stages[index - 1] for unit in previous]
            else:
                backward = []
            for unit in stage:
                unit.set_modules_to_forward_prefetch(
                    [target for target in dict.fromkeys(forward) if target is not unit]
                )
                unit.set_modules_to_backward_prefetch(list(backward))
            position += 1


class StageWrapPolicy:
    """Wrap known compute stages without changing VeOmni's common model plan.

    Args:
        model: Unwrapped model; retains residual parameters on its root group.
        target_modules: Ordered (qualified name, block) pairs from VeOmni.
        backend: Namespace exposing ``fully_shard`` and ``module_type``.
        device_type: Accelerator type; CUDA and NPU support stage wrapping.
        compiled: Whether block compilation requires whole-block fallback.
        log: Optional setup-only logging callback.

    Note:
        VeOmni's builder owns ``block._fsdp_modules``. The adapter preserves
        these existing EP/FP32 groups and uses their order only for unsplit blocks.
        Output weights remain root-owned so fused losses can access them
        directly without requiring an output projection forward call.
    """

    requires_manual_prefetch = True

    def __init__(
        self,
        model: Any,
        target_modules: Iterable[tuple[str, Any]],
        backend: Any,
        device_type: str,
        compiled: bool = False,
        log: Optional[Callable[[str], None]] = None,
        mixed_precision: Any = None,
    ) -> None:
        """Resolve the model's safe stage boundaries before any FSDP wrapping."""
        self.model = model
        self.target_modules = list(target_modules)
        self.backend = backend
        self.output_dtype_is_set = bool(
            mixed_precision is not None and mixed_precision.enable and mixed_precision.output_dtype is not None
        )
        self.allow_manual_prefetch = not any(not param.requires_grad for param in model.parameters())
        can_split = (self.output_dtype_is_set and device_type == "cuda") or (
            not self.output_dtype_is_set and self.allow_manual_prefetch
        )
        self.wrap_targets = (
            _resolve_wrap_targets(self.target_modules, model, device_type, compiled) if can_split else {}
        )
        self.prefetch_stages = {}
        self.input_embeddings = None
        self.log = log if log is not None else getLogger(__name__).info

    def needs_parent_wrap(self, block: Any) -> bool:
        """Preserve the enclosing output-cast boundary when precision needs it."""
        return block not in self.wrap_targets or self.output_dtype_is_set

    def record_parent(self, block: Any) -> None:
        """Include an enclosing precision unit before its independently wrapped stages."""
        if block in self.prefetch_stages:
            self.prefetch_stages[block].insert(0, [block])

    def wrap_model_io(self, fsdp_kwargs: dict[str, Any]) -> None:
        """Wrap independent embeddings while leaving output weights on root."""
        if not self.wrap_targets or self.output_dtype_is_set:
            return

        def _optional_module(name: str) -> Optional[nn.Module]:
            """Read an optional model accessor without requiring HF base classes."""
            accessor = getattr(self.model, name, None)
            if not callable(accessor):
                return None
            try:
                module = accessor()
            except NotImplementedError:
                return None
            return module if isinstance(module, nn.Module) else None

        embedding = _optional_module("get_input_embeddings")
        head = _optional_module("get_output_embeddings")
        if embedding is not None and head is not None:
            if {id(p) for p in embedding.parameters()} & {id(p) for p in head.parameters()}:
                return
        if embedding is not None and not isinstance(embedding, self.backend.module_type):
            self.backend.fully_shard(embedding, **fsdp_kwargs)
            self.input_embeddings = embedding

    def exclude_persistent(self, params: Iterable[Any]) -> None:
        """Keep blocks with persistent ExtraParallel parameters intact.

        Args:
            params: Parameters ignored by FSDP; their original layout must survive.
        """
        persistent_ids = {id(param) for param in params}
        self.wrap_targets = {
            block: children
            for block, children in self.wrap_targets.items()
            if not any(id(param) in persistent_ids for param in block.parameters())
        }

    def wrap_submodules(self, block: Any, fsdp_kwargs: dict[str, Any]) -> None:
        """Wrap declared children after VeOmni's nested EP/FP32 groups.

        Args:
            block: Current target block with builder-owned FSDP unit bookkeeping.
            fsdp_kwargs: Caller-owned mesh, precision and offload configuration.
        """
        children = self.wrap_targets.get(block)
        if children is None:
            return
        existing_units = list(block._fsdp_modules)
        for child in children:
            if not isinstance(child, self.backend.module_type):
                self.backend.fully_shard(child, **fsdp_kwargs)
                block._fsdp_modules.append(child)
        self.prefetch_stages[block] = _build_prefetch_stages(block, children, existing_units)

    def configure_prefetch(self) -> bool:
        """Set depth-two forward and whole-block replay-aware backward targets.

        Returns:
            Whether a stage chain was configured. False leaves the caller's
            existing whole-block prefetch path responsible for scheduling.
        """
        if not self.wrap_targets or not self.allow_manual_prefetch:
            return False
        block_stages = [
            self.prefetch_stages[block] if block in self.prefetch_stages else [list(reversed(block._fsdp_modules))]
            for _, block in self.target_modules
        ]
        _configure_prefetch(block_stages)
        if self.input_embeddings is not None:
            entries = [stage for block in block_stages for stage in block]
            self.input_embeddings.set_modules_to_forward_prefetch([unit for stage in entries[:2] for unit in stage])
            self.input_embeddings.set_modules_to_backward_prefetch([])
        self.log("[hyper-forward-prefetch] policy=stage depth=2")
        names = {id(module): name for name, module in self.model.named_modules()}
        self.log(
            "[hyper-submodule-prefetch] stages="
            f"{[[[names[id(unit)] for unit in stage] for stage in stages] for stages in block_stages]}"
        )
        return True
