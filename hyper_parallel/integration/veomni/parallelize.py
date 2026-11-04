# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import types
from collections.abc import Mapping
from functools import lru_cache
from inspect import Parameter, signature
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import torch
from torch import nn
from torch.distributed.tensor.parallel import parallelize_module
from torch.utils.checkpoint import noop_context_fn
from veomni.arguments import MixedPrecisionConfig
from veomni.arguments.arguments_types import validate_low_precision_reduce_scatter_comm
from veomni.distributed.checkpoint import CheckpointFunction
from veomni.distributed.parallel_plan import get_runtime_parallel_plan
from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.torch_compile import (
    CompileConfig,
    compile_decoder_blocks,
    validate_compile_runtime,
)
from veomni.distributed.torch_parallelize import (
    _can_shard_extra_parallel_dim0,
    _is_fsdp_wrap_target,
    _move_buffers_to_device,
)
from veomni.distributed.utils import sort_fqn_by_submodule_first
from veomni.utils import logging
from veomni.utils.device import IS_NPU_AVAILABLE, get_device_type

from .module_utils import _materialize_and_load_weights, register_dtensor_adapter

logger = logging.get_logger(__name__)


@lru_cache(maxsize=1)
def get_hyper_fsdp2_api() -> Any:
    """Import HyperParallel lazily so the default Torch backend is unaffected."""
    try:
        from hyper_parallel import DTensor, HSDPModule, fully_shard
        from hyper_parallel.core.dtensor.dtensor import distribute_tensor
        from hyper_parallel.core.dtensor.placement_types import Shard
        from hyper_parallel.core.fully_shard.utils import (
            CPUOffloadPolicy,
            MixedPrecisionPolicy,
        )
    except ImportError as exc:
        raise RuntimeError(
            "fsdp_backend='hyper' requires the optional 'hyper_parallel' package. "
            "Install HyperParallel in the active training environment first."
        ) from exc

    register_dtensor_adapter(DTensor)
    return SimpleNamespace(
        fully_shard=fully_shard,
        module_type=HSDPModule,
        mixed_precision_policy=MixedPrecisionPolicy,
        cpu_offload_policy=CPUOffloadPolicy,
        shard=Shard,
        distribute_tensor=distribute_tensor,
    )


def to_hyper_device_mesh(torch_mesh: Any, name: str) -> Any:
    """Reuse VeOmni's existing process groups in a HyperParallel device mesh."""
    from hyper_parallel import DeviceMesh

    dim_names = tuple(torch_mesh.mesh_dim_names or ())
    if not dim_names:
        dim_names = tuple(f"{name}_{index}" for index in range(torch_mesh.ndim))
    groups = torch_mesh.get_all_groups()
    group = groups[0] if torch_mesh.ndim == 1 else groups
    return DeviceMesh.from_group(
        group=group,
        device_type=torch_mesh.device_type,
        mesh=torch_mesh.mesh.tolist(),
        mesh_dim_names=dim_names,
    )


def extra_parallel_gradient_scaling_factor(fsdp_group_size: int, gradient_divide_factor: int) -> float:
    """Convert Torch's custom reduce-scatter divisor to Hyper's averaged reduction."""
    return fsdp_group_size / gradient_divide_factor


def validate_reduce_comm_interval(interval: Any) -> Any:
    """Validate the positive integer reduction interval."""
    if type(interval) is not int or interval < 1:
        raise ValueError("hyper_reduce_comm_interval must be a positive integer.")


def set_reduce_comm_interval(model: Any, interval: Any) -> Any:
    """Apply a non-default reduction interval through the Hyper API."""
    if interval == 1:
        return
    setter = getattr(model, "set_reduce_comm_interval", None)
    if setter is None:
        raise RuntimeError("This HyperParallel version does not support set_reduce_comm_interval.")
    setter(interval)


def set_gradient_divide_factor(module: Any, mesh: Any, divisor: Any) -> Any:
    """Translate the native gradient divisor to Hyper averaged reduction."""
    module.set_gradient_scaling_factor(extra_parallel_gradient_scaling_factor(mesh.size(), divisor))


def shard_placement(param: Any) -> Any:
    """Resolve an explicit FSDP shard dimension when present."""
    dim = getattr(param, "_veomni_fsdp_shard_dim", None)
    return get_hyper_fsdp2_api().shard(dim) if dim is not None else None


def parallelize_model_fsdp2(
    model: "nn.Module",
    weights_path: Optional[str | Mapping[str, str]] = None,
    enable_reshard_after_forward: bool = True,
    mixed_precision: MixedPrecisionConfig = MixedPrecisionConfig(enable=True),  # noqa
    basic_modules: Optional[List[str]] = None,
    muon_expert_zero_comm: bool = False,
    low_precision_reduce_scatter_comm: bool = False,
    compile_config: Optional[CompileConfig] = None,
    should_skip_hf_weight_load: bool = False,
    **kwargs: Any,
) -> "nn.Module":
    """
    Apply ExtraParallel (e.g. Expert Parallel or Embed Parallel) + FSDP2 parallel strategy to the model.

    For Expert Parallel, the flow is as follows:
        1. Apply EP: Expert tensors [128,H,I] -> [32,H,I] local tensors per EP rank
        2. Apply FSDP2 to expert modules: Shard expert tensors along dim-1 (hidden dim)
        3. Apply FSDP2 to regular modules: Standard dim-0 sharding
        4. Result: Expert params [32,H/fsdp_size,I], regular params use standard FSDP2

    For ExtraParallel, see test_clip_grad_norm_fsdp2_ep2_emb4 with Expert Parallel + Embed Parallel, where
        ToyMoeAndEmbedModel(
            (embed_tokens): ToyEmbed()
            (decoder): ToyMoeAndEmbedDecoderLayer(
                (embed_tokens): ToyEmbed()
                (moe): ToyMoeExperts()
            )
        )
        ToyMoeAndEmbedModel._no_split_modules = ["ToyMoeAndEmbedDecoderLayer", "ToyEmbed"]
        ep_plan = {"decoder.moe.experts": Shard(0)}
        emb_plan = {"embed_tokens.weight": Shard(0), "decoder.embed_tokens.weight": Shard(0)}
        ep_size, emb_size = 2, 4
    We will use this model for illustration of Expert Parallel + Embed Parallel below.
    """
    reduce_comm_interval = kwargs.pop("hyper_reduce_comm_interval", 1)
    expert_grad_dtype_fp32 = kwargs.pop("hyper_expert_grad_dtype_fp32", False)
    validate_reduce_comm_interval(reduce_comm_interval)
    parallel_state = get_parallel_state()
    api = get_hyper_fsdp2_api()
    try:
        from hyper_parallel.integration.veomni import StageWrapPolicy
    except ImportError as exc:
        raise RuntimeError(
            "Hyper FSDP requires HyperParallel with the integration.veomni adapter. "
            "Update the optional HyperParallel installation."
        ) from exc
    fsdp_module_type = api.module_type
    mixed_precision_policy_type = api.mixed_precision_policy
    cpu_offload_policy_type = api.cpu_offload_policy
    shard_type = api.shard
    fully_shard = api.fully_shard

    expert_grad_param_ids = set()
    if expert_grad_dtype_fp32:
        if parallel_state.ep_enabled:
            raise ValueError("FP32 expert gradients require Hyper FSDP2 with ep_size=1.")
        if not mixed_precision.enable or (
            mixed_precision.param_dtype != "bfloat16" or mixed_precision.reduce_dtype != "float32"
        ):
            raise ValueError("FP32 expert gradients require BF16 parameters and FP32 gradient reduction.")
        if get_device_type() != "cuda" or getattr(getattr(model, "config", None), "model_type", None) != "qwen3_moe":
            raise ValueError("FP32 expert gradients currently support Qwen3 MoE on CUDA only.")
        parameters = signature(mixed_precision_policy_type).parameters
        if "grad_dtype_fn" not in parameters and not any(
            p.kind is Parameter.VAR_KEYWORD for p in parameters.values()
        ):
            raise RuntimeError("FP32 expert gradients require Hyper MixedPrecisionPolicy(grad_dtype_fn=...).")
        for module in model.modules():
            if module.__class__.__name__ == "Qwen3MoeExperts":
                for name in ("gate_up_proj", "down_proj"):
                    param = getattr(module, name, None)
                    if not isinstance(param, nn.Parameter) or param.ndim != 3:
                        raise ValueError(f"Qwen3 MoE expert parameter {name} must be a 3D Parameter.")
                    expert_grad_param_ids.add(id(param))
        if not expert_grad_param_ids:
            raise ValueError("FP32 expert gradients requested but no Qwen3 MoE expert parameters were found.")
        logger.info_rank0(f"FP32 expert wgrad enabled for {len(expert_grad_param_ids)} parameters.")

    use_low_precision_transport = validate_low_precision_reduce_scatter_comm(
        low_precision_reduce_scatter_comm, mixed_precision
    )
    if use_low_precision_transport:
        raise NotImplementedError("Low-precision ReduceScatter transport is not supported by Hyper FSDP2.")
    if low_precision_reduce_scatter_comm:
        logger.info_rank0("Parameter dtype matches reduce dtype; using the native backend collective.")
    model_no_split_modules = getattr(model, "_no_split_modules", None) or []
    target_classes = set(model_no_split_modules) | set(basic_modules or [])

    # Make a list of tuples that contains target classes' name and module
    # Note that all target classes should include all ExtraParallel modules.
    #   e.g. `ToyEmbed` and `ToyMoeAndEmbedDecoderLayer` include `embed_tokens.weight` and `decoder.embed_tokens.weight`
    # Note that target class A is allowed to include target class B:
    #   e.g. `ToyMoeAndEmbedDecoderLayer` includes target class `ToyEmbed`
    # Thus, target module A could include target module B.
    #   e.g. `decoder` includes `decoder.embed_tokens`
    target_modules: List[Tuple[str, nn.Module]] = [
        (fqn, mod)
        for fqn, mod in model.named_modules()
        if _is_fsdp_wrap_target(fqn, mod.__class__.__name__, target_classes)
    ]
    logger.info_rank0(f"target classes to shard: {target_classes}")
    wrap_plan = StageWrapPolicy(
        model,
        target_modules,
        api,
        get_device_type(),
        compile_config is not None and compile_config.enable,
        log=logger.info_rank0,
        mixed_precision=mixed_precision,
    )

    model._veomni_compile_enabled = False
    compile_config = compile_config or CompileConfig()

    # Step 1: Apply ExtraParallel
    #   e.g. Apply expert parallelism (slice expert tensors [128,H,I] -> [16,H,I])
    #        Apply embed parallelism (slice embed tensors [64,H] -> [16,H])
    if parallel_state.any_extra_parallel_enabled:
        parallel_plan = get_runtime_parallel_plan(model)
        assert parallel_plan is not None, (
            "ExtraParallel needs parallel plan defined in the model! \
            Please see veomni/models/transformers/qwen3_moe/parallel_plan.py for example of expert parallelism. \
            Please see tests/utils/test_extra_parallel_clip_grad_norm.py::test_clip_grad_norm_fsdp2_ep2_emb4 \
            for example of expert parallelism + embed parallelism."
        )
        # Add SpecInfo to extra_parallel modules,
        #   e.g. embed_tokens.weight, decoder.regular_mlp, decoder.embed_tokens.weight, and decoder.moe.experts
        fqn2spec_info = parallel_plan.apply(model, parallel_state.extra_parallel_fsdp_device_mesh)

        model._fqn2spec_info = fqn2spec_info
        persistent_extra_parallel_params = {
            param for fqn, param in model.named_parameters() if fqn2spec_info[fqn].persistent_fsdp_shard_dim is not None
        }
        persistent_para_names = {
            spec_info.para_name
            for spec_info in fqn2spec_info.values()
            if spec_info.persistent_fsdp_shard_dim is not None
        }
        if persistent_extra_parallel_params:
            if parallel_state.dp_replicate_enabled:
                raise NotImplementedError("Persistent ExtraParallel parameters do not support HSDP replicas yet.")
            if parallel_state.sp_enabled:
                raise NotImplementedError(
                    "Persistent ExtraParallel parameters do not support sequence parallelism yet."
                )
            for para in persistent_para_names:
                para_mesh = parallel_state.extra_parallel_fsdp_device_mesh[para]
                flat_mesh = parallel_state.extra_parallel_flat_device_mesh[para]
                if (
                    para_mesh is None
                    or para_mesh.ndim != 2
                    or para_mesh.size() != parallel_state.world_size
                    or flat_mesh is None
                ):
                    raise NotImplementedError(
                        f"Persistent {para} parameters currently require a 2D ({para}_fsdp, {para}) mesh "
                        "covering the complete world."
                    )
                logger.info_rank0(
                    f"Persistent {para} 2D layout: rows={para_mesh.size(1)} ({para}), "
                    f"columns={para_mesh.size(0)} ({para}_fsdp); parameter all-gather is disabled."
                )
        _extra_parallel_mesh = {}
        _extra_parallel_map = {}
        for para in parallel_state.extra_parallel_names:
            # A globally-enabled ExtraParallel (e.g. ``ep``) may not be declared by
            # THIS model's plan -- e.g. a text-encoder / over-encoder has no experts,
            # so ``ep`` shards nothing here even though a MoE sibling module uses it.
            # Treat such a para as a no-op for this model (``None``) instead of
            # crashing in ``get_extra_parallel_fsdp_no_shard_info`` (which requires
            # >=1 matching module). Nothing downstream wraps it (its layer para_mods
            # stay empty).
            if parallel_state.extra_parallel_enabled(para) and para in parallel_plan.extra_parallel_plan:
                _extra_parallel_mesh[para] = parallel_state.extra_parallel_fsdp_device_mesh[para]
                _extra_parallel_map[para] = parallel_plan.get_extra_parallel_fsdp_no_shard_info(model, para)
            else:
                _extra_parallel_mesh[para] = None
                _extra_parallel_map[para] = None

            logger.info_rank0(
                f"Applied {para}: tensors sliced along dimension ({para} mesh: {_extra_parallel_mesh[para]})"
            )
            logger.info_rank0(f"{para} Map: {_extra_parallel_map[para]}")

    else:
        parallel_plan = None
        fqn2spec_info = None
        persistent_extra_parallel_params = set()
        persistent_para_names = set()
        _extra_parallel_mesh = None
        _extra_parallel_map = None

    wrap_plan.exclude_persistent(persistent_extra_parallel_params)

    # Extract ExtraParallel modules from the target classes if any, then pair them.
    # Regard each target module as a layer.
    # Note that all target modules should include ExtraParallel modules.
    #     If we have ToyMoeAndEmbedModel like the above, then,
    #         layer_pairs_list = [
    #            ('decoder.embed_tokens', (ToyEmbed, {'emb': ToyEmbed, 'ep': None})),
    #            ('embed_tokens', (ToyEmbed, {'emb': ToyEmbed, 'ep': None})),
    #            ('decoder', (ToyMoeAndEmbedDecoderLayer, {'emb': ToyEmbed, 'ep': ToyMoeExperts}))
    #         ]
    layer_pairs = {}
    for layer_fqn, layer_mod in target_modules:
        layer_pair = [layer_mod]
        extra_parallel_mod = {}

        if parallel_state.any_extra_parallel_enabled:
            for para in parallel_state.extra_parallel_names:
                if _extra_parallel_map[para] is not None:
                    # Collect ALL extra-parallel (e.g. ep) modules under this layer,
                    # not just the first. A single layer may hold more than one such
                    # module (e.g. multiple experts modules in one decoder layer), and
                    # EACH must be fully_shard'd over the ep_fsdp mesh. If only the
                    # first were wrapped, the others would fall through to the regular
                    # ``fully_shard(layer_mod)`` below and be re-sharded over the
                    # FSDP mesh -- but their params were already EP-sliced (each rank
                    # holds a *different* [E/ep, ...] slice), so that second shard
                    # scrambles them and corrupts the experts.
                    # Match ``layer_fqn + "."`` (not a bare ``startswith``) so e.g.
                    # ``model.layers.1`` does not also swallow ``model.layers.10``.
                    para_mods = [
                        para_mod
                        for para_mod_fqn, para_mod in _extra_parallel_map[para].items()
                        if para_mod_fqn == layer_fqn or para_mod_fqn.startswith(layer_fqn + ".")
                    ]
                else:
                    para_mods = []
                extra_parallel_mod[para] = para_mods
        layer_pair.append(extra_parallel_mod)
        layer_pairs[layer_fqn] = tuple(layer_pair)

    logger.info_rank0(f"extra_parallel layer pairs: {layer_pairs}")

    if compile_config.enable:
        validate_compile_runtime(
            compile_config,
            device_type=get_device_type(),
            fsdp_enabled=parallel_state.fsdp_enabled,
            fsdp_mode=parallel_state.dp_mode,
            any_extra_parallel_enabled=parallel_state.any_extra_parallel_enabled,
            enable_reshard_after_forward=enable_reshard_after_forward,
        )

        compiled_count = compile_decoder_blocks(
            model,
            compile_config,
            sequence_parallel_enabled=parallel_state.sp_enabled,
            async_enabled=parallel_state.async_enabled,
        )
        if compiled_count == 0:
            raise RuntimeError("model.accelerator.torch_compile.enable found no decoder blocks to compile.")
        model._veomni_compile_enabled = True
        model._veomni_compile_uses_cuda_graphs = compile_config.uses_cuda_graphs()

    # Step 2: Update fsdp2 kwargs
    fsdp_kwargs = {
        "mesh": to_hyper_device_mesh(parallel_state.fsdp_mesh, "fsdp"),
        "reshard_after_forward": enable_reshard_after_forward,
        "shard_placement_fn": shard_placement,
    }
    if persistent_extra_parallel_params:
        # These params already carry their complete 2D DTensor placement. Every
        # ancestor FSDP group must ignore them or the parent layer/root would
        # absorb them and reintroduce a parameter all-gather.
        fsdp_kwargs["ignored_params"] = persistent_extra_parallel_params
    # prepare mp_policy kwargs
    if mixed_precision.enable:
        mp_kwargs = {
            "param_dtype": getattr(torch, mixed_precision.param_dtype) if mixed_precision.param_dtype else None,
            "reduce_dtype": getattr(torch, mixed_precision.reduce_dtype) if mixed_precision.reduce_dtype else None,
            "output_dtype": getattr(torch, mixed_precision.output_dtype) if mixed_precision.output_dtype else None,
            "cast_forward_inputs": mixed_precision.cast_forward_inputs,
        }
        if expert_grad_dtype_fp32:
            mp_kwargs["grad_dtype_fn"] = lambda param: torch.float32 if id(param) in expert_grad_param_ids else None
        mp_policy = mixed_precision_policy_type(**mp_kwargs)
        fsdp_kwargs["mp_policy"] = mp_policy
    # prepare offload_policy kwargs
    enable_fsdp_cpu_offload = kwargs.pop("enable_fsdp_offload", False)
    offload_pin_memory = kwargs.pop("fsdp_offload_pin_memory", True)
    model._fsdp_cpu_offload_enabled = enable_fsdp_cpu_offload
    if enable_fsdp_cpu_offload:
        if persistent_extra_parallel_params:
            raise NotImplementedError(
                "FSDP CPU offload is not supported with persistent ExtraParallel parameters: "
                "FSDP does not manage or stage ignored parameters."
            )
        logger.info_rank0("Enable FSDP2 CPU offload for parameters, gradients, and optimizer states.")
        # ``CPUOffloadPolicy`` defaults to ``pin_memory=True``: the offloaded CPU
        # param shards are page-locked, which (a) is accounted as non-reclaimable
        # ``Shmem`` against the container memcg and (b) cannot be swapped/evicted.
        # For a very large model (e.g. the ~1.5 TiB MoT experts) each rank pins
        # its whole shard, so N ranks/host pin ~N * shard_size of Shmem and OOM
        # the cgroup during load. ``fsdp_offload_pin_memory=False`` keeps the
        # shards in ordinary pageable anon memory (what a bespoke manual offload
        # does) at the cost of a non-pinned (slightly slower) H2D per layer.
        fsdp_kwargs["offload_policy"] = cpu_offload_policy_type(pin_memory=offload_pin_memory)

    if hasattr(model, "get_ignore_modules_in_mixed_precision"):
        modules_to_ignore_in_mixed_precision = model.get_ignore_modules_in_mixed_precision()
    else:
        modules_to_ignore_in_mixed_precision = None

    if modules_to_ignore_in_mixed_precision:
        assert isinstance(modules_to_ignore_in_mixed_precision, tuple), (
            "modules_to_ignore_in_mixed_precision needs to be a tuple!"
        )
        mp_ignored_classes = modules_to_ignore_in_mixed_precision
        fsdp_kwargs_without_mp = dict(fsdp_kwargs)
        fsdp_kwargs_without_mp.pop("mp_policy", None)
        # for high-precision modules, we do not reshard them after forward to avoid all-gather them in backward
        # these modules will stay in GPU memory so please ensure high-precision modules do not contain too many
        # parameters
        fsdp_kwargs_without_mp["reshard_after_forward"] = False
    else:
        mp_ignored_classes = None
        fsdp_kwargs_without_mp = fsdp_kwargs

    # prepare extra_parallel_fsdp2 kwargs
    extra_parallel_fsdp_kwargs = {}
    for para in parallel_state.extra_parallel_names:
        if parallel_state.extra_parallel_enabled(para):
            para_mesh = parallel_state.extra_parallel_fsdp_device_mesh[para]
            para_mesh_dim_names = para_mesh.mesh_dim_names
            # exclude para_size dimension:
            # - (ep_fsdp, ep) -> (ep_fsdp,)
            # - (ep_replicate, ep_fsdp, ep) -> (ep_replicate, ep_fsdp)
            para_fsdp_mesh = para_mesh[para_mesh_dim_names[:-1]]
            para_fsdp_kwargs = dict(fsdp_kwargs)
            para_fsdp_kwargs["mesh"] = to_hyper_device_mesh(para_fsdp_mesh, f"{para}_fsdp")
            shard_dim_for_para = 1
            # Muon zero-comm needs whole experts per rank; otherwise keep the
            # default hidden-dim sharding. A para this model's plan does not
            # declare wraps nothing here, so leave it out of the report.
            declared = parallel_plan is not None and para in (parallel_plan.extra_parallel_plan or {})
            if muon_expert_zero_comm and declared:
                para_fsdp_size = para_mesh[f"{para}_fsdp"].size()
                if _can_shard_extra_parallel_dim0(model, parallel_plan, para, para_fsdp_size):
                    shard_dim_for_para = 0
                    logger.info_rank0(
                        f"[muon_expert_zero_comm] {para}: enabling Shard(0) for "
                        f"the FSDP step ({para}_fsdp size={para_fsdp_size}); Muon will "
                        "run batched NS locally with zero communication."
                    )
                else:
                    logger.warning_rank0(
                        f"[muon_expert_zero_comm] {para}: falling back to the default "
                        "Shard(1) layout (Muon will use the all-to-all-gather path)."
                    )
            para_fsdp_kwargs["shard_placement_fn"] = lambda param, _d=shard_dim_for_para: shard_type(_d)
            extra_parallel_fsdp_kwargs[para] = para_fsdp_kwargs
            # Record the FSDP shard dim on the spec_info so the checkpointer can
            # build correct DTensor placements (EP dim vs FSDP dim) on save/load.
            for spec_info in getattr(model, "_fqn2spec_info", {}).values():
                if spec_info.para_name == para:
                    spec_info.fsdp_shard_dim = shard_dim_for_para
        else:
            extra_parallel_fsdp_kwargs[para] = None

    # Here we have a basic assumption for target module (e.g. embed_tokens, decoder) hierarchy:
    # | -- target module A (e.g. decoder)
    #   | -- target module B (e.g. decoder.embed_tokens)
    #   | -- extra parallel module C (e.g. decoder.moe)
    #     | -- no more target module or extra parallel module
    #   | -- mp modules
    #     | -- no more target module or extra parallel module
    #   | -- other module (e.g. attention, if provided)
    # e.g. Decoder Layer
    # | -- layers that are sharded by fully_shard(decode_layer) (e.g., Attention)
    # | -- experts layer (apply fully_shard separately in order to shard across EP groups on the same EP rank instead of
    # sharding globally)
    # | -- layers (declared in model.modules_to_ignore_in_mixed_precision) that need to apply fully_shard separately due
    # to different mp policy as the decoder layer
    #      (e.g., some models requires MoE TopK gate layer to have parameters in higher FP32 precision in forward).
    # NPU currently does not support the PreSumMul operation, so this operation is supported through the
    # apply_hccl_premul_sum_patch.
    # TODO(https://github.com/ByteDance-Seed/VeOmni/issues/241):
    # NPU is missing PreSumMul ReduceOp. Need to remove this condition after the issue is resolved.
    if IS_NPU_AVAILABLE and parallel_state.any_extra_parallel_enabled:
        from veomni.ops.platform.npu import apply_hccl_premul_sum_patch

        apply_hccl_premul_sum_patch()

    # Sort layer_pairs by fqn by submodule order, as fully_shard should starts from bottom modules to top modules
    #   e.g. sorted_fqn_list = ['decoder.embed_tokens', 'embed_tokens', 'decoder']
    sorted_fqn_list = sort_fqn_by_submodule_first(list(layer_pairs.keys()))
    layer_pairs_list = [(fqn, layer_pairs[fqn]) for fqn in sorted_fqn_list]
    for layer_fqn, (layer_mod, extra_parallel_mod) in layer_pairs_list:
        # register all the FSDPModule inside this decoder layer for the convenience of manual prefetching configuration
        layer_mod._fsdp_modules = []

        for para in parallel_state.extra_parallel_names:
            # para (e.g. ep, emb) enabled and this layer contains the para module(s)
            # (e.g. expert/decoder.moe, embed_tokens/decoder.embed_tokens). A layer
            # may hold more than one (e.g. multiple experts modules) -- wrap each.
            if not parallel_state.extra_parallel_enabled(para):
                continue
            for _para_mod in extra_parallel_mod[para]:
                if isinstance(_para_mod, fsdp_module_type):
                    continue
                if any(param in persistent_extra_parallel_params for param in _para_mod.parameters()):
                    # Persistent params remain on the complete 2D mesh. Any
                    # non-persistent siblings are picked up by the enclosing
                    # decoder/root FSDP group through the normal bottom-up walk.
                    continue
                # shard para module (e.g. expert/decoder.moe, embed_tokens/decoder.embed_tokens)
                fully_shard(_para_mod, **extra_parallel_fsdp_kwargs[para])
                # average para (e.g. ep) grads across para (e.g. ep) ranks
                # NOTE: in torch 2.8 and later we should use
                # experts_mod.set_gradient_divide_factor(parallel_state.ep_size)
                # but for torch 2.7 we still use set_reduce_scatter_divide_factor(parallel_state.ep_size)
                gradient_divide_factor = parallel_state.extra_parallel_gradient_divide_factor(para)
                logger.info(f"setting grad divide factor for {para} module to {gradient_divide_factor}")
                set_gradient_divide_factor(
                    _para_mod,
                    extra_parallel_fsdp_kwargs[para]["mesh"],
                    gradient_divide_factor,
                )
                layer_mod._fsdp_modules.append(_para_mod)

        # shard module that needs to ignore mixed precision control
        if mp_ignored_classes:
            for sub_mod in layer_mod.modules():
                if isinstance(sub_mod, mp_ignored_classes) and sub_mod is not layer_mod:
                    fully_shard(sub_mod, **fsdp_kwargs_without_mp)
                    layer_mod._fsdp_modules.append(sub_mod)

        wrap_plan.wrap_submodules(layer_mod, fsdp_kwargs)

        # Shard everything else in the module:
        #   Note:
        #      if we have a model and layer_pairs_list like the above,
        #      when layer_mod (also called as target module, e.g. decoder.embed_tokens),
        #      is the parent of or equal to extra_parallel_mod[para] (e.g. ToyEmbed),
        #      no need to shard layer_mod again.
        if wrap_plan.needs_parent_wrap(layer_mod) and not isinstance(layer_mod, fsdp_module_type):
            fully_shard(layer_mod, **fsdp_kwargs)
            layer_mod._fsdp_modules.append(layer_mod)
            wrap_plan.record_parent(layer_mod)
        logger.info_rank0(f"{layer_fqn=}, {layer_mod._fsdp_modules=}")

    # Keep the root unsharded between forward and backward using the backend's
    # root policy.
    # This keeps decoder/final norms and any unsplit/shared/direct-weight head
    # unsharded between forward and backward, which is what fused-linear
    # kernels (chunk_logprobs / chunk_topk_distill) need — they save the
    # lm_head weight via `save_for_backward` and would otherwise hit
    # `setStorage … storage of size 0` when the saved reference points
    # to a freed buffer. Decoder layers reshard normally (their calls
    # above pass `reshard_after_forward` explicitly).
    wrap_plan.wrap_model_io(fsdp_kwargs)
    root_fsdp_kwargs = {**fsdp_kwargs, "reshard_after_forward": False}
    fully_shard(model, **root_fsdp_kwargs)
    set_reduce_comm_interval(model, reduce_comm_interval)
    # Persistent ExtraParallel parameters are ignored by every FSDP group and
    # retain their original 2D DTensor identity. Record them after wrapping so
    # gradient clipping can reduce their unique local shards once over the
    # flattened 2D mesh instead of traversing the two mesh axes separately.
    model._persistent_extra_parallel_param_ids = {id(param) for param in persistent_extra_parallel_params}

    # configure manual prefetching when needed
    need_manual_prefetch = (
        wrap_plan.requires_manual_prefetch
        or parallel_state.any_extra_parallel_enabled
        or mp_ignored_classes is not None
    ) and kwargs.pop("enable_forward_prefetch", True) and wrap_plan.allow_manual_prefetch
    stage_prefetch = need_manual_prefetch and wrap_plan.configure_prefetch()
    if need_manual_prefetch and not stage_prefetch:
        blocks = [pair[1][0] for pair in layer_pairs_list]  # all target modules
        next_blocks = blocks[1:] + [None]
        for current_block, next_block in zip(blocks, next_blocks):
            if next_block is not None:
                prefetch_modules = next_block._fsdp_modules
                # prefetch in order of attn, gate, experts
                current_block.set_modules_to_forward_prefetch(list(reversed(prefetch_modules)))

        # configure backward prefetch
        rev_blocks = list(reversed(blocks))
        prev_blocks = rev_blocks[1:] + [None]
        for current_block, prev_block in zip(rev_blocks, prev_blocks):
            if prev_block is not None:
                prefetch_modules = prev_block._fsdp_modules
                current_block.set_modules_to_backward_prefetch(list(reversed(prefetch_modules)))

    # Handle meta initialization for FSDP2 (fallback if pre-load not done)
    assert kwargs.get("init_device") == "meta", "Please use init_device: meta for FSDP2"
    materialize_device = "cpu" if enable_fsdp_cpu_offload else get_device_type()

    _materialize_and_load_weights(
        model,
        weights_path,
        materialize_device,
        should_skip_hf_weight_load=should_skip_hf_weight_load,
        is_peft_model=kwargs.pop("is_peft_model", False),
        adapter_path=kwargs.pop("adapter_path", None),
        broadcast_from_rank0=bool(kwargs.get("broadcast_model_weights_from_rank0")),
        cpu_load_param_name=kwargs.get("cpu_load_param_name", None),
        max_load_broadcast_size=kwargs.get("max_load_broadcast_size", 20.0),
        fqn_to_index_mapping=kwargs.get("fqn_to_index_mapping"),
        ep_sharded_stream_load=bool(kwargs.get("ep_sharded_stream_load")),
        module_skip_hf_weight_load=kwargs.pop("module_skip_hf_weight_load", None),
        module_is_peft_model=kwargs.pop("module_is_peft_model", None),
        module_adapter_path=kwargs.pop("module_adapter_path", None),
        dtensor_factory=api.distribute_tensor,
    )
    if materialize_device == "cpu":
        _move_buffers_to_device(model, get_device_type())

    # Register grad norm clipping method for FSDP2
    from .clip_grad_norm import clip_grad_norm as clip_grad_norm_fn

    model.clip_grad_norm_ = types.MethodType(clip_grad_norm_fn, model)

    return model


def build_parallelize_model(
    model: "nn.Module",
    weights_path: Optional[str | Mapping[str, str]] = None,
    enable_reshard_after_forward: bool = True,
    mixed_precision: MixedPrecisionConfig = MixedPrecisionConfig(enable=True),  # noqa
    enable_gradient_checkpointing: bool = True,
    basic_modules: Optional[List[str]] = None,
    muon_expert_zero_comm: bool = False,
    low_precision_reduce_scatter_comm: bool = False,
    compile_config: Optional[CompileConfig] = None,
    should_skip_hf_weight_load: bool = False,
    **kwargs: Any,
) -> "nn.Module":
    """Apply parallel strategies to the model.

    Args:
        weights_path: ``None`` for random init, a single HF snapshot for the
            whole ``model``, or ``{child_name: snapshot_path}`` for a composed
            model (SeedOmni ``fsdp_scope='model'``). The single-model trainers
            pass ``args.model.model_path``; per-module SeedOmni wrap passes that
            module's path; the composed wrap passes one path per child.
        muon_expert_zero_comm: Shard ExtraParallel weights on dim-0 when the
            EP-local dim is divisible by ``ep_fsdp_size``.
    """
    parallel_state = get_parallel_state()
    if parallel_state.dp_mode != "fsdp2":
        raise ValueError("Hyper parallelization requires fsdp_mode='fsdp2'.")
    if low_precision_reduce_scatter_comm is not False:
        # Only literal False bypasses validation; false-like non-booleans must still raise.
        validate_low_precision_reduce_scatter_comm(
            low_precision_reduce_scatter_comm, mixed_precision, fsdp_mode=parallel_state.dp_mode
        )
    compile_config = compile_config or CompileConfig()

    if not parallel_state.fsdp_enabled:
        if kwargs.get("init_device") not in ["cuda", "npu", "mlu"]:
            raise ValueError("Only FSDP training supports `init_device=meta`.")

    if mixed_precision.enable:  # upcast to float32 before feed it to optimizer
        model = model.float()

    checkpoint_early_stop = kwargs.pop("early_stop", True)
    if enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        logger.info_rank0("Enable gradient checkpointing.")
        use_reentrant = kwargs.pop("enable_reentrant", False)
        if use_reentrant:
            torch.utils.checkpoint.CheckpointFunction = CheckpointFunction

        gradient_checkpointing_kwargs = {
            "use_reentrant": use_reentrant,
            "context_fn": kwargs.pop("recompute_context_fn", noop_context_fn),
        }
        if not use_reentrant:
            gradient_checkpointing_kwargs["early_stop"] = checkpoint_early_stop

        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        )

    if parallel_state.tp_enabled:
        logger.info_rank0("Apply tensor parallel to the model.")
        model = parallelize_module(
            model,
            device_mesh=parallel_state.tp_mesh,
        )

    if parallel_state.fsdp_enabled:
        logger.info_rank0(f"Apply data parallel to the model: {parallel_state.dp_mode}.")
        model = parallelize_model_fsdp2(
            model=model,
            weights_path=weights_path,
            enable_reshard_after_forward=enable_reshard_after_forward,
            mixed_precision=mixed_precision,
            basic_modules=basic_modules,
            muon_expert_zero_comm=muon_expert_zero_comm,
            low_precision_reduce_scatter_comm=low_precision_reduce_scatter_comm,
            compile_config=compile_config,
            should_skip_hf_weight_load=should_skip_hf_weight_load,
            **kwargs,
        )
    elif compile_config.enable:
        raise RuntimeError(
            "model.accelerator.torch_compile.enable requires FSDP2; compile without FSDP is not supported."
        )

    return model
