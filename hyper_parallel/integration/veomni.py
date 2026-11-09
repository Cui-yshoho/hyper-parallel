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
"""Explicit VeOmni construction bindings for Hyper-managed Torch DTensors.

Native loading, runtime and checkpoint functions are not rebound.
Activate before constructing the trainer; activation is not thread-safe.
"""

from contextlib import contextmanager
from functools import wraps
from importlib import import_module
from inspect import signature
from pathlib import Path
import runpy
import sys
from typing import Any, Iterable, Iterator

import torch
from torch.distributed.device_mesh import DeviceMesh as TorchMesh
from torch.distributed.tensor import DTensor as TorchDTensor
from torch.distributed.tensor import Shard as TorchShard

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.fully_shard.api import HSDPModule, fully_shard
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy, MixedPrecisionPolicy
from hyper_parallel.core.utils import clip_grad_norm_


def _modules() -> list:
    return [module for name, module in tuple(sys.modules.items()) if module is not None
            and (name == "veomni" or name.startswith("veomni."))]


class VeOmniAdapter:
    """Select Hyper construction once without replacing Torch algorithms."""

    def __init__(self) -> None:
        """Create an inactive adapter without importing VeOmni."""
        self._bindings = []

    def enable(self) -> "VeOmniAdapter":
        """Bind Hyper construction and clipping only within VeOmni modules."""
        if self._bindings:
            return self
        parallel = import_module("veomni.distributed.torch_parallelize")
        native = import_module("veomni.optim.optimizer")
        clipping = import_module("veomni.distributed.fsdp2.clip_grad_norm")
        old_shard, old_optimizer = parallel.fully_shard, native.build_optimizer
        old_clip = clipping.clip_grad_norm
        if any(getattr(function, "_hyper_adapter", None) is not None for function in (old_shard, old_clip)):
            raise RuntimeError("A VeOmni Hyper adapter is already active.")
        optimizer_signature = signature(old_optimizer)
        if not {"model", "optimizer_type", "optimizer_config"}.issubset(optimizer_signature.parameters):
            raise RuntimeError("Unsupported VeOmni optimizer factory signature.")
        clip_signature = signature(old_clip)
        if not {"model", "max_norm"}.issubset(clip_signature.parameters):
            raise RuntimeError("Unsupported VeOmni gradient clipping signature.")

        @wraps(old_shard)
        def shard(module: torch.nn.Module, **kwargs: Any) -> torch.nn.Module:
            mesh = kwargs.get("mesh")
            if isinstance(mesh, TorchMesh):
                kwargs["mesh"] = DeviceMesh.from_group(mesh.get_all_groups(), mesh.device_type,
                                                       mesh=mesh.mesh, mesh_dim_names=mesh.mesh_dim_names)
            policy = kwargs.get("mp_policy")
            if policy is not None:
                kwargs["mp_policy"] = MixedPrecisionPolicy(**{
                    key: getattr(policy, key) for key in
                    ("param_dtype", "reduce_dtype", "output_dtype", "cast_forward_inputs")})
            offload = kwargs.get("offload_policy")
            if offload is not None:
                kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=offload.pin_memory)
            placement = kwargs.get("shard_placement_fn")
            if placement is not None:
                kwargs["shard_placement_fn"] = lambda parameter: _placement(placement(parameter))
            kwargs.setdefault("reshard_after_forward", False)
            return fully_shard(module, use_torch_dtensor=True, **kwargs)

        @wraps(old_optimizer)
        def optimizer(*args: Any, **kwargs: Any) -> Any:
            bound = optimizer_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            managed = any(isinstance(module, HSDPModule) for module in values["model"].modules())
            if values["optimizer_type"] != "muon" or not managed:
                return old_optimizer(*args, **kwargs)
            return _build_muon(values, native)

        @wraps(old_clip)
        def clip(*args: Any, **kwargs: Any) -> torch.Tensor:
            if not self._bindings:
                return old_clip(*args, **kwargs)
            bound = clip_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            model = values["model"]
            if not any(isinstance(module, HSDPModule) for module in model.modules()):
                return old_clip(*args, **kwargs)
            return clip_grad_norm_(model.parameters(), values["max_norm"], values.get("norm_type", 2.0),
                                   values.get("error_if_nonfinite", False), values.get("foreach"))

        shard._hyper_adapter = self
        clip._hyper_adapter = self
        self._bindings = [(old_shard, shard), (parallel.FSDPModule, (parallel.FSDPModule, HSDPModule)),
                          (old_optimizer, optimizer), (old_clip, clip)]
        self._replace(self._bindings)
        return self

    @staticmethod
    def _replace(bindings: list) -> None:
        for module in _modules():
            for name, value in tuple(vars(module).items()):
                for original, replacement in bindings:
                    if value is original:
                        setattr(module, name, replacement)
                        break

    def disable(self) -> None:
        """Restore owned framework bindings, including late-imported aliases."""
        self._replace([(replacement, original) for original, replacement in self._bindings])
        self._bindings = []

    def __enter__(self) -> "VeOmniAdapter":
        """Activate for one non-nested application scope."""
        if self._bindings:
            raise RuntimeError("Nested adapter contexts are unsupported.")
        return self.enable()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Restore bindings when leaving the application scope."""
        self.disable()


def _placement(placement: Any) -> Any:
    if placement is None or isinstance(placement, Shard):
        return placement
    if isinstance(placement, TorchShard):
        return Shard(placement.dim)
    raise NotImplementedError("Only Shard placement callbacks are supported.")


@contextmanager
def _state_templates(optimizers: Iterable, retain: bool = False) -> Iterator[None]:
    """Expose layout-correct load destinations without fake optimizer updates."""
    temporary = []
    restored = False
    try:
        for leaf in optimizers:
            for group in leaf.param_groups:
                for parameter in group["params"]:
                    if leaf.state.get(parameter):
                        continue
                    if getattr(leaf, "is_muon", False):
                        state = {"momentum_buffer": torch.zeros_like(parameter)}
                    else:
                        step_device = parameter.device if group.get("fused") or group.get("capturable") else "cpu"
                        state = {"step": torch.zeros((), device=step_device),
                                 "exp_avg": torch.zeros_like(parameter), "exp_avg_sq": torch.zeros_like(parameter)}
                    leaf.state[parameter] = state
                    temporary.append((leaf, parameter))
        yield
        restored = retain
    finally:
        if not restored:
            for leaf, parameter in temporary:
                leaf.state.pop(parameter, None)


def _build_muon(values: dict, native: Any) -> Any:
    # VeOmni and its optimizer helpers are optional, resolved only on activation.
    utils = import_module("hyper_parallel.core.optimizer.utils")
    utils._install_logger_methods()
    detect = import_module("hyper_parallel.core.optimizer.dtensor_compat").detect_dtensor_backend
    model = values["model"]
    if not all(isinstance(parameter, TorchDTensor) for parameter in model.parameters()):
        raise NotImplementedError("This integration requires genuine Torch DTensor parameters.")
    detect([], [{"params": list(model.parameters())}])
    muon = import_module("hyper_parallel.core.optimizer.muon").Muon
    options = dict(values.get("muon_kwargs") or {})
    if values["optimizer_config"] is not None:
        options = {**native._collect_muon_kwargs(values["optimizer_config"]), **options}
    if values["param_groups"] is not None or options.get("head_group_size", 0):
        raise NotImplementedError("Custom Muon groups and head splitting need explicit translation.")
    if options.get("adjust_lr_fn", "match_rms_adamw") != "match_rms_adamw":
        raise NotImplementedError("Only RMS-matched Muon LR scaling is supported.")
    for name, parameter in model.named_parameters():
        parameter.model_name = name
    split = import_module("veomni.optim.muon").split_muon_adamw_params
    matrix, scalar, _, _ = split(model, values["no_decay_modules"], values["no_decay_params"])
    coefficients = tuple(options.get("ns_coefficients", (3.4445, -4.775, 2.0315)))
    steps = options.get("ns_steps", 5)
    implementation = options.get("ns_implementation", "std")
    callback = None
    if implementation != "std":
        run_ns = import_module("veomni.optim.muon").run_newton_schulz

        def callback(matrix: torch.Tensor, steps: int) -> torch.Tensor:
            return run_ns(matrix, ns_steps=steps, ns_coefficients=coefficients,
                          eps=options.get("eps", 1e-7), ns_implementation=implementation,
                          gram_ns_reset_iterations=options.get("gram_ns_reset_iterations", (2,)))

    lr = values["lr"] if options.get("lr") is None else options["lr"]
    leaves = {"muon": muon(matrix, lr=lr, momentum=options.get("momentum", 0.95),
                           nesterov=options.get("nesterov", True), weight_decay=options.get("weight_decay", 0.0),
                           ns_variant="custom", ns_steps=steps, ns_coefficients=[coefficients] * steps,
                           ns_epsilon=options.get("eps", 1e-7), zeropower_fn=callback)}
    for group in leaves["muon"].param_groups:
        group.setdefault("step", 0)
    if scalar:
        groups = native._make_param_groups_for_subset(model, scalar, values["weight_decay"],
                                                      values["no_decay_modules"], values["no_decay_params"])
        leaves["adamw"] = torch.optim.AdamW(groups, lr=values["lr"], betas=values["betas"], eps=values["eps"],
                                            fused=values["fused"], foreach=not values["fused"])

    class HyperMuonContainer(native.MultiOptimizer):
        """Reuse native scheduler/DCP protocols without fake optimizer steps."""

        def state_dict(self) -> dict:
            """Use native DCP serialization over transient state destinations."""
            with _state_templates(self.optimizers_dict.values()):
                return super().state_dict()

        def _sparse_state_dict(self) -> dict:
            owner = self.optimizers_dict["muon"]
            owner._broadcast_state_fused_for_ckpt()
            try:
                return super()._sparse_state_dict()
            finally:
                owner.cleanup_synced_state()

        def load_state_dict(self, state_dict: dict) -> None:
            """Keep loaded state on its owning ranks using Hyper's cleanup."""
            # Torch DCP's initializer would otherwise run a zero-LR Muon step.
            with _state_templates(self.optimizers_dict.values(), retain=True):
                super().load_state_dict(state_dict)
            self.optimizers_dict["muon"].cleanup_synced_state()

    return HyperMuonContainer(model, leaves, list(leaves))


_ADAPTER = VeOmniAdapter()


def apply_veomni_patch() -> VeOmniAdapter:
    """Apply Hyper bindings before constructing an otherwise unchanged VeOmni task."""
    return _ADAPTER.enable()


def disable() -> None:
    """Restore the default adapter's bindings."""
    _ADAPTER.disable()


def main() -> None:
    """Execute an original task; do not add a training loop or runtime."""
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m hyper_parallel.integration.veomni TASK.py [ARGS ...]")
    task = Path(sys.argv[1]).resolve()
    if not task.is_file():
        raise SystemExit(f"Task script does not exist: {task}")
    original = sys.argv
    try:
        sys.argv = [str(task), *original[2:]]
        with VeOmniAdapter():
            runpy.run_path(str(task), run_name="__main__")
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
