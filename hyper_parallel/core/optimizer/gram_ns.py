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

"""Optional Dao-AILab Gram-NS adapter for the Muon zeropower callback."""

from typing import Any, Dict, Tuple

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.optimizer.muon import _NS_ASYM5_COEFFS, _NS_LEGACY_COEFFS


def _validate_gram_device(device: torch.device) -> None:
    """Require the CUDA architectures and runtime supported by the upstream kernels."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ns_backend='gram' requires CUDA tensors on a Hopper or Blackwell GPU (SM90+)")
    capability = torch.cuda.get_device_capability(device)
    if capability < (9, 0):
        raise RuntimeError(f"ns_backend='gram' requires SM90+, got compute capability {capability}")
    cuda_version = torch.version.cuda
    if cuda_version is None or tuple(int(part) for part in cuda_version.split(".")[:2]) < (12, 9):
        raise RuntimeError(f"ns_backend='gram' requires CUDA 12.9+, got {cuda_version}")


def _coefficient_schedule(steps: int, config: Dict[str, Any]) -> Tuple[Tuple[float, float, float], ...]:
    """Preserve the native Muon coefficient schedule for each parameter group."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError(f"Gram NS steps must be a positive integer, got {steps!r}")
    variant = config["ns_variant"]
    if variant == "legacy":
        return (_NS_LEGACY_COEFFS,) * steps
    if variant == "asym5":
        return _NS_ASYM5_COEFFS[:steps]
    if variant == "custom":
        coefficients = config["ns_coefficients"]
        if coefficients is None or steps > len(coefficients):
            raise ValueError("Gram NS requires at least ns_steps custom coefficient groups")
        return tuple(tuple(coeff) for coeff in coefficients[:steps])
    raise ValueError(f"Unknown Gram NS coefficient variant {variant!r}")


class GramNewtonSchulzAdapter:
    """Call the optional package directly while caching its configured orthogonalizers."""

    def __init__(self) -> None:
        """Load the optional dependency only when the Gram backend is selected."""
        # The optional CUDA package must not be imported by the native or NPU path.
        try:
            from gram_newton_schulz import GramNewtonSchulz  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError(
                "ns_backend='gram' requires gram-newton-schulz and its CUDA dependencies. "
                "Install with: pip install gram-newton-schulz --no-build-isolation"
            ) from exc
        self._orthogonalizer_type = GramNewtonSchulz
        self._orthogonalizers: Dict[Tuple[Any, ...], Any] = {}
        self._validated_devices = set()

    def __call__(self, ns_inputs: torch.Tensor, steps: int, *, ns_config: Dict[str, Any]) -> torch.Tensor:
        """Orthogonalize the full matrices gathered by Muon using the external package.

        Args:
            ns_inputs: One matrix or a batch of complete logical matrices.
            steps: Number of native NS coefficients to use.
            ns_config: Current parameter group's ns_variant, ns_coefficients, and ns_epsilon.

        Returns:
            The package's result with the input shape and dtype.
        """
        device = ns_inputs.device
        if device not in self._validated_devices:
            _validate_gram_device(device)
            self._validated_devices.add(device)
        schedule = _coefficient_schedule(steps, ns_config)
        epsilon = ns_config["ns_epsilon"]
        key = (device, schedule, epsilon)
        orthogonalizer = self._orthogonalizers.get(key)
        if orthogonalizer is None:
            # Match VeOmni's eager package adapter; upstream handles shape and dtype.
            orthogonalizer = self._orthogonalizer_type(
                ns_coefficients=[list(coeff) for coeff in schedule],
                ns_epsilon=epsilon,
                ns_use_kernels=True,
                gram_newton_schulz_reset_iterations=[2] if len(schedule) > 2 else [],
                compile_kwargs=None,
            )
            self._orthogonalizers[key] = orthogonalizer
        return orthogonalizer(ns_inputs)
