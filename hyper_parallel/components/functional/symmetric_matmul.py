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
"""Symmetric products with static Ascend C / native matmul dispatch."""

import importlib
import math
from functools import lru_cache
from typing import Optional
from pathlib import Path

import torch


_MIN_ASCENDC_FLOPS = 20_000_000_000


def _use_ascendc(a, alpha):
    batch = 1 if a.ndim == 2 else a.shape[0]
    n, k = a.shape[-2:]
    eligible = (a.device.type == "npu" and a.dtype in (torch.float16, torch.bfloat16)
                and alpha != 0 and n > 256 and 2 * batch * n * n * k >= _MIN_ASCENDC_FLOPS)
    return eligible and _native_library_present()


@lru_cache(maxsize=1)
def _native_library_present():
    return (Path(__file__).parent / "_ascendc/symmetric_matmul/libhp_symmetric.so").is_file()


def _check_layout(tensor: torch.Tensor) -> None:
    valid = tensor.layout == torch.strided
    if valid and not tensor.is_contiguous():
        # Inspect transpose contiguity without constructing a temporary Tensor.
        shape, strides = tensor.shape, tensor.stride()
        valid = (strides[-2] == 1 and strides[-1] == shape[-2]
                 and (tensor.ndim == 2 or shape[0] <= 1 or strides[0] == shape[-2] * shape[-1]))
    if not valid:
        raise ValueError("symmetric_matmul requires contiguous matrices or their last-two-axis transposes")


def _output_shape(a, b):
    if a.ndim not in (2, 3) or b.ndim != a.ndim:
        raise ValueError("a and b must both be rank 2 or rank 3")
    if a.shape[:-2] != b.shape[:-2] or a.shape[-1] != b.shape[-2] or a.shape[-2] != b.shape[-1]:
        raise ValueError("Expected a [..., N, K] and b [..., K, N] with identical batch dimensions")
    return (*a.shape[:-2], a.shape[-2], a.shape[-2])


def _check_output(out, inputs, shape, a):
    if out.shape != shape or out.dtype != a.dtype or out.device != a.device or not out.is_contiguous():
        raise ValueError("out must be contiguous with the output shape, dtype and device")
    # Reject shared storage, even for disjoint views, to keep the write contract simple.
    for tensor in inputs:
        shared_storage = (out.device == tensor.device
                          and out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr())
        if out.numel() and tensor.numel() and shared_storage:
            raise ValueError("out must not share storage with an input")


def _check_inputs(tensors, a, backend):
    allowed = (torch.float16, torch.bfloat16) if backend == "ascendc" else (
        torch.float16, torch.bfloat16, torch.float32, torch.float64,
    )
    if a.dtype not in allowed:
        raise ValueError(f"Unsupported dtype for {backend}: {a.dtype}")
    for tensor in tensors:
        if tensor.dtype != a.dtype or tensor.device != a.device:
            raise ValueError("All active inputs must have the same dtype and device")
        _check_layout(tensor)
    if backend == "ascendc" and a.device.type != "npu":
        raise ValueError("The accelerated backends require Ascend NPU tensors")


def _validate(a, b, c, out, alpha, beta, backend):
    if backend not in ("auto", "cann", "ascendc"):
        raise ValueError(f"Unknown symmetric_matmul backend: {backend!r}")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (alpha, beta)):
        raise ValueError("alpha and beta must be finite Python scalars")
    shape = _output_shape(a, b)
    tensors = [a, b]
    if beta != 0:
        if c is None or c.shape != shape:
            raise ValueError("Nonzero beta requires c with the output shape")
        tensors.append(c)
    _check_inputs(tensors, a, backend)
    if out is not None:
        _check_output(out, [a, b] + ([] if c is None else [c]), shape, a)
        tensors.append(out)
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise ValueError("symmetric_matmul is forward-only; use it under torch.no_grad()")
    return shape


def _cann(a, b, c, out, alpha, beta):
    if alpha == 0 or a.shape[-1] == 0:
        out.zero_()
    else:
        torch.matmul(a, b, out=out)
        if alpha != 1:
            out.mul_(alpha)
    if beta != 0:
        out.add_(c.mul(beta))


@lru_cache(maxsize=1)
def _backend_module():
    # Import torch_npu only when a native kernel is selected.
    return importlib.import_module("hyper_parallel.components.functional._ascendc.symmetric_matmul.launch")


def symmetric_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    out: Optional[torch.Tensor] = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Compute alpha * (a @ b) + beta * c for mathematically symmetric operands.

    Args:
        a: Contiguous or transposed matrix [N,K], or batch [B,N,K].
        b: Matrix [K,N], or matching batch [B,K,N]. May alias a.
        c: Symmetric additive input matching the output; ignored when beta is zero.
        alpha: Finite product multiplier. Zero suppresses reads of a and b.
        beta: Finite additive multiplier. Zero suppresses reads of c.
        out: Optional contiguous output that shares no storage with any input.
        backend: "auto" selects Ascend C for installed NPU FP16/BF16 products
            with N > 256 and 2*B*N*N*K >= 20 GFLOPs; otherwise native matmul.
            "cann" or "ascendc" forces a backend for validation.

    Returns:
        Output of the input dtype. Ascend C mirrors rounded lower values;
        native matmul computes both triangles independently.

    Raises:
        ValueError: Unsupported metadata, aliasing, backend, or active autograd.

    Note:
        The caller must ensure AB and the active C are mathematically symmetric.
        This condition is not checked at runtime. Ascend C scales/adds in FP32;
        native matmul uses separate operations in the input dtype, including
        on CPU/CUDA. Backend results need not agree bit for bit. No backward.
        Selection uses metadata only, with no timing or device synchronization.
    """
    shape = _validate(a, b, c, out, alpha, beta, backend)
    result = torch.empty(shape, dtype=a.dtype, device=a.device) if out is None else out
    if result.numel() == 0:
        return result
    native = backend == "ascendc" or (backend == "auto" and _use_ascendc(a, alpha))
    if native:
        _backend_module().launch_symmetric_matmul(a, b, c, result, float(alpha), float(beta))
    else:
        _cann(a, b, c, result, alpha, beta)
    return result
