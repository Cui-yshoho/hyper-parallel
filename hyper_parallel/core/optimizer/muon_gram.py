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
"""Buffered Gram Newton-Schulz with a metadata-only algorithm cost model."""

import math
from collections.abc import Sequence

import torch

from hyper_parallel.components.functional.symmetric_matmul import symmetric_matmul


_MIN_GRAM_SAVED_FLOPS = 50_000_000_000


def use_gram_for_shape(shape: Sequence[int]) -> bool:
    """Select Gram for a five-step iteration with one restart after step two.

    Args:
        shape: Positive matrix or batched matrix dimensions.

    Returns:
        Whether ordinary-GEMM savings exceed 50 GFLOPs and the short side is
        larger than 256. Square and near-square matrices use standard NS.
    """
    if len(shape) not in (2, 3) or min(shape) <= 0:
        return False
    batch = math.prod(shape[:-2])
    short, long = sorted(shape[-2:])
    return short > 256 and 6 * batch * short**2 * (2 * long - 3 * short) >= _MIN_GRAM_SAVED_FLOPS


def _standard(value, coefficients, backend):
    shape = (*value.shape[:-2], value.shape[-2], value.shape[-2])
    gram, polynomial = value.new_empty(shape), value.new_empty(shape)
    updated = value.new_empty(value.shape)
    for linear, cubic, quintic in coefficients:
        symmetric_matmul(value, value.mT, out=gram, backend=backend)
        symmetric_matmul(gram, gram, out=polynomial, backend=backend)
        # A rejected Gram candidate must retain standard NS coefficient rounding.
        gram.mul_(cubic)
        polynomial.mul_(quintic)
        gram.add_(polynomial)
        torch.matmul(gram, value, out=updated)
        value.mul_(linear).add_(updated)
    return value


def _gram(value, coefficients, backend, reset_iterations=(2,)):
    shape = (*value.shape[:-2], value.shape[-2], value.shape[-2])
    gram, polynomial, accumulated, work, intermediate = [value.new_empty(shape) for _ in range(5)]
    rectangular = torch.empty_like(value)
    identity = torch.eye(value.shape[-2], dtype=value.dtype, device=value.device)
    resets = {int(index) for index in reset_iterations}
    symmetric_matmul(value, value.mT, out=gram, backend=backend)
    for index, (linear, cubic, quintic) in enumerate(coefficients):
        if index in resets and index != 0:
            torch.matmul(accumulated, value, out=rectangular)
            value, rectangular = rectangular, value
            symmetric_matmul(value, value.mT, out=gram, backend=backend)
        symmetric_matmul(gram, gram, gram, alpha=quintic, beta=cubic, out=polynomial, backend=backend)
        if index == 0 or index in resets:
            # A strided diagonal update incurs an AICPU view copy on Ascend.
            torch.add(polynomial, identity, alpha=linear, out=accumulated)
        else:
            symmetric_matmul(accumulated, polynomial, accumulated, beta=linear, out=work, backend=backend)
            accumulated, work = work, accumulated
        if index + 1 < len(coefficients) and index + 1 not in resets:
            symmetric_matmul(gram, polynomial, gram, beta=linear, out=intermediate, backend=backend)
            symmetric_matmul(polynomial, intermediate, intermediate, beta=linear, out=work, backend=backend)
            gram, work = work, gram
    torch.matmul(accumulated, value, out=rectangular)
    return rectangular


@torch.no_grad()
def newton_schulz_with_gram(
    inputs: torch.Tensor, coefficients: Sequence[tuple[float, float, float]], epsilon: float, backend: str = "cann",
    *, reset_iterations: Sequence[int] = (2,),
) -> torch.Tensor:
    """Choose standard or Gram NS; each symmetric product chooses its own backend.

    Args:
        inputs: Floating-point matrix/batch in the caller's computation dtype.
        coefficients: Resolved quintic polynomial coefficient schedule.
        epsilon: Normalization stability term.
        backend: Symmetric GEMM backend, independent of the algorithm selector.
        reset_iterations: Zero-based steps refreshing the Gram matrix. Zero,
            negative indices and indices outside the schedule have no effect.

    Returns:
        Update with the input shape and dtype; input storage is unmodified.

    Note:
        The Gram identities hold in exact arithmetic. Low-precision intermediate
        products need not commute exactly; symmetric kernels project those
        products onto one mirrored triangle. This is not bitwise-equivalent NS.
    """
    transposed = inputs.shape[-2] > inputs.shape[-1]
    value = inputs.mT if transposed else inputs
    value = value / (value.norm(dim=(-2, -1), keepdim=True) + epsilon)
    if use_gram_for_shape(inputs.shape):
        result = _gram(value, coefficients, backend, reset_iterations)
    else:
        result = _standard(value, coefficients, backend)
    return result.mT if transposed else result
