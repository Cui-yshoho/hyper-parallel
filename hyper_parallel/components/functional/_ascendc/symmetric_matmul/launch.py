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
"""Explicitly built Ascend C backend; no compiler or NPU dependency at package import."""

import ctypes
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
import torch_npu

# Keep the public fallback for versions without the optional raw-stream bridge.
_RAW_STREAM = getattr(torch_npu._C, "_npu_getCurrentRawStream", None)


def _current_stream(device):
    if _RAW_STREAM is not None:
        return _RAW_STREAM(device)
    return torch_npu.npu.current_stream(device).npu_stream


@lru_cache(maxsize=1)
def _library():
    path = Path(__file__).with_name("libhp_symmetric.so")
    if not path.is_file():
        raise RuntimeError("Ascend C library is absent; reinstall hyper_parallel with the CANN toolkit available")
    for name in ("libhp_epilogue_kernel.so", "libhp_fused_kernel.so"):
        ctypes.CDLL(str(path.with_name(name)), mode=ctypes.RTLD_GLOBAL)
    library = ctypes.CDLL(str(path))
    library.hp_abi_version.argtypes = []
    library.hp_abi_version.restype = ctypes.c_int
    if library.hp_abi_version() != 4:
        raise RuntimeError("Rebuild the Ascend C symmetric GEMM library: incompatible ABI")
    library.hp_plan.argtypes = [ctypes.c_int] * 7
    library.hp_plan.restype = ctypes.c_void_p
    library.hp_plan_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    library.hp_plan_info.restype = None
    library.hp_destroy.argtypes = [ctypes.c_void_p]
    library.hp_destroy.restype = None
    library.hp_launch.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_float] * 2
    library.hp_launch.restype = ctypes.c_uint32
    return library


class _Plan:
    def __init__(self, n, k, batch, tile, cores, dtype, direct):
        self.pointer = None
        self.library = _library()
        self.pointer = self.library.hp_plan(n, max(k, 16), batch, tile, cores, dtype, direct)
        if not self.pointer:
            raise RuntimeError("Ascend C Matmul tiling failed for the requested shape")
        values = (ctypes.c_int * 7)()
        self.library.hp_plan_info(self.pointer, values)
        self.n, self.k, self.tile = values[:3]
        self.base_shape = tuple(values[3:6])
        self.row_tile = values[6]

    def __del__(self):
        if self.pointer:
            self.library.hp_destroy(self.pointer)


@lru_cache(maxsize=128)
def _plan(device, n, k, batch, tile, dtype, direct):
    cores = torch_npu.npu.get_device_properties(device).cube_core_num
    return _Plan(n, k, batch, tile, cores, dtype, direct)


def _pad(tensor, rows, columns):
    if tensor.shape[-2:] == (rows, columns) and tensor.data_ptr() % 32 == 0:
        return tensor.contiguous()
    value = tensor.new_zeros((*tensor.shape[:-2], rows, columns))
    value[..., :tensor.shape[-2], :tensor.shape[-1]].copy_(tensor)
    return value


def _destination(out, n, padded_n):
    # DMA writes require aligned addresses even for otherwise contiguous views.
    if padded_n == n and out.data_ptr() % 32 == 0:
        return out
    return out.new_empty((*out.shape[:-2], padded_n, padded_n))


def launch_symmetric_matmul(
    a: torch.Tensor, b: torch.Tensor, c: Optional[torch.Tensor],
    out: torch.Tensor, alpha: float, beta: float,
) -> None:
    """Compute the lower product and its exact mirror on the current NPU stream.

    Args:
        a: Validated left operand.
        b: Validated right operand.
        c: Validated additive operand, or None.
        out: Validated destination.
        alpha: Product multiplier.
        beta: Additive multiplier.
    """
    n, k = a.shape[-2:]
    batch = 1 if a.ndim == 2 else a.shape[0]
    if max(k, batch * n * n, batch * n * k) >= 2 ** 31:
        raise ValueError("Shape exceeds the Ascend C backend indexing range")
    device = a.device.index
    guard = torch_npu.npu.device(device) if device != torch_npu.npu.current_device() else nullcontext()
    with guard:
        product_scale = alpha if k else 0.0
        direct = product_scale == 1 and beta == 0
        plan = _plan(device, n, k, batch, 0, int(a.dtype == torch.bfloat16), int(direct))
        padded_n, padded_k = plan.n, plan.k
        if product_scale != 0:
            a = _pad(a, padded_n, padded_k)
            b = _pad(b.mT, padded_n, padded_k)
        if beta != 0:
            c = _pad(c, padded_n, padded_n)
        destination = _destination(out, n, padded_n)
        scratch = destination if direct or product_scale == 0 else torch.empty(
            destination.shape, device=a.device, dtype=torch.float32,
        )
        status = plan.library.hp_launch(
            plan.pointer, _current_stream(device), a.data_ptr(), b.data_ptr(),
            c.data_ptr() if beta else out.data_ptr(), scratch.data_ptr(), destination.data_ptr(), product_scale, beta,
        )
        if status != 0:
            raise RuntimeError(f"Ascend C kernel launch failed: {status}")
        if destination is not out:
            out.copy_(destination[..., :n, :n])
