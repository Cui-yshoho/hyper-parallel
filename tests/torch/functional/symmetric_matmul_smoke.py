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
"""Installed-package Ascend C, automatic dispatch and Muon integration smoke tests."""

import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu

import hyper_parallel
from hyper_parallel.components.functional.symmetric_matmul import symmetric_matmul
from hyper_parallel.core.optimizer.muon import Muon, zeropower_via_newtonschulz5
from hyper_parallel.core.optimizer.muon_gram import use_gram_for_shape
from hyper_parallel.integration.veomni.optimizer import _build_hyper_muon_adapter


_BRIDGE = importlib.import_module("hyper_parallel.components.functional.symmetric_matmul")


def _relative(actual, reference):
    return ((actual.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-20)).item()


@torch.no_grad()
def _products():
    rows = []
    for shape in ((257, 513), (2, 1024, 4096), (32, 768, 768)):
        value = torch.randn(shape, device="npu", dtype=torch.bfloat16) / shape[-1]**0.5
        reference = symmetric_matmul(value, value.mT, backend="cann")
        native = symmetric_matmul(value, value.mT, backend="ascendc")
        automatic = symmetric_matmul(value, value.mT)
        expected = native if _BRIDGE._use_ascendc(value, 1) else reference
        assert torch.equal(automatic, expected), "Automatic dispatch did not select the expected backend"
        assert torch.equal(native, native.mT), "Native output is not bitwise symmetric"
        error = _relative(native, reference)
        assert error < 0.001, f"Unexpected pure GEMM discrepancy: {error}"
        rows.append({"shape": shape, "ascendc_selected": _BRIDGE._use_ascendc(value, 1), "relative_l2": error})
    value = torch.randn(257, 513, device="npu", dtype=torch.float16)
    backing = torch.full((257 * 257 + 2,), 123, device="npu", dtype=value.dtype)
    output = backing[1:-1].view(257, 257)
    stream = torch_npu.npu.Stream()
    stream.wait_stream(torch_npu.npu.current_stream())
    with torch_npu.npu.stream(stream):
        symmetric_matmul(value, value.mT, out=output, backend="ascendc")
    torch_npu.npu.current_stream().wait_stream(stream)
    assert backing[0].item() == 123 and backing[-1].item() == 123
    assert torch.equal(output, output.mT)
    fused = symmetric_matmul(output, output, output, alpha=0.01, beta=-0.2, backend="ascendc")
    assert torch.isfinite(fused).all() and torch.equal(fused, fused.mT)
    reference = (0.01 * (output.float() @ output.float()) - 0.2 * output.float()).to(output.dtype)
    error = _relative(fused, reference)
    assert error < 0.002, f"Unexpected fused epilogue discrepancy: {error}"
    rows.append({"fused_epilogue_relative_l2": error})
    return rows


@torch.no_grad()
def _optimizer():
    value = torch.randn(32, 768, 2048, device="npu", dtype=torch.bfloat16) * 0.01
    original = value.clone()
    update = zeropower_via_newtonschulz5(value, 5, use_gram=True)
    reference = zeropower_via_newtonschulz5(value, 5)
    error = _relative(update, reference)
    assert error < 0.05 and torch.isfinite(update).all(), f"Unexpected NS discrepancy: {error}"
    accelerated = zeropower_via_newtonschulz5(value, 5, use_gram=True, ns_backend="auto")
    accelerated_error = _relative(accelerated, reference)
    backend_error = _relative(accelerated, update)
    assert accelerated_error < 0.05, f"Unexpected accelerated NS discrepancy: {accelerated_error}"
    assert backend_error < 0.05, f"Unexpected Gram backend discrepancy: {backend_error}"
    assert torch.equal(value, original), "NS modified its input"
    parameter = torch.nn.Parameter(original.clone())
    optimizer = Muon([parameter], use_gram=True)
    parameter.grad = value.clone()
    gram_module = importlib.import_module("hyper_parallel.core.optimizer.muon_gram")
    with patch("hyper_parallel.core.optimizer.muon.newton_schulz_with_gram",
               wraps=gram_module.newton_schulz_with_gram) as call:
        optimizer.step()
        assert call.called and use_gram_for_shape(call.call_args.args[0].shape)
    state = optimizer.state_dict()
    restored = Muon([parameter])
    restored.load_state_dict(state)
    assert restored.param_groups[0]["use_gram"] is True
    assert torch.isfinite(parameter).all() and not torch.equal(parameter, original)
    return {"shape": list(value.shape), "relative_l2": error,
            "accelerated_relative_l2": accelerated_error, "gram_backend_relative_l2": backend_error,
            "step_and_state_restore": True}


@torch.no_grad()
def _veomni_step(implementation, initial, gradient):
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(initial.clone()))
    native = SimpleNamespace(
        split_muon_adamw_params=lambda *_: ([model.weight], [], [], []),
        _as_coeff_schedule=lambda coefficients, steps: (tuple(coefficients),) * steps,
    )
    options = {
        "no_decay_modules": [], "no_decay_params": [], "lr": 0.01,
        "muon_kwargs": {"ns_implementation": implementation, "ns_steps": 5, "eps": 1e-10},
    }
    # Only VeOmni grouping is mocked; the optimizer and kernels are real.
    with patch("hyper_parallel.integration.veomni.optimizer.import_module",
               side_effect=lambda name: native if name == "veomni.optim.muon" else importlib.import_module(name)):
        adapter = _build_hyper_muon_adapter(model, options, SimpleNamespace())
    core = adapter.optimizers_dict["muon"]
    assert core.zeropower_fn is None, f"Unexpected callback for {implementation}"
    assert core.param_groups[0]["momentum_mode"] == "vanilla", (
        f"Expected vanilla, got {core.param_groups[0]['momentum_mode']}"
    )
    model.weight.grad = gradient.clone()
    launcher = importlib.import_module("hyper_parallel.components.functional._ascendc.symmetric_matmul.launch")
    with patch.object(launcher, "launch_symmetric_matmul", wraps=launcher.launch_symmetric_matmul) as launch:
        adapter.step()
        count = launch.call_count
    assert torch.isfinite(model.weight).all(), f"Nonfinite parameters for {implementation}"
    assert torch.isfinite(core.state[model.weight]["momentum_buffer"]).all(), (
        f"Nonfinite momentum for {implementation}"
    )
    return {"momentum_mode": "vanilla", "ascendc_launches": count}


@torch.no_grad()
def _veomni_optimizer():
    initial = torch.randn(32, 768, 2048, device="npu", dtype=torch.bfloat16) * 0.01
    gradient = torch.randn_like(initial) * 0.05
    reports = {}
    for implementation in ("std", "gram", "gram_quack"):
        result = _veomni_step(implementation, initial, gradient)
        expected_native = implementation == "gram_quack"
        assert (result["ascendc_launches"] > 0) == expected_native, (
            f"Expected native calls={expected_native} for {implementation}, got {result}"
        )
        reports[implementation] = result
    small = _veomni_step("gram_quack", initial[0, :64, :128].contiguous(), gradient[0, :64, :128].contiguous())
    assert small["ascendc_launches"] == 0, f"Small inputs should use native matmul, got {small}"
    reports["gram_quack_small"] = small
    # Simulate an absent optional library without changing installed files.
    with patch.object(_BRIDGE, "_native_library_present", return_value=False):
        missing = _veomni_step("gram_quack", initial, gradient)
    assert missing["ascendc_launches"] == 0, f"Absent library should use native matmul, got {missing}"
    reports["gram_quack_missing_library"] = missing
    return reports


def test_installed_symmetric_muon() -> None:
    """Worker entry for the repository single-card ST launcher."""
    torch_npu.npu.set_device(0)
    torch.manual_seed(42)
    _products()
    _optimizer()
    _veomni_optimizer()
    torch_npu.npu.synchronize()


def main() -> None:
    """Verify the installed package location and exercise its bundled libraries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    location = Path(hyper_parallel.__file__).resolve()
    assert location.is_relative_to(Path(args.expected_root).resolve()), str(location)
    torch_npu.npu.set_device(0)
    torch.manual_seed(42)
    report = {"package": str(location), "device": torch_npu.npu.get_device_name(0),
              "products": _products(), "muon": _optimizer(), "veomni": _veomni_optimizer()}
    torch_npu.npu.synchronize()
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
