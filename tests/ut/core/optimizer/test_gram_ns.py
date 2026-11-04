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

"""CPU-only tests of optional Gram-NS delegation and factory integration."""

import copy
import subprocess
import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.core.optimizer import get_hyper_optimizer
from hyper_parallel.core.optimizer.gram_ns import GramNewtonSchulzAdapter, _validate_gram_device
from hyper_parallel.core.optimizer.muon import NSInputTransform, _NS_ASYM5_COEFFS, _NS_LEGACY_COEFFS
from tests.common.mark_utils import arg_mark


@contextmanager
def _optional_package(package):
    """Restore only the optional package entry, retaining imports performed by optimizers."""
    name = "gram_newton_schulz"
    missing = object()
    previous = sys.modules.get(name, missing)
    sys.modules[name] = package
    try:
        yield
    finally:
        if previous is missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def _fake_package():
    orthogonalizer = MagicMock(side_effect=lambda tensor: tensor.clone())
    factory = MagicMock(return_value=orthogonalizer)
    return SimpleNamespace(GramNewtonSchulz=factory), factory, orthogonalizer


def _split_transform(_name, tensor):
    def restore(updates: list[torch.Tensor], output: torch.Tensor) -> None:
        """Restore split logical matrices to their original column order.

        Args:
            updates: Orthogonalized column blocks.
            output: Destination matrix with the original shape.
        """
        output.copy_(torch.cat(updates, dim=-1))
    return NSInputTransform(tensors=list(tensor.chunk(2, dim=-1)), restore=restore)


class TestGramNewtonSchulzAdapter(unittest.TestCase):
    """Check package delegation, compatibility gates, and constructor isolation."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_package_receives_schedule_and_reuses_cached_instance(self):
        """
        Feature: Direct Gram-NS package delegation.
        Description: Exercise all coefficient variants, shape batches, and repeated calls with a package double.
        Expectation: The external constructor receives the native schedule and each result preserves shape and dtype.
        """
        package, factory, orthogonalizer = _fake_package()
        cases = [
            ("legacy", 3, None, (_NS_LEGACY_COEFFS,) * 3),
            ("asym5", 2, None, _NS_ASYM5_COEFFS[:2]),
            ("custom", 1, [(1.0, -0.5, 0.1), (2.0, -1.0, 0.2)], ((1.0, -0.5, 0.1),)),
        ]
        with _optional_package(package), patch(
                "hyper_parallel.core.optimizer.gram_ns._validate_gram_device") as validate:
            adapter = GramNewtonSchulzAdapter()
            for variant, steps, coefficients, expected in cases:
                config = {"ns_variant": variant, "ns_coefficients": coefficients, "ns_epsilon": 2e-6}
                for shape in ((4, 8), (8, 4), (3, 4, 8)):
                    tensor = torch.randn(shape, dtype=torch.bfloat16)
                    result = adapter(tensor, steps, ns_config=config)
                    torch.testing.assert_close(result, tensor, rtol=0, atol=0)
                    orthogonalizer.assert_called_with(tensor)
                self.assertEqual(factory.call_args.kwargs, {
                    "ns_coefficients": [list(coeff) for coeff in expected], "ns_epsilon": 2e-6,
                    "ns_use_kernels": True, "gram_newton_schulz_reset_iterations": [2] if steps > 2 else [],
                    "compile_kwargs": None,
                })
            self.assertEqual(factory.call_count, 3)
            validate.assert_called_once_with(torch.device("cpu"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_missing_dependency_has_installation_guidance(self):
        """
        Feature: Optional Gram-NS dependency.
        Description: Construct the selected backend with the package unavailable.
        Expectation: The error explains how to install the optional dependency.
        """
        with _optional_package(None):
            with self.assertRaisesRegex(ImportError, "pip install gram-newton-schulz --no-build-isolation"):
                GramNewtonSchulzAdapter()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_device_and_cuda_version_requirements(self):
        """
        Feature: Gram-NS hardware eligibility.
        Description: Check CPU, NPU, unavailable CUDA, old architectures and old CUDA runtimes.
        Expectation: Unsupported configurations fail explicitly while supported CUDA devices are accepted.
        """
        for device in (torch.device("cpu"), SimpleNamespace(type="npu")):
            with self.subTest(device=device), self.assertRaisesRegex(RuntimeError, "CUDA tensors"):
                _validate_gram_device(device)
        device = torch.device("cuda", 1)
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA tensors"):
                _validate_gram_device(device)
        with patch("torch.cuda.is_available", return_value=True), patch(
                "torch.cuda.get_device_capability", return_value=(8, 0)):
            with self.assertRaisesRegex(RuntimeError, "SM90"):
                _validate_gram_device(device)
        for version in (None, "12.8", "12.9", "13.0"):
            with self.subTest(version=version), patch("torch.cuda.is_available", return_value=True), patch(
                    "torch.cuda.get_device_capability", return_value=(9, 0)) as capability, patch(
                    "torch.version.cuda", version):
                if version in (None, "12.8"):
                    with self.assertRaisesRegex(RuntimeError, "CUDA 12.9"):
                        _validate_gram_device(device)
                else:
                    _validate_gram_device(device)
                capability.assert_called_once_with(device)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_invalid_steps_and_runtime_errors_are_not_hidden(self):
        """
        Feature: Gram-NS error boundaries.
        Description: Request invalid schedules or let the external kernel raise an error.
        Expectation: Invalid input and external runtime errors propagate without changing backends.
        """
        package, factory, orthogonalizer = _fake_package()
        tensor = torch.ones(2, 4)
        config = {"ns_variant": "custom", "ns_coefficients": [(1., 0., 0.)], "ns_epsilon": 1e-10}
        with _optional_package(package), patch(
                "hyper_parallel.core.optimizer.gram_ns._validate_gram_device"):
            adapter = GramNewtonSchulzAdapter()
            for steps in (0, -1, True, 2):
                with self.subTest(steps=steps), self.assertRaises(ValueError):
                    adapter(tensor, steps, ns_config=config)
            factory.assert_not_called()
            orthogonalizer.side_effect = RuntimeError("external kernel failure")
            with self.assertRaisesRegex(RuntimeError, "external kernel failure"):
                adapter(tensor, 1, ns_config=config)


class TestGramOptimizerFactory(unittest.TestCase):
    """Exercise the adapter through actual optimizer factory and update paths."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_factory_preserves_group_overrides_and_checkpoint(self):
        """
        Feature: Gram-NS factory and parameter-group configuration.
        Description: Step two groups with different schedules, change epsilon, and save and restore momentum.
        Expectation: Effective group options reach the package and checkpoints retain ordinary optimizer state.
        """
        model = torch.nn.Sequential(torch.nn.Linear(4, 2, bias=False), torch.nn.Linear(2, 4, bias=False))
        params = list(model.parameters())
        for param in params:
            param.grad = torch.ones_like(param)
        groups = [{"params": [params[0]], "ns_variant": "legacy", "ns_steps": 2, "ns_epsilon": 1e-5},
                  {"params": [params[1]], "ns_variant": "custom", "ns_steps": 1,
                   "ns_coefficients": [(1., 0., 0.)], "ns_epsilon": 1e-4}]
        package, factory, _ = _fake_package()
        with _optional_package(package), patch(
                "hyper_parallel.core.optimizer.gram_ns._validate_gram_device"):
            chained = get_hyper_optimizer(model, groups, [], ns_backend="gram",
                                          muon_kwargs={"momentum_mode": "vanilla"})
            opt = chained.optimizers_dict["muon"]
            self.assertIsInstance(opt.zeropower_fn, GramNewtonSchulzAdapter)
            before = [param.detach().clone() for param in params]
            chained.step()
            for param, initial in zip(params, before):
                self.assertFalse(torch.equal(param, initial))
                torch.testing.assert_close(opt.state[param]["momentum_buffer"], torch.full_like(param, 0.05))
            self.assertEqual(factory.call_args_list[0].kwargs["ns_coefficients"], [list(_NS_LEGACY_COEFFS)] * 2)
            self.assertEqual(factory.call_args_list[1].kwargs["ns_coefficients"], [[1., 0., 0.]])
            self.assertEqual([call.kwargs["ns_epsilon"] for call in factory.call_args_list], [1e-5, 1e-4])
            saved = copy.deepcopy(opt.state_dict())
            opt.load_state_dict(saved)
            opt.param_groups[0]["ns_epsilon"] = 2e-5
            chained.step()
            self.assertEqual(factory.call_count, 3)
            self.assertEqual(factory.call_args.kwargs["ns_epsilon"], 2e-5)
            self.assertTrue(all("zeropower_fn" not in group for group in saved["param_groups"]))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_transform_path_receives_group_options(self):
        """
        Feature: Gram-NS with reversible matrix transforms.
        Description: Split one logical matrix into two matrices before delegating to the package.
        Expectation: The package sees the transformed batch and the current group's epsilon.
        """
        model = torch.nn.Linear(4, 2, bias=False)
        model.weight.grad = torch.ones_like(model.weight)
        package, factory, orthogonalizer = _fake_package()
        with _optional_package(package), patch(
                "hyper_parallel.core.optimizer.gram_ns._validate_gram_device"):
            opt = get_hyper_optimizer(model, [{"params": [model.weight], "ns_epsilon": 3e-5}], [],
                                      ns_backend="gram", muon_kwargs={"ns_transform_fn": _split_transform})
            opt.step()
            self.assertEqual(tuple(orthogonalizer.call_args.args[0].shape), (2, 2, 2))
            self.assertEqual(factory.call_args.kwargs["ns_epsilon"], 3e-5)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_native_callbacks_remain_compatible_and_conflicts_fail(self):
        """
        Feature: Backend selection and callback compatibility.
        Description: Use a legacy two-argument callback and request conflicting or invalid backend selections.
        Expectation: Legacy callbacks work, native needs no Gram dependency, and conflicts fail explicitly.
        """
        model = torch.nn.Linear(4, 2, bias=False)
        model.weight.grad = torch.ones_like(model.weight)
        calls = []

        def callback(tensor: torch.Tensor, steps: int) -> torch.Tensor:
            """Exercise the original two-argument callback contract.

            Args:
                tensor: Prepared NS input.
                steps: Requested iteration count.

            Returns:
                An independent copy of the input.
            """
            calls.append(steps)
            return tensor.clone()

        with _optional_package(None):
            opt = get_hyper_optimizer(model, [{"params": [model.weight]}], [],
                                      muon_kwargs={"zeropower_fn": callback, "ns_steps": 2})
            opt.step()
            self.assertEqual(calls, [2])
            native = get_hyper_optimizer(model, [{"params": [model.weight]}], [])
            self.assertIsNone(native.optimizers_dict["muon"].zeropower_fn)
            with self.assertRaisesRegex(ValueError, "custom zeropower_fn"):
                get_hyper_optimizer(model, [{"params": [model.weight]}], [], ns_backend="gram",
                                    muon_kwargs={"muon_zeropower_fn": callback})
            with self.assertRaisesRegex(ValueError, "ns_backend"):
                get_hyper_optimizer(model, [], [], ns_backend="typo")
            adam_only = get_hyper_optimizer(model, [], [{"params": [model.weight]}], ns_backend="gram")
            self.assertEqual(list(adam_only.optimizers_dict), ["adamw"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_native_import_does_not_load_gram_dependency(self):
        """
        Feature: Optional dependency import isolation.
        Description: Import the optimizer package in a fresh interpreter.
        Expectation: Neither the Gram adapter nor the external dependency is imported.
        """
        script = (
            "import sys; import hyper_parallel.core.optimizer; "
            "assert 'hyper_parallel.core.optimizer.gram_ns' not in sys.modules; "
            "assert 'gram_newton_schulz' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, f"returncode={result.returncode}, stderr={result.stderr}")
