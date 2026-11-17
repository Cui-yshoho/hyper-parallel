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
"""CPU regressions for optional Hyper Muon algorithm construction and restore."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.integration.veomni.optimizer import HyperOptimizerAdapter, _build_hyper_muon_adapter


class TestHyperMuonCoreOptions(unittest.TestCase):
    """Keep configurable core algorithms independent from VeOmni callbacks."""

    @staticmethod
    def _build(options, config=None, supported=None):
        model = torch.nn.Linear(2, 2, bias=False)
        coefficients = options.get("ns_coefficients", (3.4445, -4.775, 2.0315))
        schedule = (tuple(tuple(row) for row in coefficients)
                    if coefficients and isinstance(coefficients[0], (list, tuple))
                    else (tuple(coefficients),) * options.get("ns_steps", 5))
        native_muon = SimpleNamespace(
            split_muon_adamw_params=MagicMock(return_value=([model.weight], [], [], [])),
            run_newton_schulz=MagicMock(return_value=torch.ones(2, 2)),
            _as_coeff_schedule=MagicMock(return_value=schedule),
        )
        factory = MagicMock(side_effect=lambda groups, **_: SimpleNamespace(param_groups=groups))
        if supported is not None:
            factory.ADDITIONAL_CONFIG_KEYS = supported
        modules = {
            "hyper_parallel.core.optimizer.utils": SimpleNamespace(_install_logger_methods=MagicMock()),
            "hyper_parallel.core.optimizer.dtensor_compat": SimpleNamespace(detect_dtensor_backend=MagicMock()),
            "hyper_parallel.core.optimizer.muon": SimpleNamespace(Muon=factory),
            "veomni.optim.muon": native_muon,
        }
        arguments = {
            "no_decay_modules": [], "no_decay_params": [], "lr": 1e-4,
            "muon_kwargs": options, "optimizer_config": config,
        }
        native_factory = SimpleNamespace(_collect_muon_kwargs=MagicMock(return_value={
            "ns_implementation": getattr(config, "muon_ns_implementation", "std"),
        }))
        with patch("hyper_parallel.integration.veomni.optimizer.import_module", side_effect=modules.__getitem__):
            adapter = _build_hyper_muon_adapter(model, arguments, native_factory)
        return adapter, factory, native_muon.run_newton_schulz

    def test_implementations_bridge_to_hyper_with_vanilla_momentum(self):
        """Feature: VeOmni NS selection.

        Description: Construct all modes and the default using existing VeOmni options.
        Expectation: The core receives the mapped backend and never a VeOmni callback.
        """
        cases = (("std", False, None), ("gram", True, "cann"),
                 ("gram_quack", True, "auto"), (None, True, "auto"))
        for implementation, use_gram, backend in cases:
            with self.subTest(implementation=implementation):
                options = {} if implementation is None else {"ns_implementation": implementation}
                adapter, factory, callback = self._build(options)
                kwargs = factory.call_args.kwargs
                self.assertEqual(kwargs["momentum_mode"], "vanilla")
                self.assertEqual(kwargs["use_gram"], use_gram)
                self.assertEqual(kwargs["ns_backend"], backend)
                self.assertEqual(kwargs["ns_variant"], "custom")
                self.assertIsNone(kwargs["zeropower_fn"])
                callback.assert_not_called()
                group = adapter.optimizers_dict["muon"].param_groups[0]
                for name in ("momentum_mode", "use_gram", "ns_backend"):
                    self.assertEqual(group[name], kwargs[name], f"Metadata mismatch for {name}: {group[name]}")
                self.assertEqual(group["ns_implementation"], implementation or "gram_quack")

    def test_existing_config_and_explicit_implementation_override(self):
        """Feature: Existing VeOmni configuration precedence.

        Description: Select Gram through config, then override with explicit kwargs.
        Expectation: Config needs no new fields and explicit implementation wins.
        """
        config = SimpleNamespace(muon_ns_implementation="gram")
        _, factory, _ = self._build({}, config=config)
        self.assertEqual(factory.call_args.kwargs["ns_backend"], "cann")
        _, factory, _ = self._build({"ns_implementation": "std"}, config=config)
        self.assertFalse(factory.call_args.kwargs["use_gram"])
        self.assertIsNone(factory.call_args.kwargs["ns_backend"])

    def test_standard_preserves_custom_coefficients_steps_and_epsilon(self):
        """Feature: Standard NS compatibility.

        Description: Supply the existing VeOmni custom coefficient triple and step count.
        Expectation: Standard NS retains repeated coefficients and custom schedule.
        """
        coefficients = (3.2, -4.1, 1.7)
        _, factory, _ = self._build({
            "ns_implementation": "std", "ns_coefficients": coefficients,
            "ns_steps": 3, "eps": 1e-8,
        })
        options = factory.call_args.kwargs
        self.assertEqual(options["ns_variant"], "custom")
        self.assertEqual(options["ns_coefficients"], [coefficients] * 3)
        self.assertEqual(options["ns_steps"], 3)
        self.assertEqual(options["ns_epsilon"], 1e-8)

    def test_gram_backends_preserve_the_veomni_default_schedule(self):
        """Feature: Native Gram coefficient semantics.

        Description: Build Gram and Gram with Ascend C using VeOmni's default triple.
        Expectation: Both preserve VeOmni's coefficients and differ only in the backend argument.
        """
        actual = []
        for implementation in ("gram", "gram_quack"):
            _, factory, _ = self._build({
                "ns_implementation": implementation,
                "ns_coefficients": [3.4445, -4.775, 2.0315],
            })
            options = dict(factory.call_args.kwargs)
            self.assertEqual(options["ns_variant"], "custom")
            self.assertEqual(options["ns_steps"], 5)
            self.assertEqual(options["ns_coefficients"], [(3.4445, -4.775, 2.0315)] * 5)
            options.pop("ns_backend")
            actual.append(options)
        self.assertEqual(actual[0], actual[1])

    def test_gram_accepts_custom_steps_coefficients_and_restarts(self):
        """Feature: Configuration translation.

        Description: Supply a custom triple or per-step schedule and non-default restarts.
        Expectation: Hyper receives the native effective schedule and restart configuration.
        """
        schedules = ((3.2, -4.1, 1.7), ((3.2, -4.1, 1.7), (2.9, -3.1, 1.2)))
        for implementation in ("gram", "gram_quack"):
            for coefficients in schedules:
                with self.subTest(implementation=implementation, coefficients=coefficients):
                    adapter, factory, _ = self._build({
                        "ns_implementation": implementation, "ns_steps": 3,
                        "ns_coefficients": coefficients, "eps": 1e-8,
                        "gram_ns_reset_iterations": [0, 1, 4],
                    })
                    expected = (list(coefficients) if isinstance(coefficients[0], tuple)
                                else [coefficients] * 3)
                    options = factory.call_args.kwargs
                    self.assertEqual(options["ns_coefficients"], expected)
                    self.assertEqual(options["ns_steps"], len(expected))
                    self.assertEqual(options["ns_epsilon"], 1e-8)
                    self.assertEqual(options["gram_reset_iterations"], (0, 1, 4))
                    group = adapter.optimizers_dict["muon"].param_groups[0]
                    self.assertEqual(group["gram_ns_reset_iterations"], (0, 1, 4))

    def test_core_overrides_are_not_exposed_through_veomni(self):
        """Feature: One VeOmni algorithm selector.

        Description: Attempt to override fixed momentum or enum-derived core options.
        Expectation: Each explicit override is rejected for every implementation.
        """
        overrides = {"momentum_mode": "legacy", "use_gram": True,
                     "ns_backend": "ascendc", "ns_variant": "asym5", "gram_reset_iterations": (1,)}
        for implementation in ("std", "gram", "gram_quack"):
            for name, value in overrides.items():
                with self.subTest(implementation=implementation, name=name), self.assertRaisesRegex(ValueError, name):
                    self._build({"ns_implementation": implementation, name: value})

    def test_old_core_reports_unavailable_required_features(self):
        """Feature: Core capability validation.

        Description: Use a core without the required momentum and Gram interfaces.
        Expectation: Construction reports missing features instead of changing algorithms.
        """
        with self.assertRaisesRegex(ValueError, "Installed Hyper Muon does not support.*momentum_mode"):
            self._build({"ns_implementation": "gram"}, supported=set())

    def test_invalid_implementations_are_rejected(self):
        """Feature: Bridged algorithm validation.

        Description: Request an unknown implementation.
        Expectation: Invalid requests fail rather than silently select another algorithm.
        """
        cases = [{"ns_implementation": "unknown"}, {"ns_implementation": None}, {"ns_implementation": []}]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self._build(options)

    @staticmethod
    def _checkpoint():
        model = torch.nn.Linear(2, 2, bias=False)
        settings = {
            "ns_implementation": "gram_quack", "ns_coefficients": None, "ns_epsilon": 1e-10,
            "ns_steps": 5, "ns_variant": "asym5", "gram_ns_reset_iterations": (2,),
            "gram_reset_iterations": (2,),
            "momentum_mode": "vanilla", "use_gram": True, "ns_backend": "auto",
        }
        leaf = SimpleNamespace(is_muon=True, param_groups=[{"params": [model.weight], **settings}])
        state = {f"param_groups.weight.{key}": value for key, value in settings.items()}
        return HyperOptimizerAdapter(model, {"muon": leaf}), state

    def test_checkpoint_rejects_changed_core_options_before_restore(self):
        """Feature: Checkpoint algorithm consistency.

        Description: Load checkpoint metadata with another momentum or backend selection.
        Expectation: Each mismatch fails before optimizer state is restored.
        """
        adapter, state = self._checkpoint()
        for name, value in (
            ("momentum_mode", "legacy"), ("use_gram", False), ("ns_backend", "cann"),
            ("ns_backend", "ascendc"), ("gram_reset_iterations", (1,)),
        ):
            with self.subTest(name=name), patch(
                "hyper_parallel.integration.veomni.optimizer.set_optim_state_dict"
            ) as restore, self.assertRaisesRegex(ValueError, name):
                adapter.load_state_dict({**state, f"param_groups.weight.{name}": value})
            restore.assert_not_called()

    def test_missing_and_partial_core_metadata_require_confirmation(self):
        """Feature: Missing checkpoint provenance.

        Description: Load absent or partially loaded core configuration metadata.
        Expectation: Explicit historical confirmation is required and reaches the restore payload.
        """
        adapter, state = self._checkpoint()
        markers = adapter.dcp_load_template(state)
        for name in ("momentum_mode", "use_gram", "ns_backend"):
            key = f"param_groups.weight.{name}"
            for marked in (False, True):
                incomplete = dict(state)
                if marked:
                    incomplete[key] = markers[key]
                else:
                    incomplete.pop(key)
                with self.subTest(name=name, marked=marked), self.assertRaisesRegex(ValueError, name):
                    adapter.load_state_dict(incomplete)
        adapter.confirm_legacy_muon_checkpoint_options(
            momentum_mode="vanilla", use_gram=True, ns_backend="auto"
        )
        for name in ("momentum_mode", "use_gram", "ns_backend"):
            state[f"param_groups.weight.{name}"] = markers[f"param_groups.weight.{name}"]
        with patch("hyper_parallel.integration.veomni.optimizer.set_optim_state_dict") as restore:
            adapter.load_state_dict(state)
        group = restore.call_args.args[2]["param_groups"][0]
        self.assertEqual(group["momentum_mode"], "vanilla")
        self.assertTrue(group["use_gram"])
        self.assertEqual(group["ns_backend"], "auto")
