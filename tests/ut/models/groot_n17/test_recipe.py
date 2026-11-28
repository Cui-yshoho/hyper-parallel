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
"""Tests for the GR00T N1.7 training adapter."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

import torch
from torch import nn

from hyper_parallel.components.losses.model_output import MaskedModelOutputLoss
from hyper_parallel.models.groot_n17.modeling import GrootBatch, GrootTrainingModel, GrootTrainingOutput
from scripts.smoke_groot_n17 import load_local_training_batch

from tests.common.mark_utils import arg_mark


class _ActionModel(nn.Module):
    """Minimal module implementing the native action-loss return contract."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute a masked synthetic objective."""
        mask = inputs["action_mask"]
        loss = (self.weight - inputs["action"]).square() * mask
        return {"loss": loss.sum() / (mask.sum() + 1e-6), "action_loss": loss, "action_mask": mask}


class _Policy(nn.Module):  # pylint: disable=too-few-public-methods
    """Stand in for the optional LeRobot policy in CPU tests."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_bf16=False)
        self._groot_model = _ActionModel()

    @staticmethod
    def _filter_groot_inputs(batch: dict[str, torch.Tensor], *, include_action: bool) -> dict[str, torch.Tensor]:
        keys = ("action", "action_mask") if include_action else ("action_mask",)
        return {key: batch[key] for key in keys}


class TestGrootRecipe(TestCase):
    """Exercise the adapter without LeRobot, Hugging Face assets or accelerators."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_batch_runtime_exposes_processed_action_mask(self) -> None:
        """Feature: VLM batch runtime for GR00T.

        Description: Apply the processor to a collated sample list.
        Expectation: The model and loss receive the same processed mask.
        """
        def processor(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {**batch, "action_mask": batch["action"].ne(0)}

        runtime = GrootBatch(processor=processor, device=torch.device("cpu"))
        samples = [{"action": torch.tensor([1, 0])}, {"action": torch.tensor([0, 2])}]
        model_inputs, loss_inputs = runtime(iter([{"samples": samples, "sample_count": 2}]))
        self.assertIs(model_inputs["action_mask"], loss_inputs["action_mask"])
        torch.testing.assert_close(loss_inputs["action_mask"], torch.tensor([[True, False], [False, True]]))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_native_loss_and_module_hooks(self) -> None:
        """Feature: GR00T model adapter.

        Description: Forward a batch with padded action elements through the native model.
        Expectation: The native hook runs and the masked numerator remains differentiable.
        """
        policy = _Policy()
        model = GrootTrainingModel(policy)
        observed_inputs = []
        action_model = policy._groot_model  # pylint: disable=protected-access
        with action_model.register_forward_pre_hook(lambda _, inputs: observed_inputs.append(inputs[0])):
            result = model({
                "action": torch.tensor([1.0, 12.0, 4.0]),
                "action_mask": torch.tensor([1.0, 0.0, 1.0]),
                "index": torch.tensor(0),
            })
        result.action_loss_sum.backward()

        self.assertEqual(len(observed_inputs), 1, "The inner module's forward hook must run once")
        self.assertNotIn("index", observed_inputs[0], "The native filter must strip dataset metadata")
        torch.testing.assert_close(result.loss, torch.tensor(5.0 / (2 + 1e-6)))
        torch.testing.assert_close(result.action_loss_sum, torch.tensor(5.0))
        torch.testing.assert_close(result.valid_action_count, torch.tensor(2))
        torch.testing.assert_close(action_model.weight.grad, torch.tensor(-2.0))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_missing_local_checkpoint_fails_before_loading_dependencies(self) -> None:
        """Feature: Offline GR00T setup.

        Description: Try to construct a batch without the local checkpoint.
        Expectation: The loader reports the missing config file before loading external dependencies.
        """
        with TemporaryDirectory() as root, self.assertRaisesRegex(FileNotFoundError, "config.json"):
            load_local_training_batch(
                checkpoint=Path(root) / "checkpoint",
                dataset_root=Path(root) / "data",
                dataset_repo="local/smoke",
                embodiment_tag="libero_sim",
                batch_size=1,
                video_backend="pyav",
            )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_action_loss_counts_coordinates_and_keeps_empty_batches_finite(self) -> None:
        """Feature: Masked loss.

        Description: Count the supervised action coordinates on populated and empty masks.
        Expectation: The mask count matches supervision and the native mean remains finite.
        """
        objective = MaskedModelOutputLoss(mask_field="action_mask", loss_family="action")
        for mask, expected_count in (([[1, 1], [0, 1]], 3), ([[0, 0], [0, 0]], 0)):
            with self.subTest(valid_actions=expected_count):
                counts = objective.count_loss_units({"action_mask": torch.tensor(mask)})
                count = counts["action_tokens"]
                torch.testing.assert_close(count, torch.tensor(expected_count))
                numerator = count.float() * 2
                expected_loss = torch.tensor(2.0 if expected_count else 0.0)
                result = objective(GrootTrainingOutput(expected_loss, numerator, count))
                torch.testing.assert_close(result["action_loss"], expected_loss)
