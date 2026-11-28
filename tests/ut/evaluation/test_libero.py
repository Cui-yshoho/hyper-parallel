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
"""Regression tests for the shared LIBERO episode loop with a synthetic simulator."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from hyper_parallel.evaluation.libero import rollout
from tests.common.mark_utils import arg_mark


class TestLiberoRollout(unittest.TestCase):
    """Verify the shared policy contract and episode termination."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_rollout_resets_and_stops_on_terminal_success(self):
        """Reset per episode and feed each new observation back to the policy."""
        observations = [{"step": index} for index in range(3)]
        action = np.zeros(7, dtype=np.float32)
        policy = SimpleNamespace(reset=Mock(), act=Mock(return_value=action))
        env = SimpleNamespace(
            task_description="pick up the bowl",
            reset=Mock(return_value=(observations[0], {})),
            step=Mock(side_effect=[
                (observations[1], 0.0, False, False, {}),
                (observations[2], 1.0, True, False, {"is_success": True}),
            ]),
        )

        record = rollout(env, policy, seed=42, max_steps=8)

        policy.reset.assert_called_once_with()
        env.reset.assert_called_once_with(seed=42)
        self.assertEqual([call.args[0] for call in policy.act.call_args_list], observations[:2])
        self.assertEqual([call.args[1] for call in policy.act.call_args_list], ["pick up the bowl"] * 2)
        self.assertEqual(record, {"seed": 42, "success": True, "steps": 2, "sum_reward": 1.0, "max_reward": 1.0})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_timeout_is_not_a_success(self):
        """Treat a rollout exhausted at the benchmark step limit as unsuccessful."""
        policy = SimpleNamespace(reset=Mock(), act=Mock(return_value=np.zeros(7)))
        env = SimpleNamespace(
            task_description="pick up the bowl", reset=Mock(return_value=({}, {})),
            step=Mock(return_value=({}, -0.1, False, False, {})),
        )

        record = rollout(env, policy, seed=0, max_steps=3)

        self.assertFalse(record["success"])
        self.assertEqual(record["steps"], 3)
        self.assertAlmostEqual(record["sum_reward"], -0.3)
        self.assertEqual(env.step.call_count, 3)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_rollout_rejects_invalid_action_before_stepping(self):
        """Keep a malformed controller action out of the simulator."""
        policy = SimpleNamespace(reset=Mock(), act=Mock(return_value=np.asarray([0.0] * 6 + [np.nan])))
        env = SimpleNamespace(task_description="move", reset=Mock(return_value=({}, {})), step=Mock())
        with self.assertRaisesRegex(ValueError, "finite 7-D action"):
            rollout(env, policy, seed=0, max_steps=1)
        env.step.assert_not_called()
