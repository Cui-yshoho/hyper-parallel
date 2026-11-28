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
"""Regression coverage for padded robot trajectory windows."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from hyper_parallel.data.embodied.trajectory import TrajectoryWindow
from tests.common.mark_utils import arg_mark


class TestTrajectoryWindow(unittest.TestCase):
    """Verify that frame padding and projected supervision keep the same anchor."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_tail_repeats_last_sampled_frame_and_masks_unavailable_actions(self):
        """Keep the last sampled video frame and mask absent action steps."""
        transform = TrajectoryWindow(
            action_steps=3, video_stride=1, camera_layout=["main", None, None],
            height=6, width=4, output_dim=4, action_indices=[1, 3],
            stats_file="stats.npy", action_stats_key="action", state_stats_key="state", unchanged_dimensions=[1],
        )
        stats = {"min": [-1, -1], "max": [1, 1]}
        raw_sample = {
            "main": torch.stack([torch.full((3, 2, 2), value, dtype=torch.uint8) for value in (10, 20, 99, 99)]),
            "main_is_pad": torch.tensor([False, False, True, True]),
            "action": torch.tensor([[0.0, 3.0], [0.5, 4.0], [1.0, 5.0]]),
            "action_is_pad": torch.tensor([False, False, True]),
            "observation.state": torch.tensor([0.5, 2.0]),
            "task": " test task ",
        }
        with tempfile.TemporaryDirectory() as root:
            np.save(Path(root) / "stats.npy", {"action": stats, "state": stats})
            sample, = transform.bind(Path(root))(raw_sample)
        self.assertEqual([image.getpixel((0, 0))[0] for image in sample["video"]], [10, 20, 20, 20])
        self.assertEqual(sample["prompt"], "test task")
        torch.testing.assert_close(sample["video_mask"], torch.tensor([True, True, False, False]))
        torch.testing.assert_close(
            sample["action"], torch.tensor([[0.0, 0.0, 0.0, 3.0], [0.0, 0.5, 0.0, 4.0], [0.0] * 4]),
        )
        torch.testing.assert_close(
            sample["action_mask"], torch.tensor([[False, True, False, True]] * 2 + [[False] * 4]),
        )
        torch.testing.assert_close(sample["proprio"], torch.tensor([[0.0, 0.5, 0.0, 2.0]]))
