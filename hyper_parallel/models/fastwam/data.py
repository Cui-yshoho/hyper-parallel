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
"""Prepare FastWAM inputs from the shared LeRobot reader."""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import default_collate


class FastWAMTrajectory:
    """Apply the native processor to a LIBERO trajectory window."""

    def __init__(self, data_config: str, stats_file: str) -> None:
        """Load the native two-camera LIBERO configuration."""
        config = OmegaConf.create({"data": OmegaConf.load(data_config)}).data.train
        if config.concat_multi_camera != "horizontal" or list(config.video_size) != [224, 448]:
            raise ValueError("The FastWAM LIBERO recipe requires a horizontal 224 x 448 video")
        if config.skip_padding_as_possible:
            raise ValueError("The FastWAM LIBERO recipe preserves the original sample index and padding")
        frames = int(config.num_frames)
        stride = int(config.global_sample_stride)
        ratio = int(config.action_video_freq_ratio)
        self.camera_names = tuple(camera.key for camera in config.shape_meta.images)
        if frames < 2 or stride < 1 or ratio < 1:
            raise ValueError("FastWAM requires at least two source frames and positive sampling strides")
        if len(self.camera_names) != 2 or (frames - 1) % (4 * ratio):
            raise ValueError("FastWAM LIBERO requires two cameras and aligned video and action windows")
        video_offsets = list(range(0, frames * stride, ratio * stride))
        self.frame_offsets = {
            "action": list(range(0, (frames - 1) * stride, stride)),
            "observation.state": list(range(0, frames * stride, stride)),
            **{f"observation.images.{name}": video_offsets for name in self.camera_names},
        }
        self.processor = instantiate(config.processor, num_image_steps=len(video_offsets)).train()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(stats_file))

    def bind(self, dataset_root: Path) -> Callable[[dict[str, Any]], list[dict[str, Any]]]:
        """Return the mapping consumed by the shared local reader."""
        del dataset_root
        return self

    def __call__(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        """Build a native FastWAM sample with the original padding flags."""
        first_camera = f"observation.images.{self.camera_names[0]}"
        inputs = self.processor.preprocess({
            "task": sample["task"],
            "idx": int(sample["index"]),
            "images": {name: sample[f"observation.images.{name}"] for name in self.camera_names},
            "action": {"default": sample["action"].clone()},
            "state": {"default": sample["observation.state"]},
            "action_is_pad": sample["action_is_pad"],
            "state_is_pad": sample["observation.state_is_pad"],
            "image_is_pad": sample[f"{first_camera}_is_pad"],
        })
        frames = torch.cat(tuple(inputs["pixel_values"].unbind(0)), dim=-1)
        return [{
            "video": frames.permute(1, 0, 2, 3).mul(2).sub(1),
            "action": inputs["action"],
            "proprio": inputs["proprio"][:-1],
            "prompt": DEFAULT_PROMPT.format(task=inputs["instruction"]),
            "action_is_pad": inputs["action_is_pad"],
            "image_is_pad": inputs["image_is_pad"],
        }]


class FastWAMBatch:
    """Collate reader samples into the native forward's sample mapping."""

    def __call__(self, data_iterator: Iterator[dict[str, Any]]) -> dict[str, Any]:
        """Stack the trajectory fields consumed by the native policy."""
        batch = next(data_iterator)
        return {"sample": default_collate(batch["samples"]), "sample_count": batch["sample_count"]}
