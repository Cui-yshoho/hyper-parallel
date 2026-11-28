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
"""Build model-ready trajectory windows from LeRobot-format episode samples."""

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


class TrajectoryWindow:
    """Compose synchronized camera frames and project action and state into fixed slots."""

    def __init__(
        self, *, action_steps: int, video_stride: int, camera_layout: list[str | None],
        height: int, width: int, output_dim: int, action_indices: list[int],
        stats_file: str, action_stats_key: str, state_stats_key: str,
        unchanged_dimensions: list[int], state_indices: list[int] | None = None,
        representation: str | None = None, gripper_convention: str | None = None,
    ) -> None:
        """Configure the window, image canvas, and separate action/state statistics."""
        if action_steps < 1 or video_stride < 1:
            raise ValueError("Trajectory action_steps and video_stride must be positive")
        if len(camera_layout) != 3 or camera_layout[0] is None or min(height, width) < 3:
            raise ValueError("Trajectory camera layout requires a top camera and two optional lower slots")
        if state_indices is None:
            state_indices = action_indices
        for indices in (action_indices, state_indices):
            if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= output_dim:
                raise ValueError("Trajectory projection indices must be unique and inside output_dim")
        if unchanged_dimensions and (
            min(unchanged_dimensions) < 0 or max(unchanged_dimensions) >= min(len(action_indices), len(state_indices))
        ):
            raise ValueError("Unchanged dimensions must be inside the source action and state vectors")

        video_offsets = list(range(0, action_steps + 1, video_stride))
        self.frame_offsets = {
            "action": list(range(action_steps)),
            **{camera: video_offsets for camera in camera_layout if camera is not None},
        }
        self.camera_layout = tuple(camera_layout)
        self.width, self.height = width, height
        self.output_dim = output_dim
        self.action_indices = action_indices
        self.state_indices = state_indices
        self.stats_file = stats_file
        self.action_stats_key = action_stats_key
        self.state_stats_key = state_stats_key
        self.unchanged_dimensions = unchanged_dimensions
        self.representation = representation
        self.gripper_convention = gripper_convention

    def bind(self, dataset_root: Path) -> Callable[[dict[str, Any]], list[dict[str, Any]]]:
        """Read the local statistics once before loading any sample windows."""
        if self.representation is not None:
            conversion = dataset_root / "meta/conversion.json"
            with conversion.open(encoding="utf-8") as stream:
                if json.load(stream).get("output_representation") != self.representation:
                    raise ValueError(f"{conversion} must declare {self.representation!r} as the action representation")
        path = dataset_root / self.stats_file
        # The LeRobot export stores these named statistic blocks in a local NumPy object file.
        payload = np.load(path, allow_pickle=True).item()
        action, state = payload[self.action_stats_key], payload[self.state_stats_key]
        for block in (action, state):
            if self.representation is not None and block.get("representation", self.representation) != self.representation:
                raise ValueError(f"{path} declares incompatible normalization statistics")
            if self.gripper_convention is not None and block.get("gripper_convention") != self.gripper_convention:
                raise ValueError(f"{path} must declare gripper convention {self.gripper_convention!r}")
        action_stats = self._load_stats(action, len(self.action_indices))
        state_stats = self._load_stats(state, len(self.state_indices))
        return partial(self._transform, action_stats=action_stats, state_stats=state_stats)

    @staticmethod
    def _load_stats(block: dict[str, Any], width: int) -> tuple[np.ndarray, np.ndarray]:
        lower = np.asarray(block["min"], dtype=np.float32)
        upper = np.asarray(block["max"], dtype=np.float32)
        if lower.shape != (width,) or upper.shape != (width,):
            raise ValueError(f"Trajectory normalization stats must have shape ({width},)")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or (upper < lower).any():
            raise ValueError("Trajectory normalization stats must contain ordered finite bounds")
        return lower, np.maximum(upper - lower, 1e-6)

    def _project(
        self, values: torch.Tensor, valid: torch.Tensor, indices: list[int],
        stats: tuple[np.ndarray, np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = np.asarray(values, dtype=np.float32)
        if raw.shape[-1] != len(indices) or not np.isfinite(raw).all():
            raise ValueError("Invalid trajectory action or state vector")
        lower, scale = stats
        normalized = np.clip(((raw - lower) / scale) * 2.0 - 1.0, -1.0, 1.0)
        normalized[..., self.unchanged_dimensions] = raw[..., self.unchanged_dimensions]
        result = torch.zeros((len(values), self.output_dim), dtype=torch.float32)
        mask = torch.zeros_like(result, dtype=torch.bool)
        result[:, indices] = torch.from_numpy(normalized).masked_fill(~valid[:, None], 0)
        mask[:, indices] = valid[:, None]
        return result, mask

    def _compose_video(self, sample: dict[str, Any]) -> tuple[list[Image.Image], torch.Tensor]:
        top_height = round(self.height * 2 / 3)
        left_width = self.width // 2
        slots = (
            (0, 0, self.width, top_height),
            (0, top_height, left_width, self.height - top_height),
            (left_width, top_height, self.width - left_width, self.height - top_height),
        )
        cameras = []
        for name, (x, y, width, height) in zip(self.camera_layout, slots):
            if name is None:
                continue
            valid = ~sample[f"{name}_is_pad"]
            if not valid[0]:
                raise ValueError(f"Trajectory camera {name!r} has no frame at the anchor")
            # Repeat the last sampled frame. A reader can pad with the last episode frame,
            # which may fall between the requested video timestamps.
            cameras.append((
                sample[name], valid, (x, y), (width, height), int(valid.sum()) - 1,
            ))
        video = []
        for index in range(len(cameras[0][1])):
            frame = Image.new("RGB", (self.width, self.height))
            for frames, _, xy, size, last in cameras:
                image = Image.fromarray(frames[min(index, last)].permute(1, 2, 0).numpy())
                frame.paste(image.resize(size, resample=Image.Resampling.LANCZOS), xy)
            video.append(frame)
        return video, cameras[0][1]

    def _transform(
        self, sample: dict[str, Any], *, action_stats: tuple[np.ndarray, np.ndarray],
        state_stats: tuple[np.ndarray, np.ndarray],
    ) -> list[dict[str, Any]]:
        action, action_mask = self._project(
            sample["action"], ~sample["action_is_pad"], self.action_indices, action_stats,
        )
        proprio, proprio_mask = self._project(
            sample["observation.state"].unsqueeze(0), torch.ones(1, dtype=torch.bool), self.state_indices, state_stats,
        )
        video, video_mask = self._compose_video(sample)
        prompt = sample["task"].strip()
        if not prompt:
            raise ValueError("Trajectory task text is empty")
        return [{
            "video": video,
            "vace_video": None,
            "first_frame_image": [video[0]],
            "action": action,
            "action_mask": action_mask,
            "video_mask": video_mask,
            "proprio": proprio,
            "proprio_mask": proprio_mask,
            "prompt": prompt,
        }]
