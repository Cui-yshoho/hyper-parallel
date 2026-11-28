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
"""Construct datasets using the upstream embodied data readers."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from hyper_parallel.data.embodied.trajectory import TrajectoryWindow
from hyper_parallel.data.online.mapping import MappingTransformDataset


def build_lerobot_dataset(
    dataset_root: str, dataset_repo: str, frame_offsets: dict[str, list[int]] | None = None,
    video_backend: str = "pyav", *, episodes: list[int] | None = None, split: str | None = None,
    use_imagenet_stats: bool = False, transform: TrajectoryWindow | None = None,
) -> Dataset | MappingTransformDataset:
    """Read local LeRobot episodes with per-field windows in source frames.

    Args:
        dataset_root: Local LeRobot dataset directory.
        dataset_repo: Dataset repository identifier.
        frame_offsets: Frame offsets relative to each sample's anchor, keyed by dataset field.
        video_backend: LeRobot video decoder.
        episodes: Optional episode selection.
        split: Optional split defined by LeRobot dataset metadata.
        use_imagenet_stats: Expose ImageNet RGB stats in dataset metadata for downstream normalization.
        transform: Optional trajectory sample transform configured by the Trainer.
    """
    dataset_path = Path(dataset_root).expanduser().resolve()
    info_file = dataset_path / "meta/info.json"
    if not info_file.is_file():
        raise FileNotFoundError(f"Required local LeRobot dataset metadata is missing: {info_file}")
    # Only LeRobot-backed datasets require this optional dependency.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # pylint: disable=import-outside-toplevel
    from lerobot.utils.constants import IMAGENET_STATS  # pylint: disable=import-outside-toplevel

    if transform is not None:
        frame_offsets = {**transform.frame_offsets, **(frame_offsets or {})}
    if frame_offsets or split is not None:
        with info_file.open(encoding="utf-8") as stream:
            info = json.load(stream)
    if split is not None:
        if split not in info.get("splits", {}):
            raise ValueError(f"Unknown LeRobot split {split!r} in {info_file}")
        start, stop = map(int, info["splits"][split].split(":"))
        selected = list(range(start, stop)) if episodes is None else [ep for ep in episodes if start <= ep < stop]
        exclusion_file = dataset_path / "meta/excluded_episodes.json"
        if exclusion_file.is_file():
            with exclusion_file.open(encoding="utf-8") as stream:
                excluded = set(json.load(stream)["episode_indices"])
            selected = [ep for ep in selected if ep not in excluded]
        if not selected:
            raise ValueError(f"LeRobot split {split!r} contains no selected episodes")
        # None retains LeRobot's optimized all-episodes loading path.
        episodes = None if selected == list(range(info["total_episodes"])) else selected

    delta_timestamps = None
    if frame_offsets:
        unknown_fields = frame_offsets.keys() - info["features"].keys()
        if unknown_fields:
            raise ValueError(f"LeRobot frame offsets reference missing fields: {sorted(unknown_fields)}")
        empty_windows = [field for field, offsets in frame_offsets.items() if not offsets]
        if empty_windows:
            raise ValueError(f"LeRobot frame offsets must be non-empty: {sorted(empty_windows)}")
        fps = info["fps"]
        delta_timestamps = {field: [offset / fps for offset in offsets] for field, offsets in frame_offsets.items()}

    sample_transform = transform.bind(dataset_path) if transform is not None else None
    dataset = LeRobotDataset(
        dataset_repo, root=dataset_path, episodes=episodes, delta_timestamps=delta_timestamps,
        video_backend=video_backend, return_uint8=True,
    )
    if use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            if key not in dataset.meta.depth_keys:
                dataset.meta.stats.setdefault(key, {}).update(
                    {name: torch.tensor(value) for name, value in IMAGENET_STATS.items()}
                )
    if sample_transform is not None:
        return MappingTransformDataset(dataset, sample_transform)
    return dataset


def build_openwam_dataset(dataset_root: str, dataset_config: str) -> Dataset:
    """Keep the upstream OpenWAM reader config and replace its storage path.

    Args:
        dataset_root: Local directory for the configured dataset.
        dataset_config: Path to the upstream dataset YAML.
    """
    # Only OpenWAM-backed datasets require these optional dependencies.
    from omegaconf import OmegaConf  # pylint: disable=import-outside-toplevel
    from openwam.dataloader import build_dataset  # pylint: disable=import-outside-toplevel

    config = OmegaConf.load(dataset_config)
    config.dataset_dir = dataset_root
    return build_dataset(config)
