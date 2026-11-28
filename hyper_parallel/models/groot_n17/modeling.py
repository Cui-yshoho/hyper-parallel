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
"""Native LeRobot policy, processor and model build hooks for GR00T N1.7."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

import torch
from torch import distributed as dist, nn
from torch.utils.data import default_collate

from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models.build_options import get_device_id, get_device_type

if TYPE_CHECKING:
    from lerobot.policies.groot.modeling_groot import GrootPolicy


class GrootTrainingOutput(NamedTuple):
    """Native mean loss and its numerator/count for trainer-side aggregation."""

    loss: torch.Tensor
    action_loss_sum: torch.Tensor
    valid_action_count: torch.Tensor


class GrootBatch:
    """Build VLM model and loss inputs with the LeRobot processor."""

    def __init__(self, processor: Any, device: torch.device, pp_shared_data: bool = False) -> None:
        """Bind the Trainer's LeRobot processor and execution device."""
        if processor is None:
            raise ValueError("GR00T requires a LeRobot processor for raw training samples")
        if pp_shared_data:
            raise NotImplementedError("GR00T does not support pipeline-parallel shared data")
        self.processor = processor
        self.device = device

    def __call__(self, data_iterator: Iterator[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        """Process the current collated sample list and expose its action mask."""
        samples = next(data_iterator)["samples"]
        inputs = {
            key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in self.processor(default_collate(samples)).items()
        }
        return inputs, {"action_mask": inputs["action_mask"]}


class GrootTrainingModel(nn.Module):
    """Expose the native GR00T loss and enter the inner model through module hooks."""

    def __init__(self, policy: "GrootPolicy") -> None:
        """Keep the upstream policy as the owner of model parameters."""
        super().__init__()
        self.policy = policy

    @property
    def config(self) -> Any:
        """Expose the native Transformers model configuration."""
        return self.policy._groot_model.config  # pylint: disable=protected-access

    def forward(
            self, batch: Optional[dict[str, Any]] = None, *, use_cache: bool = False, **fields: Any,
    ) -> GrootTrainingOutput:
        """Forward one preprocessed LeRobot batch using the upstream objective.

        Args:
            batch: Preprocessed inputs supplied as one mapping.
            use_cache: Decoder cache flag supplied by the Trainer.
            **fields: Preprocessed inputs supplied as named arguments.
        """
        if use_cache:
            raise ValueError("GR00T action training does not support a decoding cache")
        # The pinned LeRobot policy keeps its input filtering and model private.
        # Reuse both, but call the inner module so its sharding hooks can run.
        if batch is not None and fields:
            raise ValueError("Pass GR00T inputs as either a batch or named fields")
        inputs = self.policy._filter_groot_inputs(  # pylint: disable=protected-access
            fields if batch is None else batch, include_action=True,
        )
        device_type = next(self.policy.parameters()).device.type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=self.policy.config.use_bf16):
            outputs = self.policy._groot_model(inputs)  # pylint: disable=protected-access
        return GrootTrainingOutput(
            loss=outputs["loss"],
            action_loss_sum=outputs["action_loss"].sum(dtype=torch.float32),
            valid_action_count=outputs["action_mask"].count_nonzero(),
        )

    def build_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimizer using the LeRobot policy preset and parameter groups."""
        return self.policy.config.get_optimizer_preset().build(self.policy.get_optim_params())


def require_local_file(root: Path, filename: str) -> None:
    """Require a local checkpoint or dataset asset before constructing it.

    Args:
        root: Root directory for the asset.
        filename: Required relative filename.
    """
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required local GR00T recipe asset is missing: {path}")


def build_groot_policy_config(checkpoint: str | Path, embodiment_tag: str) -> Any:
    """Build the LeRobot policy configuration from a local N1.7 checkpoint.

    Args:
        checkpoint: Local GR00T checkpoint directory.
        embodiment_tag: GR00T embodiment identifier.
    """
    checkpoint = Path(checkpoint).expanduser().resolve()
    require_local_file(checkpoint, "config.json")
    # LeRobot is an optional training dependency.
    from huggingface_hub.constants import HF_HUB_OFFLINE  # pylint: disable=import-outside-toplevel
    from lerobot.policies.groot.configuration_groot import (  # pylint: disable=import-outside-toplevel
        GrootConfig,
        is_raw_groot_n1_7_checkpoint,
    )

    if not HF_HUB_OFFLINE:
        raise ValueError("Set HF_HUB_OFFLINE=1 before launching the GR00T recipe")
    require_local_file(checkpoint, "model.safetensors.index.json")
    if not is_raw_groot_n1_7_checkpoint(str(checkpoint)):
        raise ValueError(f"Expected the raw GR00T N1.7 checkpoint at {checkpoint}")
    return GrootConfig(
        device="cpu",
        pretrained_path=checkpoint,
        base_model_path=str(checkpoint),
        embodiment_tag=embodiment_tag,
        use_flash_attention=False,
        optimizer_fused=False,
    )


def build_groot_frame_offsets(checkpoint: str | Path, embodiment_tag: str) -> dict[str, list[int]]:
    """Return the action window requested by the GR00T checkpoint and embodiment."""
    config = build_groot_policy_config(checkpoint, embodiment_tag)
    return {"action": config.action_delta_indices}


def build_groot_policy(checkpoint: Path, embodiment_tag: str, dataset_meta: Any) -> "GrootPolicy":
    """Build the native LeRobot policy and resolve dataset image feature shapes.

    Args:
        checkpoint: Local GR00T checkpoint directory.
        embodiment_tag: GR00T embodiment identifier.
        dataset_meta: LeRobot metadata for the training dataset.
    """
    # LeRobot is an optional training dependency.
    from lerobot.policies.factory import make_policy  # pylint: disable=import-outside-toplevel

    policy = make_policy(build_groot_policy_config(checkpoint, embodiment_tag), ds_meta=dataset_meta)
    for name, feature in policy.config.image_features.items():
        source = dataset_meta.features.get(name, {})
        # LeRobot already handles "channel(s)". This LIBERO dataset also
        # declares HWC videos with an "rgb" axis but returns CHW tensors.
        if tuple(source.get("names", ())) == ("height", "width", "rgb"):
            height, width, channels = source["shape"]
            policy.config.input_features[name] = replace(feature, shape=(channels, height, width))
    return policy


def load_local_groot_model(
        checkpoint: str, dataset_root: str, dataset_repo: str, embodiment_tag: str = "libero_sim",
) -> GrootTrainingModel:
    """Build the original GR00T policy on CPU using local LeRobot dataset metadata.

    Args:
        checkpoint: Local GR00T checkpoint directory.
        dataset_root: Local LeRobot dataset directory.
        dataset_repo: Dataset repository identifier.
        embodiment_tag: GR00T embodiment identifier.
    """
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    dataset_path = Path(dataset_root).expanduser().resolve()
    require_local_file(checkpoint_path, "config.json")
    require_local_file(dataset_path, "meta/info.json")
    # Optional LeRobot dataset import.
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # pylint: disable=import-outside-toplevel

    metadata = LeRobotDatasetMetadata(dataset_repo, root=dataset_path)
    return GrootTrainingModel(build_groot_policy(checkpoint_path, embodiment_tag, metadata))


def build_groot_model(
        checkpoint: str, dataset_root: str, dataset_repo: str, embodiment_tag: str = "libero_sim",
        model_init_dtype: str = "float32",
        distributed_setup: Optional[DistributedSetup] = None,
) -> GrootTrainingModel:
    """Load the native FP32 policy and apply the configured data-parallel sharding.

    Args:
        checkpoint: Local GR00T checkpoint directory.
        dataset_root: Local LeRobot dataset directory.
        dataset_repo: Dataset repository identifier.
        embodiment_tag: GR00T embodiment identifier.
        model_init_dtype: Initializer parameter dtype.
        distributed_setup: Trainer mesh and FSDP options.
    """
    if not dist.is_initialized():
        raise ValueError("Building this GR00T recipe requires an initialized training process group")
    if model_init_dtype != "float32":
        raise ValueError("The GR00T recipe initializes its model parameters in FP32")
    world_size = dist.get_world_size()
    if distributed_setup is None:
        raise ValueError("The GR00T recipe requires the Trainer's distributed setup")
    mesh = distributed_setup.mesh_context
    if mesh.dp_size != world_size or mesh.dp_shard_size != world_size:
        raise ValueError("GR00T requires a pure data-parallel mesh with dp_shard_size equal to world size")
    model = load_local_groot_model(checkpoint, dataset_root, dataset_repo, embodiment_tag)
    model.to(torch.device(get_device_type(), get_device_id())).train()
    if world_size > 1:
        strategy = distributed_setup.strategy_config
        FSDP2Manager(
            config=strategy, mesh_context=mesh, fp32_main_params=distributed_setup.fp32_main_params,
        ).parallelize(model)
    return model


def build_groot_processors(model: GrootTrainingModel, processor_dir: Optional[str] = None) -> tuple[Any, Any]:
    """Build the LeRobot preprocessor and action decoder from the dataset statistics.

    Args:
        model: Loaded policy with its dataset metadata.
        processor_dir: Optional local processor asset directory.
    """
    # Optional LeRobot processor import.
    from lerobot.policies.factory import make_pre_post_processors  # pylint: disable=import-outside-toplevel

    config = model.policy.config
    # make_policy attaches the same upstream metadata used to resolve policy features.
    dataset_meta = config._runtime_dataset_meta  # pylint: disable=protected-access
    overrides = None
    if processor_dir is not None:
        processor_path = Path(processor_dir).expanduser().resolve()
        if not processor_path.is_dir():
            raise FileNotFoundError(f"Local GR00T processor directory is missing: {processor_path}")
        overrides = {"groot_n1_7_vlm_encode_v1": {"model_name": str(processor_path)}}
    return make_pre_post_processors(
        policy_cfg=config, pretrained_path=config.base_model_path,
        dataset_stats=dataset_meta.stats, dataset_meta=dataset_meta, preprocessor_overrides=overrides,
    )


def build_groot_processor(model: GrootTrainingModel, processor_dir: Optional[str] = None) -> Any:
    """Build the input processor used for training.

    Args:
        model: Loaded policy with its dataset metadata.
        processor_dir: Optional local processor asset directory.
    """
    preprocessor, _ = build_groot_processors(model, processor_dir)
    return preprocessor
