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
"""Native LeRobot π0.5 checkpoint loading and HyperParallel training targets."""

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple

import draccus
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_IMAGES
from safetensors.torch import load_file
from torch import distributed as dist, nn
from torch.utils.data import default_collate
from transformers import PretrainedConfig

from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models.build_options import get_device_id, get_device_type


class Pi05TrainingOutput(NamedTuple):
    """Expose the native action objective to the Trainer."""

    loss: torch.Tensor


def load_pi05_config(checkpoint: Path, overrides: tuple[str, ...] = ()) -> PI05Config:
    """Read the local policy configuration with the upstream parser."""
    return PI05Config.from_pretrained(
        checkpoint, local_files_only=True,
        cli_overrides=["--device=cpu", "--compile_model=false", *overrides],
    )


def load_pi05_weights(checkpoint: Path, config: PI05Config) -> PI05Policy:
    """Restore native π0.5 weights, failing on a missing or incompatible checkpoint."""
    weights = checkpoint / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    # Match the native loader's checkpoint mapping while propagating strict load failures.
    policy = PI05Policy(config)
    state = policy._convert_openpi_state_dict(load_file(weights))  # pylint: disable=protected-access
    policy.load_state_dict(state, strict=True)
    return policy


def build_pi05_frame_offsets(checkpoint: str) -> dict[str, list[int]]:
    """Read the action horizon from the policy config in units of source frames."""
    config = load_pi05_config(Path(checkpoint))
    return {ACTION: config.action_delta_indices}


class Pi05TrainingModel(nn.Module):
    """Present the LeRobot policy through the standard Trainer forward contract."""

    def __init__(self, policy: PI05Policy) -> None:
        """Keep the native policy as the owner of trainable parameters."""
        super().__init__()
        self.policy = policy
        self.config = PretrainedConfig(pi05=draccus.encode(policy.config))

    def forward(self, batch: dict[str, Any], *, use_cache: bool = False) -> Pi05TrainingOutput:
        """Return the native LeRobot flow-matching objective."""
        if use_cache:
            raise ValueError("π0.5 training does not support a decoding cache")
        loss, _ = self.policy(batch)
        return Pi05TrainingOutput(loss)


def build_pi05_model(
    checkpoint: str, distributed_setup: DistributedSetup, model_init_dtype: str = "float32",
    train_expert_only: bool = True,
) -> Pi05TrainingModel:
    """Load the FP32 base policy and apply data-parallel sharding."""
    if not dist.is_initialized():
        raise ValueError("The π0.5 training recipe requires an initialized process group")
    world_size = dist.get_world_size()
    mesh = distributed_setup.mesh_context
    if mesh.dp_size != world_size or mesh.dp_shard_size != world_size:
        raise ValueError("π0.5 requires a pure data-parallel mesh with dp_shard_size equal to world size")
    if model_init_dtype != "float32":
        raise ValueError("The π0.5 training recipe requires model_init_dtype=float32")
    directory = Path(checkpoint)
    config = load_pi05_config(directory, (
        "--dtype=float32", "--gradient_checkpointing=false",
        f"--train_expert_only={str(train_expert_only).lower()}",
    ))
    # Keep loading-time parameter initialization out of the training RNG stream.
    with torch.random.fork_rng(devices=[]):
        policy = load_pi05_weights(directory, config)
    model = Pi05TrainingModel(policy).to(torch.device(get_device_type(), get_device_id())).train()
    if world_size > 1:
        # The paired Gemma streams invoke layer internals directly. Keep them within
        # a single forward boundary and leave the frozen backbone replicated.
        FSDP2Manager(
            config=distributed_setup.strategy_config,
            mesh_context=mesh,
            fp32_main_params=distributed_setup.fp32_main_params,
        ).parallelize(model, ignored_params={param for param in model.parameters() if not param.requires_grad},
                      root_only=True)
    return model


def build_pi05_processor(
    model: Pi05TrainingModel, tokenizer_dir: str, stats_file: str,
) -> PolicyProcessorPipeline:
    """Build the native processor with local dataset statistics and tokenizer."""
    return build_pi05_processors(model.policy.config, Path(tokenizer_dir), Path(stats_file))[0]


def build_pi05_processors(
    config: PI05Config, tokenizer_dir: Path, stats_file: Path,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Build the matching input and action pipelines for a policy."""
    tokenizer = tokenizer_dir.expanduser().resolve()
    if not (tokenizer / "tokenizer.json").is_file():
        raise FileNotFoundError(tokenizer / "tokenizer.json")
    processor_config = replace(config, text_tokenizer_name=str(tokenizer))
    with stats_file.open(encoding="utf-8") as stream:
        stats = json.load(stream)
    return make_pre_post_processors(processor_config, dataset_stats=stats)


class Pi05Batch:
    """Adapt a batch of shared reader samples to the native processor."""

    def __init__(self, processor: PolicyProcessorPipeline, device: torch.device, pp_shared_data: bool = False) -> None:
        """Bind the native processor and the destination for loss accounting."""
        if pp_shared_data:
            raise ValueError("The π0.5 training recipe does not support pipeline-parallel shared data")
        self.processor = processor
        self.device = device

    def __call__(self, data_iterator: Iterator[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        """Prepare the policy batch and count matching native loss units."""
        batch = default_collate(next(data_iterator)["samples"])
        wrist_key = f"{OBS_IMAGES}.image2"
        if wrist_key not in batch:
            batch[wrist_key] = batch.pop(f"{OBS_IMAGES}.wrist_image")
        for key in (f"{OBS_IMAGES}.image", f"{OBS_IMAGES}.image2"):
            if batch[key].dtype == torch.uint8:
                batch[key] = batch[key].float().div_(255)
        processed = self.processor(batch)
        # LeRobot π0.5 averages the complete action chunk, including episode-tail repeats.
        action_mask = torch.ones_like(processed[ACTION][..., 0], dtype=torch.bool, device=self.device)
        return {"batch": processed}, {"action_mask": action_mask}
