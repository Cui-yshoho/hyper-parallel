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
"""Native FastWAM model construction and eager video encoding."""

from pathlib import Path
from types import MethodType
from typing import Any, NamedTuple

import torch
from fastwam.runtime import create_fastwam
from omegaconf import OmegaConf
from torch import distributed as dist, nn
from transformers import PretrainedConfig

from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models.build_options import get_device_id, get_device_type


class FastWAMOutput(NamedTuple):
    """Native joint loss and its detached video and action metrics."""

    loss: torch.Tensor
    extra_metrics: dict[str, torch.Tensor]


def _get_npu_video_freqs(expert: nn.Module, frames: int, height: int, width: int) -> torch.Tensor:
    """Concatenate the upstream RoPE tables in their real FP64 representation."""
    frame_freqs, height_freqs, width_freqs = (torch.view_as_real(freqs) for freqs in expert.freqs)
    axes = (
        frame_freqs[:frames].view(frames, 1, 1, -1, 2).expand(frames, height, width, -1, 2),
        height_freqs[:height].view(1, height, 1, -1, 2).expand(frames, height, width, -1, 2),
        width_freqs[:width].view(1, 1, width, -1, 2).expand(frames, height, width, -1, 2),
    )
    # Ascend cat accepts FP64 but rejects complex128. The original RoPE
    # operation still receives the same complex128 values and layout.
    freqs = torch.cat(axes, dim=-2).reshape(frames * height * width, 1, -1, 2)
    return torch.view_as_complex(freqs)


class FastWAMTrainingModel(nn.Module):
    """Run the native joint forward with an eager VAE encoder."""

    def __init__(self, policy: nn.Module, config: PretrainedConfig) -> None:
        """Keep native experts, weights, and loss as the owners of model state."""
        super().__init__()
        self.policy = policy
        self.config = config
        # The pinned upstream video path lazily installs CUDA graphs in this slot.
        # Binding the same native encode callable preserves its scale and output clone.
        self.policy._vae_encode_compiled = self.policy.vae.model.encode  # pylint: disable=protected-access
        if self.policy.device.type == "npu":
            self.policy.video_expert.get_freqs = MethodType(_get_npu_video_freqs, self.policy.video_expert)
        self.policy.requires_grad_(False)
        self.policy.dit.requires_grad_(True)
        if self.policy.proprio_encoder is not None:
            self.policy.proprio_encoder.requires_grad_(True)
        self.train()

    def train(self, mode: bool = True) -> "FastWAMTrainingModel":
        """Keep frozen encoders in eval mode, matching upstream training."""
        super().train(mode)
        self.policy.eval()
        self.policy.dit.train(mode)
        if self.policy.proprio_encoder is not None:
            self.policy.proprio_encoder.train(mode)
        return self

    def forward(self, sample: dict[str, Any]) -> FastWAMOutput:
        """Compute native joint video and action loss in the DiT forward."""
        loss, metrics = self.policy(sample)
        return FastWAMOutput(loss, {name: loss.detach().new_tensor(value) for name, value in metrics.items()})


def load_fastwam_policy(
    *, checkpoint: str, wan_dir: str, model_config: str, data_config: str, device: torch.device, dtype: torch.dtype,
) -> tuple[nn.Module, PretrainedConfig]:
    """Build unconditioned FastWAM from a local released checkpoint and frozen Wan assets.

    The released checkpoint supplies both trainable experts and the proprio projector.
    The Wan directory supplies the VAE, text encoder, and tokenizer.
    """
    wan = Path(wan_dir).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    for path in (
        checkpoint_path, wan / "Wan2.2_VAE.pth", wan / "models_t5_umt5-xxl-enc-bf16.pth",
        wan / "google/umt5-xxl/tokenizer_config.json", wan / "google/umt5-xxl/spiece.model",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = OmegaConf.create({"model": OmegaConf.load(model_config), "data": OmegaConf.load(data_config)})
    settings = OmegaConf.to_container(config.model, resolve=True)
    if settings.pop("_target_") != "fastwam.runtime.create_fastwam":
        raise ValueError("Expected the native FastWAM base model configuration")
    settings.update(
        model_id=str(wan), tokenizer_model_id=str(wan), redirect_common_files=False,
        load_text_encoder=True, skip_dit_load_from_pretrain=True, action_dit_pretrained_path=None,
        compile_training_denoise=False,
    )
    policy = create_fastwam(**settings, device=str(device), model_dtype=dtype)
    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    policy.mot.load_state_dict(weights["mot"], strict=True)
    if policy.proprio_encoder is not None:
        policy.proprio_encoder.load_state_dict(weights["proprio_encoder"], strict=True)
    return policy, PretrainedConfig(fastwam=settings)


def build_fastwam_model(
    checkpoint: str, wan_dir: str, model_config: str, data_config: str, distributed_setup: DistributedSetup,
    model_init_dtype: str = "bfloat16",
) -> FastWAMTrainingModel:
    """Build a FastWAM training model on the Trainer's data-parallel mesh."""
    if not dist.is_initialized():
        raise ValueError("The FastWAM training recipe requires an initialized process group")
    mesh = distributed_setup.mesh_context
    world_size = dist.get_world_size()
    if mesh.dp_size != world_size or mesh.dp_shard_size != world_size:
        raise ValueError("FastWAM requires a pure data-parallel mesh with dp_shard_size equal to world size")
    if model_init_dtype != "bfloat16":
        raise ValueError("The FastWAM recipe initializes its model in bfloat16")
    with torch.random.fork_rng(devices=[]):
        policy, config = load_fastwam_policy(
            checkpoint=checkpoint, wan_dir=wan_dir, model_config=model_config, data_config=data_config,
            device=torch.device(get_device_type(), get_device_id()), dtype=torch.bfloat16,
        )
    model = FastWAMTrainingModel(policy, config)
    if world_size > 1:
        FSDP2Manager(
            config=distributed_setup.strategy_config, mesh_context=mesh,
            fp32_main_params=distributed_setup.fp32_main_params,
        ).parallelize(
            model, ignored_params={parameter for parameter in model.parameters() if not parameter.requires_grad},
            root_only=True,
        )
    return model
