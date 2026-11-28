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
"""OpenWAM Alpha model construction and native joint training objective."""

from typing import Any, NamedTuple

import torch
from omegaconf import OmegaConf
from openwam.dataloader.collate import openwam_collate
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
from openwam.train.utils.optimizer_groups import build_trainable_parameters
from transformers import PretrainedConfig

from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models.build_options import get_device_id, get_device_type


class OpenWAMOutput(NamedTuple):
    """Joint objective and detached losses reported by the native model."""

    loss: torch.Tensor
    extra_metrics: dict[str, torch.Tensor]


class OpenWAMTrainingModel(torch.nn.Module):
    """Run native joint video/action flow matching."""

    def __init__(
        self, architecture: BaseWAMArchitecture, config: PretrainedConfig, *,
        lambda_video: float, lambda_action: float, offload_frozen_encoders: bool,
    ) -> None:
        """Keep the upstream architecture as the owner of model state."""
        super().__init__()
        self.architecture = architecture
        self.config = config
        self.lambda_video = lambda_video
        self.lambda_action = lambda_action
        self.offload_frozen_encoders = offload_frozen_encoders

    def forward(self, inputs: dict[str, Any]) -> OpenWAMOutput:
        """Run the native joint objective inside the sharded forward.

        Args:
            inputs: Encoded inputs returned by the condition model.
        """
        result = self.architecture.compute_loss(
            **inputs, lambda_video=self.lambda_video, lambda_action=self.lambda_action,
        )
        return OpenWAMOutput(
            result["loss"], {"video_loss": result["loss_video"], "action_loss": result["loss_action"]},
        )

    def build_optimizer(self, lr: float = 1e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
        """Build AdamW over the upstream OpenWAM parameter groups.

        Args:
            lr: Learning rate.
            weight_decay: AdamW weight decay.
        """
        return torch.optim.AdamW(
            build_trainable_parameters(self), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay, foreach=False,
        )


class OpenWAMCondition(torch.nn.Module):
    """Use the frozen encoders already owned by the OpenWAM architecture."""

    def __init__(self, model: OpenWAMTrainingModel, device: torch.device) -> None:
        """Bind the existing encoders without registering them a second time."""
        super().__init__()
        self.encode = model.architecture.prepare_inputs
        backbone = model.architecture.video_backbone
        self.encoders = (backbone.text_encoder, backbone.vae)
        self.device = device
        self.offload = model.offload_frozen_encoders

    def get_condition(self, samples: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        """Encode the current micro-batch and release encoder device memory."""
        try:
            if self.offload:
                for encoder in self.encoders:
                    encoder.to(device=self.device)
            return {"inputs": self.encode(openwam_collate(samples))}
        finally:
            if self.offload:
                for encoder in self.encoders:
                    encoder.to(device="cpu")

    @staticmethod
    def process_condition(**condition: Any) -> dict[str, Any]:
        """Pass through encoded inputs for the joint model objective."""
        return condition


class OpenWAMDiTProvider:
    """Connect the native OpenWAM encoders to the DiT condition stage."""

    @staticmethod
    def build_condition_model(
        *, model: OpenWAMTrainingModel, device: torch.device, **_: Any,
    ) -> OpenWAMCondition:
        """Reuse the encoders restored by the model builder."""
        return OpenWAMCondition(model, device)


def build_openwam_model(
    checkpoint: str,
    distributed_setup: DistributedSetup,
    model_init_dtype: str = "bfloat16",
    lambda_video: float = 1.0,
    lambda_action: float = 1.0,
    offload_frozen_encoders: bool = True,
    model_provider: str | None = None,
) -> OpenWAMTrainingModel:
    """Restore the self-contained foundation checkpoint and native SFT runtime.

    Args:
        checkpoint: Local OpenWAM foundation checkpoint directory.
        distributed_setup: Trainer mesh and FSDP options.
        model_init_dtype: Initializer parameter dtype.
        lambda_video: Weight of the video objective.
        lambda_action: Weight of the action objective.
        offload_frozen_encoders: Move the frozen text encoder and VAE to CPU after encoding.
        model_provider: Provider path read by the Trainer from the model target.
    """
    del model_provider
    world_size = torch.distributed.get_world_size()
    mesh = distributed_setup.mesh_context
    if mesh.dp_size != world_size or mesh.dp_shard_size != world_size:
        raise ValueError("OpenWAM requires a pure data-parallel mesh with dp_shard_size equal to world size")
    if model_init_dtype != "bfloat16":
        raise ValueError("The OpenWAM baseline uses bfloat16 weights")
    # Skeleton initialization should not advance the run's training-noise RNG.
    with torch.random.fork_rng(devices=[]):
        _, architecture, checkpoint_cfg = build_architecture_from_ckpt_dir(checkpoint, weights_required=True)
    architecture.set_dtype_device(torch.bfloat16, torch.device(get_device_type(), get_device_id()))
    architecture.train()
    frozen = architecture.freeze_modules(list(checkpoint_cfg.model.freeze))
    if not {"video_backbone.text_encoder", "video_backbone.vae"}.issubset(frozen):
        raise ValueError("OpenWAM conditioning requires a frozen text encoder and VAE")
    architecture.init_training_schedulers()
    architecture.set_training_runtime(use_gradient_checkpointing=True)
    config = PretrainedConfig(openwam_model=OmegaConf.to_container(checkpoint_cfg.model, resolve=True))
    model = OpenWAMTrainingModel(
        architecture, config, lambda_video=lambda_video, lambda_action=lambda_action,
        offload_frozen_encoders=offload_frozen_encoders,
    )
    if world_size > 1:
        # MoT calls the blocks' projections directly, so use one joint
        # FSDP unit at the recipe wrapper's forward boundary.
        FSDP2Manager(
            config=distributed_setup.strategy_config,
            mesh_context=mesh,
            fp32_main_params=distributed_setup.fp32_main_params,
        ).parallelize(
            model,
            ignored_params={param for param in model.parameters() if not param.requires_grad},
            root_only=True,
        )
    return model
