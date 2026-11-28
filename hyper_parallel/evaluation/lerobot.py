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
"""Adapt LeRobot policies and processor pipelines to the shared LIBERO runner."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.envs.configs import LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import prepare_canonical_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import policy_output_to_transition, transition_to_policy_action
from lerobot.processor.relative_action_processor import bind_relative_anchor
from lerobot.sim.integrations.libero import canonical_observation
from lerobot.utils.constants import ACTION


class LeRobotLiberoPolicy:
    """Keep LeRobot observation, policy, and action processing in the common rollout."""

    def __init__(
        self, policy: PreTrainedPolicy, policy_dir: Path, *, wrist_camera: str = "wrist_image",
        input_device: torch.device | None = None,
        preprocessor_overrides: dict[str, Any] | None = None,
    ) -> None:
        """Load the exported processor pipelines and relative-action settings."""
        self.policy = policy
        self.wrist_camera = wrist_camera
        self.input_device = input_device
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy.config, pretrained_path=policy_dir, preprocessor_overrides=preprocessor_overrides,
        )
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(LiberoEnvConfig(), policy.config)
        bind_relative_anchor(policy, self.preprocessor)

    def reset(self) -> None:
        """Clear the action queue and episode-specific processor state."""
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def act(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        """Run the native LeRobot processor pipelines around one policy action."""
        batched = {key: value[None] for key, value in canonical_observation(observation).items()}
        inputs = prepare_canonical_observation(batched)
        inputs["task"] = [instruction]
        if self.wrist_camera != "wrist_image":
            inputs[f"observation.images.{self.wrist_camera}"] = inputs.pop("observation.images.wrist_image")
        inputs = self.preprocessor(self.env_preprocessor(inputs))
        if self.input_device is not None:
            inputs = {
                name: value.to(self.input_device) if torch.is_tensor(value) else value
                for name, value in inputs.items()
            }
        with torch.inference_mode():
            output = self.policy.select_action(inputs)
        action = self.postprocessor(transition_to_policy_action(policy_output_to_transition(output)))
        return self.env_postprocessor({ACTION: action})[ACTION][0].detach().cpu().numpy()

    def close(self) -> None:
        """Complete the local policy lifecycle."""
