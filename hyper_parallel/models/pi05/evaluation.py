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
"""Load a local π0.5 policy for the shared LIBERO evaluation flow."""

from pathlib import Path

import torch

from hyper_parallel.evaluation.lerobot import LeRobotLiberoPolicy
from hyper_parallel.models.pi05.modeling import load_pi05_config, load_pi05_weights


def load_policy(policy_dir: Path, device: str, tokenizer_dir: Path | None = None) -> LeRobotLiberoPolicy:
    """Strictly reload π0.5 weights and use the checkpoint's saved processor statistics."""
    weights = policy_dir / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    tokenizer_dir = (tokenizer_dir or policy_dir / "tokenizer").expanduser().resolve()
    if not (tokenizer_dir / "tokenizer.json").is_file():
        raise FileNotFoundError(f"Missing tokenizer.json in {tokenizer_dir}. Use --tokenizer-dir to set its directory")
    config = load_pi05_config(policy_dir, ("--gradient_checkpointing=false",))
    policy = load_pi05_weights(policy_dir, config)
    inference_device = torch.device(device)
    policy.to(inference_device).eval()
    # The checkpoint saves a CUDA device step. Move inputs after preprocessing so the
    # saved pipeline also works with devices that LeRobot's device step cannot resolve.
    return LeRobotLiberoPolicy(
        policy, policy_dir, wrist_camera="image2", input_device=inference_device,
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(tokenizer_dir)},
            "device_processor": {"device": "cpu"},
        },
    )
