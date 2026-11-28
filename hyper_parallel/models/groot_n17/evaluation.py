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
"""Load an exported GR00T N1.7 policy for the shared LIBERO runner."""

from pathlib import Path

import torch
from lerobot.policies.groot.modeling_groot import GrootPolicy

from hyper_parallel.evaluation.lerobot import LeRobotLiberoPolicy as GrootLiberoPolicy


def load_policy(policy_dir: Path, device: str) -> GrootLiberoPolicy:
    """Reload the exported GR00T checkpoint and processors, then place the policy."""
    # The pinned LeRobot device config accepts CPU; the model itself supports Ascend.
    policy = GrootPolicy.from_pretrained(policy_dir, local_files_only=True, strict=True)
    policy.to(torch.device(device)).eval()
    return GrootLiberoPolicy(policy, policy_dir)
