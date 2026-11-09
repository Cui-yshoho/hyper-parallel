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
"""Launch CPU/Gloo checks of Hyper FSDP managing real Torch DTensor storage."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_torch_dtensor_storage_gloo():
    """Feature: FSDP storage. Description: Compare representations. Expectation: equal updates."""
    worker = Path(__file__).with_name("_test_torch_dtensor_storage.py")
    torchrun_case(str(worker), "test_torch_dtensor_storage", num_proc=4)
