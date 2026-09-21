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
"""Architecture identity and visual-boundary roles for Qwen3-VL-MoE."""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_model_adapter


def _load_sharding_rules():
    """Describe Qwen3-VL vision projections not covered by HF naming rules."""
    from hyper_parallel.distributed.tensor_parallel.param_role import (  # pylint: disable=C0415
        ParamRole,
    )

    return [
        ("attn.qkv", ParamRole.FUSED_QKV),
        ("attn.proj", ParamRole.ROWWISE),
        ("linear_fc1", ParamRole.COLWISE),
        (".mlp.linear_fc2", ParamRole.COLWISE),
        ("linear_fc2", ParamRole.ROWWISE),
    ]


QWEN3_VL_MOE_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen3VLMoeForConditionalGeneration",
    model_type="qwen3_vl_moe",
    sharding_rules=_load_sharding_rules,
)

register_model_adapter(QWEN3_VL_MOE_ADAPTER_SPEC)
