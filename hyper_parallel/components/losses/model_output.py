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
"""Loss module that reads the loss produced by a model."""

__all__ = ["MaskedModelOutputLoss", "ModelOutputLoss"]

from typing import Any, Dict, Optional, Union

# AutoModels loss components implement the Transformers/PyTorch Trainer API.
import torch

from hyper_parallel.data.constants import IGNORE_INDEX


class ModelOutputLoss(torch.nn.Module):
    """Return the loss field from a Transformers-style model output."""

    def forward(  # pylint: disable=unused-argument
        self,
        *,
        model_output: Any,
        labels: Optional[torch.Tensor],
        valid_token_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Read the model-computed loss.

        Args:
            model_output: Model output exposing a ``loss`` attribute.
            labels: Unshifted labels used by legacy callers that do not supply
                an aligned token count.
            valid_token_count: Trainer count after target alignment and masking.
                When omitted, preserve the legacy unshifted-label convention.

        Returns:
            The loss tensor or named loss mapping from ``model_output.loss``.
        """
        local_loss = model_output.loss
        if not isinstance(local_loss, torch.Tensor):
            return local_loss

        # Packed inputs may already be shifted: dropping their first target
        # again can erase the only supervised token on a CP shard. Trainer owns
        # target alignment, so use its count whenever it is available.
        if valid_token_count is not None:
            has_valid_labels = valid_token_count > 0
        elif labels is not None:
            has_valid_labels = labels[..., 1:].ne(IGNORE_INDEX).any()
        else:
            return local_loss
        local_loss = torch.where(has_valid_labels, local_loss, torch.zeros_like(local_loss))

        return local_loss


class MaskedModelOutputLoss(torch.nn.Module):
    """Weight a native mean loss by its mask across micro-batches and ranks."""

    preserve_single_microbatch_loss = True

    def __init__(self, mask_field: str, loss_family: str) -> None:
        """Select the supervision mask field and the loss family to aggregate."""
        super().__init__()
        self.mask_field = mask_field
        self.loss_name = f"{loss_family}_loss"
        self.count_name = f"{loss_family}_tokens"

    def count_loss_units(self, loss_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Count supervised elements using the configured loss mask.

        Args:
            loss_inputs: Fields used by the configured loss.
        """
        return {self.count_name: loss_inputs[self.mask_field].count_nonzero()}

    def forward(
        self, model_output: Any, labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Read the native mean computed over the same supervision mask.

        Args:
            model_output: Output exposing the model-computed mean loss.
            labels: Optional Trainer labels.
        """
        del labels
        return {self.loss_name: model_output.loss}
