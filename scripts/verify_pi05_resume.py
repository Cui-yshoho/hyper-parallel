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
"""Trace and compare the continuous and resumed steps of a π0.5 FSDP run."""

import hashlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from hyper_parallel.trainer.callbacks.base import Callback
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.state import TrainerState
from hyper_parallel.trainer.vlm_trainer import VLMTrainer


class Pi05ResumeTrace(Callback):
    """Record the delivered frame indices and local action-head shard after each optimizer step."""

    def __init__(self, trainer: VLMTrainer, output_dir: Path) -> None:
        super().__init__(trainer.base)
        self.path = output_dir / f"resume-trace-rank{self.trainer.global_rank}.jsonl"
        self.sample_ids: list[int] = []
        if self.path.exists():
            raise FileExistsError(self.path)
        output_dir.mkdir(parents=True, exist_ok=True)
        get_batch = self.trainer.get_batch

        def traced_batch(iterator: Iterator[dict[str, Any]]) -> Any:
            # Observe exactly the samples the native iterator delivers, including after restore.
            batch = next(iterator)
            self.sample_ids.extend(int(sample["index"]) for sample in batch["samples"])
            return get_batch(iter((batch,)))

        self.trainer.get_batch = traced_batch

    def on_step_end(
        self, state: TrainerState, loss: float, loss_dict: dict[str, Any], grad_norm: float, **kwargs: Any,
    ) -> None:
        """Write one record after the native optimizer and scheduler have stepped."""
        del kwargs
        weight = self.trainer.model.get_parameter("policy.model.action_out_proj.weight").detach()
        if hasattr(weight, "to_local"):
            weight = weight.to_local()
        row = {
            "step": state.global_step,
            "sample_ids": self.sample_ids,
            "learning_rates": [float(group["lr"]) for group in self.trainer.optimizer.param_groups],
            "loss": float(loss),
            "action_loss": float(loss_dict["action_loss"]),
            "grad_norm": float(grad_norm),
            "weight_sha256": hashlib.sha256(weight.cpu().contiguous().numpy().tobytes()).hexdigest(),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        self.sample_ids.clear()


def _load_trace(directory: Path, rank: int) -> list[dict[str, Any]]:
    trace = directory / f"resume-trace-rank{rank}.jsonl"
    return [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]


def compare(continuous: Path, resumed: Path) -> None:
    """Require bitwise-identical action-head shards and the same trace on steps 3 and 4."""
    for rank in range(2):
        uninterrupted = _load_trace(continuous, rank)
        restarted = _load_trace(resumed, rank)
        if [row["step"] for row in uninterrupted] != [1, 2, 3, 4]:
            raise ValueError(f"Rank {rank} must record continuous steps 1 through 4")
        if restarted != uninterrupted[2:]:
            raise ValueError(f"Rank {rank} resume trace differs: expected {uninterrupted[2:]}, got {restarted}")
        for previous, current in zip(uninterrupted[1:], uninterrupted[2:]):
            if previous["weight_sha256"] == current["weight_sha256"]:
                raise ValueError(f"Rank {rank} action-head shard did not update at step {current['step']}")
        print(f"Rank {rank} exact steps 3 and 4: {json.dumps(restarted)}")


def main() -> None:
    """Run the existing Trainer with tracing, or compare the traces from two runs."""
    if len(sys.argv) == 4 and sys.argv[1] == "compare":
        compare(Path(sys.argv[2]), Path(sys.argv[3]))
        return
    config = parse_training_args()
    trainer = VLMTrainer(config)
    trace = Pi05ResumeTrace(trainer, Path(config.checkpoint.checkpoint_dir))
    trainer.base._callbacks.append(trace)  # pylint: disable=protected-access
    trainer.train()


if __name__ == "__main__":
    main()
