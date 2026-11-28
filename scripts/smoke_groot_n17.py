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
"""Validate the GR00T N1.7 recipe with one local batch and one parameter update."""

import argparse
import logging
import math
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from hyper_parallel.models.groot_n17.modeling import (
    GrootTrainingModel,
    build_groot_policy,
    build_groot_policy_config,
    build_groot_processor,
    require_local_file,
)

logger = logging.getLogger(__name__)


def load_local_training_batch(
        *,
        checkpoint: Path,
        dataset_root: Path,
        dataset_repo: str,
        embodiment_tag: str,
        batch_size: int,
        video_backend: str,
        processor_dir: Path | None = None,
) -> tuple[GrootTrainingModel, dict[str, Any]]:
    """Load a raw N1.7 checkpoint and the first processed batch for baseline capture."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    policy_config = build_groot_policy_config(checkpoint, embodiment_tag)
    dataset_root = dataset_root.expanduser().resolve()
    require_local_file(dataset_root, "meta/info.json")
    # Keep this baseline on LeRobot's native dataset factory.
    from lerobot.configs.default import DatasetConfig  # pylint: disable=import-outside-toplevel
    from lerobot.configs.train import TrainPipelineConfig  # pylint: disable=import-outside-toplevel
    from lerobot.datasets.factory import make_dataset  # pylint: disable=import-outside-toplevel

    dataset = make_dataset(TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=dataset_repo, root=str(dataset_root), video_backend=video_backend),
        policy=policy_config,
    ))
    if len(dataset) < batch_size:
        raise ValueError(f"Requested {batch_size} samples from a dataset of length {len(dataset)}")
    policy = build_groot_policy(checkpoint.expanduser().resolve(), embodiment_tag, dataset.meta)
    model = GrootTrainingModel(policy)
    processor = build_groot_processor(model, str(processor_dir) if processor_dir is not None else None)
    raw_batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size)))
    return model, processor(raw_batch)


def main() -> None:
    """Compare adapter loss and capture one official LeRobot optimizer update."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--embodiment-tag", default="libero_sim")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--processor-dir", type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-output", type=Path)
    args = parser.parse_args()

    # LeRobot and Accelerate are optional dependencies of this model-specific script.
    from accelerate import Accelerator  # pylint: disable=import-outside-toplevel
    from accelerate.utils import set_seed  # pylint: disable=import-outside-toplevel
    from lerobot.scripts.lerobot_train import update_policy  # pylint: disable=import-outside-toplevel
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker  # pylint: disable=import-outside-toplevel

    device = torch.device(args.device)
    if device.type != "cpu" and device.index not in (None, 0):
        parser.error("Use a single visible device and select logical device 0")
    accelerator = Accelerator(cpu=device.type == "cpu", mixed_precision="no")
    if accelerator.num_processes != 1 or accelerator.device.type != device.type:
        parser.error(f"Expected a single {device.type} device, found {accelerator.device}")
    device = accelerator.device
    set_seed(args.seed)
    model, batch = load_local_training_batch(
        checkpoint=args.checkpoint,
        dataset_root=args.dataset_root,
        dataset_repo=args.dataset_repo,
        embodiment_tag=args.embodiment_tag,
        batch_size=args.batch_size,
        video_backend=args.video_backend,
        processor_dir=args.processor_dir,
    )
    model.to(device).train()
    optimizer_config = model.policy.config.get_optimizer_preset()
    optimizer = optimizer_config.build(model.policy.get_optim_params())
    policy, prepared_optimizer = accelerator.prepare(model.policy, optimizer)

    # Replay noise, time samples and dropout for the native update.
    devices = [] if device.type == "cpu" else [device.index if device.index is not None else 0]
    with torch.no_grad(), torch.random.fork_rng(devices=devices, device_type=device.type):
        result = model(batch)
    if not torch.isfinite(result.loss) or result.valid_action_count.item() == 0:
        raise RuntimeError("GR00T smoke requires a finite loss and valid action elements")

    trainable_names = [name for name, parameter in policy.named_parameters() if parameter.requires_grad]
    parameters = dict(policy.named_parameters())
    if any(parameters[name].dtype != torch.float32 for name in trainable_names):
        raise RuntimeError("The LeRobot baseline requires FP32 trainable parameters")
    frozen_name, frozen_parameter = next(
        (name, parameter) for name, parameter in parameters.items() if not parameter.requires_grad
    )
    frozen_before = frozen_parameter.detach().view(-1)[:512].cpu().clone()
    probe = {}

    def _capture_probe(_optimizer: Any, _args: Any, _kwargs: Any) -> None:
        """Capture a representative gradient before the native update."""
        if any(parameter.grad is not None for parameter in parameters.values() if not parameter.requires_grad):
            raise RuntimeError("The LeRobot step produced gradients on frozen GR00T parameters")
        for name in trainable_names:
            parameter = parameters[name]
            if parameter.grad is None:
                continue
            stride = max(parameter.numel() // 512, 1)
            gradient = parameter.grad.detach().view(-1)[::stride][:512].cpu()
            if gradient.count_nonzero():
                probe.update(
                    name=name, stride=stride, gradient=gradient,
                    before=parameter.detach().view(-1)[::stride][:512].cpu().clone(),
                )
                return
        raise RuntimeError("The LeRobot step found no nonzero gradients in the parameter probes")

    metrics = MetricsTracker(
        args.batch_size, args.batch_size, 1,
        {name: AverageMeter(name, ":.6g") for name in ("loss", "grad_norm", "lr", "update_s")},
    )
    rng = {"cpu": torch.get_rng_state()}
    if device.type != "cpu":
        rng[device.type] = torch.get_device_module(device).get_rng_state(device)
    with optimizer.register_step_pre_hook(_capture_probe):
        metrics, _ = update_policy(metrics, policy, batch, prepared_optimizer,
                                   optimizer_config.grad_clip_norm, accelerator)
    torch.testing.assert_close(result.loss.cpu(), torch.tensor(metrics.loss.val))
    if not math.isfinite(metrics.grad_norm.val) or metrics.grad_norm.val <= 0:
        raise RuntimeError(f"Invalid LeRobot gradient norm: {metrics.grad_norm.val}")
    probe["after"] = parameters[probe["name"]].detach().view(-1)[::probe["stride"]][:512].cpu()
    if torch.equal(probe["before"], probe["after"]):
        raise RuntimeError("LeRobot did not update the probed trainable parameter")
    torch.testing.assert_close(frozen_parameter.detach().view(-1)[:512].cpu(), frozen_before, rtol=0, atol=0)
    if args.baseline_output is not None:
        parameter_names = {id(parameter): name for name, parameter in parameters.items()}
        optimizer_groups = [
            {**group, "params": [parameter_names[id(parameter)] for parameter in group["params"]]}
            for group in optimizer.param_groups
        ]
        with args.baseline_output.open("xb") as destination:
            torch.save({
                "seed": args.seed, "batch": batch, "rng": rng,
                "assets": {
                    "checkpoint": str(args.checkpoint.resolve()), "dataset": str(args.dataset_root.resolve()),
                    "processor": str(args.processor_dir.resolve()) if args.processor_dir is not None else None,
                },
                "versions": {name: version(name) for name in ("torch", "lerobot", "transformers", "accelerate")},
                "loss": metrics.loss.val, "grad_norm": metrics.grad_norm.val, "lr": metrics.lr.val,
                "valid_action_count": result.valid_action_count.item(), "trainable_names": trainable_names,
                "optimizer_groups": optimizer_groups,
                "probe": probe, "frozen_probe_name": frozen_name, "frozen_probe": frozen_before,
            }, destination)
    logger.info(
        "LeRobot GR00T N1.7 loss=%g valid_action_count=%d grad_norm=%g lr=%g probe=%s",
        metrics.loss.val, result.valid_action_count.item(), metrics.grad_norm.val, metrics.lr.val, probe["name"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
