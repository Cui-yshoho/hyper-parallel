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
"""Export a HyperParallel OpenWAM Alpha DCP checkpoint for the official policy server."""

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf
from openwam.train.utils.checkpointing import find_latest_weights, save_config, save_normalization_stats
from safetensors import safe_open
from safetensors.torch import save_file

from hyper_parallel.core.distributed_checkpoint.api import load
from hyper_parallel.core.distributed_checkpoint.filesystem_storage import FileSystemReader
from hyper_parallel.core.distributed_checkpoint.metadata import TensorStorageMetadata
from hyper_parallel.data.embodied.dataset import build_openwam_dataset

logger = logging.getLogger(__name__)


def export_openwam_alpha(
    training_checkpoint: Path, foundation: Path, dataset_root: Path, dataset_config: Path,
    output_dir: Path, step: int,
) -> None:
    """Consolidate all model shards and attach the native model and LIBERO assets.

    Args:
        training_checkpoint: Directory containing the HyperParallel DCP checkpoint.
        foundation: Directory containing the original OpenWAM weights and tokenizer.
        dataset_root: Local LIBERO dataset directory.
        dataset_config: Path to the upstream dataset YAML.
        output_dir: Destination for the deployable policy.
        step: Optimizer step used in the output filename.
    """
    if output_dir.exists():
        raise FileExistsError(f"OpenWAM export already exists: {output_dir}")
    if step < 1:
        raise ValueError("Export step must be positive")
    reader = FileSystemReader(training_checkpoint)
    metadata = reader.load_metadata()
    prefix = "model.architecture."
    model_entries = {
        name.removeprefix(prefix): entry
        for name, entry in metadata.state_dict_metadata.items()
        if name.startswith(prefix)
    }
    foundation_weights = find_latest_weights(str(foundation))
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(foundation_weights, framework="pt", device="cpu") as source:
        foundation_keys = set(source.keys())
        if set(model_entries) != foundation_keys:
            missing = sorted(foundation_keys - model_entries.keys())
            unexpected = sorted(model_entries.keys() - foundation_keys)
            raise ValueError(f"OpenWAM checkpoint keys differ from the foundation. missing={missing}, extra={unexpected}")
        for name, entry in model_entries.items():
            if not isinstance(entry, TensorStorageMetadata) or list(entry.size) != source.get_slice(name).get_shape():
                raise ValueError(f"OpenWAM checkpoint shape differs for {name}")
            dtype = source.get_tensor(name).dtype
            if entry.properties.dtype.removeprefix("torch.") != str(dtype).removeprefix("torch."):
                raise ValueError(f"OpenWAM checkpoint dtype differs for {name}")
            state_dict[name] = torch.empty(entry.size, dtype=dtype)
    load(
        {"model": {f"architecture.{name}": value for name, value in state_dict.items()}},
        storage_reader=reader,
        no_dist=True,
    )

    cfg = OmegaConf.load(foundation / "config.yaml")
    cfg.dataloader = OmegaConf.load(dataset_config)
    cfg.dataloader.dataset_dir = str(dataset_root)
    cfg.training = {"mixed_precision": "bf16"}
    dataset = build_openwam_dataset(str(dataset_root), str(dataset_config))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temp:
        staged = Path(temp) / output_dir.name
        staged.mkdir()
        save_config(str(staged), cfg)
        save_normalization_stats(str(staged), dataset)
        if not (staged / "normalization_stats.npy").is_file():
            raise FileNotFoundError("The LIBERO reader must supply normalization_stats.npy for deployment")
        shutil.copytree(foundation / "tokenizer", staged / "tokenizer")
        save_file(state_dict, staged / f"checkpoint_step_{step}.safetensors")
        staged.rename(output_dir)
    logger.info("Exported %d OpenWAM tensors to %s", len(state_dict), output_dir)


def main() -> None:
    """Read export paths and write the official OpenWAM checkpoint directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument("--foundation", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    export_openwam_alpha(
        args.training_checkpoint, args.foundation, args.dataset_root, args.dataset_config, args.output_dir, args.step,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
