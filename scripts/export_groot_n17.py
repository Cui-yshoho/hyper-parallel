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
"""Convert a HyperParallel GR00T DCP checkpoint to a local LeRobot policy."""

import argparse
import logging
from pathlib import Path

from hyper_parallel.core.distributed_checkpoint.api import load
from hyper_parallel.models.groot_n17.modeling import build_groot_processors, load_local_groot_model

logger = logging.getLogger(__name__)


def main() -> None:
    """Load the model shards on CPU and save the official policy and processors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True, help="Original NVIDIA GR00T N1.7 checkpoint directory")
    parser.add_argument("--processor-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-repo", default="nvidia/LIBERO_LeRobot_v3")
    parser.add_argument("--embodiment-tag", default="libero_sim")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    if not (args.training_checkpoint / ".metadata").is_file():
        parser.error(f"Expected a completed HyperParallel DCP checkpoint at {args.training_checkpoint}")

    model = load_local_groot_model(args.checkpoint, args.dataset_root, args.dataset_repo, args.embodiment_tag)
    state = {"model": model.state_dict()}
    load(state, checkpoint_id=args.training_checkpoint, no_dist=True)
    model.load_state_dict(state["model"], strict=True)
    preprocessor, postprocessor = build_groot_processors(model, args.processor_dir)

    args.output_dir.mkdir(parents=True)
    model.policy.save_pretrained(args.output_dir)
    preprocessor.save_pretrained(args.output_dir)
    postprocessor.save_pretrained(args.output_dir)
    logger.info("Exported LeRobot policy and processors to %s", args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
