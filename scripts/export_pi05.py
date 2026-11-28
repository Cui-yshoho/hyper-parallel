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
"""Export a HyperParallel π0.5 DCP checkpoint as a local LeRobot policy."""

import argparse
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from transformers.initialization import no_init_weights

from hyper_parallel.core.distributed_checkpoint.api import load
from hyper_parallel.models.pi05.modeling import Pi05TrainingModel, build_pi05_processors, load_pi05_config

logger = logging.getLogger(__name__)


def export_pi05(
    training_checkpoint: Path, base_checkpoint: Path, tokenizer_dir: Path, stats_file: Path, output_dir: Path,
) -> None:
    """Consolidate the training model and package its original normalization pipeline."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not (training_checkpoint / ".metadata").is_file():
        raise FileNotFoundError(training_checkpoint / ".metadata")
    config = load_pi05_config(base_checkpoint, ("--dtype=float32", "--gradient_checkpointing=false"))
    preprocessor, postprocessor = build_pi05_processors(config, tokenizer_dir, stats_file)
    # DCP supplies the complete model, including the replicated backbone.
    with no_init_weights():
        model = Pi05TrainingModel(PI05Policy(config))
    state = {"model": model.state_dict()}
    load(state, checkpoint_id=training_checkpoint, no_dist=True)
    model.load_state_dict(state["model"], strict=True)
    # Transformers normally ties the embedding and LM head during initialization.
    model.policy.model.paligemma_with_expert.paligemma.tie_weights(recompute_mapping=False)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / output_dir.name
        model.policy.save_pretrained(staging)
        preprocessor.save_pretrained(staging)
        postprocessor.save_pretrained(staging)
        staging.rename(output_dir)
    logger.info("Exported π0.5 policy and processors to %s", output_dir)


def main() -> None:
    """Resolve the local training assets and export a LeRobot policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--stats-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_pi05(args.training_checkpoint, args.base_checkpoint, args.tokenizer_dir, args.stats_file, args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
