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
"""Evaluate robot policies with one LIBERO rollout and reporting protocol."""

import argparse
import json
import logging
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from statistics import mean
from time import monotonic
from typing import Any, Protocol

import numpy as np
import torch

logger = logging.getLogger(__name__)


class LiberoPolicy(Protocol):
    """A policy acting on a raw LIBERO observation and task instruction."""

    def reset(self) -> None:
        """Clear policy and processor state between episodes."""

    def act(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        """Return one 7-D LIBERO controller action."""

    def close(self) -> None:
        """Release policy runtime resources."""


def rollout(env: Any, policy: LiberoPolicy, *, seed: int, max_steps: int) -> dict[str, Any]:
    """Run one episode until success, simulator termination, or the step limit."""
    if max_steps < 1:
        raise ValueError("LIBERO max_steps must be positive")
    policy.reset()
    observation, _ = env.reset(seed=seed)
    rewards = []
    success = False
    for _ in range(max_steps):
        action = np.asarray(policy.act(observation, env.task_description), dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError(f"LIBERO expects a finite 7-D action, got shape {action.shape}")
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        success = bool(info.get("is_success", False))
        if success or terminated or truncated:
            break
    return {
        "seed": seed, "success": success, "steps": len(rewards),
        "sum_reward": sum(rewards), "max_reward": max(rewards),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    # The evaluation environment and its metric utilities are optional dependencies.
    from lerobot.utils.eval_stats import success_summary  # pylint: disable=import-outside-toplevel

    return {
        **success_summary([record["success"] for record in records]),
        "avg_sum_reward": mean(record["sum_reward"] for record in records),
        "avg_max_reward": mean(record["max_reward"] for record in records),
        "video_paths": [],
        "predicted_video_paths": [],
    }


def evaluate_libero(
    policy: LiberoPolicy, *, suite: str, task_ids: list[int] | None = None,
    episodes: int = 1, seed: int = 42, max_steps: int | None = None, settle_steps: int = 10,
) -> dict[str, Any]:
    """Run both local and external model policies using the same LIBERO environment settings."""
    if episodes < 1 or settle_steps < 0 or (max_steps is not None and max_steps < 1):
        raise ValueError("LIBERO requires positive episodes and max_steps, and nonnegative settle_steps")
    # LIBERO and its simulator are optional for HyperParallel training.
    from lerobot.sim.integrations.libero import env_fns  # pylint: disable=import-outside-toplevel

    task_factories = env_fns(
        suite, n_envs=1,
        gym_kwargs={
            "task_ids": task_ids, "obs_type": "pixels_agent_pos",
            "observation_height": 256, "observation_width": 256, "num_steps_wait": settle_steps,
        },
        camera_name_mapping={"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"},
    )
    started = monotonic()
    per_task = []
    group_records = defaultdict(list)
    with torch.no_grad():
        for group, tasks in task_factories.items():
            for task_id, (factory,) in tasks.items():
                with closing(factory()) as env:
                    # The LIBERO integration exposes its suite-specific budget on this attribute.
                    step_limit = env._max_episode_steps if max_steps is None else max_steps
                    records = [
                        {"episode_ix": episode, **rollout(env, policy, seed=seed + episode, max_steps=step_limit)}
                        for episode in range(episodes)
                    ]
                group_records[group].extend(records)
                summary = _summarize(records)
                logger.info("%s task %d: %d/%d successful episodes", group, task_id,
                            summary["n_success"], summary["n_episodes"])
                per_task.append({
                    "task_group": group, "task_id": task_id, "max_steps": step_limit,
                    "n_episodes": summary["n_episodes"], "n_success": summary["n_success"],
                    "pc_success": summary["pc_success"], "pc_success_ci95": summary["pc_success_ci95"],
                    "metrics": {
                        "sum_rewards": [record["sum_reward"] for record in records],
                        "max_rewards": [record["max_reward"] for record in records],
                        "successes": [record["success"] for record in records],
                        "video_paths": [], "predicted_video_paths": [], "per_episode": records,
                    },
                })
    overall = _summarize([record for records in group_records.values() for record in records])
    elapsed = monotonic() - started
    overall.update(eval_s=elapsed, eval_ep_s=elapsed / overall["n_episodes"])
    return {
        "settings": {"suite": suite, "episodes_per_task": episodes, "seed": seed, "settle_steps": settle_steps},
        "per_task": per_task,
        "per_group": {group: _summarize(records) for group, records in group_records.items()},
        "overall": overall,
    }


def main(default_model: str | None = None) -> None:
    """Load an exported policy and run the shared LIBERO evaluator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("groot_n17", "openwam_alpha", "pi05"), required=default_model is None,
                        default=default_model)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per task")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, help="Override the suite's per-episode step budget")
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--deploy-config", type=Path, help="OpenWAM inference overrides")
    parser.add_argument("--tokenizer-dir", type=Path, help="Local PaliGemma tokenizer directory for π0.5")
    args = parser.parse_args()
    if args.episodes < 1 or args.settle_steps < 0 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("Specify positive episodes and max-steps, and nonnegative settle-steps")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if not args.policy_dir.is_dir():
        parser.error(f"Policy directory does not exist: {args.policy_dir}")
    if args.deploy_config is not None and args.model != "openwam_alpha":
        parser.error("--deploy-config applies to OpenWAM")
    if args.tokenizer_dir is not None and args.model != "pi05":
        parser.error("--tokenizer-dir applies to π0.5")

    # Keep the simulator and model-specific dependencies optional at module import.
    from accelerate.utils import set_seed  # pylint: disable=import-outside-toplevel

    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)
    if args.model == "groot_n17":
        from hyper_parallel.models.groot_n17.evaluation import load_policy  # pylint: disable=import-outside-toplevel

        policy = load_policy(args.policy_dir, args.device)
    elif args.model == "pi05":
        from hyper_parallel.models.pi05.evaluation import load_policy  # pylint: disable=import-outside-toplevel

        policy = load_policy(args.policy_dir, args.device, tokenizer_dir=args.tokenizer_dir)
    else:
        from hyper_parallel.models.openwam_alpha.evaluation import load_policy  # pylint: disable=import-outside-toplevel

        policy = load_policy(args.policy_dir, args.device, args.deploy_config)
    with closing(policy):
        result = evaluate_libero(
            policy, suite=args.suite, task_ids=args.task_ids, episodes=args.episodes,
            seed=args.seed, max_steps=args.max_steps, settle_steps=args.settle_steps,
        )
    result["model"] = args.model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
