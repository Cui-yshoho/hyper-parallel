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
"""Adapt OpenWAM's exported LIBERO policy to the shared evaluator."""

import secrets
import socket
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from typing import Any

import numpy as np
from benchmarks.libero.openwam2libero_interface import OpenWAMLiberoPolicy
from benchmarks.utils.transport import WSPolicyClient
from omegaconf import OmegaConf


def _stop_server(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _wait_for_server(process: subprocess.Popen, port: int, token: str) -> None:
    deadline = monotonic() + 600
    while monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"OpenWAM policy server exited during startup (code {process.returncode})")
        try:
            with WSPolicyClient(f"ws://127.0.0.1:{port}", timeout=2, open_timeout=2) as client:
                response = client.ping()
        except ConnectionRefusedError:
            sleep(0.5)
            continue
        if response.get("type") != "pong" or response.get("readiness_token") != token:
            raise RuntimeError(f"Unexpected policy service on localhost:{port}")
        return
    raise TimeoutError("OpenWAM policy server did not become ready within 10 minutes")


class OpenWAMPolicy:
    """Map the shared simulator observation to OpenWAM's raw LIBERO fields."""

    def __init__(self, bridge: OpenWAMLiberoPolicy, server: subprocess.Popen) -> None:
        """Bind the upstream camera and EEF10 policy adapter."""
        self.bridge = bridge
        self.server = server

    def reset(self) -> None:
        """Reset inference state between episodes."""
        self.bridge.reset()

    def act(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        """Send raw cameras and quaternion state for native OpenWAM processing."""
        state = observation["robot_state"]
        pixels = observation["pixels"]
        raw_observation = {
            "agentview_image": pixels["image"],
            "robot0_eye_in_hand_image": pixels["wrist_image"],
            "robot0_eef_pos": state["eef"]["pos"],
            "robot0_eef_quat": state["eef"]["quat"],
            "robot0_gripper_qpos": state["gripper"]["qpos"],
        }
        return self.bridge.act(raw_observation, instruction)

    def close(self) -> None:
        """Release the upstream policy executor."""
        try:
            self.bridge.close()
        finally:
            _stop_server(self.server)


def load_policy(policy_dir: Path, device: str, deploy_config: Path | None = None) -> OpenWAMPolicy:
    """Load an exported OpenWAM policy and its checkpoint-owned normalization statistics."""
    # LIBERO consumes actions; the upstream deploy preset also skips video decoding.
    config = OmegaConf.create({"optimization": {"decode_video": False}})
    if deploy_config is not None:
        config = OmegaConf.merge(config, OmegaConf.load(deploy_config))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    token = secrets.token_hex(24)
    with TemporaryDirectory(prefix="hyper-openwam-libero-") as directory:
        config_path = Path(directory) / "deploy.yaml"
        OmegaConf.save(config, config_path)
        process = subprocess.Popen([
            sys.executable, "-m", "openwam.deploy.server",
            "--ckpt-dir", str(policy_dir.resolve()), "--config", str(config_path),
            "--device", device, "--host", "127.0.0.1", "--port", str(port),
            "--readiness-token", token,
        ])
        try:
            _wait_for_server(process, port, token)
            bridge = OpenWAMLiberoPolicy(host="127.0.0.1", port=port)
        except BaseException:
            _stop_server(process)
            raise
    return OpenWAMPolicy(bridge, process)
