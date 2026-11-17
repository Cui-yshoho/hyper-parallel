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
"""Hardware-independent target selection for packaged Ascend C builds."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from setup_ascendc import ascend_soc_version


class TestAscendSocVersion(unittest.TestCase):
    """Prevent accidentally building 910B binaries for a different NPU generation."""

    def test_explicit_target_does_not_load_runtime(self):
        """Cross-compilation requires no driver or NPU on the build machine."""
        target = "Ascend950DT_9582"
        with patch.dict(os.environ, {"HP_ASCEND_SOC_VERSION": target}):
            with patch("setup_ascendc.ctypes.CDLL") as loader:
                actual = ascend_soc_version(Path("/cann"))
        self.assertEqual(actual, target, f"Expected explicit target={target}, got actual={actual}")
        loader.assert_not_called()

    def test_detects_full_device_variant(self):
        """The runtime identifier retains the variant required by the compiler."""
        for target in ("Ascend910B3", "Ascend950DT_9582"):
            with self.subTest(target=target):
                with patch.dict(os.environ, {"HP_ASCEND_SOC_VERSION": ""}):
                    with patch("setup_ascendc.ctypes.CDLL") as loader:
                        loader.return_value.aclrtGetSocName.return_value = target.encode("ascii")
                        actual = ascend_soc_version(Path("/cann"))
                self.assertEqual(actual, target, f"Expected detected target={target}, got actual={actual}")
                loader.assert_called_once_with("/cann/lib64/libascendcl.so")

    def test_missing_runtime_requires_explicit_target(self):
        """A compiler-only toolkit must never silently select another generation."""
        with patch.dict(os.environ, {"HP_ASCEND_SOC_VERSION": ""}):
            with patch("setup_ascendc.ctypes.CDLL", side_effect=OSError("driver unavailable")):
                with self.assertRaisesRegex(RuntimeError, "HP_ASCEND_SOC_VERSION"):
                    ascend_soc_version(Path("/cann"))

    def test_missing_device_requires_explicit_target(self):
        """A null runtime result cannot become a misleading successful build."""
        with patch.dict(os.environ, {"HP_ASCEND_SOC_VERSION": ""}):
            with patch("setup_ascendc.ctypes.CDLL") as loader:
                loader.return_value.aclrtGetSocName.return_value = None
                with self.assertRaisesRegex(RuntimeError, "HP_ASCEND_SOC_VERSION"):
                    ascend_soc_version(Path("/cann"))
