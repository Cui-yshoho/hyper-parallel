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
"""CANN-only native build hook, without importing Torch during packaging."""

import ctypes
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Optional

from setuptools import Extension
from setuptools.command.build_ext import build_ext


_PACKAGE = "hyper_parallel.components.functional._ascendc.symmetric_matmul"
_LIBRARIES = ("libhp_symmetric.so", "libhp_epilogue_kernel.so", "libhp_fused_kernel.so")


def cann_toolkit() -> Optional[Path]:
    """Find an installed CANN compiler toolkit, or return None on non-NPU hosts."""
    for key in ("ASCEND_HOME_PATH", "ASCEND_CANN_PACKAGE_PATH"):
        if os.environ.get(key):
            return Path(os.environ[key]).resolve()
    for candidate in ("/usr/local/Ascend/ascend-toolkit/latest", "/usr/local/Ascend/cann"):
        path = Path(candidate)
        if (path / "compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake").is_file():
            return path.resolve()
    return None


def ascendc_extensions() -> list[Extension]:
    """Enable automatic compilation when CANN is present, with an explicit opt-out."""
    mode = os.environ.get("HP_BUILD_ASCENDC", "auto").lower()
    if mode not in ("auto", "on", "off"):
        raise ValueError("HP_BUILD_ASCENDC must be auto, on or off")
    if mode == "off":
        return []
    toolkit = cann_toolkit()
    if toolkit is None:
        if mode == "on":
            raise RuntimeError("HP_BUILD_ASCENDC=on requires ASCEND_HOME_PATH pointing to the CANN toolkit")
        logging.getLogger(__name__).info("No CANN toolkit found; using native Torch matmul fallback")
        return []
    cmake_file = toolkit / "compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
    if not cmake_file.is_file():
        raise RuntimeError(f"CANN compiler is missing: {cmake_file}")
    return [Extension(_PACKAGE + ".libhp_symmetric", sources=[])]


def ascend_soc_version(toolkit: Path) -> str:
    """Select an explicit cross-compilation target or query CANN without importing Torch.

    Args:
        toolkit: CANN installation containing the host runtime library.

    Returns:
        The full compiler SoC identifier, including the device variant.

    Raises:
        RuntimeError: No device can be queried and no explicit target was provided.
    """
    target = os.environ.get("HP_ASCEND_SOC_VERSION", "").strip()
    if target:
        return target
    try:
        runtime = ctypes.CDLL(str(toolkit / "lib64/libascendcl.so"))
        query = runtime.aclrtGetSocName
        query.argtypes = []
        query.restype = ctypes.c_char_p
        name = query()
        if name:
            return name.decode("ascii")
    except (OSError, AttributeError) as exc:
        raise RuntimeError("Cannot query Ascend SoC; set HP_ASCEND_SOC_VERSION for cross-compilation") from exc
    raise RuntimeError("CANN returned no Ascend SoC; set HP_ASCEND_SOC_VERSION for cross-compilation")


class AscendCBuildExt(build_ext):
    """Build and package all three C ABI libraries for wheels and editable installs."""

    def get_ext_filename(self, fullname: str) -> str:
        """Use the stable C ABI filename instead of a CPython extension suffix.

        Args:
            fullname: Dotted extension module name supplied by setuptools.

        Returns:
            Relative filename of the built extension library.
        """
        if fullname.rsplit(".", 1)[-1] == "libhp_symmetric":
            return os.path.join(*fullname.split(".")) + ".so"
        return super().get_ext_filename(fullname)

    def build_extension(self, ext: Extension) -> None:
        """Compile the native payload with CANN and stage it in the build package.

        Args:
            ext: Extension descriptor for the symmetric GEMM host library.
        """
        cmake = shutil.which("cmake")
        if cmake is None:
            raise RuntimeError("Building Ascend C requires cmake >= 3.16 on PATH")
        source = Path(__file__).resolve().parent.joinpath(*_PACKAGE.split("."))
        build = Path(self.build_temp).resolve() / "symmetric_matmul"
        destination = Path(self.get_ext_fullpath(ext.name)).parent
        soc = ascend_soc_version(cann_toolkit())
        jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4")
        subprocess.run([cmake, "-S", str(source), "-B", str(build),
                        f"-DASCEND_CANN_PACKAGE_PATH={cann_toolkit()}",
                        f"-DSOC_VERSION={soc}", "-DCMAKE_BUILD_TYPE=Release"], check=True)
        subprocess.run([cmake, "--build", str(build), "--parallel", jobs], check=True)
        destination.mkdir(parents=True, exist_ok=True)
        for name in _LIBRARIES:
            shutil.copy2(build / "lib" / name, destination / name)

    def copy_extensions_to_source(self) -> None:
        """Place dependent libraries beside the main library for editable imports."""
        if not self.extensions:
            return
        super().copy_extensions_to_source()
        build_py = self.get_finalized_command("build_py")
        destination = Path(build_py.get_package_dir(_PACKAGE))
        source = Path(self.build_lib).joinpath(*_PACKAGE.split("."))
        for name in _LIBRARIES[1:]:
            shutil.copy2(source / name, destination / name)

    def get_outputs(self) -> list[str]:
        """Include dependent libraries in wheel and editable build output records."""
        outputs = super().get_outputs()
        if self.extensions:
            folder = Path(self.get_ext_fullpath(self.extensions[0].name)).parent
            outputs.extend(str(folder / name) for name in _LIBRARIES[1:])
        return outputs

    def get_output_mapping(self) -> dict[str, str]:
        """Map dependent libraries into setuptools strict editable link trees."""
        mapping = super().get_output_mapping()
        if self.inplace and self.extensions:
            build_py = self.get_finalized_command("build_py")
            source = Path(build_py.get_package_dir(_PACKAGE))
            built = Path(self.build_lib).joinpath(*_PACKAGE.split("."))
            mapping.update({str(built / name): str(source / name) for name in _LIBRARIES[1:]})
        return mapping
