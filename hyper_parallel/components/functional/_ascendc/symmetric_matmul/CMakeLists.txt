# Copyright 2026 Huawei Technologies Co., Ltd
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(hp_symmetric LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(SOC_VERSION "Ascend910B3" CACHE STRING "Target Ascend SoC")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN toolkit")
include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
ascendc_library(hp_epilogue_kernel SHARED epilogue.cpp)
ascendc_library(hp_fused_kernel SHARED fused.cpp)
add_library(hp_symmetric SHARED plan.cpp)
target_include_directories(hp_symmetric PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
    ${ASCEND_CANN_PACKAGE_PATH}/asc/include)
target_link_directories(hp_symmetric PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(hp_symmetric PRIVATE hp_epilogue_kernel hp_fused_kernel tiling_api platform ascendcl dl)
set_target_properties(hp_symmetric hp_epilogue_kernel hp_fused_kernel PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN;${ASCEND_CANN_PACKAGE_PATH}/lib64")
