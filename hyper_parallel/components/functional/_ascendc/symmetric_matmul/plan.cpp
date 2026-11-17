// Copyright 2026 Huawei Technologies Co., Ltd
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <new>
#include <limits>
#include <numeric>
#include <cstring>
#include "tiling/matmul/matmul_tiling.h"
#include "params.h"

extern "C" uint32_t aclrtlaunch_hp_epilogue(uint32_t, void *, void *, void *, void *, SymmetricPlan *, float, float);
extern "C" uint32_t aclrtlaunch_hp_fused(uint32_t, void *, void *, void *, void *, void *, void *, SymmetricPlan *,
                                         float, float);
namespace {
using CubeTiling = AscendC::tiling::TCubeTiling;

bool make_tiling(int n, int k, int rows, int columns, int dtype, int direct, CubeTiling &result) {
  auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  matmul_tiling::MatmulApiTiling tiler(*platform);
  using namespace matmul_tiling;
  auto type = dtype == 0 ? DataType::DT_FLOAT16 : DataType::DT_BF16;
  tiler.SetAType(TPosition::GM, CubeFormat::ND, type, false);
  tiler.SetBType(TPosition::GM, CubeFormat::ND, type, true);
  tiler.SetCType(TPosition::GM, CubeFormat::ND, direct ? type : DataType::DT_FLOAT);
  tiler.SetShape(rows, columns, k);
  tiler.SetOrgShape(n, n, k);
  tiler.SetMatmulConfigParams(1);
  tiler.SetBias(false);
  tiler.SetBufferSpace(-1, -1, -1);
  return tiler.GetTiling(result) == 0;
}

int aligned_size(int value, int alignment) {
  int64_t result = (static_cast<int64_t>(value) + alignment - 1) / alignment * alignment;
  return result <= std::numeric_limits<int>::max() ? static_cast<int>(result) : 0;
}

bool indexed_shape(int n, int k, int batch) {
  if (n <= 0 || k <= 0 || batch <= 0) return false;
  int64_t rows = static_cast<int64_t>(batch) * n;
  return rows <= std::numeric_limits<int>::max() / std::max(n, k);
}

int task_size(const CubeTiling &tiling) {
  if (tiling.baseM <= 0 || tiling.baseN <= 0) return 0;
  // Task ownership is square; Cube and Vector subtiles retain independent shapes.
  int64_t result = std::lcm<int64_t>(64, std::lcm<int64_t>(tiling.baseM, tiling.baseN));
  return result <= std::numeric_limits<int>::max() ? static_cast<int>(result) : 0;
}
}  // namespace

extern "C" void *hp_plan(int n, int k, int batch, int tile, int cores, int dtype, int direct) {
  if (!indexed_shape(n, k, batch) || cores <= 0) return nullptr;
  int probe_n = aligned_size(n, 64);
  int padded_k = aligned_size(k, 16);
  CubeTiling tiling;
  if (!probe_n || !padded_k || !make_tiling(probe_n, padded_k, probe_n, probe_n, dtype, direct, tiling)) return nullptr;
  if (tile == 0) tile = std::min(task_size(tiling), probe_n);
  if (tile <= 0 || tile % 64 != 0) return nullptr;
  int padded_n = probe_n;
  if (!indexed_shape(padded_n, padded_k, batch)) return nullptr;
  int row_tile = tile;
  if (!make_tiling(padded_n, padded_k, row_tile, tile, dtype, direct, tiling)) return nullptr;
  auto *plan = new (std::nothrow) SymmetricPlan{};
  if (!plan) return nullptr;
  plan->n = padded_n;
  plan->k = padded_k;
  plan->batch = batch;
  plan->tile = tile;
  plan->cores = cores;
  plan->dtype = dtype;
  plan->direct = direct;
  plan->row_tile = row_tile;
  static_assert(sizeof(tiling) <= sizeof(plan->tiling));
  std::memcpy(plan->tiling, &tiling, sizeof(tiling));
  return plan;
}

extern "C" void hp_plan_info(void *pointer, int *values) {
  auto *plan = static_cast<SymmetricPlan *>(pointer);
  CubeTiling tiling;
  std::memcpy(&tiling, plan->tiling, sizeof(tiling));
  values[0] = plan->n;
  values[1] = plan->k;
  values[2] = plan->tile;
  values[3] = tiling.baseM;
  values[4] = tiling.baseN;
  values[5] = tiling.baseK;
  values[6] = plan->row_tile;
}

extern "C" void hp_destroy(void *plan) { delete static_cast<SymmetricPlan *>(plan); }
extern "C" uint32_t hp_launch(void *plan, void *stream, void *a, void *b, void *c, void *scratch, void *out,
                              float alpha, float beta) {
  auto *config = static_cast<SymmetricPlan *>(plan);
  if (alpha != 0) {
    return aclrtlaunch_hp_fused(config->cores, stream, a, b, c, scratch, out, config, alpha, beta);
  }
  return aclrtlaunch_hp_epilogue(config->cores * 2, stream, scratch, c, out, config, alpha, beta);
}

extern "C" int hp_abi_version() { return 4; }
