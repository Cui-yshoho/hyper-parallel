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
#pragma once
#include "lib/matmul_intf.h"
#include "params.h"
#include "schedule.h"

template <class Matmul, class T, class Output>
__aicore__ inline void produce_region(Matmul &mm, GlobalTensor<T> &a, GlobalTensor<T> &b, GlobalTensor<Output> &out,
                                      const SymmetricPlan &plan, const TaskRegion &region) {
  if (region.rows == 0) return;
  int64_t input_batch = static_cast<int64_t>(region.batch) * plan.n * plan.k;
  int64_t output_batch = static_cast<int64_t>(region.batch) * plan.n * plan.n;
  mm.SetSingleShape(region.rows, region.columns, plan.k);
  mm.SetTensorA(a[input_batch + region.row * plan.k]);
  mm.SetTensorB(b[input_batch + region.col * plan.k], true);
  mm.IterateAll(out[output_batch + region.row * plan.n + region.col]);
  mm.End();
}

template <class T, bool Fused = false, class Output = float>
__aicore__ inline void product(GM_ADDR a, GM_ADDR b, GM_ADDR scratch, SymmetricPlan &plan) {
  if ASCEND_IS_AIV {
    return;
  }
  TPipe pipe;
  using Left = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
  using Right = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
  using Result = matmul::MatmulType<TPosition::GM, CubeFormat::ND, Output>;
  matmul::MatmulImpl<Left, Right, Result, Result, CFG_MDL> mm;
  TCubeTiling tiling;
  auto *words = reinterpret_cast<uint32_t *>(&tiling);
  for (int i = 0; i < sizeof(tiling) / sizeof(uint32_t); ++i) words[i] = plan.tiling[i];
  mm.SetSubBlockIdx(0);
  mm.Init(&tiling, &pipe);
  GlobalTensor<T> ga, gb;
  GlobalTensor<Output> gc;
  ga.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(a));
  gb.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(b));
  gc.SetGlobalBuffer(reinterpret_cast<__gm__ Output *>(scratch));
  int tasks = task_count<true>(plan);
  // Two ready/consumed slots bound producer lead and prevent flag reuse before
  // both AIVs finish. GM-only MatmulImpl does not use the KFC synchronization path.
  int iteration = 0;
  for (int task = GetBlockIdx(); task < tasks; task += GetBlockNum(), ++iteration) {
    if constexpr (Fused) {
      if (iteration >= 2) CrossCoreWaitFlag<0x2>(8 + iteration % 2);
    }
    auto region = task_region<true>(plan, task);
    produce_region(mm, ga, gb, gc, plan, region);
    if constexpr (Fused) CrossCoreSetFlag<0x2, PIPE_FIX>(6 + iteration % 2);
  }
  if constexpr (Fused) {
    for (int slot = 0; slot < 2 && slot < iteration; ++slot) CrossCoreWaitFlag<0x2>(8 + slot);
  }
}

template <class T, bool Fused = false>
__aicore__ inline void dispatch_product(GM_ADDR a, GM_ADDR b, GM_ADDR scratch, SymmetricPlan &plan) {
  if (plan.direct)
    product<T, Fused, T>(a, b, scratch, plan);
  else
    product<T, Fused, float>(a, b, scratch, plan);
}
