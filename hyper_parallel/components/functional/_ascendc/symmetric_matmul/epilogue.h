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
#include "params.h"
#include "schedule.h"

struct EpilogueGeometry {
  int core, cores;
};

__aicore__ inline bool lower_tile(int row, int col, int limit) { return row >= col && row < limit && col < limit; }

template <bool Fused>
__aicore__ inline EpilogueGeometry epilogue_geometry(const SymmetricPlan &plan) {
  return {static_cast<int>(Fused ? GetBlockIdx() / 2 : GetBlockIdx()),
          static_cast<int>(Fused ? plan.cores : GetBlockNum())};
}

__aicore__ inline void initialize_diagonal_mask(TBuf<TPosition::VECCALC> &selection) {
  constexpr int tile = 64;
  auto bits = selection.Get<uint64_t>();
  for (int row = 0; row < tile; ++row) {
    bits.SetValue(row, row == 63 ? ~uint64_t(0) : (uint64_t(1) << (row + 1)) - 1);
  }
}

template <class T>
__aicore__ inline void transpose_tile(LocalTensor<T> &r, LocalTensor<T> &tr) {
  constexpr int tile = 64;
  uint64_t sources[16], destinations[16];
  for (int block = 0; block < tile / 16; ++block) {
    for (int line = 0; line < 16; ++line) {
      sources[line] = reinterpret_cast<uint64_t>(r[(block * 16 + line) * tile].GetPhyAddr());
      destinations[line] = reinterpret_cast<uint64_t>(tr[line * tile + block * 16].GetPhyAddr());
    }
    TransDataTo5HD<uint16_t>(destinations, sources, {false, false, tile / 16, tile, 1});
  }
  PipeBarrier<PIPE_V>();
}

template <bool Fused>
__aicore__ inline void wait_epilogue_tile(int iteration) {
  if constexpr (Fused) CrossCoreWaitFlag<0x2>(6 + iteration % 2);
}

template <bool Fused>
__aicore__ inline void finish_epilogue_tile(int iteration) {
  if constexpr (Fused) CrossCoreSetFlag<0x2, PIPE_MTE3>(8 + iteration % 2);
}

template <class T, bool Fused = false>
__aicore__ inline void epilogue(GM_ADDR scratch, GM_ADDR c, GM_ADDR out, SymmetricPlan &plan, float alpha, float beta) {
  TPipe pipe;
  constexpr int tile = 64;
  TBuf<TPosition::VECCALC> values, additive, rounded, transposed, selection;
  pipe.InitBuffer(values, tile * tile * sizeof(float));
  pipe.InitBuffer(additive, tile * tile * sizeof(float));
  pipe.InitBuffer(rounded, tile * tile * sizeof(T));
  pipe.InitBuffer(transposed, tile * tile * sizeof(T));
  pipe.InitBuffer(selection, tile * tile / 8);
  initialize_diagonal_mask(selection);
  auto v = values.Get<float>();
  auto add = additive.Get<float>();
  auto r = rounded.Get<T>();
  auto tr = transposed.Get<T>();
  GlobalTensor<float> src;
  GlobalTensor<T> cg, dst;
  src.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch));
  cg.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(c));
  dst.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));
  auto geometry = epilogue_geometry<Fused>(plan);
  int tasks = task_count<Fused>(plan);
  int core = geometry.core;
  int cores = geometry.cores;
  int iteration = 0;
  for (int task = core; task < tasks; task += cores, ++iteration) {
    auto region = task_region<Fused>(plan, task);
    wait_epilogue_tile<Fused>(iteration);
    for (int subrow = Fused ? GetSubBlockIdx() : 0; subrow < region.rows / tile; subrow += Fused ? 2 : 1) {
      for (int subcol = 0; subcol < region.columns / tile; ++subcol) {
        int row = region.row / tile + subrow;
        int col = region.col / tile + subcol;
        if (!lower_tile(row, col, plan.n / tile)) continue;
        int64_t lower = static_cast<int64_t>(region.batch) * plan.n * plan.n + row * tile * plan.n + col * tile;
        int64_t upper = static_cast<int64_t>(region.batch) * plan.n * plan.n + col * tile * plan.n + row * tile;
        if (plan.direct) {
          DataCopy(r, dst[lower], {tile, tile * 2 / 32, static_cast<uint16_t>((plan.n - tile) * 2 / 32), 0});
        } else {
          if (alpha != 0 && plan.k != 0) {
            DataCopy(v, src[lower], {tile, tile * 4 / 32, static_cast<uint16_t>((plan.n - tile) * 4 / 32), 0});
            PipeBarrier<PIPE_ALL>();
            Muls(v, v, alpha, tile * tile);
          } else {
            Duplicate(v, float(0), tile * tile);
          }
          if (beta != 0) {
            DataCopy(r, cg[lower], {tile, tile * 2 / 32, static_cast<uint16_t>((plan.n - tile) * 2 / 32), 0});
            PipeBarrier<PIPE_ALL>();
            Cast(add, r, RoundMode::CAST_NONE, tile * tile);
            PipeBarrier<PIPE_V>();
            Muls(add, add, beta, tile * tile);
            PipeBarrier<PIPE_V>();
            Add(v, v, add, tile * tile);
          }
          PipeBarrier<PIPE_V>();
          Cast(r, v, RoundMode::CAST_RINT, tile * tile);
        }
        PipeBarrier<PIPE_ALL>();
        transpose_tile(r, tr);
        if (row == col) {
          Select(r.template ReinterpretCast<half>(), selection.Get<uint8_t>(), r.template ReinterpretCast<half>(),
                 tr.template ReinterpretCast<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, tile * tile);
        }
        PipeBarrier<PIPE_ALL>();
        if (!plan.direct || row == col)
          DataCopy(dst[lower], r, {tile, tile * 2 / 32, 0, static_cast<uint16_t>((plan.n - tile) * 2 / 32)});
        if (row != col)
          DataCopy(dst[upper], tr, {tile, tile * 2 / 32, 0, static_cast<uint16_t>((plan.n - tile) * 2 / 32)});
        PipeBarrier<PIPE_ALL>();
      }
    }
    finish_epilogue_tile<Fused>(iteration);
  }
}
