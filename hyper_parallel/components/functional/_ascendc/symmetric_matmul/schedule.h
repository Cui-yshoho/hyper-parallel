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
#include "kernel_operator.h"
#include "params.h"
using namespace AscendC;

__aicore__ inline void decode(int task, int tiles, int &batch, int &row, int &col) {
  int count = tiles * (tiles + 1) / 2;
  batch = task / count;
  int local = task % count;
  int left = 0, right = tiles;
  while (left + 1 < right) {
    int middle = (left + right) / 2;
    if (static_cast<int64_t>(middle) * (middle + 1) / 2 <= local)
      left = middle;
    else
      right = middle;
  }
  row = left;
  col = local - row * (row + 1) / 2;
}

struct TaskRegion {
  int batch, row, col, rows, columns;
};

template <bool Fused>
__aicore__ inline int task_count(const SymmetricPlan &plan) {
  int tile = Fused ? plan.tile : 64;
  int row_tile = Fused ? plan.row_tile : 64;
  int tiles = (plan.n + tile - 1) / tile;
  int strips = (tile + row_tile - 1) / row_tile;
  return plan.batch * tiles * (tiles + 1) / 2 * strips;
}

template <bool Fused>
__aicore__ inline TaskRegion task_region(const SymmetricPlan &plan, int task) {
  int tile = Fused ? plan.tile : 64;
  int row_tile = Fused ? plan.row_tile : 64;
  int strips = (tile + row_tile - 1) / row_tile;
  int batch, row, col;
  decode(task / strips, (plan.n + tile - 1) / tile, batch, row, col);
  int offset = task % strips * row_tile;
  int start_row = row * tile + offset;
  int start_col = col * tile;
  int rows = min(row_tile, min(tile - offset, max(0, plan.n - start_row)));
  int columns = min(tile, plan.n - start_col);
  // A diagonal strip must not write above later strips' mirrored destinations.
  if (row == col) columns = min(columns, offset + rows);
  return {batch, start_row, start_col, rows, columns};
}
