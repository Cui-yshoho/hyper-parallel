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
#include "product.h"
#include "epilogue.h"

extern "C" __global__ __aicore__ void hp_fused(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR scratch, GM_ADDR out,
                                               SymmetricPlan plan, float alpha, float beta) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  if ASCEND_IS_AIC {
    if (plan.dtype == 0)
      dispatch_product<half, true>(a, b, scratch, plan);
    else
      dispatch_product<bfloat16_t, true>(a, b, scratch, plan);
  } else {
    if (plan.dtype == 0)
      epilogue<half, true>(scratch, c, out, plan, alpha, beta);
    else
      epilogue<bfloat16_t, true>(scratch, c, out, plan, alpha, beta);
  }
}
