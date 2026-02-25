/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_ACLNN_SELECT_KERNEL_H
#define CATLASS_ACLNN_SELECT_KERNEL_H

#include <limits>
#include "tiling_params.h"

namespace CatlassTiling {
void SelectKernel(TilingParams &tilingParams);
void SelectKernelWithCache(TilingParams &tilingParams);
// void SelectKernelWithPredictor(TilingParams &tilingParams);
} // namespace CatlassTiling

#endif  // CATLASS_ACLNN_SELECT_KERNEL_H
