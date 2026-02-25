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

#ifndef CATLASS_ACLNN_ADJUST_TILING_H
#define CATLASS_ACLNN_ADJUST_TILING_H

#include "platform_info.h"
#include "tiling_params.h"

namespace CatlassTiling {
void DoTilingLayout00(TilingParams& tilingParams, PlatformInfo& platformInfo);
void DoTilingLayout01(TilingParams& tilingParams, PlatformInfo& platformInfo);
void DoTilingLayout10(TilingParams& tilingParams, PlatformInfo& platformInfo);
void DoTilingLayout11(TilingParams& tilingParams, PlatformInfo& platformInfo);

using FuncType = void (*)(TilingParams& tilingParams, PlatformInfo& platformInfo);
static std::array<std::array<FuncType, 2>, 2> DoTiling = {
    {{{DoTilingLayout00, DoTilingLayout01}}, {{DoTilingLayout10, DoTilingLayout11}}}};
} // namespace CatlassTiling
#endif  // CATLASS_ACLNN_ADJUST_TILING_H