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

#include "catlass_dynamic_matmul.h"
#include "catlass_dynamic_matmul_tiling_data.h"
#include "kernel/kernel_utils.h"

template <uint32_t DATATYPE_IN, uint32_t DATATYPE_OUT, uint32_t KERNEL_TYPE,
          uint32_t PADDING_TAGA, uint32_t PADDING_TAGB, uint32_t PADDING_TAGC, 
          uint32_t LAYOUT_TAGA, uint32_t LAYOUT_TAGB, uint32_t LAYOUT_TAGC, 
          uint32_t DISPATCH_POLICY_TAG>
__global__ __aicore__
void catlass_dynamic_matmul(GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR workspace, GM_ADDR tiling) {
    __gm__ uint8_t* user = AscendC::GetUserWorkspace(workspace);
    REGISTER_TILING_DEFAULT(CatlassDynamicMatmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(CatlassDynamicMatmulTilingData, tilingData, tiling);

    // half(fp16)
    if constexpr (
        DATATYPE_IN == static_cast<uint32_t>(CatlassDynamicMatmul::CatlassDataType::CATLASS_DT_FLOAT16) &&
        DATATYPE_OUT == static_cast<uint32_t>(CatlassDynamicMatmul::CatlassDataType::CATLASS_DT_FLOAT16)
    ) {
        CatlassDynamicMatmul::DynamicMatmulKernelWrapper<half, half, KERNEL_TYPE,
            PADDING_TAGA, PADDING_TAGB, PADDING_TAGC, LAYOUT_TAGA, LAYOUT_TAGB, LAYOUT_TAGC,
            DISPATCH_POLICY_TAG>(gmA, gmB, gmC, user, tilingData);
    }

    // bfloat16(bf16)
    if constexpr (
        DATATYPE_IN == static_cast<uint32_t>(CatlassDynamicMatmul::CatlassDataType::CATLASS_DT_BF16) &&
        DATATYPE_OUT == static_cast<uint32_t>(CatlassDynamicMatmul::CatlassDataType::CATLASS_DT_BF16)
    ) {
        CatlassDynamicMatmul::DynamicMatmulKernelWrapper<bfloat16_t, bfloat16_t, KERNEL_TYPE,
            PADDING_TAGA, PADDING_TAGB, PADDING_TAGC, LAYOUT_TAGA, LAYOUT_TAGB, LAYOUT_TAGC,
            DISPATCH_POLICY_TAG>(gmA, gmB, gmC, user, tilingData);
    }
}