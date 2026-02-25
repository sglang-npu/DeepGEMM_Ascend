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

 #ifndef __CATLASS_DYNAMIC_MATMUL_H__
 #define __CATLASS_DYNAMIC_MATMUL_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "catlass_dynamic_matmul_tiling_data.h"
#include "catlass_dynamic_matmul_tiling_key.h"
#include "kernel/kernel_utils.h"

#include "kernel/common_matmul_kernel.h"
#include "kernel/small_matmul_kernel.h"
#include "kernel/padding_common_matmul_kernel.h"
#include "kernel/padding_streamk_matmul_kernel.h"

namespace CatlassDynamicMatmul {
template <class InDtype, class OutDtype, uint32_t KERNEL_TYPE,
          uint32_t PADDING_TAGA, uint32_t PADDING_TAGB, uint32_t PADDING_TAGC,
          uint32_t LAYOUT_TAGA, uint32_t LAYOUT_TAGB, uint32_t LAYOUT_TAGC,
          uint32_t DISPATCH_POLICY_TAG>
__aicore__ inline void DynamicMatmulKernelWrapper(
    GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR workspace, const CatlassDynamicMatmulTilingData &tilingData)
{
    if constexpr (KERNEL_TYPE == static_cast<uint32_t>(CatlassKernelType::CATLASS_COMMON_MATMUL)) {
        DynamicKernel::CommonKernelWrapper<
            InDtype, OutDtype, LAYOUT_TAGA, LAYOUT_TAGB, LAYOUT_TAGC>(gmA, gmB, gmC, tilingData);
        return;
    }

    if constexpr (KERNEL_TYPE == static_cast<uint32_t>(CatlassKernelType::CATLASS_SMALL_MATMUL)) {
        DynamicKernel::SmallKernelWrapper<
            InDtype, OutDtype, LAYOUT_TAGA, LAYOUT_TAGB, LAYOUT_TAGC>(gmA, gmB, gmC, tilingData);
        return;
    }

    if constexpr (KERNEL_TYPE == static_cast<uint32_t>(CatlassKernelType::CATLASS_PADDING_MATMUL)) {
        DynamicKernel::PaddingCommonKernelWrapper<
            InDtype, OutDtype, LAYOUT_TAGA, LAYOUT_TAGB, LAYOUT_TAGC,
            PADDING_TAGA, PADDING_TAGB, PADDING_TAGC>(gmA, gmB, gmC, workspace, tilingData);
        return;
    }

    if constexpr (KERNEL_TYPE == static_cast<uint32_t>(CatlassKernelType::CATLASS_STREAMK_MATMUL)) {
        DynamicKernel::PaddingStreamkMatmulKernelWrapper<
            InDtype, OutDtype, LAYOUT_TAGA, LAYOUT_TAGB, LAYOUT_TAGC,
            PADDING_TAGA, PADDING_TAGB>(gmA, gmB, gmC, workspace, tilingData);
        return;
    }
}
} // namespace CatlassDynamicMatmul


 #endif // __CATLASS_DYNAMIC_MATMUL_H__