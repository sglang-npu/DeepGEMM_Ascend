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

#include "catlass_dynamic_matmul_tiling.h"
#include "op_tiling/utils.h"
#include "op_tiling/select_kernel.h"

using namespace CatlassTiling;

namespace {
constexpr uint64_t RPC_WORKSIZE = 200UL;
constexpr uint64_t MB_SIZE = 1024 * 1024UL;

void SetMacroTilingKey(const TilingParams &tilingParams, gert::TilingContext *context)
{
    uint32_t DATATYPE_IN = static_cast<uint32_t>(context->GetInputTensor(0)->GetDataType());
    uint32_t DATATYPE_OUT = static_cast<uint32_t>(context->GetInputTensor(1)->GetDataType());
    uint32_t KERNEL_TYPE = static_cast<uint32_t>(tilingParams.kernelSerial);
    uint32_t PADDING_TAGA = static_cast<uint32_t>(tilingParams.paddingTagA);
    uint32_t PADDING_TAGB = static_cast<uint32_t>(tilingParams.paddingTagB);
    uint32_t PADDING_TAGC = static_cast<uint32_t>(tilingParams.paddingTagC);
    uint32_t LAYOUT_TAGA = static_cast<uint32_t>(tilingParams.layoutTagA);
    uint32_t LAYOUT_TAGB = static_cast<uint32_t>(tilingParams.layoutTagB);
    uint32_t LAYOUT_TAGC = static_cast<uint32_t>(tilingParams.layoutTagC);
    uint32_t DISPATCH_POLICY_TAG = static_cast<uint32_t>(tilingParams.dispatchPolicyTag);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(DATATYPE_IN, DATATYPE_OUT, KERNEL_TYPE,
        PADDING_TAGA, PADDING_TAGB, PADDING_TAGC, LAYOUT_TAGA, LAYOUT_TAGB, LAYOUT_TAGC, DISPATCH_POLICY_TAG);
    context->SetTilingKey(tilingKey);
}

ge::graphStatus SetTilingContext(const TilingParams &tilingParams, gert::TilingContext *context)
{
    CatlassDynamicMatmulTilingData* tilingData = context->GetTilingData<CatlassDynamicMatmulTilingData>();
    if (tilingData == nullptr) {
        std::cerr << "[DGA] [ERROR] cannot get tiling data." << std::endl;
        return ge::GRAPH_FAILED;
    }
    if (memset_s(tilingData, sizeof(CatlassDynamicMatmulTilingData), 0, sizeof(CatlassDynamicMatmulTilingData)) != EOK) {
        std::cerr << "[DGA] [ERROR] cannot allocate tiling data ptr." << std::endl;
        return ge::GRAPH_FAILED;
    }

    tilingData->strideA = tilingParams.strideA;
    tilingData->strideB = tilingParams.strideB;
    tilingData->strideC = tilingParams.strideC;
    tilingData->m = tilingParams.m;
    tilingData->n = tilingParams.n;
    tilingData->k = tilingParams.k;
    tilingData->m1 = tilingParams.m1;
    tilingData->n1 = tilingParams.n1;
    tilingData->k1 = tilingParams.k1;
    tilingData->swizzleOffset = tilingParams.swizzleOffset;
    tilingData->swizzleDirection = tilingParams.swizzleDirection;
    tilingData->splitkFactor = tilingParams.splitkFactor;

    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));
    tilingData->fftsAddr = fftsAddr;

    context->SetBlockDim(tilingParams.blockDim);
    SetMacroTilingKey(tilingParams, context);
    return ge::GRAPH_SUCCESS;
}
}

namespace optiling {
ge::graphStatus CatlassDynamicMatmulTilingFunc(gert::TilingContext *context)
{
    auto selfShape = context->GetInputShape(0)->GetOriginShape();
    auto mat2Shape = context->GetInputShape(1)->GetOriginShape();

    // todo use mat stride to check transpose
    bool isATrans = false;
    bool isBTrans = true;

    bool dimNumCheck = (selfShape.GetDimNum() == 2) && (mat2Shape.GetDimNum() == 2);
    if (!dimNumCheck) {
        std::cerr << "[DGA] [ERROR] self dim or mat2 dim is not 2." << std::endl;
        return ge::GRAPH_FAILED;
    }

    uint32_t m = selfShape.GetDim(0);
    uint32_t ka = selfShape.GetDim(1);
    uint32_t kb = mat2Shape.GetDim(0);
    uint32_t n = mat2Shape.GetDim(1);
    bool dimSizeCheck = (ka == kb);
    if (!dimSizeCheck) {
        std::cerr << "[DGA] [ERROR] self dimk is not equal with mat2 dimk" << std::endl;
        return ge::GRAPH_FAILED;
    }

    LayoutTag layoutTagA = isATrans ? LayoutTag::TagColumnMajor : LayoutTag::TagRowMajor;
    LayoutTag layoutTagB = isBTrans ? LayoutTag::TagColumnMajor : LayoutTag::TagRowMajor;
    LayoutTag layoutTagC = LayoutTag::TagRowMajor;

    TilingParams tilingParams(m, n, ka, layoutTagA, layoutTagB, layoutTagC);
    SelectKernelWithCache(tilingParams);

    auto setRes = SetTilingContext(tilingParams, context);
    if (setRes != ge::GRAPH_SUCCESS) {
        std::cerr << "[DGA] [ERROR] set tiling context failed." << std::endl;
        return ge::GRAPH_FAILED;
    }

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        std::cerr << "[DGA] [ERROR] cannot get current workspace." << std::endl;
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = RPC_WORKSIZE * MB_SIZE;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling