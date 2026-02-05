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
 
#ifndef CATLASS_PADDING_COMMON_MATMUL_KERNEL_H
#define CATLASS_PADDING_COMMON_MATMUL_KERNEL_H

#include "../catlass_dynamic_matmul_tiling_data.h"
#include "kernel_utils.h"
#include "acl/acl.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/kernel/dynamic_padding_common_matmul.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace DynamicKernel {
template <class ElementA, class LayoutA,
          class ElementB, class LayoutB,
          class ElementC, class LayoutC,
          Catlass::Gemm::Kernel::PaddingTag paddingTagA,
          Catlass::Gemm::Kernel::PaddingTag paddingTagB,
          Catlass::Gemm::Kernel::PaddingTag paddingTagC>
__aicore__ inline void PaddingCommonMatmulKernel(GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmW,
    const CatlassDynamicMatmulTilingData &tilingData)
{
    using PaddingTag = Catlass::Gemm::Kernel::PaddingTag;
    using ArchTag = Catlass::Arch::AtlasA2;
    AscendC::SetSyncBaseAddr(tilingData.fftsAddr);
    Catlass::Arch::Resource<ArchTag> resource;

    // get tiling data
    int64_t strideA = static_cast<int64_t>(tilingData.strideA);
    int64_t strideB = static_cast<int64_t>(tilingData.strideB);
    int64_t strideC = static_cast<int64_t>(tilingData.strideC);

    uint32_t m = tilingData.m;
    uint32_t n = tilingData.n;
    uint32_t k = tilingData.k;

    uint32_t m1 = static_cast<uint32_t>(tilingData.m1);
    uint32_t n1 = static_cast<uint32_t>(tilingData.n1);
    uint32_t k1 = static_cast<uint32_t>(tilingData.k1);

    uint32_t swizzleOffset = static_cast<uint32_t>(tilingData.swizzleOffset);
    uint32_t swizzleDirection = static_cast<uint32_t>(tilingData.swizzleDirection);

    // construct params
    Catlass::GemmCoord problemShape(m, n, k);
    Catlass::GemmCoord l1TileShape(m1, n1, k1);
    LayoutA layoutA{m, k, strideA};
    LayoutB layoutB{k, n, strideB};
    LayoutC layoutC{m, n, strideC};

    // construct padding template params
    using PaddingBuilderA = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagA, ArchTag, ElementA, LayoutA>;
    using PaddingBuilderB = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagB, ArchTag, ElementB, LayoutB>;
    using RemovePaddingBuilderC = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagC, ArchTag, ElementC, LayoutC>;
    using PaddingA = typename PaddingBuilderA::Padding;
    using PaddingB = typename PaddingBuilderB::Padding;
    using RemovePaddingC = typename RemovePaddingBuilderC::Padding;

    // compute workspace
    size_t sizeWA = 0, sizeWB = 0;
    sizeWA = PaddingWorkspaceSize<ElementA, PaddingBuilderA, paddingTagA>(m, k, m1, k1);
    sizeWB = PaddingWorkspaceSize<ElementB, PaddingBuilderB, paddingTagB>(k, n, k1, n1);

    GM_ADDR gmWA = gmW;
    GM_ADDR gmWB = gmW + sizeWA;
    GM_ADDR gmWC = nullptr;
    if constexpr (paddingTagC == PaddingTag::PADDING_ND) {
        gmWC = gmW + sizeWA + sizeWB;
    }

    // construct kernel
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = Catlass::Gemm::MmadAtlasA2DynamicCommon<enableUnitFlag, enableShuffleK>;

    using AType = Catlass::Gemm::GemmType<ElementA, typename PaddingBuilderA::LayoutAfterPadding>;
    using BType = Catlass::Gemm::GemmType<ElementB, typename PaddingBuilderB::LayoutAfterPadding>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;

    using TileCopy = TileCopyDynamicOptimized<AType, BType, CType>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<DispatchPolicy, void, void, AType, BType, CType, void, TileCopy>;
    using BlockEpilogue = void;

    using BlockScheduler = typename Catlass::Gemm::Block::DynamicGemmIdentityBlockSwizzle;
    // kernel level
    using MatmulKernel = Catlass::Gemm::Kernel::DynamicPaddingCommonMatmul<
        PaddingA, PaddingB, BlockMmad, BlockEpilogue, BlockScheduler, RemovePaddingC>;
    typename MatmulKernel::Params params{
        problemShape, l1TileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC,
        gmWA, gmWB, gmWC, swizzleOffset, swizzleDirection};
    // call a kernel
    MatmulKernel matmul;
    matmul(params, resource);
}

template <class InDtype, class OutDtype, uint32_t LAYOUT_TAGA, uint32_t LAYOUT_TAGB, uint32_t LAYOUT_TAGC,
    uint32_t PADDING_TAGA, uint32_t PADDING_TAGB, uint32_t PADDING_TAGC>
__aicore__ inline void PaddingCommonKernelWrapper(
    GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmW, const CatlassDynamicMatmulTilingData &tilingData)
{
    using PaddingTag = Catlass::Gemm::Kernel::PaddingTag;
    constexpr bool layoutA = 
        (LAYOUT_TAGA == static_cast<uint32_t>(CatlassDynamicMatmul::CatlassLayoutTag::CATLASS_ROW_MAJOR));
    constexpr bool layoutB = 
        (LAYOUT_TAGB == static_cast<uint32_t>(CatlassDynamicMatmul::CatlassLayoutTag::CATLASS_ROW_MAJOR));
    
    using LayoutA = std::conditional_t<layoutA, Catlass::layout::RowMajor, Catlass::layout::ColumnMajor>;
    using LayoutB = std::conditional_t<layoutB, Catlass::layout::RowMajor, Catlass::layout::ColumnMajor>;
    // layoutC only support row major
    using LayoutC = Catlass::layout::RowMajor;

    constexpr PaddingTag paddingTagA = static_cast<PaddingTag>(PADDING_TAGA);
    constexpr PaddingTag paddingTagB = static_cast<PaddingTag>(PADDING_TAGB);
    constexpr PaddingTag paddingTagC = static_cast<PaddingTag>(PADDING_TAGC);

    PaddingCommonMatmulKernel<InDtype, LayoutA, InDtype,
        LayoutB, OutDtype, LayoutC, paddingTagA, paddingTagB, paddingTagC>(gmA, gmB, gmC, tilingData);
}
} // namespace DynamicKernel
#endif // CATLASS_PADDING_COMMON_MATMUL_KERNEL_H