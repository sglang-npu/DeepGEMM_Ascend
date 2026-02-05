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

#ifndef CATLASS_KERNEL_UTILS_H
#define CATLASS_KERNEL_UTILS_H

#include "catlass/arch/arch.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/kernel/padding_matmul.hpp"

namespace CatlassDynamicMatmul {
enum class CatlassDataType : uint32_t
{
    CATLASS_DT_FLOAT16 = 1,
    CATLASS_DT_BF16 = 27,
};

enum class CatlassKernelType : uint32_t
{
    CATLASS_COMMON_MATMUL = 0,
    CATLASS_SMALL_MATMUL = 1,
    CATLASS_PADDING_MATMUL = 2,
    CATLASS_MULTICORE_SPLITK_MATMUL = 3,
    CATLASS_STREAMK_MATMUL = 4,
    CATLASS_NO_PADDING_SINGLECORE_SPLITK_MATMUL = 5,
    CATLASS_SINGLECORE_SPLITK_SMALL_MATMUL = 6,
    CATLASS_SINGLECORE_SPLITK_KLOOP_OUTER_MATMUL = 7,
    CATLASS_SINGLECORE_SPLITK_KLOOP_MIDDLE_MATMUL = 8,
    CATLASS_AIV_MATMUL = 9,
};

enum class CatlassLayoutTag : uint32_t
{
    CATLASS_ROW_MAJOR = 0,
    CATLASS_COLUMN_MAJOR = 1,
};

enum class CatlassAivDispatchPolicy : uint32_t
{
    CATLASS_MATMUL_AIV_DEFAULT = 0,
    CATLASS_MATMUL_AIV_SIMPLE = 1,
    CATLASS_MATMUL_AIV_TRANS = 2,
};
} // namespace CatlassDynamicMatmul

namespace DynamicKernel {
template <class AType, class BType, class CType, class BiasType = void>
struct TileCopyDynamicOptimized :
    public Catlass::Gemm::Tile::TileCopy<Catlass::Arch::AtlasA2, AType, BType, CType, BiasType> {
    using ArchTag = Catlass::Arch::AtlasA2;
    using CopyGmToL1A = typename Catlass::Gemm::Tile::CopyGmToL1DynamicOptimized<ArchTag, AType>;
    using CopyGmToL1B = typename Catlass::Gemm::Tile::CopyGmToL1DynamicOptimized<ArchTag, BType>;
};

template <class Element, class PaddingBuilder, Catlass::Gemm::Kernel::PaddingTag paddingTag>
__aicore__ inline size_t PaddingWorkspaceSize(uint32_t in, uint32_t out, uint32_t inL1, uint32_t outL1)
{
    size_t sizeW = 0;
    if constexpr (paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
        sizeW = PaddingBuilder::Padding::GetWorkspaceSize(in, out, inL1, outL1);
    } else if constexpr (paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
        sizeW = PaddingBuilder::Padding::GetWorkspaceSize(in, out, 512 / sizeof(Element));
    } else if constexpr (paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
        sizeW = PaddingBuilder::Padding::GetWorkspaceSize(in, out);
    }
    return sizeW;
}
} // namespace DynamicKernel

#endif // CATLASS_KERNEL_UTILS_H