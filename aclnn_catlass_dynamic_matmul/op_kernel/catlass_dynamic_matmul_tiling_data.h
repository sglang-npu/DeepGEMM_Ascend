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

#ifndef CATLASS_DYNAMIC_MATMUL_ACLNN_TILING_DATA_H
#define CATLASS_DYNAMIC_MATMUL_ACLNN_TILING_DATA_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#pragma pack(push, 8)
struct alignas(8) CatlassDynamicMatmulTilingData {
    uint64_t strideA{0};
    uint64_t strideB{0};
    uint64_t strideC{0};
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    uint16_t m1{0};
    uint16_t n1{0};
    uint16_t k1{0};
    uint8_t swizzleOffset{1};
    uint8_t swizzleDirection{0};
    uint16_t splitkFactor{1};
    uint64_t fftsAddr{0};
};
#pragma pack(pop)

#endif // CATLASS_DYNAMIC_MATMUL_ACLNN_TILING_DATA_H