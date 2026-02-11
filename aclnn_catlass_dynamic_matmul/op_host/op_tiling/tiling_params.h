/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TILING_PARAMS_H
#define TILING_PARAMS_H

#include <cstdint>

enum class LayoutTag : uint8_t { TagRowMajor = 0, TagColumnMajor = 1 };
enum class PaddingTag : uint8_t { PADDING_NONE = 0, PADDING_ND = 1, PADDING_BLOCK_ND = 2, PADDING_NZ = 3 };

struct TilingParams {
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
    // The following parameters are only used in tiling and are not read by the kernel.
    uint8_t layoutTagA{0};
    uint8_t layoutTagB{0};
    uint8_t layoutTagC{0};
    uint8_t paddingTagA{0};
    uint8_t paddingTagB{0};
    uint8_t paddingTagC{0};
    uint8_t blockDim{0};
    uint8_t kernelSerial{0};
    uint8_t dispatchPolicyTag{0};

    TilingParams() {}

    TilingParams(uint32_t m_, uint32_t n_, uint32_t k_, 
        LayoutTag layoutTagA_, LayoutTag layoutTagB_, LayoutTag layoutTagC_)
        : m(m_), n(n_), k(k_), layoutTagA(static_cast<uint8_t>(layoutTagA_)),
          layoutTagB(static_cast<uint8_t>(layoutTagB_)), layoutTagC(static_cast<uint8_t>(layoutTagC_))
    {
        strideA = k;
        strideB = n;
        strideC = n;
        if (layoutTagA_ == LayoutTag::TagColumnMajor) {
            strideA = m;
        }
        if (layoutTagB_ == LayoutTag::TagColumnMajor) {
            strideB = k;
        }
        if (layoutTagC_ == LayoutTag::TagColumnMajor) {
            strideC = m;
        }

        swizzleOffset = 3;
        swizzleDirection = (m > n) ? 0 : 1;
    }
};

#endif // TILING_PARAMS_H
