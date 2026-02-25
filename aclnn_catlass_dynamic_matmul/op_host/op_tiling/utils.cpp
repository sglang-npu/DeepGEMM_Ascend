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

#include "utils.h"
#include <vector>
#include "tiling_params.h"
#include "platform_info.h"

namespace CatlassTiling {
uint32_t CeilDiv(uint32_t dividend, uint32_t divisor)
{
    return (dividend + divisor - 1) / divisor;
}

uint32_t RoundUp(uint32_t val, uint32_t align)
{
    return (val + align - 1) / align * align;
}

void BalanceWorkload(uint32_t m, uint32_t n, uint32_t& m1, uint32_t& n1, uint32_t threshold, PlatformInfo& platformInfo)
{
    uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), platformInfo.coreNum);
    while (m1 > threshold && (CeilDiv(m, m1 - 16) * CeilDiv(n, n1) <= maxBlocks)) {
        m1 -= 16;
    }
    if (m < m1) {
        m1 = RoundUp(m, 16);
    }
    if (n < n1) {
        n1 = RoundUp(n, 16);
    }
}

void SetTile(TilingParams &tilingParams, uint32_t m1, uint32_t n1, uint32_t k1)
{
    // To save space, tiling parameters (m1, n1, k1) are stored as uint16_t.
    tilingParams.m1 = static_cast<uint16_t>(m1);
    tilingParams.n1 = static_cast<uint16_t>(n1);
    tilingParams.k1 = static_cast<uint16_t>(k1);
}

bool IsExStrideLimit(uint32_t rows, uint32_t cols, uint32_t layoutTag)
{
    if (static_cast<LayoutTag>(layoutTag) == LayoutTag::TagColumnMajor) {
        return rows >= 65536;
    } else {
        return cols >= 65536;
    }
}

bool JudgeSpace(uint32_t m1, uint32_t n1, uint32_t k1, PlatformInfo& platformInfo, uint32_t dataSize)
{
    bool judgeL1 = (m1 * k1 * 2 * dataSize + k1 * n1 * 2 * dataSize <= platformInfo.l1Size);
    bool judgeL0C = (m1 * n1 * 4 <= platformInfo.l0CSize) ? true : false;
    return judgeL1 && judgeL0C;
}

uint32_t GetMaxK1(uint32_t m1, uint32_t n1, PlatformInfo& platformInfo, uint32_t dataSize)
{
    std::vector<uint32_t> k1List = {1024, 512, 256, 128};
    uint32_t k1 = 512 / dataSize;
    for (const auto &k1t : k1List) {
        if (JudgeSpace(m1, n1, k1t, platformInfo, dataSize)) {
            k1 = k1t;
            break;
        }
    }
    return k1;
}

std::string GetEnv(const std::string& name)
{
    const char* envVal = std::getenv(name.c_str());
    return (envVal != nullptr) ? std::string(envVal) : "";
}
}