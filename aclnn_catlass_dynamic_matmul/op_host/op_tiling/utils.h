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

#ifndef CATLASS_ACLNN_UTILS_H
#define CATLASS_ACLNN_UTILS_H

#include <string>
#include <acl/acl.h>
#include "tiling_params.h"
#include "platform_info.h"

namespace CatlassTiling {
uint32_t CeilDiv(uint32_t dividend, uint32_t divisor);

uint32_t RoundUp(uint32_t val, uint32_t align);

void BalanceWorkload(uint32_t m, uint32_t n, uint32_t& m1, uint32_t& n1, uint32_t threshold, PlatformInfo& platformInfo);

void SetTile(TilingParams &tilingParams, uint32_t m1, uint32_t n1, uint32_t k1);

bool IsExStrideLimit(uint32_t rows, uint32_t cols, uint32_t layoutTag);

bool JudgeSpace(uint32_t m1, uint32_t n1, uint32_t k1, PlatformInfo& platformInfo, uint32_t dataSize);

uint32_t GetMaxK1(uint32_t m1, uint32_t n1, PlatformInfo& platformInfo, uint32_t dataSize);

std::string GetEnv(const std::string& name);
} // namespace CatlassTiling

extern "C" int rtGetC2cCtrlAddr(uint64_t *, uint32_t *);

// Macro function for unwinding rt errors.
#define RT_CHECK(status)                                                                    \
    do {                                                                                    \
        int32_t error = status;                                                             \
        if (error != 0) {                                                                   \
            std::cerr << __FILE__ << ":" << __LINE__ << "rtError:" << error << std::endl;   \
        }                                                                                   \
    } while(0)

#endif  // CATLASS_ACLNN_UTILS_H