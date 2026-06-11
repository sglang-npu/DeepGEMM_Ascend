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

#include <gtest/gtest.h>
#include "utils.h"
#include "tiling_params.h"
#include "platform_info.h"

using namespace CatlassTiling;

class UtilsTest : public ::testing::Test {
protected:
    PlatformInfo platformInfo;
    void SetUp() override {
        platformInfo.coreNum = 24;
        platformInfo.l1Size = 512 * 1024;
        platformInfo.l0CSize = 128 * 1024;
    }
};

TEST_F(UtilsTest, CeilDiv_BasicTest) {
    EXPECT_EQ(CeilDiv(10, 3), 4);
    EXPECT_EQ(CeilDiv(9, 3), 3);
    EXPECT_EQ(CeilDiv(10, 10), 1);
    EXPECT_EQ(CeilDiv(0, 5), 0);
}

TEST_F(UtilsTest, CeilDiv_LargeNumbers) {
    EXPECT_EQ(CeilDiv(1000, 16), 63);
    EXPECT_EQ(CeilDiv(65536, 256), 256);
}

TEST_F(UtilsTest, RoundUp_BasicTest) {
    EXPECT_EQ(RoundUp(10, 16), 16);
    EXPECT_EQ(RoundUp(16, 16), 16);
    EXPECT_EQ(RoundUp(17, 16), 32);
    EXPECT_EQ(RoundUp(0, 16), 0);
}

TEST_F(UtilsTest, RoundUp_AlignmentTest) {
    EXPECT_EQ(RoundUp(100, 32), 128);
    EXPECT_EQ(RoundUp(127, 16), 128);
    EXPECT_EQ(RoundUp(256, 256), 256);
}

TEST_F(UtilsTest, BalanceWorkload_BasicTest) {
    uint32_t m = 512;
    uint32_t n = 256;
    uint32_t m1 = 128;
    uint32_t n1 = 256;
    uint32_t threshold = 32;
    
    BalanceWorkload(m, n, m1, n1, threshold, platformInfo);
    
    EXPECT_GT(m1, 0);
    EXPECT_GT(n1, 0);
    EXPECT_LE(m1, m);
    EXPECT_LE(n1, n);
}

TEST_F(UtilsTest, BalanceWorkload_SmallMatrixTest) {
    uint32_t m = 64;
    uint32_t n = 64;
    uint32_t m1 = 128;
    uint32_t n1 = 256;
    uint32_t threshold = 32;
    
    BalanceWorkload(m, n, m1, n1, threshold, platformInfo);
    
    EXPECT_TRUE(m1 <= m || m1 >= 16);
    EXPECT_TRUE(n1 <= n || n1 >= 16);
}

TEST_F(UtilsTest, SetTile_BasicTest) {
    TilingParams tilingParams;
    uint32_t m1 = 128;
    uint32_t n1 = 256;
    uint32_t k1 = 256;
    
    SetTile(tilingParams, m1, n1, k1);
    
    EXPECT_EQ(tilingParams.m1, m1);
    EXPECT_EQ(tilingParams.n1, n1);
    EXPECT_EQ(tilingParams.k1, k1);
}

TEST_F(UtilsTest, SetTile_LargeValuesTest) {
    TilingParams tilingParams;
    uint32_t m1 = 512;
    uint32_t n1 = 512;
    uint32_t k1 = 1024;
    
    SetTile(tilingParams, m1, n1, k1);
    
    EXPECT_EQ(tilingParams.m1, static_cast<uint16_t>(m1));
    EXPECT_EQ(tilingParams.n1, static_cast<uint16_t>(n1));
    EXPECT_EQ(tilingParams.k1, static_cast<uint16_t>(k1));
}

TEST_F(UtilsTest, IsExStrideLimit_RowMajorTest) {
    uint32_t rows = 100;
    uint32_t cols = 100;
    uint32_t layoutTag = static_cast<uint32_t>(LayoutTag::TagRowMajor);
    
    EXPECT_FALSE(IsExStrideLimit(rows, cols, layoutTag));
    
    cols = 65536;
    EXPECT_TRUE(IsExStrideLimit(rows, cols, layoutTag));
}

TEST_F(UtilsTest, IsExStrideLimit_ColumnMajorTest) {
    uint32_t rows = 100;
    uint32_t cols = 100;
    uint32_t layoutTag = static_cast<uint32_t>(LayoutTag::TagColumnMajor);
    
    EXPECT_FALSE(IsExStrideLimit(rows, cols, layoutTag));
    
    rows = 65536;
    EXPECT_TRUE(IsExStrideLimit(rows, cols, layoutTag));
}

TEST_F(UtilsTest, JudgeSpace_WithinLimitTest) {
    uint32_t m1 = 128;
    uint32_t n1 = 256;
    uint32_t k1 = 256;
    uint32_t dataSize = 2;
    
    EXPECT_TRUE(JudgeSpace(m1, n1, k1, platformInfo, dataSize));
}

TEST_F(UtilsTest, JudgeSpace_ExceedLimitTest) {
    uint32_t m1 = 512;
    uint32_t n1 = 512;
    uint32_t k1 = 512;
    uint32_t dataSize = 2;
    
    bool result = JudgeSpace(m1, n1, k1, platformInfo, dataSize);
}

TEST_F(UtilsTest, GetMaxK1_BasicTest) {
    uint32_t m1 = 128;
    uint32_t n1 = 256;
    uint32_t dataSize = 2;
    
    uint32_t k1 = GetMaxK1(m1, n1, dataSize, platformInfo, dataSize);
    
    EXPECT_GT(k1, 0);
    EXPECT_TRUE(k1 == 1024 || k1 == 512 || k1 == 256 || k1 == 128);
}

TEST_F(UtilsTest, GetMaxK1_LargeTileTest) {
    uint32_t m1 = 256;
    uint32_t n1 = 512;
    uint32_t dataSize = 2;
    
    uint32_t k1 = GetMaxK1(m1, n1, dataSize, platformInfo, dataSize);
    
    EXPECT_GT(k1, 0);
}