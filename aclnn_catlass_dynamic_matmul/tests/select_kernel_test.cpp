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
#include "select_kernel.h"
#include "tiling_params.h"
#include "platform_info.h"

using namespace CatlassTiling;

class SelectKernelTest : public ::testing::Test {
protected:
    PlatformInfo platformInfo;
    void SetUp() override {
        platformInfo.coreNum = 24;
        platformInfo.l1Size = 512 * 1024;
        platformInfo.l0CSize = 128 * 1024;
    }
};

TEST_F(SelectKernelTest, SelectKernel_BasicTest) {
    TilingParams tilingParams(512, 512, 512, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    SelectKernel(tilingParams);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
    EXPECT_GT(tilingParams.k1, 0);
    EXPECT_GT(tilingParams.blockDim, 0);
    EXPECT_GE(tilingParams.kernelSerial, 0);
}

TEST_F(SelectKernelTest, SelectKernel_SmallMatrixTest) {
    TilingParams tilingParams(64, 64, 128, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    SelectKernel(tilingParams);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
    EXPECT_GT(tilingParams.k1, 0);
}

TEST_F(SelectKernelTest, SelectKernel_LargeKTest) {
    TilingParams tilingParams(512, 512, 8192, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    SelectKernel(tilingParams);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
    EXPECT_GT(tilingParams.k1, 0);
}

TEST_F(SelectKernelTest, SelectKernel_LargeMTest) {
    TilingParams tilingParams(4096, 512, 512, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    SelectKernel(tilingParams);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
}

TEST_F(SelectKernelTest, SelectKernel_LargeNTest) {
    TilingParams tilingParams(512, 4096, 512, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    SelectKernel(tilingParams);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
}

TEST_F(SelectKernelTest, SelectKernel_DifferentLayoutTest) {
    TilingParams tilingParams(512, 512, 512, 
        LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor, LayoutTag::TagRowMajor);
    
    SelectKernel(tilingParams);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
}

TEST_F(SelectKernelTest, SelectKernelWithCache_BasicTest) {
    TilingParams tilingParams1(512, 512, 512, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    SelectKernelWithCache(tilingParams1);
    
    EXPECT_GT(tilingParams1.m1, 0);
    EXPECT_GT(tilingParams1.n1, 0);
    
    TilingParams tilingParams2(512, 512, 512, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    SelectKernelWithCache(tilingParams2);
    
    EXPECT_EQ(tilingParams2.m1, tilingParams1.m1);
    EXPECT_EQ(tilingParams2.n1, tilingParams1.n1);
}

TEST_F(SelectKernelTest, SelectKernelWithCache_DifferentParamsTest) {
    TilingParams tilingParams1(512, 512, 512, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    SelectKernelWithCache(tilingParams1);
    
    TilingParams tilingParams2(1024, 512, 512, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    SelectKernelWithCache(tilingParams2);
    
    EXPECT_TRUE(tilingParams2.m1 != tilingParams1.m1 || tilingParams2.n == tilingParams1.n);
}

TEST_F(SelectKernelTest, SelectKernel_SingleRowTest) {
    TilingParams tilingParams(1, 512, 512, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    SelectKernel(tilingParams);
    
    EXPECT_GT(tilingParams.n1, 0);
    EXPECT_GT(tilingParams.k1, 0);
}

TEST_F(SelectKernelTest, SelectKernel_SingleColumnTest) {
    TilingParams tilingParams(512, 1, 512, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    SelectKernel(tilingParams);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.k1, 0);
}