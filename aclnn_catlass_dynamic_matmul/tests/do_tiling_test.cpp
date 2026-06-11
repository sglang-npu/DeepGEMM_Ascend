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
#include "do_tiling.h"
#include "tiling_params.h"
#include "platform_info.h"

using namespace CatlassTiling;

class DoTilingTest : public ::testing::Test {
protected:
    PlatformInfo platformInfo;
    void SetUp() override {
        platformInfo.coreNum = 24;
        platformInfo.l1Size = 512 * 1024;
        platformInfo.l0CSize = 128 * 1024;
    }
};

TEST_F(DoTilingTest, DoTilingLayout00_BasicTest) {
    TilingParams tilingParams(512, 512, 512, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    DoTilingLayout00(tilingParams, platformInfo);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
    EXPECT_GT(tilingParams.k1, 0);
}

TEST_F(DoTilingTest, DoTilingLayout00_LargeKTest) {
    TilingParams tilingParams(512, 512, 65536, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    DoTilingLayout00(tilingParams, platformInfo);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
    EXPECT_GT(tilingParams.k1, 0);
}

TEST_F(DoTilingTest, DoTilingLayout00_SmallNTest) {
    TilingParams tilingParams(512, 64, 256, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    DoTilingLayout00(tilingParams, platformInfo);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
    EXPECT_GT(tilingParams.k1, 0);
}

TEST_F(DoTilingTest, DoTilingLayout01_BasicTest) {
    TilingParams tilingParams(512, 512, 512, 
        LayoutTag::TagRowMajor, LayoutTag::TagRowMajor, LayoutTag::TagRowMajor);
    
    DoTilingLayout01(tilingParams, platformInfo);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
    EXPECT_GT(tilingParams.k1, 0);
}

TEST_F(DoTilingTest, DoTilingLayout01_MGreaterThanNTest) {
    TilingParams tilingParams(1024, 256, 512, 
        LayoutTag::TagRowMajor, LayoutTag::TagRowMajor, LayoutTag::TagRowMajor);
    
    DoTilingLayout01(tilingParams, platformInfo);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
}

TEST_F(DoTilingTest, DoTilingLayout01_NGreaterThanMTest) {
    TilingParams tilingParams(256, 1024, 512, 
        LayoutTag::TagRowMajor, LayoutTag::TagRowMajor, LayoutTag::TagRowMajor);
    
    DoTilingLayout01(tilingParams, platformInfo);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
}

TEST_F(DoTilingTest, DoTilingLayout10_BasicTest) {
    TilingParams tilingParams(512, 512, 512, 
        LayoutTag::TagColumnMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    DoTilingLayout10(tilingParams, platformInfo);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
    EXPECT_GT(tilingParams.k1, 0);
}

TEST_F(DoTilingTest, DoTilingLayout10_MGreaterThanNTest) {
    TilingParams tilingParams(1024, 256, 512, 
        LayoutTag::TagColumnMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    DoTilingLayout10(tilingParams, platformInfo);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
}

TEST_F(DoTilingTest, DoTilingLayout10_LargeMTest) {
    TilingParams tilingParams(65536, 256, 512, 
        LayoutTag::TagColumnMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    DoTilingLayout10(tilingParams, platformInfo);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
}

TEST_F(DoTilingTest, DoTilingLayout11_BasicTest) {
    TilingParams tilingParams(512, 512, 512, 
        LayoutTag::TagColumnMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    DoTilingLayout11(tilingParams, platformInfo);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
    EXPECT_GT(tilingParams.k1, 0);
}

TEST_F(DoTilingTest, DoTilingLayout11_LargeMTest) {
    TilingParams tilingParams(1024, 512, 512, 
        LayoutTag::TagColumnMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    DoTilingLayout11(tilingParams, platformInfo);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
}

TEST_F(DoTilingTest, DoTilingLayout11_SmallMTest) {
    TilingParams tilingParams(64, 512, 512, 
        LayoutTag::TagColumnMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    DoTilingLayout11(tilingParams, platformInfo);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
}