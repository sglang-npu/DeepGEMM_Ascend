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
#include "cache.h"
#include "tiling_params.h"

using namespace CatlassTiling::Cache;

class CacheTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
    void TearDown() override {
    }
};

TEST_F(CacheTest, GetInstance_Test) {
    TilingCache& cache1 = TilingCache::GetInstance();
    TilingCache& cache2 = TilingCache::GetInstance();
    
    EXPECT_EQ(&cache1, &cache2);
}

TEST_F(CacheTest, GetTiling_NotFoundTest) {
    TilingCache& cache = TilingCache::GetInstance();
    TilingParams tilingParams(9999, 9999, 9999, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    bool found = cache.GetTiling(tilingParams);
    
    EXPECT_FALSE(found);
}

TEST_F(CacheTest, SetTiling_BasicTest) {
    TilingCache& cache = TilingCache::GetInstance();
    TilingParams tilingParams(512, 512, 512, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    tilingParams.m1 = 128;
    tilingParams.n1 = 256;
    tilingParams.k1 = 256;
    
    cache.SetTiling(tilingParams);
    
    TilingParams retrievedParams(512, 512, 512, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    bool found = cache.GetTiling(retrievedParams);
    
    EXPECT_TRUE(found);
    EXPECT_EQ(retrievedParams.m1, tilingParams.m1);
    EXPECT_EQ(retrievedParams.n1, tilingParams.n1);
    EXPECT_EQ(retrievedParams.k1, tilingParams.k1);
}

TEST_F(CacheTest, SetTiling_DuplicateTest) {
    TilingCache& cache = TilingCache::GetInstance();
    TilingParams tilingParams1(1024, 1024, 1024, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    tilingParams1.m1 = 128;
    tilingParams1.n1 = 256;
    
    cache.SetTiling(tilingParams1);
    
    TilingParams tilingParams2(1024, 1024, 1024, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    tilingParams2.m1 = 256;
    tilingParams2.n1 = 512;
    
    cache.SetTiling(tilingParams2);
    
    TilingParams retrievedParams(1024, 1024, 1024, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    cache.GetTiling(retrievedParams);
    
    EXPECT_EQ(retrievedParams.m1, tilingParams1.m1);
    EXPECT_EQ(retrievedParams.n1, tilingParams1.n1);
}

TEST_F(CacheTest, GetTiling_MultipleEntriesTest) {
    TilingCache& cache = TilingCache::GetInstance();
    
    TilingParams params1(256, 256, 256, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    params1.m1 = 64;
    cache.SetTiling(params1);
    
    TilingParams params2(512, 512, 512, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    params2.m1 = 128;
    cache.SetTiling(params2);
    
    TilingParams retrieved1(256, 256, 256, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    EXPECT_TRUE(cache.GetTiling(retrieved1));
    EXPECT_EQ(retrieved1.m1, params1.m1);
    
    TilingParams retrieved2(512, 512, 512, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    EXPECT_TRUE(cache.GetTiling(retrieved2));
    EXPECT_EQ(retrieved2.m1, params2.m1);
}