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
#include "predictor.h"
#include "tiling_params.h"

using namespace CatlassTiling;

class PredictorTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
    void TearDown() override {
    }
};

TEST_F(PredictorTest, GetInstance_Test) {
    Predictor& predictor1 = Predictor::GetInstance();
    Predictor& predictor2 = Predictor::GetInstance();
    
    EXPECT_EQ(&predictor1, &predictor2);
}

TEST_F(PredictorTest, Predict_BasicTest) {
    TilingParams tilingParams(512, 512, 512, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    Predictor::GetInstance().Predict(tilingParams);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
    EXPECT_GT(tilingParams.k1, 0);
}

TEST_F(PredictorTest, Predict_SmallMatrixTest) {
    TilingParams tilingParams(64, 64, 128, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    Predictor::GetInstance().Predict(tilingParams);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
}

TEST_F(PredictorTest, Predict_LargeKTest) {
    TilingParams tilingParams(512, 512, 8192, 
        LayoutTag::TagRowMajor, LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor);
    
    Predictor::GetInstance().Predict(tilingParams);
    
    EXPECT_GT(tilingParams.k1, 0);
}

TEST_F(PredictorTest, Predict_DifferentLayoutTest) {
    TilingParams tilingParams(512, 512, 512, 
        LayoutTag::TagColumnMajor, LayoutTag::TagRowMajor, LayoutTag::TagRowMajor);
    
    Predictor::GetInstance().Predict(tilingParams);
    
    EXPECT_GT(tilingParams.m1, 0);
    EXPECT_GT(tilingParams.n1, 0);
}