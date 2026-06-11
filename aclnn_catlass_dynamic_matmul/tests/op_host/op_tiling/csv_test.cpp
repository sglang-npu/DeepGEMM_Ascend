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
#include <fstream>
#include <cstdio>
#include "csv.h"

using namespace CatlassTiling::CSV;

class CsvTest : public ::testing::Test {
protected:
    std::string testFilePath = "test_csv_temp.csv";
    
    void SetUp() override {
        CreateTestFile();
    }
    
    void TearDown() override {
        std::remove(testFilePath.c_str());
    }
    
    void CreateTestFile() {
        std::ofstream file(testFilePath);
        file << "m,n,k,m1,n1,k1,kernelSerial,paddingTagA,paddingTagB,paddingTagC,blockDim\n";
        file << "512,512,512,128,256,256,0,0,0,0,24\n";
        file << "1024,1024,1024,256,256,256,1,1,0,0,24\n";
        file.close();
    }
    
    void CreateEmptyFile() {
        std::ofstream file(testFilePath);
        file.close();
    }
};

TEST_F(CsvTest, Document_InitFromFile_Test) {
    Document doc(testFilePath);
    
    EXPECT_TRUE(doc.IsEnable());
    EXPECT_EQ(doc.GetRowCount(), 2);
}

TEST_F(CsvTest, Document_GetRowCount_Test) {
    Document doc(testFilePath);
    
    EXPECT_EQ(doc.GetRowCount(), 2);
}

TEST_F(CsvTest, Document_GetCell_String_Test) {
    Document doc(testFilePath);
    
    std::string mValue = doc.GetCell<std::string>("m", 0);
    EXPECT_EQ(mValue, "512");
    
    std::string nValue = doc.GetCell<std::string>("n", 1);
    EXPECT_EQ(nValue, "1024");
}

TEST_F(CsvTest, Document_GetCell_UInt32_Test) {
    Document doc(testFilePath);
    
    uint32_t mValue = doc.GetCell<uint32_t>("m", 0);
    EXPECT_EQ(mValue, 512);
    
    uint32_t kValue = doc.GetCell<uint32_t>("k", 1);
    EXPECT_EQ(kValue, 1024);
}

TEST_F(CsvTest, Document_SaveRow_Test) {
    Document doc(testFilePath);
    
    std::vector<std::string> newRow = {"2048", "2048", "2048", "512", "512", "512", "2", "0", "0", "0", "24"};
    doc.SaveRow(newRow);
    
    Document updatedDoc(testFilePath);
    EXPECT_EQ(updatedDoc.GetRowCount(), 3);
    
    uint32_t mValue = updatedDoc.GetCell<uint32_t>("m", 2);
    EXPECT_EQ(mValue, 2048);
}

TEST_F(CsvTest, Document_InitRowHead_Test) {
    std::string emptyFilePath = "test_csv_empty.csv";
    std::remove(emptyFilePath.c_str());
    
    Document doc(emptyFilePath);
    
    std::remove(emptyFilePath.c_str());
}

TEST_F(CsvTest, Document_GetCell_InvalidColumn_Test) {
    Document doc(testFilePath);
    
    EXPECT_THROW(doc.GetCell<std::string>("invalid_column", 0), std::out_of_range);
}

TEST_F(CsvTest, Document_GetCell_InvalidRow_Test) {
    Document doc(testFilePath);
    
    EXPECT_THROW(doc.GetCell<std::string>("m", 999), std::out_of_range);
}