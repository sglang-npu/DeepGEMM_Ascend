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

#include "csv.h"

#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace {
std::vector<std::string> gRowName = {
    "m", "n", "k", "m1", "n1", "k1",
    "kernelSerial", "paddingTagA", "paddingTagB", "paddingTagC", "blockDim"
};
}

namespace CatlassTiling {
namespace CSV {
void Document::Init(const std::string& filePath)
{
    bool initFileSuc = InitFromFile(filePath);
    bool initRowHeadSuc = InitRowHead();
    enable_ = initFileSuc && initRowHeadSuc;
}

bool Document::InitFromFile(const std::string& filePath, const char delimiter)
{
    srcFilePath_ = filePath;
    std::fstream file(srcFilePath_, std::ios::in | std::ios::out);
    if (!file.is_open()) {
        std::cerr << "[DGA] [ERROR] Open csv file failed. File path is " << srcFilePath_ << std::endl;
        return false;
    }

    std::string line;
    std::getline(file, line);
    std::istringstream headerStream(line);
    std::string header;
    size_t index = 0;
    while (std::getline(headerStream, header, delimiter)) {
        columnNameToIndex[header] = index++;
        csvHead_.push_back(header);
    }

    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        std::string cell;
        std::vector<std::string> row;
        row.reserve(index);
        while (std::getline(lineStream, cell, delimiter)) {
            row.push_back(cell);
        }
        data_.push_back(row);
    }
    file.close();
    return true;
}

bool Document::InitRowHead(const char delimiter)
{
    if (csvHead_.size() != 0) {
        return true;
    }
    // init row head to file
    std::ofstream file(srcFilePath_);
    if (!file.is_open()) {
        std::cerr << "[DGA] [ERROR] Failed to open file for writing. File path is " << srcFilePath_ << std::endl;
        return false;
    }

    for (size_t i = 0; i < gRowName.size(); ++i) {
        if (i > 0) {
            file << delimiter;
        }
        file << gRowName[i];
        // init row head to data
        columnNameToIndex[gRowName[i]] = i;
        csvHead_.push_back(gRowName[i]);
    }
    file << "\n";
    file.close();
    return true;
}

size_t Document::GetRowCount() const
{
    return data_.size();
}

void Document::SaveRow(const std::vector<std::string>& rowInput, const char delimiter)
{
    if (rowInput.size() != csvHead_.size()) {
        std::cerr << "[DGA] [ERROR] RowInput size is not equal with csvHead size." << std::endl;
        return;
    }

    std::ofstream file(srcFilePath_, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "[DGA] [ERROR] Failed to open file for writing. File path is " << srcFilePath_ << std::endl;
        return;
    }

    for (size_t i = 0; i < rowInput.size(); ++i) {
        if (i > 0) {
            file << delimiter;
        }
        file << rowInput[i];
    }
    file << "\n";
    file.close();
}

template <>
std::string Document::GetCell(const std::string &columnName, const size_t rowIdx) const
{
    return data_[rowIdx][columnNameToIndex.at(columnName)];
}

template <>
uint32_t Document::GetCell(const std::string &columnName, const size_t rowIdx) const
{
    try {
        return std::stoul(data_[rowIdx][columnNameToIndex.at(columnName)]);
    } catch (...) {
        std::cerr << "[DGA] [ERROR] try to trans " << columnName << " to uint32_t failed." << std::endl;
        return 0;
    }
}
} // namespace CSV
} // namespace CatlassTiling