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

#ifndef CATLASS_TILING_CSV_H
#define CATLASS_TILING_CSV_H

#include <vector>
#include <string>
#include <map>

namespace CatlassTiling {
namespace CSV {
class Document {
public:
    Document(const std::string& filePath)
    {
        Init(filePath);
    }

    size_t GetRowCount() const;

    template <typename T = std::string>
    T GetCell(const std::string &columnName, const size_t rowIdx) const;

    void SaveRow(const std::vector<std::string>& rowInput, const char delimiter = ',');

    bool IsEnable()
    {
        return enable_;
    }

private:
    void Init(const std::string& filePath);

    bool InitFromFile(const std::string& filePath, const char delimiter = ',');

    bool InitRowHead(const char delimiter = ',');

private:
    std::string srcFilePath_;
    std::vector<std::string> csvHead_;
    std::map<std::string, size_t> columnNameToIndex;
    std::vector<std::vector<std::string>> data_;
    bool enable_;
};
} // namespace CSV
} // namespace CatlassTiling

#endif // CATLASS_TILING_CSV_H