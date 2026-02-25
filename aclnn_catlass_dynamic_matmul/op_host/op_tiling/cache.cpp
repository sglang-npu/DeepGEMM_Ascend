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

#include "cache.h"
#include <vector>
#include <iostream>
#include "tiling_params.h"
#include "csv.h"
#include "utils.h"


namespace CatlassTiling {
namespace Cache {
TilingCache::TilingCache()
{
    std::string cacheFilePath = GetEnv("CACHE_FILE_PATH");
    Init(cacheFilePath);
}

void TilingCache::Init(const std::string cacheFilePath)
{
    if (cacheFilePath == "") {
        // no cache ENV, will not use cache file
        return;
    }

    std::unique_ptr<CSV::Document> tempDoc(new CSV::Document(cacheFilePath));
    // std::unique_ptr<CSV::Document> tempDoc = std::make_unique<CSV::Document>();
    if (tempDoc == nullptr || !tempDoc->IsEnable()) {
        // Document create failed, will not use cache file
        std::cerr << "[DGA] [ERROR] Create file cache failed." << std::endl;
        return;
    }
    size_t rowCount = tempDoc->GetRowCount();
    for (size_t i = 0; i < rowCount; i++) {
        uint32_t m = tempDoc->GetCell<uint32_t>("m", i);
        uint32_t n = tempDoc->GetCell<uint32_t>("n", i);
        uint32_t k = tempDoc->GetCell<uint32_t>("k", i);
        // todo: layout tag is key
        LayoutTag layoutTagA = LayoutTag::TagRowMajor;
        LayoutTag layoutTagB = LayoutTag::TagColumnMajor;
        LayoutTag layoutTagC = LayoutTag::TagRowMajor;

        TilingParams tempParams(m, n, k, layoutTagA, layoutTagB, layoutTagC);
        tempParams.m1 = tempDoc->GetCell<uint32_t>("m1", i);
        tempParams.n1 = tempDoc->GetCell<uint32_t>("n1", i);
        tempParams.k1 = tempDoc->GetCell<uint32_t>("k1", i);
        tempParams.kernelSerial = tempDoc->GetCell<uint32_t>("kernelSerial", i);
        tempParams.paddingTagA = tempDoc->GetCell<uint32_t>("paddingTagA", i);
        tempParams.paddingTagB = tempDoc->GetCell<uint32_t>("paddingTagB", i);
        tempParams.paddingTagC = tempDoc->GetCell<uint32_t>("paddingTagC", i);
        tempParams.blockDim = tempDoc->GetCell<uint32_t>("blockDim", i);

        CacheKey tempKey = std::make_tuple(m, n, k);
        data_[tempKey] = std::move(tempParams);
    }
    cacheDoc_ = std::move(tempDoc);
    enableFile_ = true;
}

bool TilingCache::GetTiling(TilingParams& tilingParams)
{
    CacheKey tempKey = std::make_tuple(tilingParams.m, tilingParams.n, tilingParams.k);
    if (data_.find(tempKey) != data_.end()) {
        tilingParams = data_[tempKey];
        return true;
    }
    return false;
}

void TilingCache::SetTiling(const TilingParams& tilingParams)
{
    CacheKey tempKey = std::make_tuple(tilingParams.m, tilingParams.n, tilingParams.k);
    if (data_.find(tempKey) != data_.end()) {
        return;
    }
    data_[tempKey] = tilingParams;
    if (enableFile_) {
        std::vector<std::string> rowInput;
        rowInput.push_back(std::to_string(tilingParams.m));
        rowInput.push_back(std::to_string(tilingParams.n));
        rowInput.push_back(std::to_string(tilingParams.k));
        rowInput.push_back(std::to_string(tilingParams.m1));
        rowInput.push_back(std::to_string(tilingParams.n1));
        rowInput.push_back(std::to_string(tilingParams.k1));
        rowInput.push_back(std::to_string(tilingParams.kernelSerial));
        rowInput.push_back(std::to_string(tilingParams.paddingTagA));
        rowInput.push_back(std::to_string(tilingParams.paddingTagB));
        rowInput.push_back(std::to_string(tilingParams.paddingTagC));
        rowInput.push_back(std::to_string(tilingParams.blockDim));
        cacheDoc_->SaveRow(rowInput);
    }
}
} // namespace Cache
} // namespace CatlassTiling
