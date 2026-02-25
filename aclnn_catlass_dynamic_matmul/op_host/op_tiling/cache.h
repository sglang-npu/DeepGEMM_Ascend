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

#ifndef CATLASS_ACLNN_TILING_CACHE_H
#define CATLASS_ACLNN_TILING_CACHE_H

#include <map>
#include <memory>

#include "tiling_params.h"
#include "tools/csv.h"

using CacheKey = std::tuple<uint32_t, uint32_t, uint32_t>;
using CacheValue = TilingParams;

namespace CatlassTiling {
namespace Cache {
class TilingCache {
public:
    bool GetTiling(TilingParams& tilingParams);

    void SetTiling(const TilingParams& tilingParams);

    static TilingCache& GetInstance()
    {
        static TilingCache cacheInstance;
        return cacheInstance;
    }

private:
    void Init(const std::string cacheFilePath);

    TilingCache();

    ~TilingCache() {}
    TilingCache(const TilingCache&) = delete;
    TilingCache &operator=(const TilingCache&) = delete;

private:
    std::map<CacheKey, CacheValue> data_;
    std::unique_ptr<CSV::Document> cacheDoc_ = nullptr;
    bool enableFile_ = false;
};

} // namespace Cache
} // namespace CatlassTiling

#endif // CATLASS_ACLNN_TILING_CACHE_H