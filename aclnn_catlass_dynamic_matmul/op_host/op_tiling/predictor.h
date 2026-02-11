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

#ifndef CATLASS_ACLNN_TILING_PREDICTOR_H
#define CATLASS_ACLNN_TILING_PREDICTOR_H

#include <string>
#include <vector>
#include <Python.h>
#include "tiling_params.h"

namespace CatlassTiling {
class Predictor {
public:
    static Predictor& GetInstance()
    {
        static Predictor predictorInstance;
        return predictorInstance;
    }

    void Predict(TilingParams& tilingParams);

private:
    void Init();
    Predictor();
    ~Predictor();
    Predictor(const Predictor&) = delete;
    Predictor &operator=(const Predictor&) = delete;

    std::vector<uint32_t> UsePredictApi(uint32_t m, uint32_t n, uint32_t k);

private:
    PyObject* pInstance_ = nullptr;
    PyObject* pMethod_ = nullptr;
    bool inPython_ = true;
};
} // namespace CatlassTiling

#endif // CATLASS_ACLNN_TILING_PREDICTOR_H