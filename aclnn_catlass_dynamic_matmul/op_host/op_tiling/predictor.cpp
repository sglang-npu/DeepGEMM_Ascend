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

#include "predictor.h"

#include <string>
#include <stdexcept>
#include <string>
#include <Python.h>
#include "utils.h"
#include "tiling_params.h"

namespace {
constexpr uint32_t PARAM_NUM = 8;
const char* PREDICTOR_MODULE_NAME = "get_best_config";
const char* PREDICTOR_CLASS_NAME = "GetBestConfig";
const char* PREDICTOR_FUNC_NAME = "predict";

void ReleasePythonObject(PyObject* pObject) {
    if (pObject) {
        Py_DECREF(pObject);
        pObject = nullptr;
    }
}
}

namespace CatlassTiling {
void Predictor::Init()
{
    // init python interpreter
    if (!Py_IsInitialized()) {
        inPython_ = false;
        Py_Initialize();
        if (!Py_IsInitialized()) {
            throw std::runtime_error("[DGA] [ERROR] initialize Python interpreter failed.");
        }
    }

    // init python module
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* pModule = nullptr;
    PyObject* pClass = nullptr;
    PyObject* pArgs = nullptr;
    try {
        // 1. import module
        pModule = PyImport_ImportModule(PREDICTOR_MODULE_NAME);
        if (!pModule) {
            PyErr_Print();
            throw std::runtime_error("[DGA] [ERROR] import predictor module failed.");
        }
        // 2. import predictor class
        pClass = PyObject_GetAttrString(pModule, PREDICTOR_CLASS_NAME);
        if (!pClass) {
            PyErr_Print();
            throw std::runtime_error("[DGA] [ERROR] import predictor class failed.");
        }
        // 3. create predictor instance
        pArgs = PyTuple_New(0);
        pInstance_ = PyObject_CallObject(pClass, pArgs);
        if (!pInstance_) {
            PyErr_Print();
            throw std::runtime_error("[DGA] [ERROR] create predictor instance failed.");
        }
        // 4. create predictor method
        pMethod_ = PyObject_GetAttrString(pInstance_, PREDICTOR_FUNC_NAME);
        if (!pMethod_ || !PyCallable_Check(pMethod_)) {
            PyErr_Print();
            throw std::runtime_error("[DGA] [ERROR] get or check predictor function failed.");
        }
    } catch (...) {
        ReleasePythonObject(pArgs);
        ReleasePythonObject(pClass);
        ReleasePythonObject(pModule);
        throw;
    }
    ReleasePythonObject(pArgs);
    ReleasePythonObject(pClass);
    ReleasePythonObject(pModule);
    PyGILState_Release(gstate);
}

Predictor::Predictor()
{
    Init();
}

Predictor::~Predictor()
{
    PyGILState_STATE gstate = PyGILState_Ensure();
    ReleasePythonObject(pMethod_);
    ReleasePythonObject(pInstance_);
    PyGILState_Release(gstate);
    if (!inPython_ && Py_IsInitialized()) {
        Py_Finalize();
    }
}

std::vector<uint32_t> Predictor::UsePredictApi(uint32_t m, uint32_t n, uint32_t k)
{
    PyGILState_STATE gstate = PyGILState_Ensure();

    // 1. create input tuple
    PyObject* pArgs = PyTuple_New(3);
    PyTuple_SetItem(pArgs, 0, PyLong_FromLong(m));
    PyTuple_SetItem(pArgs, 1, PyLong_FromLong(n));
    PyTuple_SetItem(pArgs, 2, PyLong_FromLong(k));

    // 2. call function
    PyObject* pResult = PyObject_CallObject(pMethod_, pArgs);
    ReleasePythonObject(pArgs);
    if (!pResult) {
        PyErr_Print();
        PyGILState_Release(gstate);
        throw std::runtime_error("[DGA] [ERROR] call predict function failed.");
    }

    // 3. convert python result
    std::vector<uint32_t> result(PARAM_NUM, 0);
    PyObject* pItem = nullptr;
    for (int i = 0; i < PARAM_NUM; i++) {
        pItem = PyTuple_GetItem(pResult, i);
        result[i] = static_cast<uint32_t>(PyLong_AsLong(pItem));
    }
    ReleasePythonObject(pItem);

    ReleasePythonObject(pResult);
    PyGILState_Release(gstate);
    return result;
}

void Predictor::Predict(TilingParams& tilingParams)
{
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t k = tilingParams.k;

    auto predictRes = UsePredictApi(m, n, k);

    // NOTE: The result should be output in order.
    tilingParams.m1 = static_cast<uint16_t>(predictRes[0]);
    tilingParams.n1 = static_cast<uint16_t>(predictRes[1]);
    tilingParams.k1 = static_cast<uint16_t>(predictRes[2]);
    tilingParams.kernelSerial = static_cast<uint8_t>(predictRes[3]);
    tilingParams.paddingTagA = static_cast<uint8_t>(predictRes[4]);
    tilingParams.paddingTagB = static_cast<uint8_t>(predictRes[5]);
    tilingParams.paddingTagC = static_cast<uint8_t>(predictRes[6]);
    tilingParams.blockDim = static_cast<uint8_t>(predictRes[7]);
}
}