/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "golden.hpp"
#include "helper.hpp"
#include "catlass/layout/layout.hpp"
#include "dynamic_optimized_matmul.h"

enum class RunMode : uint32_t {
    OriginCatlass = 0,
    CustomTiling = 1,
    OnlinePredict = 2,
    OfflineCache = 3
};

bool ParseArgs(int argc, const char **argv, TilingParams &tilingParams, RunMode &runMode, bool &checkRes)
{
    if (argc < 9) {
        std::cerr << "Wrong Usage! Please check the arguments." << std::endl;
        return false;
    }
    uint32_t run_mode = std::atoi(argv[2]);
    runMode = static_cast<RunMode>(run_mode);
    uint32_t check_res = std::atoi(argv[3]);
    checkRes = (check_res != 0);

    uint32_t m = std::atoi(argv[4]);
    uint32_t n = std::atoi(argv[5]);
    uint32_t k = std::atoi(argv[6]);
    LayoutTag layoutTagA = static_cast<LayoutTag>(std::atoi(argv[7]));
    LayoutTag layoutTagB = static_cast<LayoutTag>(std::atoi(argv[8]));
    LayoutTag layoutTagC = LayoutTag::TagRowMajor;
    tilingParams.SetParams(m, n, k, layoutTagA, layoutTagB, layoutTagC);

    switch (runMode) {
        case RunMode::OriginCatlass:
            // program, device, run_mode==0, checkRes, m, n, k, layoutTagA, layoutTagB
            break;
        case RunMode::CustomTiling:
            // program, device, run_mode==1, checkRes, m, n, k, layoutTagA, layoutTagB, m1, n1, k1
            tilingParams.m1 = std::atoi(argv[9]);
            tilingParams.n1 = std::atoi(argv[10]);
            tilingParams.k1 = std::atoi(argv[11]);
            break;
        case RunMode::OnlinePredict:
            // program, device, run_mode==2, checkRes, m, n, k, layoutTagA, layoutTagB
            break;
        case RunMode::OfflineCache:
            // program, device, run_mode==3, checkRes, m, n, k, layoutTagA, layoutTagB
            break;
        default:
            std::cerr << "[DGA] Invalid run mode: " << run_mode << std::endl;
            return false;
    }
    return true;

}

bool DoTiling(TilingParams &tilingParams, const RunMode &runMode)
{
    PlatformInfo platformInfo;
    bool tilingSucc = false;

    switch (runMode) {
        case RunMode::OriginCatlass:
            DoTilingAndSelectKernel<fp16_t>(tilingParams, platformInfo);
            tilingSucc = true;
            break;
        case RunMode::CustomTiling:
            SelectKernel<fp16_t>(tilingParams, platformInfo);
            tilingSucc = true;
            break;
        case RunMode::OnlinePredict:
            tilingSucc = SelectKernelWithPredictor(tilingParams);
            break;
        case RunMode::OfflineCache:
            tilingSucc = SelectKernelWithCache(tilingParams);
            break;
        default:
            break;
    }
    PrintTilingParams<fp16_t>(tilingParams, platformInfo);
    return tilingSucc;
}

void CheckRes(TilingParams &tilingParams, std::vector<fp16_t> &hostA, std::vector<fp16_t> &hostB, std::vector<fp16_t> &hostC)
{
    std::cout << "[DGA] Check result... " << std::endl;
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t k = tilingParams.k;
    LayoutTag layoutTagA = static_cast<LayoutTag>(tilingParams.layoutTagA);
    LayoutTag layoutTagB = static_cast<LayoutTag>(tilingParams.layoutTagB);
    LayoutTag layoutTagC = static_cast<LayoutTag>(tilingParams.layoutTagC);
    size_t lenC = static_cast<size_t>(m) * n;

    std::vector<float> hostGolden(lenC);
    Catlass::GemmCoord problemShape{m, n, k};
    if (layoutTagA == LayoutTag::TagRowMajor && layoutTagB == LayoutTag::TagRowMajor) {
        Catlass::layout::RowMajor layoutA{m, k};
        Catlass::layout::RowMajor layoutB{k, n};
        Catlass::layout::RowMajor layoutC{m, n};
        Catlass::golden::ComputeMatmul(problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);
    } else if (layoutTagA == LayoutTag::TagRowMajor && layoutTagB == LayoutTag::TagColumnMajor) {
        Catlass::layout::RowMajor layoutA{m, k};
        Catlass::layout::ColumnMajor layoutB{k, n};
        Catlass::layout::RowMajor layoutC{m, n};
        Catlass::golden::ComputeMatmul(problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);
    } else if (layoutTagA == LayoutTag::TagColumnMajor && layoutTagB == LayoutTag::TagRowMajor) {
        Catlass::layout::ColumnMajor layoutA{m, k};
        Catlass::layout::RowMajor layoutB{k, n};
        Catlass::layout::RowMajor layoutC{m, n};
        Catlass::golden::ComputeMatmul(problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);
    } else {
        Catlass::layout::ColumnMajor layoutA{m, k};
        Catlass::layout::ColumnMajor layoutB{k, n};
        Catlass::layout::RowMajor layoutC{m, n};
        Catlass::golden::ComputeMatmul(problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);
    }
    std::vector<uint64_t> errorIndices = Catlass::golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }
}

void RunCompute(aclrtStream &stream, TilingParams &tilingParams, const RunMode &runMode, bool checkRes)
{
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t k = tilingParams.k;
    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;

    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeC = lenC * sizeof(fp16_t);

    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    std::vector<fp16_t> hostC(lenC);

    Catlass::golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);
    Catlass::golden::FillRandomData<fp16_t>(hostB, -5.0f, 5.0f);

    uint8_t *dA, *dB, *dC, *dW, *dTilingParams;

    ACL_CHECK(aclrtMalloc((void **)&dA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&dB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&dC, sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&dTilingParams, sizeof(TilingParams), ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMemcpy(dA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    size_t workspaceSize = DynamicOptimizedMatmulGetWorkspace(tilingParams);
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc((void **)&dW, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    ACL_CHECK(aclrtMemcpy(
        dTilingParams, sizeof(TilingParams), &tilingParams, sizeof(TilingParams), ACL_MEMCPY_HOST_TO_DEVICE));

    ExecuteDynamicOptimizedMatmul(stream, fftsAddr, dA, dB, dC, dW, dTilingParams, tilingParams);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, dC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    if (checkRes) {
        CheckRes(tilingParams, hostA, hostB, hostC);
    }

    ACL_CHECK(aclrtFree(dA));
    ACL_CHECK(aclrtFree(dB));
    ACL_CHECK(aclrtFree(dC));
    ACL_CHECK(aclrtFree(dTilingParams));
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(dW));
    }
}

int main(int argc, const char **argv)
{
    const uint32_t deviceId = std::atoi(argv[1]);
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclInit(nullptr));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    TilingParams tilingParams;
    RunMode runMode;
    bool checkRes;
    if (!ParseArgs(argc, argv, tilingParams, runMode, checkRes)) {
        std::cerr << "[DGA] Parse arguments failed." << std::endl;
        return -1;
    }
    if (!DoTiling(tilingParams, runMode)) {
        std::cerr << "[DGA] Get tiling params failed." << std::endl;
        return -1;
    }
    RunCompute(stream, tilingParams, runMode, checkRes);

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
}
