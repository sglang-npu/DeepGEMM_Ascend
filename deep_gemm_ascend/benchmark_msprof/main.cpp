/**
 * @file main.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

/**
usage:
    export LD_LIBRARY_PATH=$(pwd)/build/lib:$LD_LIBRARY_PATH
    ./ascendc_kernels_bbit [rank_id] [m] [n] [k] [m_sections] [n_sections] [m_sec_o_blocks] [n_sec_o_blocks] [k_o_iter_blocks] [db_o_blocks]
 */
#include "acl/acl.h"
#include "benchmark_util.h"

#include "aclrtlaunch_mmad_custom.h"

int32_t main(int32_t argc, char *argv[])
{
    // 0 init acl
    uint32_t rank_id = 0;
    String2UInt32(argv[1], &rank_id);
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(rank_id));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    // 1 parse param

    std::vector<uint32_t> params;
    ParseInputParams(argc, argv, params);
    uint32_t m = params[0];
    uint32_t n = params[1];
    uint32_t k = params[2];
    uint32_t m_sections = params[3];
    uint32_t n_sections = params[4];

    size_t paramSize = 28 * sizeof(uint32_t);
    uint32_t *pHost;
    uint32_t *pDevice;
    // CHECK_ACL(aclrtMallocHost((void **)(&pHost), paramSize));
    pHost = params.data();
    CHECK_ACL(aclrtMalloc((void **)&pDevice, paramSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(pDevice, paramSize, pHost, paramSize, ACL_MEMCPY_HOST_TO_DEVICE));

    size_t aSize = m * k * sizeof(int16_t); // uint16_t represent half
    size_t bSize = k * n * sizeof(int16_t); // uint16_t represent half
    size_t cSize = m * n * sizeof(float);
    uint32_t blockDim = m_sections * n_sections;

    uint8_t *aHost;
    uint8_t *aDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&aHost), aSize));
    CHECK_ACL(aclrtMalloc((void **)&aDevice, aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x1_gm.bin", aSize, aHost, aSize);
    CHECK_ACL(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *bHost;
    uint8_t *bDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&bHost), bSize));
    CHECK_ACL(aclrtMalloc((void **)&bDevice, bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x2_gm.bin", bSize, bHost, bSize);
    CHECK_ACL(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *cHost;
    uint8_t *cDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&cHost), cSize));
    CHECK_ACL(aclrtMalloc((void **)&cDevice, cSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ACLRT_LAUNCH_KERNEL(mmad_custom)(blockDim, stream, aDevice, bDevice, cDevice, pDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(cHost, cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output.bin", cHost, cSize);

    CHECK_ACL(aclrtFree(pDevice));
    // CHECK_ACL(aclrtFreeHost(pHost));
    CHECK_ACL(aclrtFree(aDevice));
    CHECK_ACL(aclrtFreeHost(aHost));
    CHECK_ACL(aclrtFree(bDevice));
    CHECK_ACL(aclrtFreeHost(bHost));
    CHECK_ACL(aclrtFree(cDevice));
    CHECK_ACL(aclrtFreeHost(cHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(rank_id));
    CHECK_ACL(aclFinalize());
    return 0;
}
