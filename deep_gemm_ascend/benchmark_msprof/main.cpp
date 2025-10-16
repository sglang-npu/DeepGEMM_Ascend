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

    // todo - read left input_tensor 

    // todo - read right input_tensor

    return 0
}
