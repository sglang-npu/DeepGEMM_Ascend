#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "../framework/csrc/jit/get_best_config.hpp"

#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)
#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

constexpr uint32_t PARAMS_NUM = 11;

bool String2UInt32(const char *str, uint32_t *result)
{
    if (str == nullptr || *str == '\0') {
        return false;
    }
    uint32_t value = 0;
    const char *ptr = str;

    while (*ptr != '\0') {
        uint32_t digit = *ptr - '0';
        value *= 10;
        value += digit;
        ptr++;
    }
    *result = value;
    return true;
}

bool ParseInputParams(int32_t argc, char *argv[], std::vector<uint32_t> &param_list)
{
    if (argc != PARAMS_NUM) {
        ERROR_LOG("params num is lower than %d", PARAMS_NUM);
        return false;
    }

    std::vector<uint32_t> params;
    for (size_t i = 2; i < PARAMS_NUM; i++) {
        uint32_t temp;
        if (!String2UInt32(argv[i], &temp)) {
            ERROR_LOG("convert argv[%zu] failed", i);
            return false;
        }
        params.push_back(temp);
    }

    uint32_t m = params[0];
    uint32_t n = params[1];
    uint32_t k = params[2];
    uint32_t m_sections = params[3];
    uint32_t n_sections = params[4];
    uint32_t m_sec_o_blocks = params[5];
    uint32_t n_sec_o_blocks = params[6];
    uint32_t k_o_iter_blocks = params[7];
    uint32_t db_o_blocks = params[8];

    const auto& config = deep_gemm_ascend::get_bench_config(m, n, k,
        m_sections, n_sections, m_sec_o_blocks, n_sec_o_blocks, k_o_iter_blocks, db_o_blocks);

    std::vector<uint32_t> config_list{
        config.m, config.n, config.k,
        config.m_sections, config.n_sections, config.m_sec_o_blocks, config.n_sec_o_blocks,
        config.k_o_iter_blocks, config.db_o_blocks, config.batch,
        config.k_iters, config.m_blocks, config.n_blocks, config.k_blocks, config.m_sc_blocks, config.n_sc_blocks,
        config.m_o_fix, config.n_o_fix, config.k_o_fix, config.db_o_num, config.m_parts, config.n_parts,
        config.r_m_parts, config.r_n_parts, config.r_m_blocks, config.r_n_blocks, config.r_k_blocks, config.r_db_num
    };
    param_list = config_list;
    return true;
}


