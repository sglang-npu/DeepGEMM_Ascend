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

