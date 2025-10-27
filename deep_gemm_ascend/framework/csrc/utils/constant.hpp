#pragma once

#include <string>

namespace deep_gemm_ascend{
namespace utils{
    const std::string KERNEL_CODE_NAME = "mmad.cpp";
    const std::string KERNEL_FATBIN_NAME = "mmad_kernels.o";
}
}