#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "jit_kernels/impls/gemm.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME deep_gemm_cpp
#endif

namespace deep_gemm_ascend {

void run_mmad_custom(const at::Tensor &x, const at::Tensor &y, at::Tensor &z)
{
    mmad_custom(x, y, z);
}

void run_mmad_cache(const at::Tensor &x, const at::Tensor &y, at::Tensor &z, const char *filePath)
{
    mmad_cache(x, y, z, filePath);
}

void grouped_gemm_int8_int8_bf16_nt(const at::Tensor &x, const at::Tensor &y, at::Tensor &z, const char *filePath)
{
    std::cout << "now in deepgemm ascend code!!!" << std::endl;
}
}

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepGEMM C++ library";

    m.def("run_mmad_custom", &deep_gemm_ascend::run_mmad_custom, "");
    m.def("run_mmad_cache", &deep_gemm_ascend::run_mmad_cache, "");
    m.def("grouped_gemm_int8_int8_bf16_nt", &deep_gemm_ascend::grouped_gemm_int8_int8_bf16_nt, "");
}
