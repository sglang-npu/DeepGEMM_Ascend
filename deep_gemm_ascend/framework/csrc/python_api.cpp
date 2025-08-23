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

void test_api()
{
    return;
}

void run_mmad_cache(const at::Tensor &x, const at::Tensor &y, at::Tensor &z, const char *filePath)
{
    mmad_cache(x, y, z, filePath);
}
}

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepGEMM C++ library";

    m.def("run_mmad_custom", &deep_gemm_ascend::run_mmad_custom, "");
    m.def("test_api", &deep_gemm_ascend::test_api, "");
    m.def("run_mmad_cache", &deep_gemm_ascend::run_mmad_cache, "");
}
