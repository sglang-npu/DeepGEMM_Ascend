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

void run_mmad_rtc(const at::Tensor &x, const at::Tensor &y, at::Tensor &z)
{
    mmad_rtc(x, y, z);
}

void run_mmad_bench(const at::Tensor &x, const at::Tensor &y, at::Tensor &z, at::Tensor &params)
{
    mmad_bench(x, y, z, params);
}
}

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepGEMM C++ library";

    m.def("run_mmad_custom", &deep_gemm_ascend::run_mmad_custom, "");
    m.def("run_mmad_rtc", &deep_gemm_ascend::run_mmad_rtc, "");
    m.def("run_mmad_bench", &deep_gemm_ascend::run_mmad_bench, "");
}
