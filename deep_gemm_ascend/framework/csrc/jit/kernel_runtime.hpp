#pragma once

#include "handle.hpp"
#include "../utils/system.hpp"

namespace deep_gemm_ascend {

class KernelRuntime final {
public:
    LibraryHandle binHandle = nullptr;
    KernelHandle kernel = nullptr;

    explicit KernelRuntime(const std::string& binPath) {
        // todo 1 check param
        // todo 2 parse param

        // 3 load file
        kernel = load_kernel(binPath.c_str(), "mmad_custom", &binHandle);
    }

    void ConstructArgs(LaunchArgsHandle &argsHandle, LaunchParamHandle &paramHandle, 
        const at::Tensor &x, const at::Tensor &y, at::Tensor &z)
    {
        construct_launch_args(kernel, argsHandle, paramHandle, x, y, z);
    }

    ~KernelRuntime() noexcept(false) {
        unload_library(binHandle);
    }
};

class LaunchRuntime {
};

} // namespace deep_gemm_ascend