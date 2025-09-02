#pragma once

#include "handle.hpp"
#include "../utils/system.hpp"

namespace deep_gemm_ascend {

class KernelRuntime final {
public:
    LibraryHandle binHandle = nullptr;
    KernelHandle kernel = nullptr;

    explicit KernelRuntime(const std::filesystem::path& binPath) {
        // todo 1 check param
        // todo 2 parse param
        // fixme 2.5 build code
        std::string codeStrPath = binPath.string();
        std::string rootPath = "/home/q30063557/code/dga/";
        std::string srcPath = rootPath + "kernel_files/";
        std::string buildPath = rootPath + "kernel_files/kernel_code_1/build/";

        std::string command = "cmake " + srcPath +
            " -B " + buildPath +
            " -DSOC_VERSION=Ascend910B3 " +
            " -DKERNEL_SRC_PATH=" + codeStrPath +
            " && cmake --build " + buildPath;
        std::cout << "run command: " << command << std::endl;
        const auto& [return_code, output] = call_external_command(command);
        std::cout << "run command result: " << return_code << std::endl;
        // std::cout << "run command output: " << output.c_str() << std::endl;
        std::string binNewPath = rootPath + "kernel_files/out/fatbin/mmad_kernels/mmad_kernels.o";
        std::cout << "bin path : " << binNewPath.c_str() << std::endl;

        // 3 load file
        kernel = load_kernel(binNewPath.c_str(), "mmad_custom", &binHandle);
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