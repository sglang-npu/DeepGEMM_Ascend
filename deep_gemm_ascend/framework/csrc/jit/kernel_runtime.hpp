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

template <typename Derived>
class LaunchRuntime {
public:
    template <typename Args>
    static std::string generate(const Args& args) {
        const auto& code = Derived::generate_impl(args);
        return code;
    }

    // template <typename Args>
    // static void launch(const std::shared_ptr<KernelRuntime>& kernel_runtime, const Args& args) {
        // const auto& kernel = kernel_runtime->kernel;
        // const auto& stream = at::cuda::getCurrentCUDAStream();
        // const LaunchArgs& launch_args = args.launch_args;

        // const dim3& grid_dim = {static_cast<unsigned>(launch_args.grid_dim.first),
        //                         static_cast<unsigned>(launch_args.grid_dim.second),
        //                         1};
        // const dim3& block_dim = {static_cast<unsigned>(launch_args.num_threads), 1, 1};
        // auto config = construct_launch_config(kernel, stream, launch_args.smem_size,
        //                                       grid_dim, block_dim, launch_args.cluster_dim);

        // // Launch in the derived class
        // if (get_env<int>("DG_JIT_DEBUG")) {
        //     printf("Launch kernel with {%d, %d} x %d, shared memory: %d bytes, cluster: %d, stream: %ld\n",
        //            launch_args.grid_dim.first, launch_args.grid_dim.second, launch_args.num_threads,
        //            launch_args.smem_size, launch_args.cluster_dim, stream.id());
        // }
        // Derived::launch_impl(kernel, config, args);
    // }
};


} // namespace deep_gemm_ascend