#ifndef GEMM_HPP
#define GEMM_HPP

#include <torch/extension.h>

#include "acl/acl.h"
#include "aclrtlaunch_mmad_custom.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../../jit/handle.hpp"
#include "../../jit/kernel_runtime.hpp"

namespace deep_gemm_ascend {
static void mmad_custom(const at::Tensor &x, const at::Tensor &y, at::Tensor &z)
{
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t blockDim = 8;
    ACLRT_LAUNCH_KERNEL(mmad_custom)(
        blockDim,
        acl_stream,
        const_cast<void *>(x.storage().data()),
        const_cast<void *>(y.storage().data()),
        const_cast<void *>(z.storage().data())
    );
}

static void mmad_cache(const at::Tensor &x, const at::Tensor &y, at::Tensor &z, const std::filesystem::path &binPath)
{
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream acl_stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&acl_stream));
    uint32_t blockDim = 8;
    
    LibraryHandle binHandle = nullptr;
    KernelHandle kernel = nullptr;
    LaunchArgsHandle argsHandle = nullptr;
    LaunchParamHandle paramHandle = nullptr;

    kernel = load_kernel(binPath, "mmad_custom", &binHandle);
    construct_launch_args(kernel, argsHandle, paramHandle, x, y, z);

    CHECK_ACL(aclrtLaunchKernelWithConfig(kernel, blockDim, acl_stream, nullptr, argsHandle, nullptr));
    CHECK_ACL(aclrtSynchronizeStream(acl_stream));

    unload_library(binHandle);
    CHECK_ACL(aclrtDestroyStream(acl_stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
}

static void mmad_rtc(const at::Tensor &x, const at::Tensor &y, at::Tensor &z, const std::filesystem::path &codePath)
{
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t blockDim = 8;

    KernelHandle kernel = nullptr;
    LaunchArgsHandle argsHandle = nullptr;
    LaunchParamHandle paramHandle = nullptr;
    KernelRuntime runtime(codePath);

    runtime.ConstructArgs(argsHandle, paramHandle, x, y, z);
    kernel = runtime.kernel;

    CHECK_ACL(aclrtLaunchKernelWithConfig(kernel, blockDim, acl_stream, nullptr, argsHandle, nullptr));
    CHECK_ACL(aclrtSynchronizeStream(acl_stream));
}
}

#endif