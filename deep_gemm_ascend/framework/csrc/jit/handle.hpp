#pragma once

#include <filesystem>
#include "../utils/exception.hpp"

namespace deep_gemm_ascend {
// Use AscendC runtime API
using LibraryHandle = aclrtBinHandle;
using KernelHandle = aclrtFuncHandle;
using LaunchArgsHandle = aclrtArgsHandle;
using LaunchParamHandle = aclrtParamHandle;

static KernelHandle load_kernel(const std::filesystem::path &filePath,
    const char *func_name, LibraryHandle *binHandlePtr = nullptr)
{
    LibraryHandle binHandle = nullptr;
    KernelHandle funcHandle = nullptr;
    // filePath = rootPath + "/out/fatbin/mmad_kernels/mmad_kernels.o";
    std::cout << "bin file path is " << filePath.c_str() << std::endl;
    CHECK_ACL(aclrtBinaryLoadFromFile(filePath.c_str(), nullptr, &binHandle));
    CHECK_ACL(aclrtBinaryGetFunction(binHandle, func_name, &funcHandle));

    if (binHandlePtr != nullptr)
        *binHandlePtr = binHandle;
    return funcHandle;
}

static void unload_library(LibraryHandle& binHandle)
{
    CHECK_ACL(aclrtBinaryUnLoad(binHandle));
}

static void construct_launch_args(const KernelHandle& kernel, LaunchArgsHandle& argsHandle,
    LaunchParamHandle& paramHandle, const at::Tensor &x, const at::Tensor &y, at::Tensor &z)
{
    CHECK_ACL(aclrtKernelArgsInit(kernel, &argsHandle));
    auto xDevice = const_cast<void *>(x.storage().data());
    auto yDevice = const_cast<void *>(y.storage().data());
    auto zDevice = const_cast<void *>(z.storage().data());
    CHECK_ACL(aclrtKernelArgsAppend(argsHandle, (void **)&xDevice, sizeof(uintptr_t), &paramHandle));
    CHECK_ACL(aclrtKernelArgsAppend(argsHandle, (void **)&yDevice, sizeof(uintptr_t), &paramHandle));
    CHECK_ACL(aclrtKernelArgsAppend(argsHandle, (void **)&zDevice, sizeof(uintptr_t), &paramHandle));
    CHECK_ACL(aclrtKernelArgsFinalize(argsHandle));
}

static auto launch_kernel(const KernelHandle& kernel, uint32_t blockDim,
    aclrtStream acl_stream, LaunchArgsHandle& argsHandle) 
{
    CHECK_ACL(aclrtLaunchKernelWithConfig(kernel, blockDim, acl_stream, nullptr, argsHandle, nullptr));
    CHECK_ACL(aclrtSynchronizeStream(acl_stream));
}

} // namespace deep_gemm_ascend
