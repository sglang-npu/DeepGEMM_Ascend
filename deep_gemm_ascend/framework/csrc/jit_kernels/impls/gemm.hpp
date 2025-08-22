#ifndef GEMM_HPP
#define GEMM_HPP

#include <torch/extension.h>

#include "aclrtlaunch_mmad_custom.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

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
}

#endif