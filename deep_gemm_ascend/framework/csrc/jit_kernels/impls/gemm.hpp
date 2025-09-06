#ifndef GEMM_HPP
#define GEMM_HPP

#include <torch/extension.h>

#include "acl/acl.h"
#include "aclrtlaunch_mmad_custom.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../../jit/handle.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../jit/compiler.hpp"
#include "../../jit/generate_code.hpp"
#include "../../utils/format.hpp"

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

class RTCRuntime final : public LaunchRuntime<RTCRuntime> {
public:
    struct Args {
        int k_iters;
        uint32_t batch;

        uint32_t m, n, k;
        uint32_t m_sections, n_sections;
        uint32_t m_blocks, n_blocks, k_blocks;
        uint32_t m_sc_blocks, n_sc_blocks;

        uint32_t m_sec_o_blocks, n_sec_o_blocks, k_o_iter_blocks, db_o_blocks;

        uint32_t m_o_fix, n_o_fix, k_o_fix;
        uint32_t db_o_num;

        uint32_t m_parts, n_parts;
        uint32_t r_m_parts, r_n_parts;

        uint32_t r_m_blocks, r_n_blocks, r_k_blocks;
        uint32_t r_db_num;
    };
    
    static std::string generate_impl(const Args& args) {
        std::string code = utils::FILE_HEAD;
        code += fmt::format(utils::FILE_ARGS,
            args.k_iters, args.batch, args.m, args.k, args.n,
            args.m_sections, args.n_sections,
            args.m_blocks, args.n_blocks, args.k_blocks,
            args.m_sc_blocks, args.n_sc_blocks,
            args.m_sec_o_blocks, args.n_sec_o_blocks,
            args.k_o_iter_blocks, args.db_o_blocks,
            args.m_o_fix, args.n_o_fix, args.k_o_fix, args.db_o_num,
            args.m_parts, args.n_parts, args.r_m_parts, args.r_n_parts,
            args.r_m_blocks, args.r_n_blocks, args.r_k_blocks, args.r_db_num);
        code += utils::FILE_EXEC;
        return code;
    }
};

static void mmad_rtc(const at::Tensor &x, const at::Tensor &y, at::Tensor &z)
{
    // 0 check input
    // 1 generate args
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t blockDim = 8;

    KernelHandle kernel = nullptr;
    LaunchArgsHandle argsHandle = nullptr;
    LaunchParamHandle paramHandle = nullptr;
    // 2 generate code
    const RTCRuntime::Args args = {
        .k_iters = 19, .batch = 0,
        .m = 96, .n = 1536, .k = 5952, 
        .m_sections = 1, .n_sections = 1,
        .m_blocks = 6, .n_blocks = 96, .k_blocks = 372,
        .m_sc_blocks = 6, .n_sc_blocks = 96,
        .m_sec_o_blocks = 3, .n_sec_o_blocks = 8,
        .k_o_iter_blocks = 20, .db_o_blocks = 10,
        .m_o_fix = 0, .n_o_fix = 0, .k_o_fix = 0, .db_o_num = 2,
        .m_parts = 2, .n_parts = 12, .r_m_parts = 2, .r_n_parts = 12,
        .r_m_blocks = 0, .r_n_blocks = 0, .r_k_blocks = 2, .r_db_num = 2,
    };
    const auto& code = RTCRuntime::generate(args);
    // 3 build code
    std::shared_ptr<CMakeCompiler> compiler = std::make_shared<CMakeCompiler>();

    const Compiler::CompileArgs& compile_args = {
        .m = 96, .n = 1536, .k = 5952,
        .kernelType = 0,
    };
    const std::string& kernel_name = "mmad_custom";
    auto runtime = compiler->build(code, compile_args, kernel_name);

    // 4 launch kernel
    runtime->ConstructArgs(argsHandle, paramHandle, x, y, z);
    kernel = runtime->kernel;

    CHECK_ACL(aclrtLaunchKernelWithConfig(kernel, blockDim, acl_stream, nullptr, argsHandle, nullptr));
    CHECK_ACL(aclrtSynchronizeStream(acl_stream));
}
}

#endif