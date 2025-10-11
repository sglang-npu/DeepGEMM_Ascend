#ifndef GEMM_HPP
#define GEMM_HPP

#include <torch/extension.h>
#include "../../jit/get_best_config.hpp"
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
        uint32_t k_iters;
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
    uint32_t blockDim = 1;
    uint32_t batch = x.size(0);
    uint32_t m = x.size(1);
    uint32_t n = y.size(2);
    uint32_t k = y.size(1);
    const auto& config = get_best_config(batch, m, n, k);

    KernelHandle kernel = nullptr;
    LaunchArgsHandle argsHandle = nullptr;
    LaunchParamHandle paramHandle = nullptr;
    // 2 generate code
    const RTCRuntime::Args args = {
        .k_iters = config.k_iters, .batch = config.batch,
        .m = config.m, .n = config.n, .k = config.k,
        .m_sections = config.m_sections, .n_sections = config.n_sections,
        .m_blocks = config.m_blocks, .n_blocks = config.n_blocks, .k_blocks = config.k_blocks,
        .m_sc_blocks = config.m_sc_blocks, .n_sc_blocks = config.n_sc_blocks,
        .m_sec_o_blocks = config.m_sec_o_blocks, .n_sec_o_blocks = config.n_sec_o_blocks,
        .k_o_iter_blocks = config.k_o_iter_blocks, .db_o_blocks = config.db_o_blocks,
        .m_o_fix = config.m_o_fix, .n_o_fix = config.n_o_fix, .k_o_fix = config.k_o_fix, .db_o_num = config.db_o_num,
        .m_parts = config.m_parts, .n_parts = config.n_parts, .r_m_parts = config.r_m_parts, .r_n_parts = config.r_n_parts,
        .r_m_blocks = config.r_m_blocks, .r_n_blocks = config.r_n_blocks, .r_k_blocks = config.r_k_blocks, .r_db_num = config.r_db_num,
    };
    const auto& code = RTCRuntime::generate(args);
    // 3 build code
    const Compiler::CompileArgs& compile_args = {
        .m = m, .n = n, .k = k,
        .kernelType = 0,
    };
    const std::string& kernel_name = "mmad_custom";
    auto runtime = compiler->build(code, compile_args, kernel_name);

    // 4 launch kernel
    runtime->ConstructArgs(argsHandle, paramHandle, x, y, z);
    kernel = runtime->kernel;

    CHECK_ACL(aclrtLaunchKernelWithConfig(kernel, blockDim, acl_stream, nullptr, argsHandle, nullptr));
    CHECK_ACL(aclrtSynchronizeStream(acl_stream)); // 阻塞应用程序，直到指定 Stream 中的所有任务都完成。
}
}

#endif