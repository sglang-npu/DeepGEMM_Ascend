#ifndef GEMM_BENCH_HPP
#define GEMM_BENCH_HPP

#include <iostream>
#include <vector>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include "../../jit/get_best_config.hpp"
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../../jit/handle.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../jit/compiler.hpp"
#include "../../jit/generate_code.hpp"
#include "../../utils/format.hpp"

namespace deep_gemm_ascend {
class BenchRuntime final : public LaunchRuntime<BenchRuntime> {
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
        code += utils::FILE_BENCH;
        code += utils::FILE_EXEC;
        return code;
    }
};

static void mmad_bench(const at::Tensor &x, const at::Tensor &y, at::Tensor &z, at::Tensor params)
{
    // 0 check input
    uint32_t m_sections = params[0].item<int>();
    uint32_t n_sections = params[1].item<int>();
    uint32_t m_sec_o_blocks = params[2].item<int>();
    uint32_t n_sec_o_blocks = params[3].item<int>();
    uint32_t k_o_iter_blocks = params[4].item<int>();
    uint32_t db_o_blocks = params[5].item<int>();
    // 1 generate args
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t blockDim = m_sections * n_sections;
    uint32_t m = x.size(0);
    uint32_t n = y.size(1);
    uint32_t k = y.size(0);
    const auto& config = get_bench_config(m, n, k,
        m_sections, n_sections, m_sec_o_blocks, n_sec_o_blocks, k_o_iter_blocks, db_o_blocks);

    // 2 generate code
    std::vector<uint32_t> config_list_ori{config.m, config.k, config.n, config.batch,
        config.k_iters, config.m_blocks, config.n_blocks, config.k_blocks, config.m_sc_blocks, config.n_sc_blocks,
        config.m_o_fix, config.n_o_fix, config.k_o_fix, config.db_o_num, config.m_parts, config.n_parts,
        config.r_m_parts, config.r_n_parts, config.r_m_blocks, config.r_n_blocks, config.r_k_blocks, config.r_db_num};

    std::vector<int> config_list;
    config_list.reserve(config_list_ori.size());
    for (auto v : config_list_ori) {
        config_list.push_back(static_cast<int>(v));
    }

    at::Tensor cpu_tensor = at::from_blob(config_list.data(), {static_cast<int64_t>(config_list.size())}, at::kInt);
    at::Tensor new_data = cpu_tensor.to(params.device());
    params.slice(0, 6, 28).copy_(new_data);
    // std::cout << "params is: " << params << std::endl;

    BenchRuntime::Args args;
    const auto& code = BenchRuntime::generate(args);
    // 3 build code
    const Compiler::CompileArgs& compile_args = {
        .m = 1, .n = 1, .k = 1,
        .kernelType = 1,
    };
    const std::string& kernel_name = "mmad_custom";
    auto runtime = compiler->build(code, compile_args, kernel_name);

    // 4 launch kernel
    KernelHandle kernel = nullptr;
    LaunchArgsHandle argsHandle = nullptr;
    LaunchParamHandle paramHandle = nullptr;
    kernel = runtime->kernel;
    CHECK_ACL(aclrtKernelArgsInit(kernel, &argsHandle));
    auto xDevice = const_cast<void *>(x.storage().data());
    auto yDevice = const_cast<void *>(y.storage().data());
    auto zDevice = const_cast<void *>(z.storage().data());
    auto pDevice = const_cast<void *>(params.storage().data());
    CHECK_ACL(aclrtKernelArgsAppend(argsHandle, (void **)&xDevice, sizeof(uintptr_t), &paramHandle));
    CHECK_ACL(aclrtKernelArgsAppend(argsHandle, (void **)&yDevice, sizeof(uintptr_t), &paramHandle));
    CHECK_ACL(aclrtKernelArgsAppend(argsHandle, (void **)&zDevice, sizeof(uintptr_t), &paramHandle));

    CHECK_ACL(aclrtKernelArgsAppend(argsHandle, (void **)&pDevice, sizeof(uintptr_t), &paramHandle));
    CHECK_ACL(aclrtKernelArgsFinalize(argsHandle));

    CHECK_ACL(aclrtLaunchKernelWithConfig(kernel, blockDim, acl_stream, nullptr, argsHandle, nullptr));
    CHECK_ACL(aclrtSynchronizeStream(acl_stream));
}
}

#endif