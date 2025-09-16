#pragma once

#include <string>
#include "cache.hpp"
#include "../utils/system.hpp"
#include "../utils/format.hpp"
#include "../utils/constant.hpp"

namespace deep_gemm_ascend {
class Compiler {
    std::string rootPath_;
    std::string socVersion_;
public:
    struct CompileArgs {
        uint32_t m, n, k;
        uint32_t kernelType;
    };
    Compiler() {
        rootPath_ = get_env<std::string>("DGA_ROOT_DIR");
        socVersion_ = "Ascend910B3";
    }

    virtual ~Compiler() = default;

    std::shared_ptr<KernelRuntime> build(const std::string& code,
        const CompileArgs& compile_args, const std::string& kernel_name) const 
    {
        // 1 get from cache
        const auto kernel_signature = fmt::format("m{}n{}k{}_type{}",
            compile_args.m, compile_args.n, compile_args.k, compile_args.kernelType);
        std::string kernel_dir = fmt::format("{}/deep_gemm_ascend/cache/kernel_{}/", rootPath_, kernel_signature);
        if (const auto& runtime = kernel_runtime_cache->get(kernel_dir); runtime != nullptr) {
            std::cout << "use kernel cache" << std::endl;
            return runtime;
        }
        std::cout << "compile new kernel" << std::endl;

        // 2 compile new cache
        // 2.1 put code to code path
        std::filesystem::create_directories(kernel_dir);
        std::filesystem::path code_path = kernel_dir + utils::KERNEL_CODE_NAME;
        OutputKernelFile(code, code_path);

        // 2.2 compile code
        compile(kernel_dir);

        // 3 create runtime
        const auto& runtime = kernel_runtime_cache->get(kernel_dir);
        // todo: assert runtime not be nullptr
        return runtime;
    }

    virtual void compile(const std::string& kernel_dir) const = 0;
private:
    void OutputKernelFile(const std::string& code, const std::filesystem::path& code_path) const
    {
        // put code to code path
        std::cout << "put code to code path: " << code_path << std::endl;
        std::ofstream ofs(code_path);
        ofs << code;
        ofs.close();
    }
};

class CMakeCompiler final: public Compiler {
public:
    CMakeCompiler() {
        rootPath_ = get_env<std::string>("DGA_ROOT_DIR");
        socVersion_ = "Ascend910B3";
    }

    void compile(const std::string& kernel_dir) const override {
        std::string cmakePath = rootPath_ + "/deep_gemm_ascend/cache/";
        std::string buildPath = kernel_dir + "/build/";

        // 1. compile kernel code
        std::string command = "cmake " + cmakePath +
            " -B " + buildPath +
            " -DSOC_VERSION=" + socVersion_ +
            " -DKERNEL_SRC_PATH=" + kernel_dir + utils::KERNEL_CODE_NAME +
            " && cmake --build " + buildPath;
        std::cout << "run cmake command: " << command << std::endl;
        const auto& [return_code, output] = call_external_command(command);
        std::cout << "run cmake command result: " << return_code << std::endl;
        // std::cout << "run cmake command output: " << output << std::endl;
    
        // 2. copy kernel bin to code directory
        std::string bin_path = cmakePath + "/out/fatbin/mmad_kernels/" + utils::KERNEL_FATBIN_NAME;
        CopyBinFile(bin_path, kernel_dir);
        std::cout << "bin path : " << bin_path.c_str() << std::endl;
    }

    ~CMakeCompiler() = default;
private:
    void CopyBinFile(const std::string& bin_path, const std::string& kernel_dir) const
    {
        std::string command = "cp -f " + bin_path + " " + kernel_dir;
        std::cout << "run move command: " << command << std::endl;
        const auto& [return_code, output] = call_external_command(command);
        std::cout << "run move command result: " << return_code << std::endl;
        // std::cout << "run move command output: " << output << std::endl;
    }

    std::string rootPath_;
    std::string socVersion_;
};

static auto compiler = std::make_shared<CMakeCompiler>();
} // namespace deep_gemm_ascend