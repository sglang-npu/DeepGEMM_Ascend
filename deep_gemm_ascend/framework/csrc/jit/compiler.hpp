#pragma once

#include <string>
#include "../utils/system.hpp"
#include "../utils/format.hpp"

namespace deep_gemm_ascend {
const std::string KERNEL_FILE_NAME = "mmad.cpp";
class Compiler {
    std::string rootPath_;
    std::string socVersion_;
public:
    struct CompileArgs {
        uint32_t m, n, k;
        uint32_t kernelType;
    };
    Compiler() {
        rootPath_ = "/home/t00937989/DeepGEMM_Ascend/";
        socVersion_ = "Ascend910B3";
    }

    virtual ~Compiler() = default;

    std::shared_ptr<KernelRuntime> build(const std::string& code,
        const CompileArgs& compile_args, const std::string& kernel_name) const {
        // todo 1 get from cache
        const auto kernel_signature = fmt::format("m{}n{}k{}_type{}_",
            compile_args.m, compile_args.n, compile_args.k, compile_args.kernelType);
        std::string code_dir = fmt::format("{}/deep_gemm_ascend/cache/kernel_{}", rootPath_, kernel_signature);

        // 2 compile new cache
        // 2.1 put code to code path
        std::filesystem::path code_path = code_dir + KERNEL_FILE_NAME;
//        OutputKernelFile(code, code_path);
        // 2.2 compile code
        std::string bin_dir;
        compile(code_dir, bin_dir);

        // 3 create runtime
        std::shared_ptr<KernelRuntime> runtime = std::make_shared<KernelRuntime>(bin_dir);
        return runtime;
    }

    virtual void compile(const std::string& code_dir, std::string& bin_dir) const = 0;
private:
    void OutputKernelFile(const std::string& code, const std::filesystem::path& code_path) const
    {
        // put code to code path
        std::ofstream ofs(code_path);
        ofs << code;
        ofs.close();
    }
};

class CMakeCompiler final: public Compiler {
private:
    std::string rootPath_;
    std::string socVersion_;
public:
    CMakeCompiler() {
        rootPath_ = "/home/t00937989/DeepGEMM_Ascend/";
        socVersion_ = "Ascend910B3";
    }

    void compile(const std::string& code_dir, std::string& bin_dir) const override {
        std::string cmakePath = rootPath_ + "/deep_gemm_ascend/cache/";
        std::string buildPath = code_dir + "_build/";

        std::string command = "cmake " + cmakePath +
            " -B " + buildPath +
            " -DSOC_VERSION=" + socVersion_ +
            " -DKERNEL_SRC_PATH=" + code_dir + KERNEL_FILE_NAME +
            " && cmake --build " + buildPath;
        std::cout << "run command: " << command << std::endl;
        const auto& [return_code, output] = call_external_command(command);
        std::cout << "run command result: " << return_code << std::endl;
        std::cout << "run command output: " << output << std::endl;
        bin_dir = cmakePath + "/out/fatbin/mmad_kernels/mmad_kernels.o";
        std::cout << "bin path : " << bin_dir.c_str() << std::endl;
    }

    ~CMakeCompiler() = default;
};

} // namespace deep_gemm_ascend