#pragma once

#include <string>
#include "../utils/system.hpp"

namespace deep_gemm_ascend {

class Compiler {
public:
    Compiler() {
    }

    virtual ~Compiler() = default;

    std::shared_ptr<KernelRuntime> build(const std::string& codePath) const {
        // todo 1 get from cache
        // 2 compile code
        std::string binPath;
        compile(codePath, binPath);
        // 3 create runtime
        std::shared_ptr<KernelRuntime> runtime = std::make_shared<KernelRuntime>(binPath);
        return runtime;
    }

    virtual void compile(const std::string& codePath, std::string& binPath) const = 0;
};

class CMakeCompiler final: public Compiler {
    std::string rootPath_;
    std::string socVersion_;
public:
    CMakeCompiler() {
        rootPath_ = "/home/q30063557/code/dga/";
        socVersion_ = "Ascend910B3";
    }

    void compile(const std::string& codePath, std::string& binPath) const override {
        std::string cmakePath = rootPath_ + "kernel_files/";
        std::string buildPath = codePath + "_build/";

        std::string command = "cmake " + cmakePath +
            " -B " + buildPath +
            " -DSOC_VERSION=" + socVersion_ +
            " -DKERNEL_SRC_PATH=" + codePath +
            " && cmake --build " + buildPath;
        std::cout << "run command: " << command << std::endl;
        const auto& [return_code, output] = call_external_command(command);
        binPath = rootPath_ + "kernel_files/out/fatbin/mmad_kernels/mmad_kernels.o";
        std::cout << "bin path : " << binPath.c_str() << std::endl;
    }

    ~CMakeCompiler() = default;
};

} // namespace deep_gemm_ascend