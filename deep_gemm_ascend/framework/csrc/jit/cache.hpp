#pragma once

#include <filesystem>
#include <memory>
#include <unordered_map>

#include "kernel_runtime.hpp"

namespace deep_gemm_ascend {

class KernelRuntimeCache {
    std::unordered_map<std::string, std::shared_ptr<KernelRuntime>> cache;

public:
    KernelRuntimeCache() = default;

    std::shared_ptr<KernelRuntime> get(const std::filesystem::path& kernel_dir) {
        // Hit the runtime cache
        const auto& iterator = cache.find(kernel_dir);
        if (iterator != cache.end()) {
            std::cout << "use kernel cache in map" << std::endl;
            return iterator->second;
        }

        if (KernelRuntime::check_validity(kernel_dir)) {
            std::cout << "make kernel cache from dir" << std::endl;
            return cache[kernel_dir] = std::make_shared<KernelRuntime>(kernel_dir);
        }
        return nullptr;
    }
};

static auto kernel_runtime_cache = std::make_shared<KernelRuntimeCache>();

} // namespace deep_gemm_ascend