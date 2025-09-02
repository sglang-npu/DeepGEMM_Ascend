#pragma once

#include <exception>
#include <string>
#include <sstream>

namespace deep_gemm_ascend {

class DGAException final : public std::exception {
    std::string message = {};

public:
    explicit DGAException(const char *name, const char* file, const int line, const std::string& error) {
        message = std::string(name) + " error (" + file + ":" + std::to_string(line) + "): " + error;
    }

    const char *what() const noexcept override {
        return message.c_str();
    }
};

#ifndef DGA_STATIC_ASSERT
#define DGA_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

// #ifndef DGA_HOST_ASSERT
// #define DGA_HOST_ASSERT(cond) \
// do { \
//     if (not (cond)) { \
//         throw DGAException("Assertion", __FILE__, __LINE__, #cond); \
//     } \
// } while (0)
// #endif

#ifndef DGA_HOST_ASSERT
#define DGA_HOST_ASSERT(cond) \
do { \
    if (not (cond)) { \
        std::cerr << __FILE__ << ":" << __LINE__ << " dgaError: " << #cond << std::endl; \
    } \
} while (0)
#endif

#ifndef CHECK_ACL
#define CHECK_ACL(cond) \
    do { \
        aclError __ret = cond; \
        if (__ret != ACL_ERROR_NONE) { \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        } \
    } while (0);
#endif

} // namespace deep_gemm_ascend