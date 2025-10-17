#!/bin/bash
set -e

DGA_ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR=${DGA_ROOT_DIR}/deep_gemm_ascend/framework/build

export CC=/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin/bisheng
export CXX=/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin/bisheng

cd ${DGA_ROOT_DIR}/deep_gemm_ascend/framework

if [[ -e "${BUILD_DIR}" ]]; then
    echo "clean cmake cache"
    rm -rf "${BUILD_DIR}"
else
    mkdir -p "${BUILD_DIR}"
fi

cmake -B ${BUILD_DIR} -DSOC_VERSION="Ascend910B3"
cmake --build ${BUILD_DIR} -j4