#!/bin/bash
set -e

BASH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR="${BASH_DIR}"/build

if [[ -e "${BUILD_DIR}" ]]; then
    echo "clean cmake cache"
    rm -rf "${BUILD_DIR}"
else
    mkdir -p "${BUILD_DIR}"
fi

cmake -B "${BUILD_DIR}" -DSOC_VERSION="Ascend910B3"
cmake --build "${BUILD_DIR}" -j4

cp "${BUILD_DIR}"/ascendc_kernels_bbit "${BASH_DIR}"
# cp "${BUILD_DIR}"/lib/libascendc_kernels_npu.so "${BASH_DIR}"
# rm -rf "${BUILD_DIR}"

