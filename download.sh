#!/bin/bash
set -e

DGA_ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
THIRD_DIR=${DGA_ROOT_DIR}/deep_gemm_ascend/third-party

if [[ -e "${THIRD_DIR}/fmt" ]]; then
    echo "third party fmt directory already exit, skip to download."
    exit 0
else
    echo "fmt directory is not exit, create new one."
    mkdir -p "${THIRD_DIR}/fmt"
fi
echo "get fmt from git."
cd "${THIRD_DIR}"
git clone https://github.com/fmtlib/fmt.git