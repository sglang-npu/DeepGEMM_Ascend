#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export DGA_ROOT_DIR=${SCRIPT_DIR}

export CC=/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin/bisheng
export CXX=/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin/bisheng

export PYTHONPATH=$DGA_ROOT_DIR/deep_gemm_ascend/framework/build:$PYTHONPATH
export PYTHONPATH=$DGA_ROOT_DIR/deep_gemm_ascend/framework:$PYTHONPATH
