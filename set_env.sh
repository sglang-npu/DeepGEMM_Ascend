SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export DGA_ROOT_DIR=${SCRIPT_DIR}

export PYTHONPATH=$DGA_ROOT_DIR/deep_gemm_ascend/framework/build:$PYTHONPATH
export PYTHONPATH=$DGA_ROOT_DIR/deep_gemm_ascend/framework:$PYTHONPATH

export LD_LIBRARY_PATH=$DGA_ROOT_DIR/deep_gemm_ascend/benchmark_msprof/build/lib:$LD_LIBRARY_PATH
