# dev 
## 1、pull remote branch
```bash
# pull remote pr & branch 
git fetch origin
git branch

# checkout local branch from remote branch 
git checkout origin/sr_1_dga_framework -b sr_1_dga_framework
git branch -vv
```

## 2、compile 
```bash 
# set home dir to project root path, exec only once
export DGA_ROOT_DIR=$(pwd)

# build python api
cd $DGA_ROOT_DIR/deep_gemm_ascend/framework
rm -rf build/
mkdir build
cmake -B build -DSOC_VERSION="Ascend910B3" \
    -DKERNEL_SRC_PATH=$DGA_ROOT_DIR/deep_gemm_ascend/framework/deep_gemm_ascend/include/impls/mmad.cpp
cmake --build build -j

# build kernel bin
cd $DGA_ROOT_DIR/deep_gemm_ascend/framework/deep_gemm_ascend/include/impls
rm -rf build/ out/
mkdir build
cmake -B build -DSOC_VERSION="Ascend910B3" \
    -DKERNEL_SRC_PATH=$DGA_ROOT_DIR/deep_gemm_ascend/framework/deep_gemm_ascend/include/impls/mmad.cpp
cmake --build build -j
```

## 3、runtime
```bash
# set kernel bin path, exec only once
export KERNEL_BIN_PATH=$(pwd)/out/fatbin/mmad_kernels/mmad_kernels.o
export PYTHONPATH=$DGA_ROOT_DIR/deep_gemm_ascend/framework/build:$PYTHONPATH
export PYTHONPATH=$DGA_ROOT_DIR/deep_gemm_ascend/framework:$PYTHONPATH

# execute python test
cd $DGA_ROOT_DIR/deep_gemm_ascend/framework/tests/
python3 test.py

# ps.we will get wrong result because kernel file is not complete
```