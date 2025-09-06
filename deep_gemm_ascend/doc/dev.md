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
source set_env.sh

# download third library
cd $DGA_ROOT_DIR/deep_gemm_ascend/third-party
bash download.sh

# build python api
cd $DGA_ROOT_DIR/deep_gemm_ascend/framework
rm -rf build/
mkdir build
cmake -B build -DSOC_VERSION="Ascend910B3" \
    -DKERNEL_SRC_PATH=$DGA_ROOT_DIR/deep_gemm_ascend/framework/deep_gemm_ascend/include/impls/mmad.cpp
cmake --build build -j
```

## 3、runtime
```bash
# execute python test
cd $DGA_ROOT_DIR/deep_gemm_ascend/framework/tests/
python3 test.py

# ps.we will get wrong result because kernel file is not complete
```