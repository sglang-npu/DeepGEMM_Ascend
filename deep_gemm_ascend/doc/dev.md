# dev 
## 1.pull remote branch
```bash
# pull remote pr & branch 
git fetch origin
git branch

# checkout local branch from remote branch 
git checkout origin/sr_1_dga_framework -b sr_1_dga_framework
git branch -vv
```

## 2.prepare environment
```bash 
# download third library
bash download.sh
```

## 3.compile
```bash 
# set home dir to project root path
source set_env.sh

# build python api
bash build.sh
```

## 4.runtime
```bash
# execute python test
cd $DGA_ROOT_DIR/deep_gemm_ascend/framework/tests/
python3 test.py

# ps.we will get wrong result because kernel file is not complete
```