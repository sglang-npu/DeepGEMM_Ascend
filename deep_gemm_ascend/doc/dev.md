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

# set home dir to project root path
source set_env.sh
```

## 3.compile
```bash
# build python api
cd $DGA_ROOT_DIR/
bash build.sh

# build benchmark bin
cd $DGA_ROOT_DIR/deep_gemm_ascend/benchmark_msprof/
bash build.sh

```

## 4.runtime
```bash
# execute python test
cd $DGA_ROOT_DIR/deep_gemm_ascend/framework/tests/
python3 test.py

# execute python benchmark test
cd $DGA_ROOT_DIR/deep_gemm_ascend/framework/tests/
python3 bench_main.py \
    --m 96 --n 1536 --k 5952  --m_sections 1 --n_sections 1  --m_sec_o_blocks 3 --n_sec_o_blocks 8 --k_o_iter_blocks 20 --db_o_blocks 10

# execute python benchmark
cd $DGA_ROOT_DIR/deep_gemm_ascend/framework/benchmark/
bash multi_start.sh benchmark.py 8
```