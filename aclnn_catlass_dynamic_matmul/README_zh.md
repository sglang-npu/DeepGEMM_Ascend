# aclnn catlass dynamic matmul
本目录可以编译产出以catlass的dynamic matmul为算子基础得到的aclnn接口

## 1.编译
根据实际环境选择device类型，目前可配置为`a2`、`a3`两种

```bash
bash build.sh --device-type <device>
```
运行前要修改 `build.sh` 中 `PYTHON_INCLUDE_DIR` 和 `PYTHON_LIBRARY_DIR` 变量为实际路径

编译产物是目录 `msopgen/build_out` 下的run包，直接运行即可安装

## 2.使用
aclnn接口可以配合pytorch使用，也可独立在c++中调用

调用时，需要注意环境变量 `ASCEND_CUSTOM_OPP_PATH` 的配置，若安装run包时为默认安装，则无需配置该变量

## 3.注意事项
本产品为测试产品，可能有部分bug未解决，可能需要自行处理

## 4.脚本参数
```bash
--device-type <device> 设备类型 配置选项为 'a2'/'a3'
--clean 清理第三方产物 清理后退出脚本
```