#!/bin/bash

set -e

REPO_URL="https://gitcode.com/cann/catlass.git"
EXAMPLE_NAME="102_dynamic_optimized_matmul"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR="${SCRIPT_DIR}/catlass"
BUILD_LOG="${SCRIPT_DIR}/build.log"

if [ -z "$1" ]; then
    echo "用法: $0 <patch文件路径>"
    echo "示例: $0 /path/to/changes.patch"
    exit 1
fi

PATCH_FILE="$1"

if [ ! -f "$PATCH_FILE" ]; then
    echo "错误: Patch文件不存在: $PATCH_FILE"
    exit 1
fi

PATCH_FILE=$(realpath "$PATCH_FILE")

echo "=========================================="
echo "步骤 1: 克隆远程仓库"
echo "=========================================="
echo "目标目录: ${REPO_DIR}"

if [ -d "${REPO_DIR}" ]; then
    echo "目录已存在，删除旧目录..."
    rm -rf "${REPO_DIR}"
fi

git clone ${REPO_URL} ${REPO_DIR}

echo ""
echo "=========================================="
echo "步骤 2: 应用Patch文件"
echo "=========================================="
echo "Patch文件: ${PATCH_FILE}"

cd ${REPO_DIR}
git apply "${PATCH_FILE}" || {
    echo "错误: 补丁应用失败"
    exit 1
}

echo ""
echo "=========================================="
echo "步骤 3: 执行构建"
echo "=========================================="
bash scripts/build.sh ${EXAMPLE_NAME} 2>&1 | tee "${BUILD_LOG}"

echo ""
echo "=========================================="
echo "步骤 4: 设置环境变量"
echo "=========================================="

LD_LIBRARY_CMD=$(grep "export LD_LIBRARY_PATH.*catlass/output/shared_lib" "${BUILD_LOG}" | tail -1)

if [ -n "$LD_LIBRARY_CMD" ]; then
    echo "找到环境变量设置命令: ${LD_LIBRARY_CMD}"
    eval "$LD_LIBRARY_CMD"
    echo "已设置 LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
else
    echo "警告: 未找到 LD_LIBRARY_PATH 设置命令"
fi

echo ""
echo "=========================================="
echo "步骤 5: 执行程序"
echo "=========================================="
cd ${REPO_DIR}/output/bin
echo "执行目录: $(pwd)"
echo "执行命令: ./102_dynamic_optimized_matmul 8 1 16 578 2014 0 0 1 16 64 512"
./102_dynamic_optimized_matmul 8 1 16 578 2014 0 0 1 16 64 512

echo ""
echo "=========================================="
echo "执行完成！"
echo "仓库目录: ${REPO_DIR}"
echo "构建日志: ${BUILD_LOG}"
echo "=========================================="
