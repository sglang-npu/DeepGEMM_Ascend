#!/bin/bash

# 检查参数是否正确
if [ $# -lt 2 ]; then
    echo "usage: $0 <python_script> <num_runs> [--operator_type TYPE] [--core_num NUM] [其他参数...]"
    echo ""
    echo "参数说明:"
    echo "  python_script: Python脚本路径（如 benchmark_catlass.py）"
    echo "  num_runs: 进程数量（1-8之间的整数）"
    echo "  --operator_type: 算子类型（可选），可选值: smallmatmul"
    echo "  --core_num: AI Core数量（可选），默认20"
    echo "  --shapes_file: shapes.xlsx文件路径（可选），如果提供则从文件读取shape"
    echo "  其他参数: 会原样传递给Python脚本"
    echo ""
    echo "示例:"
    echo "  $0 benchmark_catlass.py 8"
    echo "  $0 benchmark_catlass.py 8 --operator_type smallmatmul"
    echo "  $0 benchmark_catlass.py 8 --operator_type smallmatmul --core_num 20"
    echo "  $0 benchmark_catlass.py 8 --operator_type smallmatmul --core_num 20 --shapes_file ./shapes.xlsx"
    echo "  $0 benchmark_catlass.py 8 --operator_type smallmatmul --core_num 20 --catlass_bin_path /path/to/bin"
    exit 1
fi

PYTHON_SCRIPT="$1"
NUM_RUNS="$2"
MONITOR_INTERVAL=5  # 监控间隔时间，单位：秒

# 验证NUM_RUNS是否为1-8之间的整数
if ! [[ "$NUM_RUNS" =~ ^[1-8]$ ]]; then
    echo "错误: 进程数量必须是1到8之间的整数"
    exit 1
fi

# 检查脚本文件是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: $PYTHON_SCRIPT 不存在"
    exit 1
fi

# 提取额外的参数（从第3个参数开始）
shift 2  # 移除前两个参数
EXTRA_ARGS="$@"  # 保存所有额外参数

# 使用关联数组存储进程ID和对应的rank_id
declare -A PIDS

# 启动所有进程
echo "============================================="
echo "启动 $NUM_RUNS 个进程"
if [ -n "$EXTRA_ARGS" ]; then
    echo "额外参数: $EXTRA_ARGS"
fi
echo "============================================="

for ((i=0; i<NUM_RUNS; i++)); do
    echo "启动进程 $i"
    # 构建完整的命令，包含所有参数
    if [ -n "$EXTRA_ARGS" ]; then
        python3 "$PYTHON_SCRIPT" --rank_id $((i+1)) --process_num $NUM_RUNS $EXTRA_ARGS &
    else
        python3 "$PYTHON_SCRIPT" --rank_id $((i+1)) --process_num $NUM_RUNS &
    fi
    pid=$!
    PIDS[$i]=$pid
    echo "进程 $i 启动，PID: $pid"
done

# 监控进程并自动重启
echo ""
echo "开始监控进程，每隔 $MONITOR_INTERVAL 秒检查一次..."
echo "按 Ctrl+C 停止监控"
echo ""

while true; do
    # 遍历所有进程
    for rank_id in "${!PIDS[@]}"; do
        pid=${PIDS[$rank_id]}
        # 检查进程是否存在
        if ! ps -p $pid > /dev/null 2>&1; then
            echo "[$(date +%H:%M:%S)] 进程 $rank_id (PID: $pid) 已终止，正在重启..."
            # 重启进程时同样传递所有参数
            if [ -n "$EXTRA_ARGS" ]; then
                python3 "$PYTHON_SCRIPT" --rank_id $rank_id --process_num $NUM_RUNS $EXTRA_ARGS &
            else
                python3 "$PYTHON_SCRIPT" --rank_id $rank_id --process_num $NUM_RUNS &
            fi
            new_pid=$!
            PIDS[$rank_id]=$new_pid
            echo "[$(date +%H:%M:%S)] 进程 $rank_id 已重启，新PID: $new_pid"
        fi
    done
    # 等待指定的监控间隔
    sleep $MONITOR_INTERVAL
done
