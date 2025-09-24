#!/bin/bash

# 检查参数是否正确
if [ $# -ne 2 ]; then
    echo "usage: $0 <python_script> <num_runs>"
    echo "示例: $0 my_script.py 8"
    echo "注意: num_runs必须是1到8之间的整数"
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

# 使用关联数组存储进程ID和对应的rank_id
declare -A PIDS

# 启动所有进程
for ((i=0; i<NUM_RUNS; i++)); do
    echo "启动进程 $i"
    # 将NUM_RUNS和rank_id都传递给Python程序
    python3 "$PYTHON_SCRIPT" --rank_id $i --process_num $NUM_RUNS &
    pid=$!
    PIDS[$i]=$pid
    echo "进程 $i 启动，PID: $pid"
done

# 监控进程并自动重启
echo "开始监控进程，每隔 $MONITOR_INTERVAL 秒检查一次..."
while true; do
    # 遍历所有进程
    for rank_id in "${!PIDS[@]}"; do
        pid=${PIDS[$rank_id]}
        # 检查进程是否存在
        if ! ps -p $pid > /dev/null; then
            echo "进程 $rank_id (PID: $pid) 已终止，正在重启..."
            # 重启进程时同样传递两个参数
            python3 "$PYTHON_SCRIPT" --rank_id $rank_id --process_num $NUM_RUNS &
            new_pid=$!
            PIDS[$rank_id]=$new_pid
            echo "进程 $rank_id 已重启，新PID: $new_pid"
        fi
    done
    # 等待指定的监控间隔
    sleep $MONITOR_INTERVAL
done
