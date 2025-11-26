#!/bin/bash

# 检查参数是否正确
if [ $# -lt 3 ]; then
    echo "usage: $0 <python_script> <num_runs> <npu_ids> [--operator_type TYPE] [--core_num NUM] [其他参数...]"
    echo ""
    echo "参数说明:"
    echo "  python_script: Python脚本路径（如 catlass_benchmark_cli.py）"
    echo "  num_runs: 进程数量（1-8之间的整数）"
    echo "  npu_ids: NPU设备ID列表，用逗号分隔（如: 0,1,2,3），数量必须与num_runs一致"
    echo "  --operator_type: 算子类型（可选），可选值: smallmatmul"
    echo "  --core_num: AI Core数量（可选），默认20"
    echo "  --shapes_file: shapes.xlsx文件路径（可选），如果提供则从文件读取shape"
    echo "  其他参数: 会原样传递给Python脚本"
    echo ""
    echo "示例:"
    echo "  $0 catlass_benchmark_cli.py 4 0,1,2,3"
    echo "  $0 catlass_benchmark_cli.py 4 0,1,2,3 --operator_type smallmatmul"
    echo "  $0 catlass_benchmark_cli.py 4 0,1,2,3 --operator_type smallmatmul --core_num 20"
    echo "  $0 catlass_benchmark_cli.py 4 0,1,2,3 --operator_type smallmatmul --core_num 20 --shapes_file ./shapes.xlsx"
    echo "  $0 catlass_benchmark_cli.py 4 0,1,2,3 --operator_type smallmatmul --core_num 20 --catlass_bin_path /path/to/bin"
    exit 1
fi

PYTHON_SCRIPT="$1"
NUM_RUNS="$2"
NPU_IDS="$3"
MONITOR_INTERVAL=5  # 监控间隔时间，单位：秒

# 验证NUM_RUNS是否为1-8之间的整数
if ! [[ "$NUM_RUNS" =~ ^[1-8]$ ]]; then
    echo "错误: 进程数量必须是1到8之间的整数"
    exit 1
fi

# 验证NPU_IDS格式
NPU_COUNT=$(echo "$NPU_IDS" | tr ',' '\n' | wc -l)
if [ "$NPU_COUNT" -ne "$NUM_RUNS" ]; then
    echo "错误: NPU ID数量 ($NPU_COUNT) 与进程数量 ($NUM_RUNS) 不一致"
    exit 1
fi

# 检查脚本文件是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: $PYTHON_SCRIPT 不存在"
    exit 1
fi

# 提取额外的参数（从第4个参数开始）
shift 3  # 移除前三个参数
EXTRA_ARGS="$@"  # 保存所有额外参数

# 将脚本路径转换为模块路径，便于 python -m 形式执行
SCRIPT_MODULE=${PYTHON_SCRIPT%.py}
SCRIPT_MODULE=${SCRIPT_MODULE#./}
SCRIPT_MODULE=${SCRIPT_MODULE//\//.}

# 使用关联数组存储进程ID和对应的rank_id，以及完成状态
declare -A PIDS
declare -A COMPLETED

# 启动所有进程
echo "============================================="
echo "启动 $NUM_RUNS 个进程"
echo "NPU IDs: $NPU_IDS"
if [ -n "$EXTRA_ARGS" ]; then
    echo "额外参数: $EXTRA_ARGS"
fi
echo "============================================="

for ((i=0; i<NUM_RUNS; i++)); do
    echo "启动进程 $i"
    # 构建完整的命令，包含所有参数
    if [ -n "$EXTRA_ARGS" ]; then
        echo "命令: python3 -m $SCRIPT_MODULE --rank_id $i --process_num $NUM_RUNS --npu_ids $NPU_IDS $EXTRA_ARGS"
        python3 -m "$SCRIPT_MODULE" --rank_id $i --process_num $NUM_RUNS --npu_ids "$NPU_IDS" $EXTRA_ARGS &
    else
        echo "命令: python3 -m $SCRIPT_MODULE --rank_id $i --process_num $NUM_RUNS --npu_ids $NPU_IDS"
        python3 -m "$SCRIPT_MODULE" --rank_id $i --process_num $NUM_RUNS --npu_ids "$NPU_IDS" &
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
            # 获取退出码（只会阻塞极短时间，因为进程已结束）
            wait $pid
            exit_code=$?

            if [ $exit_code -eq 0 ]; then
                echo "[$(date +%H:%M:%S)] 进程 $rank_id (PID: $pid) 已正常完成，退出码0，不再重启"
                unset PIDS[$rank_id]
                COMPLETED[$rank_id]=1
            else
                echo "[$(date +%H:%M:%S)] 进程 $rank_id (PID: $pid) 异常退出 (code=$exit_code)，正在重启..."
                # 重启进程时同样传递所有参数
                if [ -n "$EXTRA_ARGS" ]; then
                    echo "命令: python3 -m $SCRIPT_MODULE --rank_id $rank_id --process_num $NUM_RUNS --npu_ids $NPU_IDS $EXTRA_ARGS"
                    python3 -m "$SCRIPT_MODULE" --rank_id $rank_id --process_num $NUM_RUNS --npu_ids "$NPU_IDS" $EXTRA_ARGS &
                else
                    echo "命令: python3 -m $SCRIPT_MODULE --rank_id $rank_id --process_num $NUM_RUNS --npu_ids $NPU_IDS"
                    python3 -m "$SCRIPT_MODULE" --rank_id $rank_id --process_num $NUM_RUNS --npu_ids "$NPU_IDS" &
                fi
                new_pid=$!
                PIDS[$rank_id]=$new_pid
                echo "[$(date +%H:%M:%S)] 进程 $rank_id 已重启，新PID: $new_pid"
            fi
        fi
    done
    # 如果所有进程都已完成，则退出监控
    if [ ${#PIDS[@]} -eq 0 ]; then
        echo ""
        echo "所有进程均已完成，退出监控。"
        break
    fi
    # 等待指定的监控间隔
    sleep $MONITOR_INTERVAL
done
