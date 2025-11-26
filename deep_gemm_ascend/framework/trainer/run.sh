#!/bin/bash

# -------------------------- 基础配置 --------------------------
SCRIPT_PATH="trainer.py"  # 训练脚本路径
DATA_PATH="merged_excel"     # 数据文件路径
BASE_OUTPUT_DIR="./exp_results"       # 实验结果根目录
AVAILABLE_NPUS=(0 1 2 3 4 5 6 7)    # 可用NPU列表
TOTAL_NPUS=${#AVAILABLE_NPUS[@]}      # 可用NPU数量

# -------------------------- 定义参数空间（通过for循环组合） --------------------------
LOSSES=("mse" "smoothl1")                  # 损失函数列表
OPTIMIZERS=("adam" "adamw")                # 优化器列表
LEARNING_RATES=("0.005" "0.001" "0.01")    # 学习率列表
HIDDEN_DIMS_LIST=("64,128,64" "32,64,32" "32,128,32" "64,256,64" ) # 隐藏层维度列表
BATCH_SIZE_LIST=("128" "256" "512" "1024")       # 批次大小列表

# -------------------------- 自动生成实验列表 --------------------------
EXPERIMENT_LIST=()
for loss in "${LOSSES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            for hidden_dims in "${HIDDEN_DIMS_LIST[@]}"; do
                for batch_size in "${BATCH_SIZE_LIST[@]}"; do
                    # 生成唯一实验名称（由参数组合构成）
                    exp_name="loss=${loss}_opt=${optimizer}_lr=${lr}_hidden=${hidden_dims}_batch=${batch_size}"
                    # 将实验参数作为一个整体添加到列表（用特殊分隔符|分隔参数）
                    EXPERIMENT_LIST+=("$exp_name|$loss|$optimizer|$lr|$hidden_dims|$batch_size")
                done
            done
        done
    done
done

# 打印生成的实验数量
echo "============================================="
echo "自动生成 ${#EXPERIMENT_LIST[@]} 个实验组合"
echo "============================================="

# -------------------------- 分配实验到NPU（单NPU串行，多NPU并行） --------------------------
# 创建输出根目录
mkdir -p "$BASE_OUTPUT_DIR"

# 为每个NPU创建实验队列
declare -A NPU_QUEUES
for npu_id in "${AVAILABLE_NPUS[@]}"; do
    NPU_QUEUES[$npu_id]=()
done

# 循环分配实验到NPU（轮询方式）
for exp_idx in "${!EXPERIMENT_LIST[@]}"; do
    npu_idx=$((exp_idx % TOTAL_NPUS))
    npu_id=${AVAILABLE_NPUS[$npu_idx]}
    NPU_QUEUES[$npu_id]+="${EXPERIMENT_LIST[$exp_idx]}\n"
done

# -------------------------- 执行实验 --------------------------
for npu_id in "${AVAILABLE_NPUS[@]}"; do
    (
        experiments="${NPU_QUEUES[$npu_id]}"
        # 计算实际实验数量（去除空行）
        exp_count=$(echo -e "$experiments" | grep -v '^$' | wc -l | tr -d ' ')
        
        echo "============================================="
        echo "NPU $npu_id 开始执行 $exp_count 个实验（串行）"
        echo "============================================="
        
        exp_seq=0
        # 使用IFS和read正确解析每一行
        while IFS= read -r exp_entry; do
            if [ -z "$exp_entry" ]; then
                continue
            fi
            
            # 使用|作为分隔符解析参数
            IFS='|' read -r exp_name loss optimizer lr hidden_dims batch_size <<< "$exp_entry"
            
            exp_unique_id="npu${npu_id}_exp${exp_seq}_${exp_name}"
            exp_dir="${BASE_OUTPUT_DIR}/${exp_unique_id}"
            mkdir -p "$exp_dir"
            
            model_save="${exp_dir}/${exp_unique_id}_model.pth"
            scaler_save="${exp_dir}/${exp_unique_id}_scaler.npz"
            log_file="${exp_dir}/${exp_unique_id}_train.log"
            
            echo "------------------------------------------------"
            echo "NPU $npu_id 实验 $((exp_seq + 1))/$exp_count: $exp_name"
            echo "参数: loss=$loss, opt=$optimizer, lr=$lr, hidden=$hidden_dims, batch_size=$batch_size"
            echo "日志: $log_file"
            echo "开始时间: $(date +%H:%M:%S)"
            echo "------------------------------------------------"
            
            # 执行实验
            python "$SCRIPT_PATH" \
                --data-path "$DATA_PATH" \
                --batch-size "$batch_size" \
                --hidden-dims "$hidden_dims" \
                --epochs 500 \
                --patience 20 \
                --loss "$loss" \
                --optimizer "$optimizer" \
                --lr "$lr" \
                --npu-id "$npu_id" \
                --model-save-path "$model_save" \
                --scaler-save-path "$scaler_save" \
                > "$log_file" 2>&1
            
            echo "------------------------------------------------"
            echo "NPU $npu_id 实验 $((exp_seq + 1))/$exp_count 完成"
            echo "结束时间: $(date +%H:%M:%S)"
            echo "结果目录: $exp_dir"
            echo "------------------------------------------------"
            
            ((exp_seq++))
        done <<< "$(echo -e "$experiments")"
        
        echo "============================================="
        echo "NPU $npu_id 所有 $exp_count 个实验执行完毕"
        echo "============================================="
    ) &
done

echo "所有NPU进程已启动，总实验数：${#EXPERIMENT_LIST[@]}"
echo "等待所有实验完成..."
wait
echo "所有实验执行完毕！结果保存在：$BASE_OUTPUT_DIR"
