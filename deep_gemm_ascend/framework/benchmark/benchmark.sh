#!/bin/bash

# 要监控的Python脚本
SCRIPT="benchmark.py"
# 监控间隔时间(毫秒)
INTERVAL=100
CHECKPOINTS_FILE="checkpoints.json"
# 日志文件
LOG_FILE="monitor.log"

# 记录日志函数
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S.%3N')] $1" >> "$LOG_FILE"
}

# 检查脚本是否存在
if [ ! -f "$SCRIPT" ]; then
    log "错误: 脚本 $SCRIPT 不存在"
    exit 1
fi

log "开始监控 $SCRIPT，间隔 $INTERVAL 毫秒"
python3 "$SCRIPT" &
PID=$!
log "启动 $SCRIPT，PID: $PID"


# 监控进程是否存活
while true; do
    # 检查进程是否存在
    if ! ps -p $PID > /dev/null; then
        # 进程不存在，重启它
        RESTART_COUNT=$((RESTART_COUNT + 1))
        echo "[$(date)] 检测到进程已退出，第 $RESTART_COUNT 次重启..." 

        python3 "$SCRIPT" & # todo 设置 checkpoints_file 如果checkpoints 没有值，则是正常退出。
        log "$SCRIPT (PID: $PID) 已退出，准备重启"`
        break
    fi
        # 等待指定毫秒数(使用sleep的小数形式)
        sleep $(echo "$INTERVAL / 1000" | bc -l)
done