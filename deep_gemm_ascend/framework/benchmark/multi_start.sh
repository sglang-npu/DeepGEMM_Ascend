#!/bin/bash

if [ $# -ne 1]; then
    echo "usage $0 <python_script>"
    exit
fi

PYTHON_SCRIPT="$1"
NUM_RUNS=8

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "$PYTHON_SCRIPT not exist"
    exit 1
fi

PIDS=()

for ((i=0; i<NUM_RUNS; i++)); do
    echo "process $i start"
    python3 "$PYTHON_SCRIPT" --rank_id $i &
    PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
    echo "$pid has done"
done
