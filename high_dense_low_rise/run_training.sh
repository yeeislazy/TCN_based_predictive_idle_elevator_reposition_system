#!/bin/bash

########################################
# Activate conda environment (重要)
########################################
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

echo ">>> Conda env activated"

########################################
# User Config
########################################

TRAIN_SCRIPTS=(
    "model_training-v2_restart_noweightclamp.py"
    "model_training-v5_restart_focalloss_0.25.py"
    "model_training-v5_restart_focalloss_0.75.py"
)

MAX_JOBS=3
TB_LOGDIR="runs"
TB_PORT=6006

########################################
# Start TensorBoard
########################################

if ! pgrep -f "tensorboard.*$TB_LOGDIR" > /dev/null; then
    echo ">>> Starting TensorBoard..."
    nohup tensorboard --logdir "$TB_LOGDIR" --port "$TB_PORT" > tensorboard.log 2>&1 &
else
    echo ">>> TensorBoard already running"
fi

########################################
# Helper: count running training scripts
########################################

count_running_jobs() {
    local count=0
    for script in "${TRAIN_SCRIPTS[@]}"; do
        if pgrep -f "python.*$script" > /dev/null; then
            ((count++))
        fi
    done
    echo "$count"
}

########################################
# Launch training jobs with queue
########################################

echo ">>> Starting training queue (max $MAX_JOBS concurrent)..."

for script in "${TRAIN_SCRIPTS[@]}"; do

    # 如果这个脚本已经在跑，跳过（防止重复训练）
    if pgrep -f "python.*$script" > /dev/null; then
        echo ">>> Skip $script (already running)"
        continue
    fi

    # 等待空位
    while true; do
        running=$(count_running_jobs)
        if (( running < MAX_JOBS )); then
            break
        fi
        sleep 10
    done

    LOGFILE="log_${script%.py}.txt"
    echo ">>> Launching: $script (log: $LOGFILE)"
    nohup python "$script" > "$LOGFILE" 2>&1 &
    sleep 5
done

########################################
# Wait for ALL training jobs to finish
########################################

echo ">>> Waiting for all training jobs to finish..."

while true; do
    running=$(count_running_jobs)
    if (( running == 0 )); then
        break
    fi
    echo ">>> Still running: $running job(s)..."
    sleep 30
done

########################################
# Shutdown
########################################

echo ">>> All training finished."
echo ">>> Shutting down system in 60 seconds..."
sleep 60

sudo /usr/sbin/shutdown -h now
