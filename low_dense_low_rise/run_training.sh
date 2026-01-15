#!/bin/bash

# ============================
#  User Configurable Settings
# ============================

# Training scripts (modify if needed)
TRAIN_SCRIPTS=(
    "model_training-v1_norestart_noweightclamp.py"
    "model_training-v2_restart_noweightclamp.py"
    "model_training-v3_restart_logitsclamp_weightclamp.py"
    "model_training-v4_restart_thresholdsearch.py"
    "model_training-v5_restart_focalloss.py"
    "model_training-v6_restart_thresholdsearch_focalloss.py"
)

# Maximum concurrent jobs
MAX_JOBS=3

# TensorBoard settings
TB_LOGDIR="runs"
TB_PORT=6006

# ============================
#      Start TensorBoard
# ============================

echo ">>> Starting TensorBoard on port $TB_PORT ..."
nohup tensorboard --logdir "$TB_LOGDIR" --port "$TB_PORT" > tensorboard.log 2>&1 &
echo ">>> TensorBoard started (log: tensorboard.log)"
echo ">>> Access using SSH tunnel: ssh -L 16006:localhost:$TB_PORT <YOUR_ADDR>"

# ============================
#      Training Execution
# ============================

echo ">>> Starting training scripts (max $MAX_JOBS at once)..."

job_count=0

for script in "${TRAIN_SCRIPTS[@]}"; do

    # If queue is full, wait
    while (( job_count >= MAX_JOBS )); do
        sleep 5

        # Count active python jobs
        job_count=$(pgrep -fc "python.*model_training")
    done

    # Run training script
    LOGFILE="log_${script%.py}.txt"
    echo ">>> Launching: $script  (log: $LOGFILE)"
    
    nohup python "$script" > "$LOGFILE" 2>&1 &

    ((job_count++))
    sleep 2
done

echo ">>> All scripts launched."
echo ">>> Use: pgrep -fc python   to check running jobs."
echo ">>> Logs saved in: log_*.txt"
