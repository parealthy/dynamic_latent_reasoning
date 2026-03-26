#!/bin/bash
set -euo pipefail

# Model
SLOW_THINKING_MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
REASONING_NET_PATH="Qwen/Qwen3-Embedding-0.6B"
LATENT_TRAJECTORY_LENGTH=256
OUTPUT_DIR="checkpoints/DSR1-Qwen-1.5B-RFT"
# RESUME_FROM_CHECKPOINT=""
RUN_NAME="latent-reasoning-rft"

# Data
DATASET_NAME="BytedTsinghua-SIA/DAPO-Math-17k"
REWARD_METRIC="accuracy"

# GRPO
BETA=0.0
NUM_GENERATIONS=8
TEMPERATURE=1.0

# Training
DEEPSPEED_CONFIG="configs/deepspeed_zero2.yaml"
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=1e-5
NUM_EPOCHS=1
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=2048
LOGGING_STEPS=10
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=3
DATALOADER_NUM_WORKERS=8
BF16=true
TF32=false

# Distributed
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-4}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-23458}
TOTAL_PROCESSES=$((NUM_GPUS_PER_NODE * NUM_NODES))

if [ "${NODE_RANK}" -eq 0 ]; then
    mkdir -p "$OUTPUT_DIR"
fi

echo "------------------------------------------------"
echo "Distributed Training Configuration:"
echo "  NUM_GPUS_PER_NODE: $NUM_GPUS_PER_NODE"
echo "  NUM_NODES: $NUM_NODES"
echo "  NODE_RANK: $NODE_RANK"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "------------------------------------------------"

accelerate launch \
    --config_file "$DEEPSPEED_CONFIG" \
    --num_processes "$TOTAL_PROCESSES" \
    --num_machines "$NUM_NODES" \
    --machine_rank "$NODE_RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    rft.py \
    --slow_thinking_model_path "$SLOW_THINKING_MODEL_PATH" \
    --reasoning_net_path "$REASONING_NET_PATH" \
    --latent_trajectory_length "$LATENT_TRAJECTORY_LENGTH" \
    --dataset_name "$DATASET_NAME" \
    --reward_metric "$REWARD_METRIC" \
    --beta "$BETA" \
    --num_generations "$NUM_GENERATIONS" \
    --temperature "$TEMPERATURE" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --num_train_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --bf16 "$BF16" \
    --tf32 "$TF32" \
    2>&1 | tee -a "$OUTPUT_DIR/train-node${NODE_RANK}-$(date +%Y%m%d-%H%M%S).log"
