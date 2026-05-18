#!/bin/bash
# Phase 1: Matryoshka SFT for Adaptive Latent Trajectory (ALT).
# Trains the latent trajectory to be prefix-consistent: every prefix
# k ∈ LENGTH_CANDIDATES must already be sufficient on its own. This is the
# prerequisite for the DifficultyEstimator trained in Phase 2.
set -euo pipefail

# ---- Offline HF / local cache ----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="${HF_HOME:-../.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-../.hf_cache/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-../.hf_cache/transformers}"

# LRT local dataset root used by utils/load_data.py
export LRT_DATA_ROOT="${LRT_DATA_ROOT:-../datasets}"

MODEL_ROOT="${MODEL_ROOT:-../models}"

# Model
SLOW_THINKING_MODEL_PATH="${MODEL_ROOT}/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
REASONING_NET_PATH="${MODEL_ROOT}/Qwen/Qwen3-Embedding-0.6B"
LATENT_TRAJECTORY_LENGTH=256
OUTPUT_DIR="checkpoints/DSR1-Qwen-1.5B-ALT-SFT"
# RESUME_FROM_CHECKPOINT=""
RUN_NAME="latent-reasoning-alt-sft"

# --- ALT specific ---
USE_ADAPTIVE_LENGTH=true
LENGTH_CANDIDATES="64,128,192,256"

# Data
DATASET_NAME="open-r1/OpenR1-Math-220k"

# Training
DEEPSPEED_CONFIG="configs/deepspeed_zero2.yaml"
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=16
LEARNING_RATE=3e-4
NUM_EPOCHS=3
PROMPT_MAX_LENGTH=1024
COMPLETION_MAX_LENGTH=2048
LOGGING_STEPS=20
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=3
DATALOADER_NUM_WORKERS=8
BF16=true
TF32=false

# Distributed
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-4}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-23460}
TOTAL_PROCESSES=$((NUM_GPUS_PER_NODE * NUM_NODES))

if [ "${NODE_RANK}" -eq 0 ]; then
    mkdir -p "$OUTPUT_DIR"
fi

echo "------------------------------------------------"
echo "ALT-SFT (Matryoshka prefix-consistency) Configuration:"
echo "  MODEL_ROOT:      $MODEL_ROOT"
echo "  LRT_DATA_ROOT:   $LRT_DATA_ROOT"
echo "  USE_ADAPTIVE_LENGTH: $USE_ADAPTIVE_LENGTH"
echo "  LENGTH_CANDIDATES:   $LENGTH_CANDIDATES"
echo "  LATENT_TRAJECTORY_LENGTH: $LATENT_TRAJECTORY_LENGTH"
echo "------------------------------------------------"
echo "Distributed Training:"
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
    sft.py \
    --slow_thinking_model_path "$SLOW_THINKING_MODEL_PATH" \
    --reasoning_net_path "$REASONING_NET_PATH" \
    --latent_trajectory_length "$LATENT_TRAJECTORY_LENGTH" \
    --use_adaptive_length "$USE_ADAPTIVE_LENGTH" \
    --length_candidates "$LENGTH_CANDIDATES" \
    --dataset_name "$DATASET_NAME" \
    --prompt_max_length "$PROMPT_MAX_LENGTH" \
    --completion_max_length "$COMPLETION_MAX_LENGTH" \
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
