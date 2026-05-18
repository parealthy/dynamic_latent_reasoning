#!/bin/bash
# Phase 2: GRPO with Difficulty-Aware Routing for Adaptive Latent Trajectory (ALT).
#
# Loads a Matryoshka-SFT checkpoint and trains the DifficultyEstimator via
# REINFORCE on GRPO advantage. Trajectory cost reward injects the
# "shortest sufficient k" preference into the advantage signal.
set -euo pipefail

# Model
SLOW_THINKING_MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
REASONING_NET_PATH="Qwen/Qwen3-Embedding-0.6B"
LATENT_TRAJECTORY_LENGTH=256
OUTPUT_DIR="checkpoints/DSR1-Qwen-1.5B-ALT-RFT"
# Resume from the Matryoshka-SFT checkpoint produced by train_sft_alt_1.5B.sh.
RESUME_FROM_CHECKPOINT="checkpoints/DSR1-Qwen-1.5B-ALT-SFT"
RUN_NAME="latent-reasoning-alt-rft"

# --- ALT specific ---
USE_ADAPTIVE_LENGTH=true
LENGTH_CANDIDATES="64,128,192,256"
# Stochastic sampling temperature for DifficultyEstimator during rollouts.
# Must be > 0 so different rollouts of the same prompt explore different k.
DIFF_SAMPLE_TEMPERATURE=1.0
# Weight λ for the REINFORCE auxiliary loss on the DifficultyEstimator.
# Typical range 0.02–0.1; start small to avoid destabilising the policy.
DIFF_REINFORCE_WEIGHT=0.05
# Weight of the trajectory cost reward. Without this, GRPO advantage cannot
# distinguish (correct, k=64) from (correct, k=256). Range 0.2–0.4.
TRAJECTORY_EFFICIENCY_WEIGHT=0.3

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
MASTER_PORT=${MASTER_PORT:-23462}
TOTAL_PROCESSES=$((NUM_GPUS_PER_NODE * NUM_NODES))

if [ "${NODE_RANK}" -eq 0 ]; then
    mkdir -p "$OUTPUT_DIR"
fi

echo "------------------------------------------------"
echo "ALT-RFT (Difficulty-Aware GRPO) Configuration:"
echo "  USE_ADAPTIVE_LENGTH:          $USE_ADAPTIVE_LENGTH"
echo "  LENGTH_CANDIDATES:            $LENGTH_CANDIDATES"
echo "  DIFF_SAMPLE_TEMPERATURE:      $DIFF_SAMPLE_TEMPERATURE"
echo "  DIFF_REINFORCE_WEIGHT:        $DIFF_REINFORCE_WEIGHT"
echo "  TRAJECTORY_EFFICIENCY_WEIGHT: $TRAJECTORY_EFFICIENCY_WEIGHT"
echo "  RESUME_FROM_CHECKPOINT:       $RESUME_FROM_CHECKPOINT"
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
    rft.py \
    --slow_thinking_model_path "$SLOW_THINKING_MODEL_PATH" \
    --reasoning_net_path "$REASONING_NET_PATH" \
    --latent_trajectory_length "$LATENT_TRAJECTORY_LENGTH" \
    --use_adaptive_length "$USE_ADAPTIVE_LENGTH" \
    --length_candidates "$LENGTH_CANDIDATES" \
    --diff_sample_temperature "$DIFF_SAMPLE_TEMPERATURE" \
    --diff_reinforce_weight "$DIFF_REINFORCE_WEIGHT" \
    --trajectory_efficiency_weight "$TRAJECTORY_EFFICIENCY_WEIGHT" \
    --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" \
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
