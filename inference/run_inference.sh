#!/bin/bash
set -euo pipefail

# ============================================================
# Latent Reasoning — Interactive Inference
#
# Usage:
#   bash inference/run_inference.sh
#
#   # Override model config (e.g. 7B)
#   SLOW_THINKING_MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#   CHECKPOINT_PATH=checkpoints/DSR1-Qwen-7B-LRT-Math \
#     bash inference/run_inference.sh
# ============================================================

# ---- Model ----
SLOW_THINKING_MODEL_PATH="${SLOW_THINKING_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
REASONING_NET_PATH="${REASONING_NET_PATH:-Qwen/Qwen3-Embedding-0.6B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/DSR1-Qwen-1.5B-LRT-Math}"
LATENT_TRAJECTORY_LENGTH="${LATENT_TRAJECTORY_LENGTH:-256}"

# ---- Generation ----
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10000}"
PROMPT_MAX_LENGTH="${PROMPT_MAX_LENGTH:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"

echo "================================================"
echo "  Latent Reasoning Interactive Inference"
echo "  Model:        $SLOW_THINKING_MODEL_PATH"
echo "  ReasoningNet: $REASONING_NET_PATH"
echo "  Checkpoint:   $CHECKPOINT_PATH"
echo "================================================"

python inference/run_inference.py \
    --model_path "$SLOW_THINKING_MODEL_PATH" \
    --reasoning_net_path "$REASONING_NET_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --latent_trajectory_length "$LATENT_TRAJECTORY_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --prompt_max_length "$PROMPT_MAX_LENGTH" \
    --temperature "$TEMPERATURE"
