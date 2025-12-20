#!/bin/bash

# NCCL Pipeline GPT-2 Launcher
# This script launches the NCCL-based pipeline training across 2 GPUs

# Default arguments
TRAIN_DATA="dev/data/tinyshakespeare/tiny_shakespeare_train.bin"
VAL_DATA="dev/data/tinyshakespeare/tiny_shakespeare_val.bin"
BATCH_SIZE=4
SEQ_LEN=1024
LEARNING_RATE=3e-4
VAL_EVERY=20
VAL_STEPS=20
SAMPLE_EVERY=20
GEN_TOKENS=64

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i) TRAIN_DATA="$2"; shift 2 ;;
        -j) VAL_DATA="$2"; shift 2 ;;
        -b) BATCH_SIZE="$2"; shift 2 ;;
        -t) SEQ_LEN="$2"; shift 2 ;;
        -l) LEARNING_RATE="$2"; shift 2 ;;
        -v) VAL_EVERY="$2"; shift 2 ;;
        -m) VAL_STEPS="$2"; shift 2 ;;
        -s) SAMPLE_EVERY="$2"; shift 2 ;;
        -g) GEN_TOKENS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "Launching NCCL Pipeline GPT-2 training with 2 GPUs..."
echo "Train data: $TRAIN_DATA"
echo "Val data: $VAL_DATA"
echo "Batch size: $BATCH_SIZE, Seq length: $SEQ_LEN"
echo ""

# Launch with torchrun (requires PyTorch distributed)
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    nccl_pipeline_gpt2.py \
    -i "$TRAIN_DATA" \
    -j "$VAL_DATA" \
    -b "$BATCH_SIZE" \
    -t "$SEQ_LEN" \
    -l "$LEARNING_RATE" \
    -v "$VAL_EVERY" \
    -m "$VAL_STEPS" \
    -s "$SAMPLE_EVERY" \
    -g "$GEN_TOKENS"
