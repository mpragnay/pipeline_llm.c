#!/bin/bash
# Reproduction script for single-batch baseline
# Usage: ./run_baseline.sh <num_gpus>

NUM_GPUS=${1:-2}

echo "Running baseline with ${NUM_GPUS} GPUs, 1 microbatch..."
mpirun -np ${NUM_GPUS} ./nccl_pipeline_gpt2 \
    --num_microbatches 1 \
    --microbatch_size 4 \
    --num_iterations 10 \
    --verbose
