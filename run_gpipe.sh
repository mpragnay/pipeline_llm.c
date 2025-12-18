#!/bin/bash
# Run with 4 microbatches (Total Batch Size = 4 * 4 = 16)
mpirun -np 2 ./nccl_pipeline_gpt2 \
    --num_microbatches 4 \
    --microbatch_size 4 \
    --num_iterations 10 \
    --verbose
