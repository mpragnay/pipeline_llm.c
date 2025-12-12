#!/bin/bash
# Test script for gradient explosion fix verification

echo "========================================"
echo "Testing Pipeline Parallelism Fix"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test configuration
NUM_ITERATIONS=5
SEQ_LEN=1024
BATCH_SIZE=4

echo ""
echo "Configuration:"
echo "  Iterations: $NUM_ITERATIONS"
echo "  Sequence Length: $SEQ_LEN"
echo "  Batch Size: $BATCH_SIZE"
echo ""

# Compile
echo "Compiling nccl_pipeline_gpt2..."
make nccl_pipeline_gpt2 2>&1 | tee compile.log

if [ ! -f "./nccl_pipeline_gpt2" ]; then
    echo -e "${RED}❌ Compilation failed!${NC}"
    echo "Check compile.log for errors"
    exit 1
fi

echo -e "${GREEN}✓ Compilation successful${NC}"
echo ""

# Run the test
echo "Running pipeline parallel training..."
echo "Command: mpirun --allow-run-as-root -np 2 ./nccl_pipeline_gpt2 --seq_len $SEQ_LEN --num_iterations $NUM_ITERATIONS"
echo ""

mpirun --allow-run-as-root -np 2 ./nccl_pipeline_gpt2 \
    --seq_len $SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --batch_size $BATCH_SIZE \
    2>&1 | tee test_output.log

echo ""
echo "========================================"
echo "Analysis"
echo "========================================"

# Extract loss values
echo ""
echo "Loss progression:"
grep "Iteration.*Loss:" test_output.log | grep "Stage 1"

# Check for issues
echo ""
echo "Checking for issues..."

# Check for NaN/Inf
nan_count=$(grep -c "NaN\|Inf" test_output.log || true)
if [ $nan_count -gt 0 ]; then
    echo -e "${RED}❌ Found NaN/Inf values ($nan_count occurrences)${NC}"
else
    echo -e "${GREEN}✓ No NaN/Inf values detected${NC}"
fi

# Check gradient explosion
grad_explosion=$(grep "Grad norm:" test_output.log | awk '{print $NF}' | awk '{if ($1 > 100) print $1}' | wc -l)
if [ $grad_explosion -gt 0 ]; then
    echo -e "${YELLOW}⚠ Gradient explosion detected (before clipping)${NC}"
else
    echo -e "${GREEN}✓ Gradients are stable${NC}"
fi

# Extract final loss
final_loss=$(grep "Iteration $NUM_ITERATIONS Loss:" test_output.log | grep "Stage 1" | awk '{print $NF}')
initial_loss=$(grep "Iteration 1 Loss:" test_output.log | grep "Stage 1" | awk '{print $NF}')

echo ""
echo "Initial Loss: $initial_loss"
echo "Final Loss:   $final_loss"

# Check if loss decreased
if [ -n "$initial_loss" ] && [ -n "$final_loss" ]; then
    result=$(awk -v init="$initial_loss" -v final="$final_loss" 'BEGIN {if (final < init) print "PASS"; else print "FAIL"}')
    
    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✓ Loss is DECREASING - Fix is working!${NC}"
    else
        echo -e "${RED}❌ Loss is INCREASING or FLAT - Issue persists!${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Could not extract loss values${NC}"
fi

echo ""
echo "Full output saved to: test_output.log"
echo "Compilation log saved to: compile.log"
echo ""

# Optional: Compare with reference implementation
echo "========================================"
echo "Optional: Compare with Reference"
echo "========================================"
echo ""
echo "To compare with reference implementation, run:"
echo "  ./train_gpt2_fp32 -b $BATCH_SIZE -t $SEQ_LEN | head -100"
echo ""
