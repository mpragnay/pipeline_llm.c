# Verbose Logging Changes Summary

## Overview

Modified the codebase to make verbose logging conditional based on the `--verbose` flag provided at runtime. When the flag is not provided, only essential output (loss, errors, and critical messages) will be displayed.

## Files Modified

### 1. `debug_utils.h`

- **Updated `check_tensor_debug()` function**:
  - Added `bool verbose = true` parameter (default true for backward compatibility)
  - Wrapped debug output in `if (verbose || has_nan > 0 || has_inf > 0)` check
  - Still checks for NaN/Inf errors regardless of verbose flag
  - Only prints detailed tensor statistics when verbose is true OR when errors are detected

### 2. `nccl_pipeline_gpt2.cu`

#### Function Signatures

- **Updated `stage_accumulate_gradients()`**:
  - Changed signature from `void stage_accumulate_gradients(PipelineStage *stage)`
  - To: `void stage_accumulate_gradients(PipelineStage *stage, int verbose)`
  - Wrapped all debug prints inside this function with `if (verbose)` checks
  - Updated `check_tensor_debug()` calls to pass the `verbose` parameter

#### Main Loop Changes

All `[DEBUG]` print statements are now wrapped with verbose checks:

**Initialization Phase:**

- `[DEBUG] Initializing training dataloader...`
- `[DEBUG] Training dataloader initialized`
- `[DEBUG] Initializing validation dataloader...`
- `[DEBUG] Validation dataloader initialized
`
- `[DEBUG] Initializing tokenizer...`
- `[DEBUG] Tokenizer initialized (init_ok=%d)`
- `[DEBUG] Creating CUDA events...`
- `[DEBUG] Starting training loop (%d iterations)...`

**Per-Iteration Phase:**

- `[DEBUG] ========== Iteration %d/%d ==========`
- `[DEBUG] Loading batch...`
- `[DEBUG] Batch loaded and copied to GPU.`
- `[DEBUG] Starting backward pass...`
- `[DEBUG] Accumulating gradients...`
- `[DEBUG] Gradient accumulation complete.`
- `[DEBUG] Clipping gradients...`
- `[DEBUG] Updating parameters with AdamW...`
- `[DEBUG] Synchronizing parameters between stages...`
- `[DEBUG] Iteration %d/%d complete.`
- `[DEBUG] Training complete!`

**Stage-Level Operations:**

- All `[Stage %d]` prints for forward/backward passes
- NCCL send/receive messages
- Gradient norm checks
- Tensor debug checks
- Accumulated gradient norm prints

## Usage

### Without verbose flag (minimal output)

```bash
mpirun -np 2 ./nccl_pipeline_gpt2 --num_iterations 10
```

**Output will show only:**

- Loss values per iteration
- Step info (learning rate, gradient norm)
- Any error messages
- Final summary

### With verbose flag (detailed output)

```bash
mpirun -np 2 ./nccl_pipeline_gpt2 --num_iterations 10 --verbose
```

**Output will show:**

- All of the above PLUS
- All [DEBUG] messages
- Stage-by-stage operation logs
- Tensor statistics
- NCCL communication details
- Forward/backward pass progress
- Gradient norms and checks

## What Still Prints (Non-Verbose)

These messages are considered essential and will always print:

1. **Loss output**: `[Stage %d] Iteration %d Loss: %.6f`
2. **Step info**: `[Stage %d] Step %d: lr=%.2e, grad_norm=%.2e`
3. **Gradient clipping info**: When gradients are clipped
4. **Device info**: GPU and NCCL initialization messages
5. **Configuration**: Training configuration at startup
6. **Error messages**: All CUDA, NCCL, and cuBLAS errors
7. **NaN/Inf detection**: Always checked and reported regardless of verbose flag

## Benefits

1. **Cleaner output**: Easier to focus on training progress without verbose logs
2. **Better debugging**: Detailed logs available when needed with `--verbose`
3. **Performance**: Slightly faster execution without verbose output
4. **Flexibility**: Can toggle verbosity without recompiling

## Notes

- The verbose flag defaults to 0 (off), so existing scripts will see reduced output
- All error checking (NaN/Inf detection) remains active regardless of verbose setting
- The lint errors shown are unrelated to these changes - they're from the clang parser not finding CUDA/NCCL headers, which is expected
