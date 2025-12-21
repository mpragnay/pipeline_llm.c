# NCCL Pipeline GPT-2

This directory contains two implementations of pipeline parallelism for GPT-2:

## Files

### 1. `pipeline_gpt2.py` - PyTorch `.to()` baseline

- Uses simple `tensor.to(device)` for inter-GPU communication
- Single-process architecture
- **Communication method**: CUDA memcpy (direct P2P transfer)
- **Sampling**: Random multinomial sampling

**Run:**

```bash
python pipeline_gpt2.py -b 4 -t 1024
```

### 2. `nccl_pipeline_gpt2.py` - NCCL-based implementation

- Uses `torch.distributed` with NCCL backend
- Multi-process architecture (requires 2 GPUs)
- **Communication method**: NCCL `dist.send()`/`dist.recv()`
- Supports automatic gradient communication via custom autograd functions
- **Micro-batching support**: `-c/--num-chunks` flag for gradient accumulation
- **Sampling**: Random multinomial sampling (matching `pipeline_gpt2.py`)

**Run:**

```bash
./run_nccl_pipeline.sh
# or manually:
torchrun --standalone --nnodes=1 --nproc_per_node=2 nccl_pipeline_gpt2.py -b 4 -t 1024

# With micro-batching (2 chunks):
torchrun --standalone --nnodes=1 --nproc_per_node=2 nccl_pipeline_gpt2.py -b 4 -t 1024 -c 2
```

### 3. `run_nccl_pipeline.sh` - Launcher script

Convenient wrapper for launching NCCL training with `torchrun`

## Command-Line Options

| Flag               | Description                                 | Default                                               |
| ------------------ | ------------------------------------------- | ----------------------------------------------------- |
| `-i`               | Train data pattern                          | `dev/data/tinyshakespeare/tiny_shakespeare_train.bin` |
| `-j`               | Val data pattern                            | `dev/data/tinyshakespeare/tiny_shakespeare_val.bin`   |
| `-b`               | Batch size                                  | 4                                                     |
| `-t`               | Sequence length                             | 1024                                                  |
| `-l`               | Learning rate                               | 0.0003                                                |
| `-v`               | Validation loss interval (steps)            | 20                                                    |
| `-m`               | Max validation batches                      | 20                                                    |
| `-s`               | Sample generation interval (steps)          | 20                                                    |
| `-g`               | Generation length (tokens)                  | 64                                                    |
| `-c, --num-chunks` | **Micro-batches for gradient accumulation** | 1 (no micro-batching)                                 |
| `--verbose`        | Enable debug logging                        | False                                                 |

## Key Differences

| Feature            | `pipeline_gpt2.py`       | `nccl_pipeline_gpt2.py`  |
| ------------------ | ------------------------ | ------------------------ |
| **Communication**  | `.to()` (CUDA memcpy)    | NCCL send/recv           |
| **Processes**      | Single process           | 2 processes (1 per GPU)  |
| **Backend**        | Direct CUDA              | torch.distributed + NCCL |
| **Gradient sync**  | Automatic (single graph) | Custom autograd hooks    |
| **Micro-batching** | No                       | Yes (`-c` flag)          |
| **Launcher**       | `python`                 | `torchrun`               |
| **Sampling**       | Multinomial              | Multinomial              |

## Architecture

Both implementations split GPT-2 (12 layers) across 2 GPUs:

- **GPU 0 (Rank 0)**: Embedding + Layers 0-5
- **GPU 1 (Rank 1)**: Layers 6-11 + LM Head

### Forward Pass

1. Rank 0: Processes input → sends activations to Rank 1
2. Rank 1: Receives activations → computes loss

### Backward Pass

1. Rank 1: Computes gradients → sends grad to Rank 0
2. Rank 0: Receives gradients → updates parameters

### Micro-batching (Optional)

When using `-c` flag:

1. Split batch into `num_chunks` micro-batches
2. Process each micro-batch sequentially through pipeline
3. Accumulate gradients across all micro-batches
4. Single optimizer update after all micro-batches complete

**Example**: `-b 8 -c 4` splits batch of 8 into 4 micro-batches of size 2

## Performance Comparison

To compare `.to()` vs NCCL:

```bash
# Baseline with .to()
python pipeline_gpt2.py -b 4 -t 1024 | grep "total average"

# NCCL version (no micro-batching)
torchrun --standalone --nnodes=1 --nproc_per_node=2 nccl_pipeline_gpt2.py -b 4 -t 1024 -c 1 | grep "total average"

# NCCL with micro-batching
torchrun --standalone --nnodes=1 --nproc_per_node=2 nccl_pipeline_gpt2.py -b 8 -t 1024 -c 4 | grep "total average"
```

## Requirements

- PyTorch with NCCL support
- 2 GPUs with CUDA
- `tiktoken` for tokenization
- GPT-2 checkpoint: `gpt2_124M.bin`

## Implementation Details

- **Custom Autograd Functions**: `NCCLSendFunction` and `NCCLRecvFunction` ensure gradients flow correctly through distributed send/recv operations
- **Gradient Accumulation**: When using micro-batching, gradients are scaled by `1/num_chunks` to maintain correct averaging
- **Multinomial Sampling**: Both implementations use `torch.multinomial` with softmax probabilities for text generation (not greedy argmax)
- **Buffer Reallocation**: Activation buffers automatically resize when switching between training (B=4) and generation (B=1)

## Extending to More GPUs

To partition across N GPUs, modify the model initialization:

1. Split layers evenly: `layers_per_gpu = 12 // N`
2. Create N ranks with appropriate layer ranges
3. Chain communication: Rank i sends to Rank i+1
