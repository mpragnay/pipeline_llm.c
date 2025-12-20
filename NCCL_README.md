# NCCL Pipeline GPT-2

This directory contains two implementations of pipeline parallelism for GPT-2:

## Files

### 1. `pipeline_gpt2.py` - PyTorch `.to()` baseline

- Uses simple `tensor.to(device)` for inter-GPU communication
- Single-process architecture
- **Communication method**: CUDA memcpy (direct P2P transfer)

**Run:**

```bash
python pipeline_gpt2.py -b 4 -t 1024
```

### 2. `nccl_pipeline_gpt2.py` - NCCL-based implementation

- Uses `torch.distributed` with NCCL backend
- Multi-process architecture (requires 2 GPUs)
- **Communication method**: NCCL `dist.send()`/`dist.recv()`
- Supports automatic gradient communication via custom autograd functions

**Run:**

```bash
./run_nccl_pipeline.sh
# or manually:
torchrun --standalone --nnodes=1 --nproc_per_node=2 nccl_pipeline_gpt2.py -b 4 -t 1024
```

### 3. `run_nccl_pipeline.sh` - Launcher script

Convenient wrapper for launching NCCL training with `torchrun`

## Key Differences

| Feature           | `pipeline_gpt2.py`       | `nccl_pipeline_gpt2.py`  |
| ----------------- | ------------------------ | ------------------------ |
| **Communication** | `.to()` (CUDA memcpy)    | NCCL send/recv           |
| **Processes**     | Single process           | 2 processes (1 per GPU)  |
| **Backend**       | Direct CUDA              | torch.distributed + NCCL |
| **Gradient sync** | Automatic (single graph) | Custom autograd hooks    |
| **Launcher**      | `python`                 | `torchrun`               |

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

## Performance Comparison

To compare `.to()` vs NCCL:

```bash
# Baseline with .to()
python pipeline_gpt2.py -b 4 -t 1024 | grep "total average"

# NCCL version
./run_nccl_pipeline.sh -b 4 -t 1024 | grep "total average"
```

## Requirements

- PyTorch with NCCL support
- 2 GPUs with CUDA
- `tiktoken` for tokenization
- GPT-2 checkpoint: `gpt2_124M.bin`

## Notes

- The NCCL version uses custom `NCCLSendFunction` and `NCCLRecvFunction` to ensure gradients flow correctly through send/recv operations
- Communication timing can be added by wrapping send/recv calls with CUDA events
- For more GPUs, extend the partitioning logic in the model initialization
