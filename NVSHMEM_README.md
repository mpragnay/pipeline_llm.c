# NVSHMEM Pipeline Parallelism for GPT-2

Pipeline parallelism implementation for GPT-2 training using NVSHMEM on 2 GPUs.

## Quick Start

```bash
# 1. On cluster, clone and setup
cd /scratch/$USER
git clone https://github.com/mpragnay/pipeline_llm.c.git
cd pipeline_llm.c

# 2. Install NVSHMEM
wget https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.4.5_cuda12-archive.tar.xz
tar -xf libnvshmem-linux-x86_64-3.4.5_cuda12-archive.tar.xz
export NVSHMEM_HOME=$PWD/libnvshmem-linux-x86_64-3.4.5_cuda12-archive
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_BOOTSTRAP=MPI

# 3. Download data
chmod +x ./dev/download_starter_pack.sh
./dev/download_starter_pack.sh

# 4. Compile and run
make nvshmem_train_gpt2
mpirun -np 2 ./nvshmem_train_gpt2 -b 4 -t 128
```

## Architecture

- **GPU 0**: Embedding + Layers 0-5
- **GPU 1**: Layers 6-11 + Loss computation
- **Communication**: NVSHMEM `putmem`/`getmem` for activations and gradients

## Requirements

- 2 NVIDIA GPUs (A100 recommended, V100+ supported)
- CUDA 11.0+
- MPI (OpenMPI or similar)
- NVSHMEM 3.4.5+
- Linux (tested on Ubuntu 20.04/22.04)

## Phase 1 Status

✅ **Implemented**:

- Split forward/backward passes
- NVSHMEM communication
- Symmetric buffer allocation
- Makefile target

⚠️ **Limitations**:

- No micro-batching (50% pipeline efficiency)
- Sequential execution (no overlap)
- ~1.0x speedup vs single GPU

## Performance Expectations

| Metric              | Phase 1 | Phase 2 (Future) |
| ------------------- | ------- | ---------------- |
| Pipeline Efficiency | ~50%    | ~85%             |
| Speedup vs 1 GPU    | 1.0x    | 1.7x             |
| Micro-batching      | No      | Yes              |

## Troubleshooting

**Compilation fails**: Check `NVSHMEM_HOME` is set and points to extracted directory.

**Runtime error "invalid argument"**: Run `git pull` to get latest fixes with `nvshmem_putmem/getmem`.

**Hangs at startup**: Check both GPUs are visible with `nvidia-smi` and `NVSHMEM_BOOTSTRAP=MPI` is set.

**Out of memory**: Reduce batch size `-b 2` or sequence length `-t 64`.

## File Structure

```
/scratch/$USER/pipeline_llm.c/
├── nvshmem_train_gpt2.cu        # Main NVSHMEM pipeline code
├── Makefile                      # Build configuration
├── gpt2_124M.bin                 # Model weights (downloaded)
├── gpt2_tokenizer.bin            # Tokenizer (downloaded)
└── dev/data/tinyshakespeare/     # Training data (downloaded)
```

## Commands

```bash
# Compile
make nvshmem_train_gpt2

# Run with different batch sizes
mpirun -np 2 ./nvshmem_train_gpt2 -b 2 -t 64   # Small (fits any GPU)
mpirun -np 2 ./nvshmem_train_gpt2 -b 4 -t 128  # Standard
mpirun -np 2 ./nvshmem_train_gpt2 -b 8 -t 256  # Large (40GB GPUs)

# Monitor GPUs
watch -n 0.5 nvidia-smi
```

## What's Next

Phase 2 will add micro-batching for 1.7x speedup:

- Split batch into micro-batches
- Overlap GPU 0 (micro-batch 2) with GPU 1 (micro-batch 1)
- Implement gradient accumulation
- Add CUDA streams for compute/communication overlap
