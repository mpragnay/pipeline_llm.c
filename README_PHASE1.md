# NVSHMEM GPT-2 Pipeline Parallelism - Phase 1

Simple 2-GPU pipeline parallelism for GPT-2 training using NVSHMEM.

## Phase 1 Status: Basic Implementation

This is the Phase 1 implementation with basic NVSHMEM infrastructure. The code includes:

- ✅ NVSHMEM initialization with MPI
- ✅ 2-GPU validation
- ✅ Per-GPU device setup
- ✅ All CUDA kernels from train_gpt2_fp32.cu
- ⏳ Forward/backward pass splitting (TODO)

## Build

```bash
cd /scratch/$USER/pipeline_llm.c

# Set NVSHMEM path
export NVSHMEM_HOME=/scratch/$USER/libnvshmem-linux-x86_64-3.4.5_cuda12-archive

# Build
make nvshmem_train_gpt2
```

## Run

```bash
# Set environment
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_DISABLE_IBRC=1
export NVSHMEM_DISABLE_IBGDA=1
export NVSHMEM_DISABLE_IBDEVX=1
export NVSHMEM_REMOTE_TRANSPORT=none

# Run with 2 GPUs
mpirun -np 2 ./nvshmem_train_gpt2 -b 4 -t 128
```

## Files

- `nvshmem_train_gpt2.cu` - Main training file with NVSHMEM support
- `NVSHMEM_README.md` - NVSHMEM installation guide
- `HANDOFF.md` - Detailed project status

## Next Steps (Not Yet Implemented)

Phase 2+:

- Split forward pass (GPU 0: layers 0-5, GPU 1: layers 6-11)
- Add NVSHMEM communication for activations
- Split backward pass with gradient communication
- Micro-batching and pipelining
- Performance optimization

See HANDOFF.md for complete implementation plan.
