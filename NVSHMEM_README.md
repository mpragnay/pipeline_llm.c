# NVSHMEM Installation and Testing Guide

This guide covers installing NVSHMEM and running the basic multi-GPU communication test.

## Prerequisites

- **GPU**: NVIDIA A100 or newer (compute capability 8.0+)
- **CUDA**: 12.x
- **MPI**: OpenMPI installed
- **OS**: Linux

Verify your setup:
```bash
# Check GPU compute capability (need 8.0+)
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Check CUDA version
nvcc --version

# Check MPI
mpirun --version
```

## 1. Download and Extract NVSHMEM

```bash
cd /scratch/$USER

# Download NVSHMEM 3.4.5 for CUDA 12
wget https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.4.5_cuda12-archive.tar.xz

# Extract
tar -xf libnvshmem-linux-x86_64-3.4.5_cuda12-archive.tar.xz

# Verify extraction
ls libnvshmem-linux-x86_64-3.4.5_cuda12-archive/
# Should see: include/  lib/  bin/
```

## 2. Set Environment Variables

Add to your shell or run before building/running:

```bash
export NVSHMEM_HOME=/scratch/$USER/libnvshmem-linux-x86_64-3.4.5_cuda12-archive
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
```

## 3. Build the Test

```bash
cd /scratch/$USER/pipeline_llm.c

# Build with NVSHMEM_HOME set
NVSHMEM_HOME=/scratch/$USER/libnvshmem-linux-x86_64-3.4.5_cuda12-archive make test_nvshmem
```

### Build Command Details

The Makefile uses these flags:
```makefile
nvcc -arch=sm_80 -std=c++17 -rdc=true \
     --extended-lambda --expt-relaxed-constexpr \
     -I$(NVSHMEM_HOME)/include -L$(NVSHMEM_HOME)/lib \
     -I/usr/lib/x86_64-linux-gnu/openmpi/include \
     -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
     test_nvshmem_basic.cu \
     -lnvshmem_host -lnvshmem_device -lcuda -lnvidia-ml -lcudart -lmpi \
     -o test_nvshmem
```

Key flags explained:
- `-arch=sm_80`: Target A100 GPU architecture
- `-rdc=true`: Relocatable device code (required for NVSHMEM device functions)
- `-lnvshmem_host -lnvshmem_device`: NVSHMEM libraries (split host/device)
- `-lmpi`: MPI for bootstrap coordination

## 4. Run the Test

```bash
cd /scratch/$USER/pipeline_llm.c

# Set runtime environment
export LD_LIBRARY_PATH=/scratch/$USER/libnvshmem-linux-x86_64-3.4.5_cuda12-archive/lib:$LD_LIBRARY_PATH
export NVSHMEM_BOOTSTRAP=MPI

# Disable InfiniBand transports (for single-node testing)
export NVSHMEM_DISABLE_IBRC=1
export NVSHMEM_DISABLE_IBGDA=1
export NVSHMEM_DISABLE_IBDEVX=1
export NVSHMEM_REMOTE_TRANSPORT=none

# Run with 2 GPUs
mpirun -np 2 ./test_nvshmem
```

## 5. One-Liner (Copy-Paste)

#### Bottom exports is a one time thing

```bash
cd /scratch/$USER/pipeline_llm.c && \
export LD_LIBRARY_PATH=/scratch/$USER/libnvshmem-linux-x86_64-3.4.5_cuda12-archive/lib:$LD_LIBRARY_PATH && \
export NVSHMEM_BOOTSTRAP=MPI && \
export NVSHMEM_DISABLE_IBRC=1 && \
export NVSHMEM_DISABLE_IBGDA=1 && \
export NVSHMEM_DISABLE_IBDEVX=1 && \
export NVSHMEM_REMOTE_TRANSPORT=none && \
```

## Run Command
```bash
mpirun -np 2 ./test_nvshmem
```

### Expected Output

```
[Rank 0] Set GPU 0 (of 2 GPUs)
[Rank 1] Set GPU 1 (of 2 GPUs)
[PE 0] NVSHMEM initialized. Total PEs: 2
[PE 1] NVSHMEM initialized. Total PEs: 2
[PE 0] Sending: 0 1 2 3 4 5 6 7 
[PE 1] Sending: 100 101 102 103 104 105 106 107 
[PE 0] Putting data to PE 1
[PE 1] Putting data to PE 0
[PE 0] Received: 100 101 102 103 104 105 106 107 
[PE 0] PASS - Got expected value 100 from PE 1
[PE 1] Received: 0 1 2 3 4 5 6 7 
[PE 1] PASS - Got expected value 0 from PE 0
```

## Troubleshooting

### Error: `atomicAdd_system is undefined`
**Cause**: GPU compute capability < 8.0 (e.g., RTX 8000 is 7.5)  
**Solution**: Use A100 or newer GPU

### Error: `ibv_modify_qp failed` / InfiniBand errors
**Cause**: InfiniBand transport not properly configured  
**Solution**: Add these environment variables:
```bash
export NVSHMEM_DISABLE_IBRC=1
export NVSHMEM_DISABLE_IBGDA=1
export NVSHMEM_DISABLE_IBDEVX=1
export NVSHMEM_REMOTE_TRANSPORT=none
```

### Error: `Total PEs: 1` (each process sees only itself)
**Cause**: NVSHMEM not using MPI bootstrap  
**Solution**: Set `NVSHMEM_BOOTSTRAP=MPI` and use `nvshmemx_init_attr()` with MPI comm

### Error: `illegal memory access` at barrier
**Cause**: CUDA device not set before NVSHMEM init  
**Solution**: Call `cudaSetDevice()` BEFORE `nvshmem_init()`

### Error: Library not found at runtime
**Solution**: Ensure `LD_LIBRARY_PATH` includes NVSHMEM lib directory

## Understanding the Test

The test performs a **ring exchange** between 2 GPUs:

```
┌─────────────┐         ┌─────────────┐
│   PE 0      │         │   PE 1      │
│   GPU 0     │         │   GPU 1     │
│             │         │             │
│ send: 0-7   │────────►│ recv: 0-7   │
│             │         │             │
│ recv: 100-  │◄────────│ send: 100-  │
│      107    │         │      107    │
└─────────────┘         └─────────────┘
```

Key NVSHMEM APIs used:
- `nvshmem_malloc()`: Allocate symmetric memory (accessible by all PEs)
- `nvshmem_int_put()`: Put data to remote PE's memory
- `nvshmem_quiet()`: Ensure all puts complete
- `nvshmem_finalize()`: Cleanup

## Next Steps

For pipeline parallelism in transformers:
1. Allocate symmetric buffers for activations and gradients
2. Use `nvshmem_float_put()` to send activations to next pipeline stage
3. Use `nvshmem_float_put()` to send gradients to previous stage
4. Overlap computation with communication using CUDA streams

## References

- [NVSHMEM Documentation](https://docs.nvidia.com/nvshmem/)
- [NVSHMEM Examples](https://github.com/NVIDIA/nvshmem)
- [CUDA IPC](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#interprocess-communication)
