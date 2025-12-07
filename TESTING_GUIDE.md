# Complete Installation, Testing & Verification Guide

## Prerequisites Check

### Hardware Requirements

```bash
# Check GPU count (need exactly 2)
nvidia-smi --query-gpu=count --format=csv,noheader

# Check GPU models (A100, V100, or compute capability 7.0+)
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Expected output:
# NVIDIA A100-SXM4-40GB, 8.0
# NVIDIA A100-SXM4-40GB, 8.0
```

### Software Requirements

```bash
# Check CUDA version (need 11.0+)
nvcc --version

# Check MPI
mpirun --version

# Check if on Linux
uname -a  # Should show Linux, not Darwin (Mac) or Windows
```

---

## Step 1: Install NVSHMEM

### Download and Install

```bash
# Go to scratch directory
cd /scratch/$USER

# Download NVSHMEM (adjust version for your CUDA)
wget https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.4.5_cuda12-archive.tar.xz

# Extract
tar -xf libnvshmem-linux-x86_64-3.4.5_cuda12-archive.tar.xz

# Set environment variables (add to ~/.bashrc for persistence)
export NVSHMEM_HOME=/scratch/$USER/libnvshmem-linux-x86_64-3.4.5_cuda12-archive
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
```

### Verify Installation

```bash
ls $NVSHMEM_HOME
# Should show: bin/  include/  lib/

ls $NVSHMEM_HOME/lib/libnvshmem*
# Should show several .so files
```

---

## Step 2: Get Data and Model

### Download GPT-2 Model Checkpoint

```bash
cd /scratch/$USER/pipeline_llm.c

# Download GPT-2 124M checkpoint (~500 MB)
wget https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/gpt2_124M.bin

# Download tokenizer
wget https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/gpt2_tokenizer.bin

# Verify downloads
ls -lh gpt2_124M.bin gpt2_tokenizer.bin
# gpt2_124M.bin should be ~475 MB
# gpt2_tokenizer.bin should be ~0.5 MB
```

### Download Training Data

```bash
# Create data directory
mkdir -p dev/data/tinyshakespeare

# Download pre-tokenized tiny shakespeare
cd dev/data/tinyshakespeare
wget https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/tinyshakespeare_train.bin -O tiny_shakespeare_train.bin
wget https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/tinyshakespeare_val.bin -O tiny_shakespeare_val.bin

# Go back to project root
cd ../../../

# Verify data files
ls -lh dev/data/tinyshakespeare/
# Should show tiny_shakespeare_train.bin (~1 MB) and tiny_shakespeare_val.bin
```

---

## Step 3: Compile the Code

### Set Environment Variables

```bash
# Required for compilation
export NVSHMEM_HOME=/scratch/$USER/libnvshmem-linux-x86_64-3.4.5_cuda12-archive
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
```

### Compile

```bash
cd /scratch/$USER/pipeline_llm.c

# Build with Makefile
make nvshmem_train_gpt2

# Expected output:
# nvcc --threads=0 -t=0 --use_fast_math -std=c++17 -O3 \
#   -arch=sm_80 \
#   -I/scratch/.../nvshmem/include -L/scratch/.../nvshmem/lib \
#   -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
#   nvshmem_train_gpt2.cu -lnvshmem -lcublas -lcublasLt -lcudart -lnvidia-ml -lmpi -o nvshmem_train_gpt2
```

### Troubleshooting Compilation

**Error: `NVSHMEM_HOME not found`**

```bash
# Make sure NVSHMEM_HOME is set correctly
echo $NVSHMEM_HOME
# Should print: /scratch/$USER/libnvshmem-linux-x86_64-3.4.5_cuda12-archive

# If not, export it again
export NVSHMEM_HOME=/scratch/$USER/libnvshmem-linux-x86_64-3.4.5_cuda12-archive
```

**Error: `cannot find -lnvshmem`**

```bash
# Check if libraries exist
ls $NVSHMEM_HOME/lib/libnvshmem.so

# Add to library path
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
```

**Error: `arch=sm_80 not supported`**

```bash
# Your GPU might have different compute capability
# Check your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# If it shows 7.0 or 7.5 (V100), edit Makefile line 307:
# Change: -arch=sm_80
# To:     -arch=sm_70  (for V100)
```

---

## Step 4: Set Runtime Environment

```bash
# Runtime library path
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH

# MPI bootstrap for NVSHMEM
export NVSHMEM_BOOTSTRAP=MPI

# Disable InfiniBand for single-node testing
export NVSHMEM_DISABLE_IBRC=1
export NVSHMEM_DISABLE_IBGDA=1
export NVSHMEM_DISABLE_IBDEVX=1
export NVSHMEM_REMOTE_TRANSPORT=none

# Make these permanent by adding to ~/.bashrc:
cat >> ~/.bashrc << 'EOF'
# NVSHMEM Environment
export NVSHMEM_HOME=/scratch/$USER/libnvshmem-linux-x86_64-3.4.5_cuda12-archive
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_DISABLE_IBRC=1
export NVSHMEM_DISABLE_IBGDA=1
export NVSHMEM_DISABLE_IBDEVX=1
export NVSHMEM_REMOTE_TRANSPORT=none
EOF

source ~/.bashrc
```

---

## Step 5: Run Tests

### Test 1: Basic Initialization (2-3 seconds)

```bash
cd /scratch/$USER/pipeline_llm.c

# Run with minimal settings
mpirun -np 2 ./nvshmem_train_gpt2 -b 1 -t 64 -v 1 -m 1

# Expected output:
+------------------------+----------------------------------------------------+
| Parameter              | Value                                              |
+------------------------+----------------------------------------------------+
| NVSHMEM PEs (GPUs)     | 2                                                  |
| This PE                | 0                                                  |
| batch size B           | 1                                                  |
| sequence length T      | 64                                                 |
+------------------------+----------------------------------------------------+
| device                 | NVIDIA A100-SXM4-40GB                              |
+------------------------+----------------------------------------------------+
| num_layers L           | 12                                                 |
| num_parameters         | 124439808                                          |
+------------------------+----------------------------------------------------+
allocated 124 MiB for model parameters
allocated 47 MiB for activations
allocated 0 MiB for NVSHMEM activation buffer
allocated 0 MiB for NVSHMEM gradient buffer
```

**✅ Pass Criteria:**

- No error messages
- Both PEs (0 and 1) initialize
- NVSHMEM buffers allocated
- Program completes without hanging

### Test 2: Small Training Run (30 seconds - 1 minute)

```bash
# Run 10 training steps with small batch
mpirun -np 2 ./nvshmem_train_gpt2 -b 2 -t 128 -v 5 -m 5

# Monitor in another terminal:
watch -n 0.5 nvidia-smi

# Expected output:
step    1/N: train loss 5.2xxx (xxx ms, xxx tok/s)
step    2/N: train loss 5.1xxx (xxx ms, xxx tok/s)
...
step    5/N: train loss 5.0xxx (xxx ms, xxx tok/s)
val loss 4.9xxx
```

**✅ Pass Criteria:**

- Loss values are reasonable (3.0 - 6.0 range)
- Loss decreases over steps
- Both GPUs show memory usage in `nvidia-smi`
- No deadlocks or hangs

### Test 3: Full Training Run (5-10 minutes)

```bash
# Run with standard settings
mpirun -np 2 ./nvshmem_train_gpt2 -b 4 -t 128 -v 10 -m 10 -s 20

# Expected behavior:
# - Trains for full epoch on tiny shakespeare
# - Validates every 10 steps
# - Generates text samples every 20 steps
```

**✅ Pass Criteria:**

- Training completes without crashes
- Loss converges (decreases to ~3.5-4.5 range)
- Text generation improves over time
- GPU utilization ~70-90% on both GPUs

---

## Step 6: Verify Correctness

### Numerical Accuracy Test

Compare pipeline results with single-GPU baseline:

```bash
# Run single-GPU version (if available)
./train_gpt2_fp32cu -b 4 -t 128 | tee single_gpu.log

# Run pipeline version
mpirun -np 2 ./nvshmem_train_gpt2 -b 4 -t 128 | tee pipeline.log

# Compare first 10 step losses
# They should match within ~0.05 tolerance
```

### Memory Usage Verification

```bash
# While training is running, check memory in another terminal:
watch -n 0.5 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv'

# Expected for B=4, T=128:
# GPU 0: ~2-3 GB memory used, 70-90% utilization
# GPU 1: ~2-3 GB memory used, 70-90% utilization
```

---

## Common Issues & Solutions

### Issue 1: `error while loading shared libraries: libnvshmem.so`

**Solution:**

```bash
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
# Make sure to do this BEFORE running mpirun
```

### Issue 2: `Error: This pipeline implementation requires exactly 2 GPUs, got X`

**Solution:**

```bash
# Check GPU visibility
nvidia-smi
CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 ./nvshmem_train_gpt2 ...
```

### Issue 3: Program hangs at initialization

**Solution:**

```bash
# Check if MPI is working
mpirun -np 2 echo "Hello from rank $OMPI_COMM_WORLD_RANK"

# Check NVSHMEM environment
env | grep NVSHMEM

# Try with MPI debug output
mpirun -np 2 --mca btl_base_verbose 10 ./nvshmem_train_gpt2 ...
```

### Issue 4: CUDA out of memory

**Solution:**

```bash
# Reduce batch size
mpirun -np 2 ./nvshmem_train_gpt2 -b 2 -t 64

# Or reduce sequence length
mpirun -np 2 ./nvshmem_train_gpt2 -b 4 -t 64
```

### Issue 5: Loss is NaN or explodes

**Solution:**

```bash
# Check model file integrity
md5sum gpt2_124M.bin

# Reduce learning rate
mpirun -np 2 ./nvshmem_train_gpt2 -b 4 -t 128 -l 0.0001

# Check for NVSHMEM communication errors in output
```

---

## Performance Expectations

### Phase 1 (Current Implementation)

| Metric                 | Expected Value            |
| ---------------------- | ------------------------- |
| Pipeline Efficiency    | ~50%                      |
| Speedup vs Single GPU  | ~1.0x (no speedup)        |
| GPU 0 Utilization      | 70-90% during layers 0-5  |
| GPU 1 Utilization      | 70-90% during layers 6-11 |
| Communication Overhead | ~5-10% of total time      |

**Why no speedup?**
Phase 1 is a **proof of concept** without micro-batching. GPUs run sequentially:

- While GPU 0 processes layers 0-5, GPU 1 is idle
- While GPU 1 processes layers 6-11, GPU 0 is idle

### Phase 2 (With Micro-batching - Future)

| Metric                | Target Value |
| --------------------- | ------------ |
| Pipeline Efficiency   | ~85%         |
| Speedup vs Single GPU | ~1.7x        |
| Both GPU Utilization  | ~85-95%      |

---

## Quick Start Commands

### One-Liner Setup and Run

```bash
# Set environment and run
export NVSHMEM_HOME=/scratch/$USER/libnvshmem-linux-x86_64-3.4.5_cuda12-archive && \
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH && \
export NVSHMEM_BOOTSTRAP=MPI && \
cd /scratch/$USER/pipeline_llm.c && \
mpirun -np 2 ./nvshmem_train_gpt2 -b 4 -t 128
```

### Monitor GPUs in Real-Time

```bash
# Open second terminal
watch -n 0.5 nvidia-smi
```

### Save Training Log

```bash
mpirun -np 2 ./nvshmem_train_gpt2 -b 4 -t 128 2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log
```

---

## Next Steps After Successful Testing

1. **Verify numerical correctness** - Compare with single-GPU baseline
2. **Measure performance metrics** - Document actual throughput and efficiency
3. **Test different batch sizes** - Find optimal B and T
4. **Profile with nsys/nvprof** - Identify bottlenecks
5. **Prepare for Phase 2** - Implement micro-batching for 1.7x speedup

---

## Getting Help

If you encounter issues not covered here:

1. Check NVSHMEM logs: Look for NVSHMEM error messages in stderr
2. Enable MPI debugging: `mpirun --mca btl_base_verbose 10 ...`
3. Check GPU status: `nvidia-smi -l 1` (updates every second)
4. Review code: See [nvshmem_train_gpt2.cu](file:///Users/kvlnraju/Desktop/courses/semester_3/bdml/proj/pipeline_llm.c/nvshmem_train_gpt2.cu)
5. Consult docs: See [HANDOFF.md](file:///Users/kvlnraju/Desktop/courses/semester_3/bdml/proj/pipeline_llm.c/HANDOFF.md) for architecture details
