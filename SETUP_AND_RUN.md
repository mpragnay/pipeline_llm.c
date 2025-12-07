# Complete Setup and Run Guide - NVSHMEM GPT-2 Training

## Prerequisites - What You Need

### Hardware

- **2 NVIDIA GPUs** (A100, V100, or any with Compute Capability 7.0+)
- **Linux OS** (Ubuntu 20.04/22.04 recommended, NVSHMEM doesn't work on Mac/Windows)
- Connected to the same node (not across nodes in this Phase 1)

### Check Your System

```bash
# Check if you have 2 GPUs
nvidia-smi

# Should show 2 GPUs, something like:
# GPU 0: NVIDIA A100-SXM4-40GB
# GPU 1: NVIDIA A100-SXM4-40GB

# Check CUDA version (need 11.0+)
nvcc --version

# Check if MPI is installed
mpirun --version

# Check compute capability (need 7.0+)
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

---

## Step 1: Install NVSHMEM

NVSHMEM is the library that enables GPU-to-GPU communication.

```bash
# Go to your scratch directory (use /scratch/$USER on cluster, or ~/ on local)
cd /scratch/$USER  # or: cd ~/

# Download NVSHMEM 3.4.5 for CUDA 12 (or compatible version)
wget https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.4.5_cuda12-archive.tar.xz

# Extract
tar -xf libnvshmem-linux-x86_64-3.4.5_cuda12-archive.tar.xz

# Set environment variable (add to ~/.bashrc to make permanent)
export NVSHMEM_HOME=/scratch/$USER/libnvshmem-linux-x86_64-3.4.5_cuda12-archive
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH

# Verify installation
ls $NVSHMEM_HOME
# Should show: bin/  include/  lib/
```

If download link doesn't work, get the latest from: https://developer.nvidia.com/nvshmem-downloads

---

## Step 2: Install OpenMPI (if not already installed)

```bash
# Check if MPI is installed
which mpirun

# If not installed, on Ubuntu:
sudo apt-get update
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev

# Verify
mpirun --version
```

---

## Step 3: Get the Data

You need GPT-2 model weights and training data.

```bash
cd /scratch/$USER/pipeline_llm.c

# Download GPT-2 124M model checkpoint
wget https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/gpt2_124M.bin

# Download tokenizer
wget https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/gpt2_tokenizer.bin

# Download tiny shakespeare dataset (for testing)
mkdir -p dev/data/tinyshakespeare
cd dev/data/tinyshakespeare
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Prepare the data (need to tokenize it - use the Python script from llm.c)
cd ../../../
# If you have train_gpt2.py, run:
python train_gpt2.py --write_tensors 0

# This creates:
# dev/data/tinyshakespeare/tiny_shakespeare_train.bin
# dev/data/tinyshakespeare/tiny_shakespeare_val.bin
```

**Alternative:** If python setup is complex, you can download pre-tokenized data:

```bash
cd dev/data
wget https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/tinyshakespeare_train.bin
wget https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/tinyshakespeare_val.bin
mv tinyshakespeare_train.bin tinyshakespeare/tiny_shakespeare_train.bin
mv tinyshakespeare_val.bin tinyshakespeare/tiny_shakespeare_val.bin
```

---

## Step 4: Compile the Code

**IMPORTANT NOTE:** The current code (`nvshmem_train_gpt2.cu`) has NVSHMEM infrastructure but the pipeline split is NOT YET IMPLEMENTED. It will run single-GPU training on each GPU independently. To make it actually work as pipeline, you need to implement the forward/backward splits (see HANDOFF.md).

For now, let's compile what we have:

```bash
cd /scratch/$USER/pipeline_llm.c

# Method 1: Manual compilation
nvcc -O3 -std=c++17 \
  -I$NVSHMEM_HOME/include \
  -L$NVSHMEM_HOME/lib \
  -I/usr/lib/x86_64-linux-gnu/openmpi/include \
  -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
  nvshmem_train_gpt2.cu \
  -lnvshmem -lcublas -lmpi -o nvshmem_train_gpt2

# Method 2: Using Makefile (need to add target first)
# Add to Makefile, then:
# make nvshmem_train_gpt2
```

If compilation succeeds, you should see `nvshmem_train_gpt2` executable.

---

## Step 5: Set Runtime Environment

Before running, set these environment variables:

```bash
# Runtime library path
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH

# MPI bootstrap for NVSHMEM
export NVSHMEM_BOOTSTRAP=MPI

# Disable InfiniBand (for single-node testing)
export NVSHMEM_DISABLE_IBRC=1
export NVSHMEM_DISABLE_IBGDA=1
export NVSHMEM_DISABLE_IBDEVX=1
export NVSHMEM_REMOTE_TRANSPORT=none
```

**Tip:** Add these to `~/.bashrc` so they persist:

```bash
echo 'export NVSHMEM_HOME=/scratch/$USER/libnvshmem-linux-x86_64-3.4.5_cuda12-archive' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export NVSHMEM_BOOTSTRAP=MPI' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 6: Run the Code

### Basic Test Run

```bash
cd /scratch/$USER/pipeline_llm.c

# Run with 2 GPUs, small batch for testing
mpirun -np 2 ./nvshmem_train_gpt2 \
  -b 2 \
  -t 64 \
  -v 10 \
  -m 5

# Explanation of arguments:
# -b 2    : batch size = 2 (small for testing)
# -t 64   : sequence length = 64 tokens (small for testing)
# -v 10   : validate every 10 steps
# -m 5    : max 5 validation batches
```

### Full Training Run

```bash
mpirun -np 2 ./nvshmem_train_gpt2 \
  -b 4 \
  -t 128 \
  -l 0.0003 \
  -v 20 \
  -m 20 \
  -s 20

# -b 4      : batch size = 4
# -t 128    : sequence length = 128
# -l 0.0003 : learning rate = 3e-4
# -v 20     : validate every 20 steps
# -m 20     : 20 validation batches
# -s 20     : sample text every 20 steps
```

### Custom Data Paths

```bash
mpirun -np 2 ./nvshmem_train_gpt2 \
  -i dev/data/tinyshakespeare/tiny_shakespeare_train.bin \
  -j dev/data/tinyshakespeare/tiny_shakespeare_val.bin \
  -b 4 -t 128
```

---

## Expected Output

### Successful Initialization

```
+------------------------+----------------------------------------------------+
| Parameter              | Value                                              |
+------------------------+----------------------------------------------------+
| NVSHMEM PEs (GPUs)     | 2                                                  |
| This PE                | 0                                                  |
| train data pattern     | dev/data/tinyshakespeare/tiny_shakespeare_train.bin|
| batch size B           | 4                                                  |
| sequence length T      | 128                                                |
+------------------------+----------------------------------------------------+
| device                 | NVIDIA A100-SXM4-40GB                              |
| TF32                   | enabled                                            |
+------------------------+----------------------------------------------------+
| num_layers L           | 12                                                 |
| num_heads NH           | 12                                                 |
| channels C             | 768                                                |
| num_parameters         | 124439808                                          |
+------------------------+----------------------------------------------------+
allocated 124 MiB for model parameters
allocated 47 MiB for activations
```

### During Training

```
val loss 5.2341
step    1/1562: train loss 5.1234 (234.5 ms, 2185 tok/s)
step    2/1562: train loss 5.0123 (231.2 ms, 2213 tok/s)
...
step   20/1562: train loss 4.3456 (229.8 ms, 2231 tok/s)
generating:
---
To be or not to be, that is the question...
---
```

### Validation Output

```
val loss 4.1234
```

---

## What the Code Does

### Current Behavior (Phase 1 - Incomplete)

- Both GPUs initialize independently
- Each GPU loads the FULL model
- **Training runs on both GPUs separately (NOT pipeline yet)**
- Loss will be computed on both GPUs independently

### Expected Behavior (After Pipeline Implementation)

- GPU 0: Runs embedding + layers 0-5
- GPU 1: Receives activations, runs layers 6-11 + loss
- Gradients flow back: GPU 1 → GPU 0
- Both GPUs update their portion of parameters

---

## Testing Checklist

- [ ] Both GPUs show in `nvidia-smi`
- [ ] NVSHMEM libraries found (check `$NVSHMEM_HOME/lib`)
- [ ] Code compiles without errors
- [ ] `mpirun -np 2` launches 2 processes
- [ ] Both PEs (0 and 1) print initialization messages
- [ ] Model loads (see "allocated X MiB" messages)
- [ ] Training steps execute
- [ ] Loss values print
- [ ] No deadlocks (program doesn't freeze)
- [ ] GPU utilization shown in `watch -n 0.5 nvidia-smi`

---

## Common Issues & Solutions

### Issue: `error while loading shared libraries: libnvshmem.so`

**Solution:**

```bash
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
```

### Issue: `Error: This pipeline implementation requires exactly 2 GPUs, got 1`

**Solution:** You need 2 GPUs. Check with `nvidia-smi`.

### Issue: Program hangs/freezes

**Solution:**

- Check both GPUs are visible: `CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 ...`
- Ensure barrier synchronization is correct (implementation issue)

### Issue: `Cannot find gpt2_124M.bin`

**Solution:** Download the checkpoint (see Step 3) and run from correct directory.

### Issue: CUDA out of memory

**Solution:** Reduce batch size (`-b 2` or `-b 1`) or sequence length (`-t 64` or `-t 32`).

---

## Next Steps

The current code has the infrastructure but NOT the pipeline splits. To complete it:

1. Implement split forward pass (GPU 0 does layers 0-5, GPU 1 does 6-11)
2. Add NVSHMEM communication for activations
3. Implement split backward pass
4. Add NVSHMEM communication for gradients

See **HANDOFF.md** and **walkthrough.md** for implementation details.

---

## Quick Reference

**One-liner to set environment and run:**

```bash
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH && \
export NVSHMEM_BOOTSTRAP=MPI && \
export NVSHMEM_DISABLE_IBRC=1 && \
export NVSHMEM_DISABLE_IBGDA=1 && \
export NVSHMEM_DISABLE_IBDEVX=1 && \
export NVSHMEM_REMOTE_TRANSPORT=none && \
mpirun -np 2 ./nvshmem_train_gpt2 -b 4 -t 128
```

**Monitor GPUs in another terminal:**

```bash
watch -n 0.5 nvidia-smi
```
