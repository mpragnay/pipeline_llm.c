
# Quick Testing Guide (concise)

This file contains the minimal steps to get a 2-node NVSHMEM/MPI training run working for the `pipeline_llm.c` project.

**Requirements:** CUDA-enabled Linux machines with MPI and NVSHMEM installed; 2 GPUs for the basic pipeline test.

**1) Minimal environment**

- **Set NVSHMEM path and runtime vars:**

```bash
export NVSHMEM_HOME=/path/to/libnvshmem
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_BOOTSTRAP=MPI
```

**2) Build**

```bash
cd /path/to/pipeline_llm.c
make nvshmem_train_gpt2
```

If your GPU compute capability differs, adjust `-arch=sm_XX` in the Makefile.

**3) Prepare model & data (examples)**

Place model and tokenizers under the repo or a data path you use. Example filenames used by scripts:

- `gpt2_124M.bin`
- tokenizer file (project uses `llmc` tokenizer)

Small datasets (e.g. tiny shakespeare) can be put under `dev/data/` for quick tests.

**4) Run a basic 2-node test**

```bash
# Example: 2 processes, small batch and seq len for quick validation
mpirun -np 2 ./nvshmem_train_gpt2 -b 1 -t 64 -v 1 -m 1
```

Expected: both ranks initialize and the program runs without hangs. If it stalls, check MPI and NVSHMEM environment variables and `nvidia-smi`.

**5) Quick training sanity check**

```bash
mpirun -np 2 ./nvshmem_train_gpt2 -b 2 -t 128 -v 5 -m 5
```

Look for reasonable loss values (not NaN) and decreasing trend.

**6) Common quick fixes**

- `libnvshmem.so` load error: ensure `LD_LIBRARY_PATH` contains `$NVSHMEM_HOME/lib`.
- MPI hangs: test with `mpirun -np 2 echo hello` and confirm environment variables are visible.
- CUDA OOM: reduce `-b` or `-t`.

**7) Notes**

- This guide is intentionally minimal — it assumes familiarity with MPI and NVSHMEM. Use the original, longer `TESTING_GUIDE.md` for full troubleshooting and advanced steps.
- For pipeline debugging start with small `B` and `T` and confirm per-rank logs.

**File edited:** `TESTING_GUIDE.md` — concise quick-start and troubleshooting.
