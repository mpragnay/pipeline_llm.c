/*
 * test_nccl_simple.cu
 *
 * A simple diagnostic tool to verify NCCL P2P communication between 2 GPUs.
 * Rank 0 Sends -> Rank 1 Recvs -> Verify Data.
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,            \
             cudaGetErrorString(e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t r = cmd;                                                      \
    if (r != ncclSuccess) {                                                    \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,            \
             ncclGetErrorString(r));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

int main(int argc, char *argv[]) {
  int rank, size;

  // 1. Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 2) {
    if (rank == 0)
      printf("Error: This test requires exactly 2 ranks/GPUs.\n");
    MPI_Finalize();
    return 0;
  }

  // 2. Set Device
  CUDACHECK(cudaSetDevice(rank));

  // 3. Initialize NCCL
  ncclUniqueId id;
  if (rank == 0)
    ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));

  // 4. Verification Data
  int count = 1024 * 1024; // 1M floats ~ 4MB
  size_t bytes = count * sizeof(float);
  float *h_data = (float *)malloc(bytes);
  float *d_data;
  CUDACHECK(cudaMalloc(&d_data, bytes));
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  if (rank == 0) {
    // --- SENDER ---
    for (int i = 0; i < count; ++i)
      h_data[i] = (float)i;
    CUDACHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    printf("[Rank 0] Sending %d elements...\n", count);
    NCCLCHECK(ncclSend(d_data, count, ncclFloat, 1, comm, stream));
    printf("[Rank 0] Send initiated.\n");

  } else if (rank == 1) {
    // --- RECEIVER ---
    // Initialize with zeros to be sure
    CUDACHECK(cudaMemset(d_data, 0, bytes));

    printf("[Rank 1] Receiving %d elements...\n", count);
    NCCLCHECK(ncclRecv(d_data, count, ncclFloat, 0, comm, stream));
    printf("[Rank 1] Recv initiated.\n");
  }

  // 5. Synchronize
  CUDACHECK(cudaStreamSynchronize(stream));

  // 6. Verify Logic
  if (rank == 1) {
    CUDACHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    int errors = 0;
    for (int i = 0; i < count; ++i) {
      if (h_data[i] != (float)i) {
        errors++;
        if (errors < 10)
          printf("Error at %d: expected %.1f got %.1f\n", i, (float)i,
                 h_data[i]);
      }
    }
    if (errors == 0) {
      printf("\n[SUCCESS] Rank 1 received all data correctly!\n");
    } else {
      printf("\n[FAILURE] Found %d errors in received data.\n", errors);
    }
  }

  // Cleanup
  CUDACHECK(cudaFree(d_data));
  free(h_data);
  ncclCommDestroy(comm);
  MPI_Finalize();
  return 0;
}
