#ifndef MOCK_CUDA_NCCL_H
#define MOCK_CUDA_NCCL_H

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// --- MPI Mocks ---
// We can't use real MPI headers if we don't have it.
// Define minimal subsets to allow compilation.
#define MPI_COMM_WORLD 0
#define MPI_BYTE 0
#define MPI_FLOAT 1
#define MPI_INT 2
typedef int MPI_Comm;
typedef int MPI_Datatype;

inline int MPI_Init(int *argc, char ***argv) {
  printf("[MOCK MPI] Init\n");
  return 0;
}
inline int MPI_Finalize() {
  printf("[MOCK MPI] Finalize\n");
  return 0;
}
inline int MPI_Comm_rank(MPI_Comm comm, int *rank) {
  char *env = getenv("MOCK_RANK");
  if (env)
    *rank = atoi(env);
  else
    *rank = 0; // Default to 0
  return 0;
}
inline int MPI_Comm_size(MPI_Comm comm, int *size) {
  *size = 2; // Default to 2 ranks
  return 0;
}
inline int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
                     MPI_Comm comm) {
  // No-op for single process, or assume 0 is root and we are 0.
  return 0;
}
inline int MPI_Barrier(MPI_Comm comm) { return 0; }

// --- CUDA Mocks ---
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaStreamDefault 0
typedef int cudaStream_t;
typedef int cudaEvent_t;
struct cudaDeviceProp {
  char name[256];
  int major;
};

// Macros to hide kernel launches
#ifdef ENABLE_CPU_MOCK
#define KERNEL_LAUNCH(name, grid, block, ...)                                  \
  printf("[MOCK] Launching kernel " #name "\n");
#define KERNEL_LAUNCH_SMEM(name, grid, block, smem, ...)                       \
  printf("[MOCK] Launching kernel " #name "\n");
#endif

// Functions
inline const char *cudaGetErrorString(cudaError_t error) { return "no error"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaSetDevice(int device) { return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
  strcpy(prop->name, "CPU_MOCK_DEVICE");
  prop->major = 8;
  return cudaSuccess;
}
inline cudaError_t cudaMalloc(void **devPtr, size_t size) {
  *devPtr = malloc(size);
  if (*devPtr == NULL) {
    printf("Malloc failed for size %lu\n", size);
    return 1;
  }
  memset(*devPtr, 0, size); // Initialize to 0
  return cudaSuccess;
}
inline cudaError_t cudaFree(void *devPtr) {
  free(devPtr);
  return cudaSuccess;
}
inline cudaError_t cudaMallocHost(void **ptr, size_t size) {
  return cudaMalloc(ptr, size);
}
inline cudaError_t cudaFreeHost(void *ptr) { return cudaFree(ptr); }
inline cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
  memset(devPtr, value, count);
  return cudaSuccess;
}

// Memcpy directions
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2
#define cudaMemcpyDeviceToDevice 3
typedef int cudaMemcpyKind;

inline cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                              cudaMemcpyKind kind) {
  memcpy(dst, src, count);
  return cudaSuccess;
}

inline cudaError_t cudaEventCreate(cudaEvent_t *event) { return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t event) { return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0) {
  return cudaSuccess;
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  return cudaSuccess;
}
inline cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
                                        cudaEvent_t end) {
  *ms = 1.0f;
  return cudaSuccess;
}

// --- cuBLAS Mocks ---
typedef int cublasHandle_t;
typedef int cublasStatus_t;
typedef int cublasMath_t;
typedef int cublasOperation_t;
typedef int cublasComputeType_t;
#define CUBLAS_STATUS_SUCCESS 0
#define CUBLAS_OP_N 0
#define CUBLAS_OP_T 1
#define CUBLAS_COMPUTE_32F 0
#define CUBLAS_COMPUTE_32F_FAST_TF32 1
#define CUBLAS_TF32_TENSOR_OP_MATH 1
#define CUBLAS_DEFAULT_MATH 0

inline cublasStatus_t cublasCreate(cublasHandle_t *handle) {
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDestroy(cublasHandle_t handle) {
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSetMathMode(cublasHandle_t handle,
                                        cublasMath_t mode) {
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSgemm(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb, int m, int n, int k,
                                  const float *alpha, const float *A, int lda,
                                  const float *B, int ldb, const float *beta,
                                  float *C, int ldc) {
  // Determine output dimensions based on transpose
  int rows = (transa == CUBLAS_OP_N) ? m : k;
  int cols = (transb == CUBLAS_OP_N)
                 ? n
                 : k; // Approximate logic, strict checking simplified
  // CPU Mock Matmul: Just fill with some dummy value or do nothing
  // For debugging flow, we don't strictly need result validity, but let's zero
  // it or set to alpha Iterating is too slow for large matrices on CPU usually.
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float *alpha, const void *A, int lda,
    long long strideA, const void *B, int ldb, long long strideB,
    const float *beta, void *C, int ldc, long long strideC, int batchCount) {
  return CUBLAS_STATUS_SUCCESS;
}

// --- NCCL Mocks ---
typedef struct ncclComm *ncclComm_t;
typedef struct {
  char internal[32];
} ncclUniqueId;
typedef int ncclResult_t;
typedef int ncclDataType_t;
#define ncclSuccess 0
#define ncclFloat 0

inline const char *ncclGetErrorString(ncclResult_t result) {
  return "no error";
}
inline ncclResult_t ncclGetUniqueId(ncclUniqueId *uniqueId) {
  memset(uniqueId, 0, sizeof(ncclUniqueId));
  return ncclSuccess;
}
inline ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks,
                                     ncclUniqueId commId, int rank) {
  *comm = (ncclComm_t)malloc(1);
  printf("[MOCK NCCL] InitRank: rank=%d, nranks=%d\n", rank, nranks);
  return ncclSuccess;
}
inline ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  free(comm);
  return ncclSuccess;
}

inline ncclResult_t ncclSend(const void *sendbuff, size_t count,
                             ncclDataType_t datatype, int peer, ncclComm_t comm,
                             cudaStream_t stream) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("[MOCK NCCL] SEND -> Rank %d (count=%lu)\n", peer, count);

  char filename[256];
  sprintf(filename, "nccl_channel_%d_%d.bin", rank, peer);
  FILE *fp = fopen(filename, "wb");
  if (fp) {
    fwrite(sendbuff, sizeof(float), count,
           fp); // Assuming float for now as mostly used
    fclose(fp);
    printf("[MOCK NCCL] Wrote data to %s\n", filename);
  } else {
    printf("[MOCK NCCL] Failed to write to %s\n", filename);
  }
  return ncclSuccess;
}

inline ncclResult_t ncclRecv(void *recvbuff, size_t count,
                             ncclDataType_t datatype, int peer, ncclComm_t comm,
                             cudaStream_t stream) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("[MOCK NCCL] RECV <- Rank %d (count=%lu)\n", peer, count);

  char filename[256];
  sprintf(filename, "nccl_channel_%d_%d.bin", peer, rank);

  // Simple polling wait for file
  printf("[MOCK NCCL] Waiting for %s...\n", filename);
  FILE *fp = NULL;
  int retries = 0;
  while (fp == NULL && retries < 20) { // Wait up to ~20 seconds
    fp = fopen(filename, "rb");
    if (fp == NULL) {
      sleep(1); // sleep 1s
      retries++;
      if (retries % 5 == 0)
        printf("[MOCK NCCL] Still waiting for %s...\n", filename);
    }
  }

  if (fp) {
    fread(recvbuff, sizeof(float), count, fp);
    fclose(fp);
    printf("[MOCK NCCL] Read data from %s\n", filename);
  } else {
    printf("[MOCK NCCL] Timeout waiting for %s! Filling with dummy data.\n",
           filename);
    float *fbuf = (float *)recvbuff;
    for (size_t i = 0; i < count && i < 10; i++)
      fbuf[i] = 1.23f + i;
  }
  return ncclSuccess;
}

// Other utilities
#define __restrict__
#define __device__
#define __global__
#define __shared__

// Cooperative Groups Mocks
namespace cooperative_groups {
struct thread_block {};
template <int N> struct thread_block_tile {};
} // namespace cooperative_groups

// Curand Mocks
typedef int curandState;
inline void curand_init(unsigned long long seed, unsigned long long sequence,
                        unsigned long long offset, curandState *state) {}
inline float curand_uniform(curandState *state) {
  return (float)rand() / RAND_MAX;
}
inline float random_f32(curandState *state) {
  return curand_uniform(state);
} // Assuming this wrapper exists or is used

#endif // MOCK_CUDA_NCCL_H
