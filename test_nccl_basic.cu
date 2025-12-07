/*
 * Basic NCCL two-GPU communication test (single node, multiple GPUs)
 * 
 * Compile: nvcc -o test_nccl_basic test_nccl_basic.cu -lnccl -lcudart
 * Run:     ./test_nccl_basic
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define N 8  // Number of elements to send
#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("CUDA error %s:%d '%s'\n",               \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("NCCL error %s:%d '%s'\n",               \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

__global__ void fill_data(float* data, int gpu_id, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Fill with pattern: GPU_ID * 100 + index
        data[idx] = (float)(gpu_id * 100 + idx);
    }
}

void print_buffer(const char* label, float* d_buf, int gpu_id, int n) {
    float* h_buf = (float*)malloc(n * sizeof(float));
    CUDACHECK(cudaMemcpy(h_buf, d_buf, n * sizeof(float), cudaMemcpyDeviceToHost));
    printf("[GPU %d] %s: ", gpu_id, label);
    for (int i = 0; i < n; i++) {
        printf("%.0f ", h_buf[i]);
    }
    printf("\n");
    free(h_buf);
}

int main(int argc, char* argv[]) {
    int nGpus = 2;
    int deviceCount;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    
    printf("Found %d GPUs\n", deviceCount);
    
    if (deviceCount < 2) {
        printf("Need at least 2 GPUs for this test. Found %d\n", deviceCount);
        printf("Running single-GPU loopback test instead...\n");
        nGpus = 1;
    }
    
    // Arrays for each GPU
    float** send_buf = (float**)malloc(nGpus * sizeof(float*));
    float** recv_buf = (float**)malloc(nGpus * sizeof(float*));
    cudaStream_t* streams = (cudaStream_t*)malloc(nGpus * sizeof(cudaStream_t));
    ncclComm_t* comms = (ncclComm_t*)malloc(nGpus * sizeof(ncclComm_t));
    
    // Initialize NCCL
    int* devs = (int*)malloc(nGpus * sizeof(int));
    for (int i = 0; i < nGpus; i++) {
        devs[i] = i;
    }
    NCCLCHECK(ncclCommInitAll(comms, nGpus, devs));
    printf("NCCL initialized with %d GPUs\n", nGpus);
    
    // Allocate memory and create streams on each GPU
    for (int i = 0; i < nGpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(&send_buf[i], N * sizeof(float)));
        CUDACHECK(cudaMalloc(&recv_buf[i], N * sizeof(float)));
        CUDACHECK(cudaStreamCreate(&streams[i]));
        
        // Initialize send buffer with unique data
        fill_data<<<1, N, 0, streams[i]>>>(send_buf[i], i, N);
        // Clear receive buffer
        CUDACHECK(cudaMemsetAsync(recv_buf[i], 0, N * sizeof(float), streams[i]));
    }
    
    // Sync all GPUs
    for (int i = 0; i < nGpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }
    
    // Print what each GPU is sending
    printf("\n=== Before Send ===\n");
    for (int i = 0; i < nGpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        print_buffer("Sending", send_buf[i], i, N);
        print_buffer("Recv buf", recv_buf[i], i, N);
    }
    
    // Perform send/receive using NCCL
    // GPU 0 sends to GPU 1, GPU 1 sends to GPU 0 (ring exchange)
    printf("\n=== Performing NCCL Send/Recv ===\n");
    
    if (nGpus >= 2) {
        // Use ncclGroupStart/End for multiple operations
        NCCLCHECK(ncclGroupStart());
        
        // GPU 0: send to GPU 1, receive from GPU 1
        CUDACHECK(cudaSetDevice(0));
        NCCLCHECK(ncclSend(send_buf[0], N, ncclFloat, 1, comms[0], streams[0]));
        NCCLCHECK(ncclRecv(recv_buf[0], N, ncclFloat, 1, comms[0], streams[0]));
        printf("[GPU 0] Sending to GPU 1, Receiving from GPU 1\n");
        
        // GPU 1: send to GPU 0, receive from GPU 0
        CUDACHECK(cudaSetDevice(1));
        NCCLCHECK(ncclSend(send_buf[1], N, ncclFloat, 0, comms[1], streams[1]));
        NCCLCHECK(ncclRecv(recv_buf[1], N, ncclFloat, 0, comms[1], streams[1]));
        printf("[GPU 1] Sending to GPU 0, Receiving from GPU 0\n");
        
        NCCLCHECK(ncclGroupEnd());
    } else {
        // Single GPU: just copy send to recv as loopback
        CUDACHECK(cudaSetDevice(0));
        CUDACHECK(cudaMemcpyAsync(recv_buf[0], send_buf[0], N * sizeof(float), 
                                   cudaMemcpyDeviceToDevice, streams[0]));
    }
    
    // Sync all GPUs
    for (int i = 0; i < nGpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }
    
    // Print what each GPU received
    printf("\n=== After Send ===\n");
    for (int i = 0; i < nGpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        print_buffer("Recv buf", recv_buf[i], i, N);
    }
    
    // Verify results
    printf("\n=== Verification ===\n");
    for (int i = 0; i < nGpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        float* h_recv = (float*)malloc(N * sizeof(float));
        CUDACHECK(cudaMemcpy(h_recv, recv_buf[i], N * sizeof(float), cudaMemcpyDeviceToHost));
        
        int expected_sender = (nGpus >= 2) ? (1 - i) : i;  // Other GPU or self
        float expected_first = (float)(expected_sender * 100);
        
        printf("[GPU %d] Expected first value: %.0f, Got: %.0f - %s\n",
               i, expected_first, h_recv[0],
               (h_recv[0] == expected_first) ? "PASS ✓" : "FAIL ✗");
        free(h_recv);
    }
    
    // Cleanup
    for (int i = 0; i < nGpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(send_buf[i]));
        CUDACHECK(cudaFree(recv_buf[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }
    
    free(send_buf);
    free(recv_buf);
    free(streams);
    free(comms);
    free(devs);
    
    printf("\nDone!\n");
    return 0;
}
