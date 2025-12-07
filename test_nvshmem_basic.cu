/*
 * Basic NVSHMEM two-GPU communication test (single node)
 * 
 * Compile: make test_nvshmem
 * Run:     NVSHMEM_BOOTSTRAP=MPI mpirun -np 2 ./test_nvshmem
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define N 8  // Number of elements to send

#define CUDA_CHECK(cmd) do { \
    cudaError_t e = cmd; \
    if (e != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

__global__ void fill_data(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx;  // Just 0, 1, 2, 3, 4, 5, 6, 7
    }
}

__global__ void verify_and_print(int* recv_buf, int my_pe, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[PE %d] Received: ", my_pe);
        for (int i = 0; i < n; i++) {
            printf("%d ", recv_buf[i]);
        }
        printf("\n");
        
        // Check if we got 0, 1, 2, ..., 7
        int pass = 1;
        for (int i = 0; i < n; i++) {
            if (recv_buf[i] != i) {
                pass = 0;
                break;
            }
        }
        if (pass) {
            printf("[PE %d] PASS - Received correct values 0 to %d\n", my_pe, n-1);
        } else {
            printf("[PE %d] FAIL - Expected 0 to %d\n", my_pe, n-1);
        }
    }
}

int main(int argc, char* argv[]) {
    // Initialize MPI first
    MPI_Init(&argc, &argv);
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // Set CUDA device based on local rank
    int n_devices;
    CUDA_CHECK(cudaGetDeviceCount(&n_devices));
    int my_device = mpi_rank % n_devices;
    CUDA_CHECK(cudaSetDevice(my_device));
    
    printf("[Rank %d] Set GPU %d (of %d GPUs)\n", mpi_rank, my_device, n_devices);
    
    // Initialize NVSHMEM with MPI communicator
    nvshmemx_init_attr_t attr;
    MPI_Comm comm = MPI_COMM_WORLD;
    attr.mpi_comm = &comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
    
    printf("[PE %d] NVSHMEM initialized. Total PEs: %d\n", my_pe, n_pes);
    
    if (n_pes < 2) {
        printf("[PE %d] Need at least 2 PEs\n", my_pe);
        nvshmem_finalize();
        MPI_Finalize();
        return 1;
    }
    
    // Create a CUDA stream for NVSHMEM operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Allocate symmetric memory
    int* send_buf = (int*)nvshmem_malloc(N * sizeof(int));
    int* recv_buf = (int*)nvshmem_malloc(N * sizeof(int));
    
    if (!send_buf || !recv_buf) {
        printf("[PE %d] Failed to allocate symmetric memory\n", my_pe);
        nvshmem_finalize();
        MPI_Finalize();
        return 1;
    }
    
    // Initialize buffers
    CUDA_CHECK(cudaMemsetAsync(recv_buf, 0xFF, N * sizeof(int), stream));
    fill_data<<<1, N, 0, stream>>>(send_buf, N);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Print send buffer on host
    int* h_send = (int*)malloc(N * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_send, send_buf, N * sizeof(int), cudaMemcpyDeviceToHost));
    printf("[PE %d] Sending: ", my_pe);
    for (int i = 0; i < N; i++) printf("%d ", h_send[i]);
    printf("\n");
    free(h_send);
    
    // Sync all PEs using MPI (safer than nvshmem_barrier_all for this test)
    MPI_Barrier(MPI_COMM_WORLD);
    
    // PE 0 sends to PE 1 only
    int next_pe = (my_pe + 1) % n_pes;
    
    if (my_pe == 0) {
        printf("[PE %d] Putting data to PE %d\n", my_pe, next_pe);
        nvshmem_int_put(recv_buf, send_buf, N, next_pe);
    }
    
    // Ensure put completes
    nvshmem_quiet();
    
    // Sync again
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Only PE 1 verifies (it received the data)
    if (my_pe == 1) {
        verify_and_print<<<1, 1, 0, stream>>>(recv_buf, my_pe, N);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    nvshmem_free(send_buf);
    nvshmem_free(recv_buf);
    
    nvshmem_finalize();
    MPI_Finalize();
    
    return 0;
}
