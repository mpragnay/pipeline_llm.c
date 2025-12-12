#ifndef DEBUG_UTILS_H
#define DEBUG_UTILS_H

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// Kernel to check for NaN/Inf and compute statistics
__global__ void check_nan_inf_kernel(const float *data, int N, int *has_nan,
                                     int *has_inf, float *min_val,
                                     float *max_val, float *sum_val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int s_has_nan;
  __shared__ int s_has_inf;
  __shared__ float s_min;
  __shared__ float s_max;
  __shared__ float s_sum;

  if (threadIdx.x == 0) {
    s_has_nan = 0;
    s_has_inf = 0;
    s_min = FLT_MAX;
    s_max = -FLT_MAX;
    s_sum = 0.0f;
  }
  __syncthreads();

  if (idx < N) {
    float val = data[idx];
    if (isnan(val)) {
      atomicAdd(&s_has_nan, 1);
    }
    if (isinf(val)) {
      atomicAdd(&s_has_inf, 1);
    }
    atomicMin((int *)&s_min, __float_as_int(val));
    atomicMax((int *)&s_max, __float_as_int(val));
    atomicAdd(&s_sum, val);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(has_nan, s_has_nan);
    atomicAdd(has_inf, s_has_inf);
    atomicMin((int *)min_val, __float_as_int(s_min));
    atomicMax((int *)max_val, __float_as_int(s_max));
    atomicAdd(sum_val, s_sum);
  }
}

// Host function to check GPU tensor for NaN/Inf
void check_tensor_debug(const char *name, const float *d_data, int size,
                        int rank, bool abort_on_error = true) {
  static int *d_has_nan = nullptr;
  static int *d_has_inf = nullptr;
  static float *d_min_val = nullptr;
  static float *d_max_val = nullptr;
  static float *d_sum = nullptr;

  // Allocate device memory for results (once)
  if (d_has_nan == nullptr) {
    cudaMalloc(&d_has_nan, sizeof(int));
    cudaMalloc(&d_has_inf, sizeof(int));
    cudaMalloc(&d_min_val, sizeof(float));
    cudaMalloc(&d_max_val, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
  }

  // Zero out counters
  int zero = 0;
  float flt_max = FLT_MAX;
  float flt_min = -FLT_MAX;
  float fzero = 0.0f;
  cudaMemcpy(d_has_nan, &zero, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_has_inf, &zero, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_min_val, &flt_max, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_max_val, &flt_min, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sum, &fzero, sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel
  int block_size = 256;
  int num_blocks = (size + block_size - 1) / block_size;
  check_nan_inf_kernel<<<num_blocks, block_size>>>(
      d_data, size, d_has_nan, d_has_inf, d_min_val, d_max_val, d_sum);
  cudaDeviceSynchronize();

  // Copy results back
  int has_nan, has_inf;
  float min_val, max_val, sum_val;
  cudaMemcpy(&has_nan, d_has_nan, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&has_inf, d_has_inf, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&min_val, d_min_val, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&max_val, d_max_val, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&sum_val, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

  float mean = sum_val / size;

  const char *status_prefix = (has_nan > 0 || has_inf > 0) ? "[ERROR]" : "[OK]";
  printf("[Stage %d] %s CHECK %s: size=%d, NaN=%d, Inf=%d, min=%.6e, max=%.6e, "
         "mean=%.6e\n",
         rank, status_prefix, name, size, has_nan, has_inf, min_val, max_val,
         mean);

  if (abort_on_error && (has_nan > 0 || has_inf > 0)) {
    printf("[Stage %d] [FATAL] NaN/Inf detected in %s! Aborting...\n", rank,
           name);
    exit(EXIT_FAILURE);
  }
}

// Compute gradient norm
__global__ void compute_norm_kernel(const float *data, int N,
                                    float *partial_sum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float s_sum[256];

  float thread_sum = 0.0f;
  if (idx < N) {
    float val = data[idx];
    thread_sum = val * val;
  }
  s_sum[threadIdx.x] = thread_sum;
  __syncthreads();

  // Reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(partial_sum, s_sum[0]);
  }
}

float compute_gradient_norm(const float *d_data, int size) {
  static float *d_norm = nullptr;
  if (d_norm == nullptr) {
    cudaMalloc(&d_norm, sizeof(float));
  }

  float zero = 0.0f;
  cudaMemcpy(d_norm, &zero, sizeof(float), cudaMemcpyHostToDevice);

  int block_size = 256;
  int num_blocks = (size + block_size - 1) / block_size;
  compute_norm_kernel<<<num_blocks, block_size>>>(d_data, size, d_norm);
  cudaDeviceSynchronize();

  float norm_sq;
  cudaMemcpy(&norm_sq, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
  return sqrtf(norm_sq);
}

#endif // DEBUG_UTILS_H
