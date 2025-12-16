/*
GPT-2 Transformer Neural Net trained in raw CUDA with Model Partitioning
This version partitions the model across GPUs to save memory.
Each GPU only holds its assigned layers' parameters and activations.
*/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// GPU / CUDA related
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
// NVSHMEM for multi-GPU communication
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
// our own utilities
#include "llmc/dataloader.h"
#include "llmc/tokenizer.h"
#include "llmc/utils.h"

// ----------------------------------------------------------------------------
// CUDA utils

int counter = 0;

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

void cublasCheck(cublasStatus_t status, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
    exit(EXIT_FAILURE);
  }
}
#define cublasCheck(status)                                                    \
  {                                                                            \
    cublasCheck((status), __FILE__, __LINE__);                                 \
  }

static cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;

namespace cg = cooperative_groups;

// ----------------------------------------------------------------------------
// Pipeline Partition Structure

typedef struct {
  int first_layer;   // First layer index (global)
  int num_layers;    // Number of layers this PE handles
  int has_embedding; // 1 if this PE does embedding (PE 0)
  int has_final_ln;  // 1 if this PE does final layernorm (PE 1)
} PipelinePartition;

void get_partition_for_pe(PipelinePartition *part, int my_pe, int n_pes,
                          int total_layers) {
  int base_layers = total_layers / n_pes;
  int remainder = total_layers % n_pes;

  // Distribute remainder: first 'remainder' PEs get one extra layer
  part->num_layers = base_layers + (my_pe < remainder ? 1 : 0);

  // Calculate first_layer: sum of layers assigned to all previous PEs
  part->first_layer = 0;
  for (int pe = 0; pe < my_pe; pe++) {
    part->first_layer += base_layers + (pe < remainder ? 1 : 0);
  }

  part->has_embedding = (my_pe == 0);
  part->has_final_ln = (my_pe == n_pes - 1);
}

// ----------------------------------------------------------------------------
// all the kernels

// Kernel for accumulating gradients on GPU (used in NVSHMEM allreduce)
__global__ void accumulate_grads_kernel(float *dst, const float *src,
                                        size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] += src[idx];
  }
}

__device__ inline float4 add_float4(const float4 &a, const float4 &b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void encoder_forward_kernel3(float4 *out, const int *inp,
                                        const float4 *wte, const float4 *wpe,
                                        int B, int T, int C) {
  int C4 = C / 4;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = B * T * C4;
  if (idx < N) {
    int bt = idx / C4;
    int b = bt / T;
    int t = bt % T;
    int c4 = idx % C4;
    int ix = inp[b * T + t];
    out[b * T * C4 + t * C4 + c4] =
        add_float4(wte[ix * C4 + c4], wpe[t * C4 + c4]);
  }
}

__global__ void encoder_backward_kernel(float *dwte, float *dwpe,
                                        const float *dout, const int *inp,
                                        int B, int T, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = B * T * C;
  if (idx < N) {
    int bt = idx / C;
    int b = bt / T;
    int t = bt % T;
    int c = idx % C;
    int ix = inp[b * T + t];
    const float *dout_btc = dout + b * T * C + t * C + c;
    float *dwte_ix = dwte + ix * C + c;
    float *dwpe_tc = dwpe + t * C + c;
    atomicAdd(dwte_ix, *dout_btc);
    atomicAdd(dwpe_tc, *dout_btc);
  }
}

__global__ void layernorm_forward_kernel3(
    float *__restrict__ out, float *__restrict__ mean, float *__restrict__ rstd,
    const float *__restrict__ inp, const float *__restrict__ weight,
    const float *__restrict__ bias, int N, int C) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
  if (idx >= N)
    return;

  const float *x = inp + idx * C;
  float sum = 0.0f;
  for (int i = warp.thread_rank(); i < C; i += warp.size()) {
    sum += x[i];
  }
  sum = cg::reduce(warp, sum, cg::plus<float>{});
  float m = sum / C;
  if (warp.thread_rank() == 0 && mean != nullptr) {
    __stcs(mean + idx, m);
  }

  sum = 0.0f;
  for (int i = warp.thread_rank(); i < C; i += warp.size()) {
    float diff = x[i] - m;
    sum += diff * diff;
  }
  sum = cg::reduce(warp, sum, cg::plus<float>{});
  float s = rsqrtf(sum / C + 1e-5f);
  if (warp.thread_rank() == 0 && rstd != nullptr) {
    __stcs(rstd + idx, s);
  }

  float *o = out + idx * C;
  for (int c = warp.thread_rank(); c < C; c += warp.size()) {
    float n = s * (__ldcs(x + c) - m);
    __stcs(o + c, n * weight[c] + bias[c]);
  }
}

__global__ void permute_kernel(float *q, float *k, float *v, const float *inp,
                               int B, int N, int NH, int d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < B * NH * N * d) {
    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int inp_idx =
        (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
    q[idx] = __ldcs(&inp[inp_idx]);
    k[idx] = __ldcs(&inp[inp_idx + NH * d]);
    v[idx] = __ldcs(&inp[inp_idx + 2 * (NH * d)]);
  }
}

__global__ void permute_kernel_backward(float *dinp, const float *dq,
                                        const float *dk, const float *dv, int B,
                                        int N, int NH, int d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < B * NH * N * d) {
    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int inp_idx =
        (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
    dinp[inp_idx] = dq[idx];
    dinp[inp_idx + NH * d] = dk[idx];
    dinp[inp_idx + 2 * (NH * d)] = dv[idx];
  }
}

__global__ void unpermute_kernel(float *inp, float *out, int B, int N, int NH,
                                 int d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < B * NH * N * d) {
    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    out[other_idx] = __ldcs(&inp[idx]);
  }
}

__global__ void unpermute_kernel_backward(float *dinp, const float *dout, int B,
                                          int N, int NH, int d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < B * NH * N * d) {
    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    dinp[idx] = dout[other_idx];
  }
}

__device__ float &vec_at(float4 &vec, int index) {
  return reinterpret_cast<float *>(&vec)[index];
}

__device__ float vec_at(const float4 &vec, int index) {
  return reinterpret_cast<const float *>(&vec)[index];
}

__global__ void softmax_forward_kernel5(float *out, float inv_temperature,
                                        const float *inp, int N, int T) {
  assert(T % 4 == 0);
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  int idx = (gridDim.x - blockIdx.x - 1) * warp.meta_group_size() +
            warp.meta_group_rank();
  if (idx >= N * T)
    return;
  int own_pos = idx % T;
  int pos_by_4 = own_pos / 4;

  const float *x = inp + idx * T;
  float maxval = -FLT_MAX;
  float sumval = 0.0f;

  const float4 *x_vec = reinterpret_cast<const float4 *>(x);
  for (int i = warp.thread_rank(); i < pos_by_4; i += warp.size()) {
    float4 v = x_vec[i];
    float old_maxval = maxval;
    for (int k = 0; k < 4; ++k) {
      maxval = fmaxf(maxval, vec_at(v, k));
    }
    sumval *= expf(inv_temperature * (old_maxval - maxval));
    for (int k = 0; k < 4; ++k) {
      sumval += expf(inv_temperature * (vec_at(v, k) - maxval));
    }
  }

  if (4 * pos_by_4 + warp.thread_rank() <= own_pos) {
    float old_maxval = maxval;
    maxval = fmaxf(maxval, x[4 * pos_by_4 + warp.thread_rank()]);
    sumval *= expf(inv_temperature * (old_maxval - maxval));
    sumval +=
        expf(inv_temperature * (x[4 * pos_by_4 + warp.thread_rank()] - maxval));
  }

  float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
  sumval *= expf(inv_temperature * (maxval - global_maxval));
  float sum = cg::reduce(warp, sumval, cg::plus<float>{});
  float norm = 1.f / sum;

  for (int i = warp.thread_rank(); i <= own_pos; i += warp.size()) {
    float ev = expf(inv_temperature * (__ldcs(x + i) - global_maxval));
    __stcs(out + idx * T + i, ev * norm);
  }
}

__global__ void residual_forward_kernel(float *out, float *inp1, float *inp2,
                                        int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    out[idx] = __ldcs(&inp1[idx]) + __ldcs(&inp2[idx]);
  }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
__global__ void gelu_forward_kernel(float *out, const float *inp, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float xi = inp[i];
    float cube = 0.044715f * xi * xi * xi;
    out[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
  }
}

__global__ void gelu_backward_kernel(float *dinp, const float *inp,
                                     const float *dout, const int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float x = inp[i];
    float cube = 0.044715f * x * x * x;
    float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
    float tanh_out = tanhf(tanh_arg);
    float coshf_out = coshf(tanh_arg);
    float sech_out = 1.0f / (coshf_out * coshf_out);
    float local_grad =
        0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR *
                                       (1.0f + 3.0f * 0.044715f * x * x);
    dinp[i] = local_grad * dout[i];
  }
}

__global__ void matmul_backward_bias_kernel4(float *dbias, const float *dout,
                                             int B, int T, int OC) {
  extern __shared__ float smem[];
  const int warp_id = threadIdx.x / warpSize;
  const int lane_id = threadIdx.x % warpSize;
  const int tl = blockIdx.x * warpSize;
  const int vstep = blockDim.x / warpSize;

  const float *dout_col = dout + tl + lane_id;
  float dout_sum = 0.0f;
  for (int row = warp_id; row < B * T; row += vstep) {
    dout_sum += dout_col[row * OC];
  }
  smem[lane_id + warp_id * warpSize] = dout_sum;
  __syncthreads();

  dout_sum = 0.0f;
  if (warp_id == 0) {
    for (int j = 0; j < vstep; j++) {
      dout_sum += smem[lane_id + j * warpSize];
    }
    dbias[tl + lane_id] += dout_sum;
  }
}

__global__ void layernorm_backward_kernel2(float *dinp, float *dweight,
                                           float *dbias, const float *dout,
                                           const float *inp,
                                           const float *weight,
                                           const float *mean, const float *rstd,
                                           int B, int T, int C) {
  extern __shared__ float shared[];
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
  int N = B * T;
  if (idx >= N)
    return;

  int b = idx / T;
  int t = idx % T;

  const float *dout_bt = dout + b * T * C + t * C;
  const float *inp_bt = inp + b * T * C + t * C;
  float *dinp_bt = dinp + b * T * C + t * C;
  const float mean_bt = mean[b * T + t];
  const float rstd_bt = rstd[b * T + t];

  float *dbias_shared = shared;
  float *dweight_shared = shared + C;

  for (int i = threadIdx.x; i < C; i += blockDim.x) {
    dbias_shared[i] = 0.0f;
    dweight_shared[i] = 0.0f;
  }
  __syncthreads();

  float dnorm_mean = 0.0f;
  float dnorm_norm_mean = 0.0f;
  for (int i = warp.thread_rank(); i < C; i += warp.size()) {
    float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
    float dnorm_i = weight[i] * dout_bt[i];
    dnorm_mean += dnorm_i;
    dnorm_norm_mean += dnorm_i * norm_bti;
  }
  dnorm_mean = cg::reduce(warp, dnorm_mean, cg::plus<float>{});
  dnorm_norm_mean = cg::reduce(warp, dnorm_norm_mean, cg::plus<float>{});
  dnorm_mean = dnorm_mean / C;
  dnorm_norm_mean = dnorm_norm_mean / C;

  for (int i = warp.thread_rank(); i < C; i += warp.size()) {
    float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
    float dnorm_i = weight[i] * dout_bt[i];
    atomicAdd(&dbias_shared[i], dout_bt[i]);
    atomicAdd(&dweight_shared[i], norm_bti * dout_bt[i]);
    float dval = 0.0f;
    dval += dnorm_i;
    dval -= dnorm_mean;
    dval -= norm_bti * dnorm_norm_mean;
    dval *= rstd_bt;
    dinp_bt[i] += dval;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < C; i += blockDim.x) {
    atomicAdd(&dbias[i], dbias_shared[i]);
    atomicAdd(&dweight[i], dweight_shared[i]);
  }
}

__global__ void softmax_autoregressive_backward_kernel(float *dpreatt,
                                                       const float *datt,
                                                       const float *att, int B,
                                                       int T, int C,
                                                       float scale) {
  constexpr const int BlockSize = 256;
  constexpr int T_per_block = 4;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  __shared__ float block_acc[32];

  int idx = blockIdx.y;
  int t0 = T - 1 - T_per_block * blockIdx.x;

  att += idx * T * T;
  datt += idx * T * T;
  dpreatt += idx * T * T;

  if (warp.meta_group_rank() == 0) {
    block_acc[warp.thread_rank()] = 0;
  }

  for (int to = 0; to < T_per_block; ++to) {
    int t = t0 - to;
    if (t < 0)
      return;
    const float *att_bth = att + t * T;
    const float *datt_bth = datt + t * T;
    float *dpreatt_bth = dpreatt + t * T;

    float local_sum = 0;
    for (int t2 = block.thread_rank(); t2 <= t; t2 += BlockSize) {
      local_sum += att_bth[t2] * datt_bth[t2];
    }

    block_acc[warp.meta_group_rank()] =
        cg::reduce(warp, local_sum, cg::plus<float>{});
    block.sync();
    local_sum =
        cg::reduce(warp, block_acc[warp.thread_rank()], cg::plus<float>{});

    for (int t3 = block.thread_rank(); t3 <= t; t3 += BlockSize) {
      float acc = __ldcs(att_bth + t3) * (__ldcs(datt_bth + t3) - local_sum);
      __stcs(dpreatt_bth + t3, scale * acc);
    }
  }
}

__device__ inline float lerp(float start, float end, float weight) {
  return fma(weight, end, fma(-weight, start, start));
}

__global__ void adamw_kernel2(float *params_memory, float *grads_memory,
                              float *m_memory, float *v_memory,
                              long num_parameters, float learning_rate,
                              float beta1, float beta2, float beta1_correction,
                              float beta2_correction, float eps,
                              float weight_decay) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_parameters)
    return;
  float grad = grads_memory[i];
  float m = m_memory[i];
  float v = v_memory[i];
  m = lerp(grad, m, beta1);
  m_memory[i] = m;
  v = lerp(grad * grad, v, beta2);
  v_memory[i] = v;
  m /= beta1_correction;
  v /= beta2_correction;
  params_memory[i] -=
      learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
}

struct SoftmaxParams {
  float Scale;
  float Offset;
};

__device__ SoftmaxParams prepare_softmax_blockwide_nofloat4(
    cg::thread_block_tile<32> &warp, int idx, const float *inp, int V, int P) {
  const float *x = inp + idx * P;
  float thread_maxval = -INFINITY;
  float thread_sumval = 0.0f;
  for (int i = V + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x) {
    float v = x[i];
    float old_maxval = thread_maxval;
    thread_maxval = fmaxf(thread_maxval, v);
    thread_sumval *= expf((old_maxval - thread_maxval));
    thread_sumval += expf(v - thread_maxval);
  }

  __shared__ float shared_maxval[32];
  __shared__ float shared_sumval[32];
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;

  float warp_maxval = cg::reduce(warp, thread_maxval, cg::greater<float>{});
  if (lane_id == 0)
    shared_maxval[warp_id] = warp_maxval;
  __syncthreads();
  warp_maxval = (lane_id < num_warps) ? shared_maxval[lane_id] : -FLT_MAX;
  float block_maxval = cg::reduce(warp, warp_maxval, cg::greater<float>{});
  thread_sumval *= expf(thread_maxval - block_maxval);
  float warp_sumval = cg::reduce(warp, thread_sumval, cg::plus<float>{});
  if (lane_id == 0)
    shared_sumval[warp_id] = warp_sumval;
  __syncthreads();
  warp_sumval = (lane_id < num_warps) ? shared_sumval[lane_id] : 0.0f;
  float block_sumval = cg::reduce(warp, warp_sumval, cg::plus<float>{});
  return SoftmaxParams{1.f / block_sumval, block_maxval};
}

__global__ void fused_classifier_kernel3(float *logits, float *losses,
                                         float *probs, const float *dlosses,
                                         const int *targets, int B, int T,
                                         int V, int P) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  int idx = blockIdx.x;
  int ix = targets[idx];

  SoftmaxParams sp =
      prepare_softmax_blockwide_nofloat4(warp, idx, logits, V, P);

  if (threadIdx.x == 0) {
    float prob = expf(logits[idx * P + ix] - sp.Offset) * sp.Scale;
    losses[idx] = -logf(prob);
  }

  float dloss = dlosses != NULL ? dlosses[idx] : 1.0f / (B * T);
  const float *logits_vec = logits + idx * P;
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    float v = __ldcs(&logits_vec[i]);
    float prob = expf(v - sp.Offset) * sp.Scale;
    if (probs != NULL) {
      probs[idx * P + i] = prob;
    }
    float indicator = (i == ix) ? 1.0f : 0.0f;
    logits[idx * P + i] = (prob - indicator) * dloss;
  }
}

__device__ float4 ld_vec(const float *address) {
  return *reinterpret_cast<const float4 *>(address);
}

__device__ void st_vec(float *address, float4 val) {
  *reinterpret_cast<float4 *>(address) = val;
}

__global__ void __launch_bounds__(16 * 16, 2)
    matmul_forward_kernel4(float *out, const float *inp, const float *weight,
                           const float *bias, int C, int OC) {
  int oc = 8 * (blockIdx.y * blockDim.y + threadIdx.y);

  __shared__ float lhs_s[128][32];
  __shared__ float rhs_s[128][32];

  inp += 128 * blockIdx.x * C;
  weight += 128 * blockIdx.y * C;
  out += 128 * blockIdx.x * OC + 128 * blockIdx.y;

  float vals[8][8] = {};
  if (bias != NULL) {
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j += 4) {
        float4 b = ld_vec(bias + oc + j);
        vals[i][j + 0] = b.x;
        vals[i][j + 1] = b.y;
        vals[i][j + 2] = b.z;
        vals[i][j + 3] = b.w;
      }
    }
  }

  int si_start = 4 * (16 * threadIdx.y + threadIdx.x);
  for (int so = 0; so < C; so += 32) {
    __syncthreads();
    int xmod8 = threadIdx.x % 8;
    int xby8 = threadIdx.x / 8;
    int xo = 4 * xmod8;
    for (int y = 2 * threadIdx.y + xby8; y < 128; y += 32) {
      st_vec(&lhs_s[y][xo], ld_vec(inp + y * C + so + xo));
      st_vec(&rhs_s[y][xo], ld_vec(weight + y * C + so + xo));
    }
    __syncthreads();

    for (int si = si_start; si < si_start + 32; si += 4) {
      float4 rhs[8];
      for (int u = 0; u < 8; ++u) {
        rhs[u] = ld_vec(&rhs_s[u + 8 * threadIdx.y][si % 32]);
      }
      for (int ii = 0; ii < 8; ++ii) {
        float4 lhs = ld_vec(&lhs_s[ii + 8 * threadIdx.x][si % 32]);
        for (int ji = 0; ji < 8; ++ji) {
          vals[ii][ji] += lhs.x * rhs[ji].x;
          vals[ii][ji] += lhs.y * rhs[ji].y;
          vals[ii][ji] += lhs.z * rhs[ji].z;
          vals[ii][ji] += lhs.w * rhs[ji].w;
        }
      }
    }
  }

  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; j += 4) {
      float4 result;
      result.x = vals[i][j + 0];
      result.y = vals[i][j + 1];
      result.z = vals[i][j + 2];
      result.w = vals[i][j + 3];
      st_vec(out + (8 * threadIdx.x + i) * OC + 8 * threadIdx.y + j, result);
    }
  }
}

// ----------------------------------------------------------------------------
// kernel launchers

void encoder_forward(float *out, const int *inp, const float *wte,
                     const float *wpe, int B, int T, int C) {
  assert(C % 4 == 0);
  const int block_size = 512;
  const int N = B * T * C;
  const int grid_size = CEIL_DIV(N / 4, block_size);
  encoder_forward_kernel3<<<grid_size, block_size>>>(
      (float4 *)out, inp, (float4 *)wte, (float4 *)wpe, B, T, C);
  cudaCheck(cudaGetLastError());
}

void encoder_backward(float *dwte, float *dwpe, const float *dout,
                      const int *inp, int B, int T, int C) {
  const int N = B * T * C;
  const int block_size = 256;
  const int grid_size = CEIL_DIV(N, block_size);
  encoder_backward_kernel<<<grid_size, block_size>>>(dwte, dwpe, dout, inp, B,
                                                     T, C);
  cudaCheck(cudaGetLastError());
}

void layernorm_forward(float *out, float *mean, float *rstd, float *inp,
                       float *weight, float *bias, int B, int T, int C) {
  const int block_size = 512;
  const int N = B * T;
  const int grid_size = CEIL_DIV(N * 32, block_size);
  layernorm_forward_kernel3<<<grid_size, block_size>>>(out, mean, rstd, inp,
                                                       weight, bias, N, C);
  cudaCheck(cudaGetLastError());
}

void matmul_forward(float *out, const float *inp, const float *weight,
                    const float *bias, int B, int T, int C, int OC) {
  int sqrt_block_size = 16;
  dim3 gridDim(CEIL_DIV(B * T, 8 * sqrt_block_size),
               CEIL_DIV(OC, 8 * sqrt_block_size));
  dim3 blockDim(sqrt_block_size, sqrt_block_size);
  matmul_forward_kernel4<<<gridDim, blockDim>>>(out, inp, weight, bias, C, OC);
  cudaCheck(cudaGetLastError());
}

void attention_forward(float *out, float *qkvr, float *att, float *inp, int B,
                       int T, int C, int NH) {
  const int block_size = 256;
  const int softmax_block_size = 256;
  int HS = C / NH;

  float *q, *k, *v;
  q = qkvr + 0 * B * T * C;
  k = qkvr + 1 * B * T * C;
  v = qkvr + 2 * B * T * C;
  int total_threads = B * NH * T * HS;
  int num_blocks = CEIL_DIV(total_threads, block_size);
  permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
  cudaCheck(cudaGetLastError());

  const float alpha = 1.0f;
  const float beta = 0.0f;
  float *preatt = inp;
  cublasCheck(cublasSgemmStridedBatched(
      cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &alpha, k, HS, T * HS,
      q, HS, T * HS, &beta, preatt, T, T * T, B * NH));

  float scale = 1.0 / sqrtf(HS);
  int grid_size = CEIL_DIV(B * NH * T * 32, softmax_block_size);
  softmax_forward_kernel5<<<grid_size, softmax_block_size>>>(att, scale, preatt,
                                                             B * NH, T);
  cudaCheck(cudaGetLastError());

  float *vaccum = inp;
  cublasCheck(cublasSgemmStridedBatched(
      cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &alpha, v, HS, T * HS,
      att, T, T * T, &beta, vaccum, HS, T * HS, B * NH));

  num_blocks = CEIL_DIV(B * T * C, block_size);
  unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
  cudaCheck(cudaGetLastError());
}

void residual_forward(float *out, float *inp1, float *inp2, int N) {
  const int block_size = 256;
  const int grid_size = CEIL_DIV(N, block_size);
  residual_forward_kernel<<<grid_size, block_size>>>(out, inp1, inp2, N);
  cudaCheck(cudaGetLastError());
}

void gelu_forward(float *out, const float *inp, int N) {
  const int block_size = 128;
  const int grid_size = CEIL_DIV(N, block_size);
  gelu_forward_kernel<<<grid_size, block_size>>>(out, inp, N);
  cudaCheck(cudaGetLastError());
}

void gelu_backward(float *dinp, const float *inp, const float *dout,
                   const int N) {
  const int block_size = 128;
  const int grid_size = CEIL_DIV(N, block_size);
  gelu_backward_kernel<<<grid_size, block_size>>>(dinp, inp, dout, N);
  cudaCheck(cudaGetLastError());
}

void matmul_backward(float *dinp, float *dweight, float *dbias, float *dout,
                     float *inp, float *weight, int B, int T, int C, int OC) {
  float one = 1.0f;
  float zero = 0.0f;
  cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B * T, OC,
                          &one, weight, C, dout, OC, &zero, dinp, C));
  cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B * T,
                          &one, inp, C, dout, OC, &one, dweight, C));
  if (dbias != NULL) {
    const int block_size = 1024;
    const int grid_size = OC / 32;
    matmul_backward_bias_kernel4<<<grid_size, block_size,
                                   block_size * sizeof(float)>>>(dbias, dout, B,
                                                                 T, OC);
    cudaCheck(cudaGetLastError());
  }
}

void layernorm_backward(float *dinp, float *dweight, float *dbias,
                        const float *dout, const float *inp,
                        const float *weight, const float *mean,
                        const float *rstd, int B, int T, int C) {
  const int block_size = 512;
  const int N = B * T;
  const int grid_size = CEIL_DIV(32 * N, block_size);
  size_t shared_mem_size = 2 * C * sizeof(float);
  layernorm_backward_kernel2<<<grid_size, block_size, shared_mem_size>>>(
      dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
  cudaCheck(cudaGetLastError());
}

void attention_backward(float *dinp, float *dqkvr, float *dpreatt, float *datt,
                        float *scratch, const float *dout, const float *qkvr,
                        const float *att, int B, int T, int C, int NH) {
  const int block_size = 256;
  int HS = C / NH;
  const float one = 1.0f;
  const float zero = 0.0f;
  const float *q, *k, *v;
  q = qkvr + 0 * B * T * C;
  k = qkvr + 1 * B * T * C;
  v = qkvr + 2 * B * T * C;
  float *dq, *dk, *dv;
  dq = dqkvr + 0 * B * T * C;
  dk = dqkvr + 1 * B * T * C;
  dv = dqkvr + 2 * B * T * C;
  int num_blocks = CEIL_DIV(B * T * C, block_size);
  unpermute_kernel_backward<<<num_blocks, block_size>>>(scratch, dout, B, T, NH,
                                                        HS);
  cudaCheck(cudaGetLastError());
  cublasCheck(cublasSgemmStridedBatched(
      cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &one, v, HS, T * HS,
      scratch, HS, T * HS, &zero, datt, T, T * T, B * NH));
  cublasCheck(cublasSgemmStridedBatched(
      cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, scratch, HS,
      T * HS, att, T, T * T, &zero, dv, HS, T * HS, B * NH));
  int hs = C / NH;
  float scale = 1.0f / sqrtf(hs);
  softmax_autoregressive_backward_kernel<<<dim3(T / 4, B * NH), 256>>>(
      dpreatt, datt, att, B, T, C, scale);
  cudaCheck(cudaGetLastError());
  cublasCheck(cublasSgemmStridedBatched(
      cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &one, k, HS, T * HS,
      dpreatt, T, T * T, &zero, dq, HS, T * HS, B * NH));
  cublasCheck(cublasSgemmStridedBatched(
      cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, q, HS, T * HS,
      dpreatt, T, T * T, &zero, dk, HS, T * HS, B * NH));
  num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
  permute_kernel_backward<<<num_blocks, block_size>>>(dinp, dq, dk, dv, B, T,
                                                      NH, HS);
  cudaCheck(cudaGetLastError());
}

void fused_classifier3(float *logits, float *losses, const float *dlosses,
                       const int *targets, int B, int T, int V, int P) {
  const int block_size = 1024;
  const int N = B * T;
  const int grid_size = N;
  fused_classifier_kernel3<<<grid_size, block_size>>>(
      logits, losses, NULL, dlosses, targets, B, T, V, P);
  cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
  int max_seq_len;
  int vocab_size;
  int padded_vocab_size;
  int num_layers;
  int num_heads;
  int channels;
} GPT2Config;

#define NUM_PARAMETER_TENSORS 16
typedef struct {
  float *wte;      // (V, C)
  float *wpe;      // (maxT, C) - only on PE 0
  float *ln1w;     // (L_local, C)
  float *ln1b;     // (L_local, C)
  float *qkvw;     // (L_local, 3*C, C)
  float *qkvb;     // (L_local, 3*C)
  float *attprojw; // (L_local, C, C)
  float *attprojb; // (L_local, C)
  float *ln2w;     // (L_local, C)
  float *ln2b;     // (L_local, C)
  float *fcw;      // (L_local, 4*C, C)
  float *fcb;      // (L_local, 4*C)
  float *fcprojw;  // (L_local, C, 4*C)
  float *fcprojb;  // (L_local, C)
  float *lnfw;     // (C) - only on final PE
  float *lnfb;     // (C) - only on final PE
} ParameterTensors;

// Full parameter sizes (for file offset calculation)
void fill_in_parameter_sizes(size_t *param_sizes, GPT2Config config) {
  int Vp = config.padded_vocab_size;
  int C = config.channels;
  int maxT = config.max_seq_len;
  int L = config.num_layers;
  param_sizes[0] = Vp * C;
  param_sizes[1] = maxT * C;
  param_sizes[2] = L * C;
  param_sizes[3] = L * C;
  param_sizes[4] = L * (3 * C) * C;
  param_sizes[5] = L * (3 * C);
  param_sizes[6] = L * C * C;
  param_sizes[7] = L * C;
  param_sizes[8] = L * C;
  param_sizes[9] = L * C;
  param_sizes[10] = L * (4 * C) * C;
  param_sizes[11] = L * (4 * C);
  param_sizes[12] = L * C * (4 * C);
  param_sizes[13] = L * C;
  param_sizes[14] = C;
  param_sizes[15] = C;
}

// Partitioned parameter sizes (for allocation)
void fill_in_parameter_sizes_partitioned(size_t *param_sizes, GPT2Config config,
                                         PipelinePartition *part) {
  int Vp = config.padded_vocab_size;
  int C = config.channels;
  int maxT = config.max_seq_len;
  int L = part->num_layers;

  param_sizes[0] = Vp * C;                             // wte - all PEs
  param_sizes[1] = part->has_embedding ? maxT * C : 0; // wpe
  param_sizes[2] = L * C;
  param_sizes[3] = L * C;
  param_sizes[4] = L * (3 * C) * C;
  param_sizes[5] = L * (3 * C);
  param_sizes[6] = L * C * C;
  param_sizes[7] = L * C;
  param_sizes[8] = L * C;
  param_sizes[9] = L * C;
  param_sizes[10] = L * (4 * C) * C;
  param_sizes[11] = L * (4 * C);
  param_sizes[12] = L * C * (4 * C);
  param_sizes[13] = L * C;
  param_sizes[14] = part->has_final_ln ? C : 0;
  param_sizes[15] = part->has_final_ln ? C : 0;
}

float *malloc_and_point_parameters(ParameterTensors *params,
                                   size_t *param_sizes, int on_device) {
  size_t num_parameters = 0;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    num_parameters += param_sizes[i];
  }
  float *params_memory;
  if (on_device) {
    cudaCheck(
        cudaMalloc((void **)&params_memory, num_parameters * sizeof(float)));
  } else {
    params_memory = (float *)mallocCheck(num_parameters * sizeof(float));
  }
  float **ptrs[] = {
      &params->wte,     &params->wpe,     &params->ln1w,     &params->ln1b,
      &params->qkvw,    &params->qkvb,    &params->attprojw, &params->attprojb,
      &params->ln2w,    &params->ln2b,    &params->fcw,      &params->fcb,
      &params->fcprojw, &params->fcprojb, &params->lnfw,     &params->lnfb};
  float *params_memory_iterator = params_memory;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    *(ptrs[i]) = params_memory_iterator;
    params_memory_iterator += param_sizes[i];
  }
  return params_memory;
}

#define NUM_ACTIVATION_TENSORS 22
typedef struct {
  float *encoded;     // (B, T, C) - only PE 0
  float *ln1;         // (L_local, B, T, C)
  float *ln1_mean;    // (L_local, B, T)
  float *ln1_rstd;    // (L_local, B, T)
  float *atty;        // (L_local, B, T, C)
  float *att;         // (L_local, B, NH, T, T)
  float *attproj;     // (L_local, B, T, C)
  float *residual2;   // (L_local, B, T, C)
  float *ln2;         // (L_local, B, T, C)
  float *ln2_mean;    // (L_local, B, T)
  float *ln2_rstd;    // (L_local, B, T)
  float *fch;         // (L_local, B, T, 4*C)
  float *fch_gelu;    // (L_local, B, T, 4*C)
  float *fcproj;      // (L_local, B, T, C)
  float *residual3;   // (L_local, B, T, C)
  float *lnf;         // (B, T, C) - only final PE
  float *lnf_mean;    // (B, T) - only final PE
  float *lnf_rstd;    // (B, T) - only final PE
  float *losses;      // (B, T) - only final PE
  float *qkvr;        // (L_local, B, T, 3*C)
  float *output;      // scratch buffer
  float *scratch_btc; // (B, T, C) - extra scratch for backward on PE 0
} ActivationTensors;

void fill_in_activation_sizes_partitioned(size_t *act_sizes, int B, int T,
                                          GPT2Config config,
                                          PipelinePartition *part) {
  size_t Vp = config.padded_vocab_size;
  size_t L = part->num_layers;
  size_t NH = config.num_heads;
  size_t C = config.channels;

  act_sizes[0] = part->has_embedding ? B * T * C : 0;
  act_sizes[1] = L * B * T * C;
  act_sizes[2] = L * B * T;
  act_sizes[3] = L * B * T;
  act_sizes[4] = L * B * T * C;
  act_sizes[5] = L * B * NH * T * T;
  act_sizes[6] = L * B * T * C;
  act_sizes[7] = L * B * T * C;
  act_sizes[8] = L * B * T * C;
  act_sizes[9] = L * B * T;
  act_sizes[10] = L * B * T;
  act_sizes[11] = L * B * T * 4 * C;
  act_sizes[12] = L * B * T * 4 * C;
  act_sizes[13] = L * B * T * C;
  act_sizes[14] = L * B * T * C;
  act_sizes[15] = part->has_final_ln ? B * T * C : 0;
  act_sizes[16] = part->has_final_ln ? B * T : 0;
  act_sizes[17] = part->has_final_ln ? B * T : 0;
  act_sizes[18] = part->has_final_ln ? B * T : 0;
  act_sizes[19] = L * B * T * 3 * C;
  act_sizes[20] = B * T * max(3 * C, max(NH * T, Vp));
  act_sizes[21] = B * T * C; // scratch_btc for backward
}

#define NUM_BACKWARD_TENSORS 3
typedef struct {
  float *bt4c;
  float *preatt;
  float *residual3;
} GradActTensors;

void fill_in_grad_act_sizes(size_t *act_sizes, int B, int T,
                            GPT2Config config) {
  size_t NH = config.num_heads;
  size_t C = config.channels;
  act_sizes[0] = B * T * 4 * C;
  act_sizes[1] = B * NH * T * T;
  act_sizes[2] = B * T * C;
}

float *malloc_and_point(float **targets[], const size_t *act_sizes, int n) {
  size_t num_activations = 0;
  for (size_t i = 0; i < n; i++) {
    num_activations += act_sizes[i];
  }
  float *acts_memory;
  cudaCheck(cudaMalloc((void **)&acts_memory, num_activations * sizeof(float)));
  float *acts_memory_iterator = acts_memory;
  for (size_t i = 0; i < n; i++) {
    *(targets[i]) = acts_memory_iterator;
    acts_memory_iterator += act_sizes[i];
  }
  return acts_memory;
}

float *malloc_and_point_activations(ActivationTensors *acts,
                                    const size_t *act_sizes) {
  float **ptrs[] = {
      &acts->encoded,  &acts->ln1,        &acts->ln1_mean,  &acts->ln1_rstd,
      &acts->atty,     &acts->att,        &acts->attproj,   &acts->residual2,
      &acts->ln2,      &acts->ln2_mean,   &acts->ln2_rstd,  &acts->fch,
      &acts->fch_gelu, &acts->fcproj,     &acts->residual3, &acts->lnf,
      &acts->lnf_mean, &acts->lnf_rstd,   &acts->losses,    &acts->qkvr,
      &acts->output,   &acts->scratch_btc};
  return malloc_and_point(ptrs, act_sizes, NUM_ACTIVATION_TENSORS);
}

float *malloc_and_point_backward(GradActTensors *acts,
                                 const size_t *act_sizes) {
  float **ptrs[] = {&acts->bt4c, &acts->preatt, &acts->residual3};
  return malloc_and_point(ptrs, act_sizes, NUM_BACKWARD_TENSORS);
}

typedef struct {
  GPT2Config config;
  PipelinePartition partition;
  ParameterTensors params;
  size_t param_sizes[NUM_PARAMETER_TENSORS];
  float *params_memory;
  size_t num_parameters;
  ParameterTensors grads;
  float *grads_memory;
  float *m_memory;
  float *v_memory;
  ActivationTensors acts;
  size_t act_sizes[NUM_ACTIVATION_TENSORS];
  float *acts_memory;
  size_t num_activations;
  GradActTensors grads_acts;
  size_t num_grad_acts;
  float *grads_acts_memory;
  int batch_size;
  int seq_len;
  int *inputs;
  int *targets;
  float mean_loss;
  float *cpu_losses;
  float *nvshmem_act_buffer;
  float *nvshmem_grad_buffer;
  size_t nvshmem_buffer_size;
  float *nvshmem_wte_grad_buffer; // symmetric buffer for wte gradient allreduce
  int *nvshmem_token_buffer;      // symmetric buffer for token broadcast

  // Microbatching
  int num_micro_batches;
  int micro_batch_size;
  int *nvshmem_fwd_flags; // symmetric buffer for forward pass ready flags
  int *nvshmem_bwd_flags; // symmetric buffer for backward pass ready flags
} GPT2;

// ----------------------------------------------------------------------------
// P2P Signaling Helpers

void host_wait_flag(int *flag_addr, int val) {
  int current_val;
  do {
    cudaCheck(cudaMemcpy(&current_val, flag_addr, sizeof(int),
                         cudaMemcpyDeviceToHost));
    // Optional: basic exponential backoff or yield to avoid PCI-E saturation
    // but for now simple poll is fine.
  } while (current_val != val);
}

__device__ void p2p_wait_flag(int *flags, int idx, int val) {
  nvshmem_int_wait_until(flags + idx, NVSHMEM_CMP_EQ, val);
}

__device__ void p2p_signal_flag(int *flags, int idx, int val, int peer) {
  nvshmem_int_p(flags + idx, val, peer);
}

void host_p2p_reset_flags(int *flags, int n) {
  cudaCheck(cudaMemset(flags, 0, n * sizeof(int)));
  nvshmem_barrier_all();
}

void gpt2_build_from_checkpoint_partitioned(GPT2 *model,
                                            const char *checkpoint_path,
                                            PipelinePartition *part) {
  int my_pe = nvshmem_my_pe();

  FILE *model_file = fopenCheck(checkpoint_path, "rb");
  int model_header[256];
  freadCheck(model_header, sizeof(int), 256, model_file);
  if (model_header[0] != 20240326) {
    fprintf(stderr, "Bad magic model file\n");
    exit(EXIT_FAILURE);
  }
  if (model_header[1] != 3) {
    fprintf(stderr, "Bad version in model file\n");
    exit(EXIT_FAILURE);
  }

  model->config.max_seq_len = model_header[2];
  model->config.vocab_size = model_header[3];
  model->config.num_layers = model_header[4];
  model->config.num_heads = model_header[5];
  model->config.channels = model_header[6];
  model->config.padded_vocab_size = model_header[7];
  model->partition = *part;

  int C = model->config.channels;
  int Vp = model->config.padded_vocab_size;
  int maxT = model->config.max_seq_len;
  int L_total = model->config.num_layers;

  // Full sizes for file offset calculation
  size_t full_param_sizes[NUM_PARAMETER_TENSORS];
  fill_in_parameter_sizes(full_param_sizes, model->config);

  // Partitioned sizes for allocation
  fill_in_parameter_sizes_partitioned(model->param_sizes, model->config, part);

  size_t num_parameters = 0;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    num_parameters += model->param_sizes[i];
  }
  model->num_parameters = num_parameters;

  model->params_memory =
      malloc_and_point_parameters(&model->params, model->param_sizes, 1);

  printf(
      "[PE %d] allocated %zu MiB for partitioned parameters (layers %d-%d)\n",
      my_pe, (num_parameters * sizeof(float)) >> 20, part->first_layer,
      part->first_layer + part->num_layers - 1);

  size_t header_size = 256 * sizeof(int);

  // Helper to read a range of floats from file
  auto read_range = [&](float *gpu_dst, size_t file_elem_offset, size_t count) {
    if (count == 0)
      return;
    float *cpu_buf = (float *)mallocCheck(count * sizeof(float));
    fseekCheck(model_file, header_size + file_elem_offset * sizeof(float),
               SEEK_SET);
    freadCheck(cpu_buf, sizeof(float), count, model_file);
    cudaCheck(cudaMemcpy(gpu_dst, cpu_buf, count * sizeof(float),
                         cudaMemcpyHostToDevice));
    free(cpu_buf);
  };

  size_t file_offset = 0;
  size_t layer_start = part->first_layer;
  size_t layer_count = part->num_layers;

  // 0: wte - all PEs need this
  read_range(model->params.wte, file_offset, Vp * C);
  file_offset += full_param_sizes[0];

  // 1: wpe - only PE 0
  if (part->has_embedding) {
    read_range(model->params.wpe, file_offset, maxT * C);
  }
  file_offset += full_param_sizes[1];

  // 2: ln1w
  read_range(model->params.ln1w, file_offset + layer_start * C,
             layer_count * C);
  file_offset += full_param_sizes[2];

  // 3: ln1b
  read_range(model->params.ln1b, file_offset + layer_start * C,
             layer_count * C);
  file_offset += full_param_sizes[3];

  // 4: qkvw
  read_range(model->params.qkvw, file_offset + layer_start * 3 * C * C,
             layer_count * 3 * C * C);
  file_offset += full_param_sizes[4];

  // 5: qkvb
  read_range(model->params.qkvb, file_offset + layer_start * 3 * C,
             layer_count * 3 * C);
  file_offset += full_param_sizes[5];

  // 6: attprojw
  read_range(model->params.attprojw, file_offset + layer_start * C * C,
             layer_count * C * C);
  file_offset += full_param_sizes[6];

  // 7: attprojb
  read_range(model->params.attprojb, file_offset + layer_start * C,
             layer_count * C);
  file_offset += full_param_sizes[7];

  // 8: ln2w
  read_range(model->params.ln2w, file_offset + layer_start * C,
             layer_count * C);
  file_offset += full_param_sizes[8];

  // 9: ln2b
  read_range(model->params.ln2b, file_offset + layer_start * C,
             layer_count * C);
  file_offset += full_param_sizes[9];

  // 10: fcw
  read_range(model->params.fcw, file_offset + layer_start * 4 * C * C,
             layer_count * 4 * C * C);
  file_offset += full_param_sizes[10];

  // 11: fcb
  read_range(model->params.fcb, file_offset + layer_start * 4 * C,
             layer_count * 4 * C);
  file_offset += full_param_sizes[11];

  // 12: fcprojw
  read_range(model->params.fcprojw, file_offset + layer_start * C * 4 * C,
             layer_count * C * 4 * C);
  file_offset += full_param_sizes[12];

  // 13: fcprojb
  read_range(model->params.fcprojb, file_offset + layer_start * C,
             layer_count * C);
  file_offset += full_param_sizes[13];

  // 14: lnfw - only final PE
  if (part->has_final_ln) {
    read_range(model->params.lnfw, file_offset, C);
  }
  file_offset += full_param_sizes[14];

  // 15: lnfb - only final PE
  if (part->has_final_ln) {
    read_range(model->params.lnfb, file_offset, C);
  }

  fcloseCheck(model_file);

  model->acts_memory = NULL;
  model->grads_memory = NULL;
  model->m_memory = NULL;
  model->v_memory = NULL;
  model->grads_acts_memory = NULL;
  model->inputs = NULL;
  model->targets = NULL;
  model->cpu_losses = NULL;
  model->batch_size = 0;
  model->seq_len = 0;
  model->mean_loss = -1.0f;
  model->nvshmem_act_buffer = NULL;
  model->nvshmem_grad_buffer = NULL;
  model->nvshmem_buffer_size = 0;
}

void gpt2_allocate_nvshmem_buffers(GPT2 *model) {
  int my_pe = nvshmem_my_pe();
  int n_pes = nvshmem_n_pes();
  int B = model->batch_size;
  int T = model->seq_len;
  int C = model->config.channels;
  int Vp = model->config.padded_vocab_size;

  size_t buffer_elements = B * T * C;
  model->nvshmem_buffer_size = buffer_elements * sizeof(float);

  model->nvshmem_act_buffer =
      (float *)nvshmem_malloc(model->nvshmem_buffer_size);
  model->nvshmem_grad_buffer =
      (float *)nvshmem_malloc(model->nvshmem_buffer_size);

  // Allocate symmetric buffer for wte gradient allreduce (needs space for all
  // PEs' contributions)
  size_t wte_grad_size = (size_t)Vp * C * sizeof(float);
  model->nvshmem_wte_grad_buffer =
      (float *)nvshmem_malloc(wte_grad_size * n_pes);

  // Allocate symmetric buffer for token broadcast during generation
  model->nvshmem_token_buffer = (int *)nvshmem_malloc(sizeof(int));

  // Allocate microbatch flags
  if (model->num_micro_batches < 1)
    model->num_micro_batches = 1;
  model->nvshmem_fwd_flags =
      (int *)nvshmem_malloc(model->num_micro_batches * sizeof(int));
  model->nvshmem_bwd_flags =
      (int *)nvshmem_malloc(model->num_micro_batches * sizeof(int));

  if (model->nvshmem_act_buffer == NULL || model->nvshmem_grad_buffer == NULL ||
      model->nvshmem_wte_grad_buffer == NULL ||
      model->nvshmem_token_buffer == NULL || model->nvshmem_fwd_flags == NULL ||
      model->nvshmem_bwd_flags == NULL) {
    printf("[PE %d] Error: Failed to allocate NVSHMEM buffers\n", my_pe);
    exit(EXIT_FAILURE);
  }

  printf("[PE %d] allocated %zu MiB for NVSHMEM buffers\n", my_pe,
         (2 * model->nvshmem_buffer_size + wte_grad_size * n_pes) >> 20);

  cudaCheck(
      cudaMemset(model->nvshmem_act_buffer, 0, model->nvshmem_buffer_size));
  cudaCheck(
      cudaMemset(model->nvshmem_grad_buffer, 0, model->nvshmem_buffer_size));
  cudaCheck(
      cudaMemset(model->nvshmem_wte_grad_buffer, 0, wte_grad_size * n_pes));
  cudaCheck(cudaMemset(model->nvshmem_fwd_flags, 0,
                       model->num_micro_batches * sizeof(int)));
  cudaCheck(cudaMemset(model->nvshmem_bwd_flags, 0,
                       model->num_micro_batches * sizeof(int)));
}

void gpt2_forward(GPT2 *model, int *inputs, int *targets, int B, int T) {
  if (model->params_memory == NULL) {
    printf("Error: model was not initialized properly.\n");
    exit(EXIT_FAILURE);
  }

  int V = model->config.vocab_size;
  int Vp = model->config.padded_vocab_size;
  int NH = model->config.num_heads;
  int C = model->config.channels;
  PipelinePartition part = model->partition;
  int my_pe = nvshmem_my_pe();
  int n_pes = nvshmem_n_pes();

  for (int i = 0; i < B * T; i++) {
    assert(0 <= inputs[i] && inputs[i] < V);
    if (targets != NULL) {
      assert(0 <= targets[i] && targets[i] < V);
    }
  }

  if (model->acts_memory == NULL) {
    model->batch_size = B;
    model->seq_len = T;
    // NOTE: This assumes 'num_micro_batches' was set before calling forward for
    // first time. It should be set in main.
    // fill_in_activation_sizes_partitioned uses B (total). That is correct.
    fill_in_activation_sizes_partitioned(model->act_sizes, B, T, model->config,
                                         &part);
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
      num_activations += model->act_sizes[i];
    }
    model->num_activations = num_activations;
    model->acts_memory =
        malloc_and_point_activations(&model->acts, model->act_sizes);
    printf("[PE %d] allocated %zu MiB for activations\n", my_pe,
           (num_activations * sizeof(float)) >> 20);
    cudaCheck(cudaMalloc((void **)&model->inputs, B * T * sizeof(int)));
    cudaCheck(cudaMalloc((void **)&model->targets, B * T * sizeof(int)));
    cudaCheck(
        cudaMallocHost((void **)&model->cpu_losses, B * T * sizeof(float)));
  } else {
    if (B != model->batch_size || T != model->seq_len) {
      printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size,
             model->seq_len, B, T);
      exit(EXIT_FAILURE);
    }
  }

  // Copy inputs/targets (Entire Batch)
  cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int),
                       cudaMemcpyHostToDevice));
  if (targets != NULL) {
    cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int),
                         cudaMemcpyHostToDevice));
  }

  ParameterTensors params = model->params;
  ActivationTensors acts = model->acts;

  int M = model->num_micro_batches;
  if (B % M != 0) {
    if (my_pe == 0)
      printf("Error: Batch size %d not divisible by microbatches %d\n", B, M);
    exit(EXIT_FAILURE);
  }
  int mb_size = B / M;
  model->micro_batch_size = mb_size;

  // Reset forward flags at start of global batch
  host_p2p_reset_flags(model->nvshmem_fwd_flags, M);

  // GPipe Loop over Microbatches
  for (int step = 0; step < M; step++) {
    // These offsets are for the current microbatch within the full allocated
    // tensor
    long idx_off = (long)step * mb_size * T; // For [B, T] tensors
    long c_off = idx_off * C;                // For [B, T, C] tensors
    long c4_off = idx_off * 4 * C;           // For [B, T, 4C] tensors
    long nh_off =
        (long)step * mb_size * NH * T; // For [B, NH, T, T] tensors (att)

    // 1. Wait for Dependency (Receive)
    if (my_pe > 0) {
      // Wait for signal from PE i-1 for this microbatch
      // Host-side polling using cudaMemcpy (safe for device memory)
      host_wait_flag(model->nvshmem_fwd_flags + step, 1);
      // nvshmem_fence() not strictly required for RAW dependency if using
      // wait_until, but safe.
    }

    // Pointer setup for residual input to the first layer of this PE
    float *residual;
    if (my_pe == 0) {
      // PE0 encodes directly into its allocated acts.encoded buffer
      float *out_ptr = acts.encoded + c_off;
      encoder_forward(out_ptr, model->inputs + idx_off, params.wte, params.wpe,
                      mb_size, T, C);
      residual = out_ptr;
    } else {
      // Other PEs read from THEIR OWN nvshmem buffer, which PE i-1 wrote to.
      // The nvshmem_act_buffer is B*T*C, so we just use the offset.
      residual = model->nvshmem_act_buffer + c_off;
    }

    // Layers Loop
    for (int l = 0; l < part.num_layers; l++) {
      // The base pointers (e.g. acts.ln1) are allocated for the full batch [L,
      // B_total, T, C] So acts.ln1 + (long)l * B * T * C gets us the Layer
      // start for the full batch. Then + c_off gets us the microbatch start
      // within that layer's allocation.

      if (l == 0) {
        // residual already set for the first layer
      } else {
        // For subsequent layers, residual comes from the previous layer's
        // output (residual3)
        residual = acts.residual3 + (long)(l - 1) * B * T * C + c_off;
      }

      // Activation pointers for the current layer 'l' and current microbatch
      // 'step'
      float *l_ln1 = acts.ln1 + (long)l * B * T * C + c_off;
      float *l_ln1_mean = acts.ln1_mean + (long)l * B * T + idx_off;
      float *l_ln1_rstd = acts.ln1_rstd + (long)l * B * T + idx_off;

      float *l_qkvr = acts.qkvr + (long)l * B * T * 3 * C +
                      idx_off * 3 * C; // 3*C elements per T

      float *l_atty = acts.atty + (long)l * B * T * C + c_off;

      float *l_att = acts.att + (long)l * B * NH * T * T +
                     nh_off * T; // NH*T elements per T

      float *l_attproj = acts.attproj + (long)l * B * T * C + c_off;
      float *l_residual2 = acts.residual2 + (long)l * B * T * C + c_off;
      float *l_ln2 = acts.ln2 + (long)l * B * T * C + c_off;
      float *l_ln2_mean = acts.ln2_mean + (long)l * B * T + idx_off;
      float *l_ln2_rstd = acts.ln2_rstd + (long)l * B * T + idx_off;

      float *l_fch = acts.fch + (long)l * B * T * 4 * C + c4_off;
      float *l_fch_gelu = acts.fch_gelu + (long)l * B * T * 4 * C + c4_off;
      float *l_fcproj = acts.fcproj + (long)l * B * T * C + c_off;
      float *l_residual3 =
          acts.residual3 + (long)l * B * T * C + c_off; // Output of this layer

      // acts.output is a scratch buffer. It's allocated for B*T*max_dim.
      // We offset it by the microbatch to avoid conflicts if PEs are processing
      // different microbatches.
      float *scratch = acts.output + idx_off * max(3 * C, max(NH * T, Vp));

      // Parameter pointers are layer-specific but not microbatch-specific
      float *l_ln1w = params.ln1w + l * C;
      float *l_ln1b = params.ln1b + l * C;
      float *l_qkvw = params.qkvw + l * 3 * C * C;
      float *l_qkvb = params.qkvb + l * 3 * C;
      float *l_attprojw = params.attprojw + l * C * C;
      float *l_attprojb = params.attprojb + l * C;
      float *l_ln2w = params.ln2w + l * C;
      float *l_ln2b = params.ln2b + l * C;
      float *l_fcw = params.fcw + l * 4 * C * C;
      float *l_fcb = params.fcb + l * 4 * C;
      float *l_fcprojw = params.fcprojw + l * C * 4 * C;
      float *l_fcprojb = params.fcprojb + l * C;

      layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b,
                        mb_size, T, C);
      matmul_forward(scratch, l_ln1, l_qkvw, l_qkvb, mb_size, T, C, 3 * C);
      attention_forward(l_atty, l_qkvr, l_att, scratch, mb_size, T, C, NH);
      matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, mb_size, T, C,
                     C);
      residual_forward(l_residual2, residual, l_attproj, mb_size * T * C);
      layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w,
                        l_ln2b, mb_size, T, C);
      matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, mb_size, T, C, 4 * C);
      gelu_forward(l_fch_gelu, l_fch, mb_size * T * 4 * C);
      matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, mb_size, T,
                     4 * C, C);
      residual_forward(l_residual3, l_residual2, l_fcproj, mb_size * T * C);
    }

    // 2. Send to Next PE (Transmit)
    if (my_pe < n_pes - 1) {
      // The output of the last layer of this PE for this microbatch
      float *last_output =
          acts.residual3 + (long)(part.num_layers - 1) * B * T * C + c_off;

      cudaCheck(
          cudaDeviceSynchronize()); // Ensure computation is done before sending

      // Put to Next PE's nvshmem_act_buffer at the correct microbatch offset
      nvshmem_putmem(model->nvshmem_act_buffer + c_off, last_output,
                     mb_size * T * C * sizeof(float), my_pe + 1);
      nvshmem_quiet(); // Ensure data visibility for the remote PE

      // Signal Next PE that this microbatch is ready
      nvshmem_int_p(model->nvshmem_fwd_flags + step, 1, my_pe + 1);
    }

    // 3. Final PE Processing (Loss Calculation)
    if (my_pe == n_pes - 1) {
      // Input to final layernorm for this microbatch
      residual =
          acts.residual3 + (long)(part.num_layers - 1) * B * T * C + c_off;

      // Logits scratch buffer for this microbatch
      float *l_logits = acts.output + idx_off * Vp;

      // Final layernorm activations for this microbatch
      float *l_lnf = acts.lnf + c_off;
      float *l_lnf_mean = acts.lnf_mean + idx_off;
      float *l_lnf_rstd = acts.lnf_rstd + idx_off;

      // Targets and losses for this microbatch
      int *l_targets = (targets != NULL) ? model->targets + idx_off : NULL;
      float *l_losses = acts.losses + idx_off;

      layernorm_forward(l_lnf, l_lnf_mean, l_lnf_rstd, residual, params.lnfw,
                        params.lnfb, mb_size, T, C);
      matmul_forward(l_logits, l_lnf, params.wte, NULL, mb_size, T, C, Vp);

      if (targets != NULL) {
        fused_classifier3(l_logits, l_losses, NULL, l_targets, mb_size, T, V,
                          Vp);
      }
    }
  }

  // After all microbatches, synchronize global mean_loss
  if (my_pe == n_pes - 1 && targets != NULL) {
    // Copy all microbatch losses from device to host
    cudaCheck(cudaMemcpy(model->cpu_losses, acts.losses, B * T * sizeof(float),
                         cudaMemcpyDeviceToHost));
    float mean_loss = 0.0f;
    for (int i = 0; i < B * T; i++) {
      mean_loss += model->cpu_losses[i];
    }
    mean_loss /= (B * T);
    model->mean_loss = mean_loss;
  } else {
    // Non-final PEs or no targets: set placeholder for backward pass check
    model->mean_loss = targets != NULL ? 0.0f : -1.0f;
  }

  // Barrier at end of FWD to ensure all stages done before BWD start.
  // This is a strict GPipe synchronization.
  nvshmem_barrier_all();
}

void gpt2_zero_grad(GPT2 *model) {
  if (model->grads_acts_memory != NULL) {
    cudaCheck(cudaMemset(model->grads_acts_memory, 0,
                         model->num_grad_acts * sizeof(float)));
  }
  if (model->grads_memory != NULL) {
    cudaCheck(cudaMemset(model->grads_memory, 0,
                         model->num_parameters * sizeof(float)));
  }
}

void gpt2_backward(GPT2 *model) {
  counter++;

  if (model->mean_loss == -1.0f) {
    printf("Error: must forward with targets before backward\n");
    exit(EXIT_FAILURE);
  }

  PipelinePartition part = model->partition;
  int my_pe = nvshmem_my_pe();
  int n_pes = nvshmem_n_pes();

  if (model->grads_memory == NULL) {
    model->grads_memory =
        malloc_and_point_parameters(&model->grads, model->param_sizes, 1);
    printf("[PE %d] allocated %zu MiB for parameter gradients\n", my_pe,
           (model->num_parameters * sizeof(float)) >> 20);

    size_t bw_act_sizes[NUM_BACKWARD_TENSORS];
    GPT2Config cfg = model->config;
    cfg.num_layers = 1;
    fill_in_grad_act_sizes(bw_act_sizes, model->batch_size, model->seq_len,
                           cfg);
    model->grads_acts_memory =
        malloc_and_point_backward(&model->grads_acts, bw_act_sizes);
    model->num_grad_acts = 0;
    for (int i = 0; i < NUM_BACKWARD_TENSORS; i++) {
      model->num_grad_acts += bw_act_sizes[i];
    }
    printf("[PE %d] allocated %zu MiB for activation gradients\n", my_pe,
           (model->num_grad_acts * sizeof(float)) >> 20);
    gpt2_zero_grad(model);
  }

  int B = model->batch_size;
  int T = model->seq_len;
  int Vp = model->config.padded_vocab_size;
  int NH = model->config.num_heads;
  int C = model->config.channels;

  ParameterTensors params = model->params;
  ParameterTensors grads = model->grads;
  ActivationTensors acts = model->acts;
  GradActTensors grads_acts = model->grads_acts;

  // GPipe Microbatching
  int M = model->num_micro_batches;
  int mb_size = model->micro_batch_size;

  host_p2p_reset_flags(model->nvshmem_bwd_flags, M);

  // Note: Gradients are accumulated into the same 'grads' buffers (which are
  // sized for L layers). This is safe because only one microbatch processes at
  // a time per PE (serial loop). 'grads_acts' (activations gradients) are
  // essentially scratch space and can be reused or offset. Current
  // implementation has 'grads_acts.bt4c' size B*T*... We can use offsets to be
  // safe or reuse if we are sure of serial execution. With B-sized allocation,
  // offsets are safer.

  for (int step = 0; step < M; step++) {
    size_t offset_elem = (size_t)step * mb_size * T * C;
    size_t offset_idx = (size_t)step * mb_size * T;

    long idx_off = (long)step * mb_size * T;
    long c_off = idx_off * C;
    long c4_off = idx_off * 4 * C;
    long nh_off = idx_off * NH * T;

    // 1. Wait for Dependency
    if (my_pe < n_pes - 1) {
      // Wait for signal from Next PE (i+1)
      host_wait_flag(model->nvshmem_bwd_flags + step, 1);
    }

    float *residual;
    float *dresidual =
        grads_acts.residual3 + c_off; // Use B-sized buffer offset

    // Last PE: backward through classifier + final layernorm
    if (my_pe == n_pes - 1) {
      // Offsets
      float *l_bt4c = grads_acts.bt4c + c4_off;
      float *l_output = acts.output + idx_off * Vp; // The logits/probs scratch
      float *l_lnf = acts.lnf + c_off;
      float *l_lnf_mean = acts.lnf_mean + idx_off;
      float *l_lnf_rstd = acts.lnf_rstd + idx_off;

      residual =
          acts.residual3 + (long)(part.num_layers - 1) * B * T * C + c_off;

      matmul_backward(l_bt4c, grads.wte, NULL, l_output, l_lnf, params.wte,
                      mb_size, T, C, Vp);
      layernorm_backward(dresidual, grads.lnfw, grads.lnfb, l_bt4c, residual,
                         params.lnfw, l_lnf_mean, l_lnf_rstd, mb_size, T, C);
    } else {
      // Not Last PE: receive gradients from Buffer
      // We read from our own buffer (written by Next PE)
      cudaCheck(cudaMemcpy(dresidual, model->nvshmem_grad_buffer + offset_elem,
                           mb_size * T * C * sizeof(float),
                           cudaMemcpyDeviceToDevice));
    }

    // Backward through layers
    for (int l = part.num_layers - 1; l >= 0; l--) {
      // Setup Input 'residual'
      if (l == 0) {
        if (my_pe == 0) {
          residual = acts.encoded + c_off;
        } else {
          residual = model->nvshmem_act_buffer + c_off;
        }
      } else {
        residual = acts.residual3 + (long)(l - 1) * B * T * C + c_off;
      }

      // Params (Accumulate into shared, no offset)
      float *dl_ln1w = grads.ln1w + l * C;
      float *dl_ln1b = grads.ln1b + l * C;
      float *dl_qkvw = grads.qkvw + l * 3 * C * C;
      float *dl_qkvb = grads.qkvb + l * 3 * C;
      float *dl_attprojw = grads.attprojw + l * C * C;
      float *dl_attprojb = grads.attprojb + l * C;
      float *dl_ln2w = grads.ln2w + l * C;
      float *dl_ln2b = grads.ln2b + l * C;
      float *dl_fcw = grads.fcw + l * 4 * C * C;
      float *dl_fcb = grads.fcb + l * 4 * C;
      float *dl_fcprojw = grads.fcprojw + l * C * 4 * C;
      float *dl_fcprojb = grads.fcprojb + l * C;

      // Acts (Offset)
      float *l_ln1 = acts.ln1 + (long)l * B * T * C + c_off;
      float *l_ln1_mean = acts.ln1_mean + (long)l * B * T + idx_off;
      float *l_ln1_rstd = acts.ln1_rstd + (long)l * B * T + idx_off;
      float *l_qkvr = acts.qkvr + (long)l * B * T * 3 * C + idx_off * 3 * C;
      float *l_atty = acts.atty + (long)l * B * T * C + c_off;
      float *l_att = acts.att + (long)l * B * NH * T * T + nh_off * T;
      float *l_attprojw = params.attprojw + l * C * C;
      float *l_residual2 = acts.residual2 + (long)l * B * T * C + c_off;

      float *l_ln2 = acts.ln2 + (long)l * B * T * C + c_off;
      float *l_ln2_mean = acts.ln2_mean + (long)l * B * T + idx_off;
      float *l_ln2_rstd = acts.ln2_rstd + (long)l * B * T + idx_off;

      float *l_fch = acts.fch + (long)l * B * T * 4 * C + c4_off;
      float *l_fch_gelu = acts.fch_gelu + (long)l * B * T * 4 * C + c4_off;
      float *l_fcprojw = params.fcprojw + l * C * 4 * C;
      float *l_fcw = params.fcw + l * 4 * C * C;
      float *l_ln2w = params.ln2w + l * C;
      float *l_ln1w = params.ln1w + l * C;
      float *l_qkvw = params.qkvw + l * 3 * C * C;

      // Grad Acts (Offset)
      // Using different scratch space for Fused Classifier Logic vs internal?
      // grads_acts.bt4c size is B*... so we iterate.
      float *dl_bt4c = grads_acts.bt4c + c4_off;
      float *dl_preatt = grads_acts.preatt + nh_off * T;
      float *scratch = acts.output + idx_off * max(3 * C, max(NH * T, Vp));

      // Only last PE has lnf allocated, others use scratch_btc
      float *dl_btc =
          (my_pe == n_pes - 1) ? acts.lnf + c_off : acts.scratch_btc + c_off;

      // Execute Kernels
      matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu,
                      l_fcprojw, mb_size, T, 4 * C, C);
      gelu_backward(dl_bt4c, l_fch, dl_bt4c, mb_size * T * 4 * C);
      matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, mb_size, T,
                      C, 4 * C);
      layernorm_backward(dresidual, dl_ln2w, dl_ln2b, dl_btc, l_residual2,
                         l_ln2w, l_ln2_mean, l_ln2_rstd, mb_size, T, C);
      matmul_backward(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty,
                      l_attprojw, mb_size, T, C, C);
      attention_backward(dl_bt4c, l_fch, dl_preatt, scratch, l_atty, dl_btc,
                         l_qkvr, l_att, mb_size, T, C, NH);
      matmul_backward(dl_btc, dl_qkvw, dl_qkvb, dl_bt4c, l_ln1, l_qkvw, mb_size,
                      T, C, 3 * C);
      layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_btc, residual, l_ln1w,
                         l_ln1_mean, l_ln1_rstd, mb_size, T, C);
    }

    // 2. Send to Prev PE
    if (my_pe > 0) {
      cudaCheck(cudaDeviceSynchronize());
      nvshmem_putmem(model->nvshmem_grad_buffer + offset_elem, dresidual,
                     mb_size * T * C * sizeof(float), my_pe - 1);
      nvshmem_quiet();

      nvshmem_int_p(model->nvshmem_bwd_flags + step, 1, my_pe - 1);
    }

    // 3. First PE: Embedding
    if (my_pe == 0) {
      encoder_backward(grads.wte, grads.wpe, dresidual, model->inputs + idx_off,
                       mb_size, T, C);
    }
  }

  // All-reduce wte gradients using NVSHMEM
  // Each PE puts its wte gradients to PE 0's buffer at its offset
  size_t wte_size = (size_t)Vp * C;

  cudaCheck(cudaDeviceSynchronize());
  nvshmem_barrier_all(); // Wait for all PEs

  // Each PE sends its grads.wte to PE 0's buffer at offset my_pe * wte_size
  nvshmem_float_put(model->nvshmem_wte_grad_buffer + my_pe * wte_size,
                    grads.wte, wte_size, 0);
  nvshmem_barrier_all();

  // PE 0 reduces all contributions using GPU kernel
  if (my_pe == 0) {
    int block_size = 256;
    int num_blocks = CEIL_DIV(wte_size, block_size);
    for (int pe = 1; pe < n_pes; pe++) {
      float *src = model->nvshmem_wte_grad_buffer + pe * wte_size;
      accumulate_grads_kernel<<<num_blocks, block_size>>>(
          model->nvshmem_wte_grad_buffer, src, wte_size);
    }
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
  }
  nvshmem_barrier_all();

  // Broadcast the reduced result from PE 0 to all PEs
  nvshmem_float_get(grads.wte, model->nvshmem_wte_grad_buffer, wte_size, 0);
  nvshmem_barrier_all();
}

void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2,
                 float eps, float weight_decay, int t) {
  if (model->m_memory == NULL) {
    cudaCheck(cudaMalloc((void **)&model->m_memory,
                         model->num_parameters * sizeof(float)));
    cudaCheck(cudaMalloc((void **)&model->v_memory,
                         model->num_parameters * sizeof(float)));
    cudaCheck(
        cudaMemset(model->m_memory, 0, model->num_parameters * sizeof(float)));
    cudaCheck(
        cudaMemset(model->v_memory, 0, model->num_parameters * sizeof(float)));
    int my_pe = nvshmem_my_pe();
    printf("[PE %d] allocated %zu MiB for AdamW optimizer state\n", my_pe,
           (2 * model->num_parameters * sizeof(float)) >> 20);
  }

  int block_size = 512;
  int num_blocks = CEIL_DIV(model->num_parameters, block_size);
  float beta1_correction = 1.0f - powf(beta1, t);
  float beta2_correction = 1.0f - powf(beta2, t);
  adamw_kernel2<<<num_blocks, block_size>>>(
      model->params_memory, model->grads_memory, model->m_memory,
      model->v_memory, model->num_parameters, learning_rate, beta1, beta2,
      beta1_correction, beta2_correction, eps, weight_decay);
  cudaCheck(cudaGetLastError());
}

void gpt2_free(GPT2 *model) {
  cudaCheck(cudaFree(model->params_memory));
  cudaCheck(cudaFree(model->grads_memory));
  cudaCheck(cudaFree(model->m_memory));
  cudaCheck(cudaFree(model->v_memory));
  cudaCheck(cudaFree(model->acts_memory));
  cudaCheck(cudaFree(model->grads_acts_memory));
  cudaCheck(cudaFree(model->inputs));
  cudaCheck(cudaFree(model->targets));
  cudaFreeHost(model->cpu_losses);
  // Free NVSHMEM buffers
  nvshmem_free(model->nvshmem_act_buffer);
  nvshmem_free(model->nvshmem_grad_buffer);
  nvshmem_free(model->nvshmem_wte_grad_buffer);
  nvshmem_free(model->nvshmem_token_buffer);
  nvshmem_free(model->nvshmem_fwd_flags);
  nvshmem_free(model->nvshmem_bwd_flags);
}

#ifndef TESTING
// ----------------------------------------------------------------------------
// sampler

#define GPT2_EOT 50256

unsigned int random_u32(unsigned long long *state) {
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) {
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_softmax(const float *logits, int n, float coin) {
  double norm = 0;
  for (int i = 0; i < n; i++) {
    norm += expf(logits[i]);
  }
  coin *= norm;
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += expf(logits[i]);
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1;
}

// ----------------------------------------------------------------------------
// Logger

typedef struct {
  FILE *logfile;
  int flush_every;
} Logger;

void logger_init(Logger *logger, const char *filename) {
  logger->flush_every = 20;
  logger->logfile = NULL;
  if (filename != NULL) {
    logger->logfile = fopenCheck(filename, "w");
  }
}

void logger_log_val(Logger *logger, int step, float val_loss) {
  if (logger->logfile != NULL) {
    fprintf(logger->logfile, "s:%d tel:%.4f\n", step, val_loss);
  }
}

void logger_log_train(Logger *logger, int step, float train_loss) {
  if (logger->logfile != NULL) {
    fprintf(logger->logfile, "s:%d trl:%.4f\n", step, train_loss);
    if (step % 10 == 0) {
      fflush(logger->logfile);
    }
  }
}

void logger_free(Logger *logger) {
  if (logger->logfile != NULL) {
    fclose(logger->logfile);
  }
}

// ----------------------------------------------------------------------------
// CLI

void error_usage() {
  fprintf(stderr, "Usage:   ./train_gpt2_partitioned [options]\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -i <string> train data filename pattern\n");
  fprintf(stderr, "  -j <string> val data filename pattern\n");
  fprintf(stderr, "  -o <string> output log file (default = NULL)\n");
  fprintf(stderr, "  -b <int>    batch size B (default = 4)\n");
  fprintf(stderr, "  -t <int>    sequence length T (default = 1024)\n");
  fprintf(stderr, "  -l <float>  learning rate (default = 3e-4f)\n");
  fprintf(stderr, "  -v <int>    val_loss_every (default = 20)\n");
  fprintf(stderr, "  -m <int>    val_max_steps (default = 20)\n");
  fprintf(stderr, "  -s <int>    sample_every (default = 20)\n");
  fprintf(stderr, "  -g <int>    genT (default = 64)\n");
  fprintf(stderr, "  -c <int>    num_micro_batches (default = 1)\n");
  exit(EXIT_FAILURE);
}

// ----------------------------------------------------------------------------
// main training loop

int main(int argc, char *argv[]) {
  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Initialize NVSHMEM
  nvshmemx_init_attr_t attr;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

  int my_pe = nvshmem_my_pe();
  int n_pes = nvshmem_n_pes();

  if (n_pes < 2) {
    if (my_pe == 0) {
      fprintf(stderr, "Error: This pipeline requires at least 2 GPUs, got %d\n",
              n_pes);
    }
    nvshmem_finalize();
    MPI_Finalize();
    return 1;
  }

  cudaCheck(cudaSetDevice(my_pe));
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, my_pe);

  cudaDeviceSynchronize();
  nvshmem_barrier_all();

  // Parse arguments
  const char *train_data_pattern =
      "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
  const char *val_data_pattern =
      "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
  const char *output_log_file = NULL;
  int B = 4;
  int T = 1024;
  float learning_rate = 3e-4f;
  int val_loss_every = 20;
  int val_max_steps = 20;
  int sample_every = 20;
  int genT = 64;
  int num_micro_batches = 1;

  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc)
      error_usage();
    if (argv[i][0] != '-')
      error_usage();
    if (strlen(argv[i]) != 2)
      error_usage();
    if (argv[i][1] == 'i')
      train_data_pattern = argv[i + 1];
    else if (argv[i][1] == 'j')
      val_data_pattern = argv[i + 1];
    else if (argv[i][1] == 'o')
      output_log_file = argv[i + 1];
    else if (argv[i][1] == 'b')
      B = atoi(argv[i + 1]);
    else if (argv[i][1] == 't')
      T = atoi(argv[i + 1]);
    else if (argv[i][1] == 'l')
      learning_rate = atof(argv[i + 1]);
    else if (argv[i][1] == 'v')
      val_loss_every = atoi(argv[i + 1]);
    else if (argv[i][1] == 'm')
      val_max_steps = atoi(argv[i + 1]);
    else if (argv[i][1] == 's')
      sample_every = atoi(argv[i + 1]);
    else if (argv[i][1] == 'g')
      genT = atoi(argv[i + 1]);
    else if (argv[i][1] == 'c')
      num_micro_batches = atoi(argv[i + 1]);
    else
      error_usage();
  }

  if (my_pe == 0) {
    printf("+------------------------+-----------------------------------------"
           "-----------+\n");
    printf("| Parameter              | Value                                   "
           "           |\n");
    printf("+------------------------+-----------------------------------------"
           "-----------+\n");
    printf("| NVSHMEM PEs (GPUs)     | %-50d |\n", n_pes);
    printf("| train data pattern     | %-50s |\n", train_data_pattern);
    printf("| val data pattern       | %-50s |\n", val_data_pattern);
    printf("| batch size B           | %-50d |\n", B);
    printf("| sequence length T      | %-50d |\n", T);
    printf("| learning rate          | %-50f |\n", learning_rate);
    printf("+------------------------+-----------------------------------------"
           "-----------+\n");
  }

  // Setup cuBLAS
  cublasCheck(cublasCreate(&cublas_handle));
  int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
  cublas_compute_type =
      enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  cublasMath_t cublas_math_mode =
      enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
  cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

  if (my_pe == 0) {
    printf("| device                 | %-50s |\n", deviceProp.name);
    printf("| TF32                   | %-50s |\n",
           enable_tf32 ? "enabled" : "disabled");
    printf("+------------------------+-----------------------------------------"
           "-----------+\n");
  }

  // Read header to get num_layers for partitioning
  FILE *tmp = fopen("gpt2_124M.bin", "rb");
  if (tmp == NULL) {
    printf("Error: could not open gpt2_124M.bin\n");
    exit(EXIT_FAILURE);
  }
  int header[256];
  fread(header, sizeof(int), 256, tmp);
  fclose(tmp);
  int total_layers = header[4];

  // Compute partition for this PE
  PipelinePartition part;
  get_partition_for_pe(&part, my_pe, n_pes, total_layers);

  if (my_pe == 0) {
    printf("| total_layers           | %-50d |\n", total_layers);
    printf("+------------------------+-----------------------------------------"
           "-----------+\n");
  }

  // Print layer distribution for each PE (synchronized)
  for (int pe = 0; pe < n_pes; pe++) {
    if (my_pe == pe) {
      printf("| PE %d layers            | %d-%d (count: %d)%-*s |\n", my_pe,
             part.first_layer, part.first_layer + part.num_layers - 1,
             part.num_layers, 50 - 25, "");
    }
    nvshmem_barrier_all();
  }
  if (my_pe == 0) {
    printf("+------------------------+-----------------------------------------"
           "-----------+\n");
  }

  // Build model with partitioning
  GPT2 model;
  gpt2_build_from_checkpoint_partitioned(&model, "gpt2_124M.bin", &part);

  if (my_pe == 0) {
    printf("| max_sequence_length T  | %-50d |\n", model.config.max_seq_len);
    printf("| vocab_size V           | %-50d |\n", model.config.vocab_size);
    printf("| padded_vocab_size Vp   | %-50d |\n",
           model.config.padded_vocab_size);
    printf("| num_layers L (total)   | %-50d |\n", model.config.num_layers);
    printf("| num_heads NH           | %-50d |\n", model.config.num_heads);
    printf("| channels C             | %-50d |\n", model.config.channels);
    printf("+------------------------+-----------------------------------------"
           "-----------+\n");
  }

  // Build data loaders
  DataLoader train_loader, val_loader;
  dataloader_init(&train_loader, train_data_pattern, B, T, 0, 1, 1);
  dataloader_init(&val_loader, val_data_pattern, B, T, 0, 1, 0);
  int train_num_batches = train_loader.num_tokens / (B * T);
  int val_num_batches = val_loader.num_tokens / (B * T);
  if (val_num_batches > val_max_steps)
    val_num_batches = val_max_steps;

  if (my_pe == 0) {
    printf("| train_num_batches      | %-50d |\n", train_num_batches);
    printf("| val_num_batches        | %-50d |\n", val_num_batches);
    printf("+------------------------+-----------------------------------------"
           "-----------+\n");
  }

  // Allocate NVSHMEM buffers
  model.batch_size = B;
  model.seq_len = T;
  model.num_micro_batches = num_micro_batches; // Set from args
  gpt2_allocate_nvshmem_buffers(&model);

  // Dummy forward to allocate activations
  int *dummy_batch = (int *)malloc(B * T * sizeof(int));
  for (int i = 0; i < B * T; i++)
    dummy_batch[i] = 0;
  gpt2_forward(&model, dummy_batch, NULL, B, T);
  free(dummy_batch);

  // Setup logger and tokenizer
  Logger logger;
  logger_init(&logger, output_log_file);

  Tokenizer tokenizer;
  tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

  unsigned long long rng_state = 1337;
  int *gen_tokens = (int *)mallocCheck(B * T * sizeof(int));
  float *cpu_logits =
      (float *)mallocCheck(model.config.vocab_size * sizeof(float));

  // Training loop
  struct timespec start, end;
  double total_sum_iteration_time_s = 0.0;

  for (int step = 0; step <= train_num_batches; step++) {
    int last_step = step == train_num_batches;

    // Validation
    if (step % val_loss_every == 0 || last_step) {
      float val_loss = 0.0f;
      dataloader_reset(&val_loader);
      for (int i = 0; i < val_num_batches; i++) {
        dataloader_next_batch(&val_loader);
        gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
        val_loss += model.mean_loss;
      }
      val_loss /= val_num_batches;
      if (my_pe == n_pes - 1) {
        printf("val loss %f\n", val_loss);
      }
      logger_log_val(&logger, step, val_loss);
    }

    // Sampling (only on last PE which has the output)
    if ((step > 0 && step % sample_every == 0) || last_step) {
      for (int i = 0; i < B * T; ++i)
        gen_tokens[i] = GPT2_EOT;
      if (my_pe == n_pes - 1)
        printf("generating:\n---\n");
      for (int t = 1; t < genT; t++) {
        gpt2_forward(&model, gen_tokens, NULL, B, T);
        if (my_pe == n_pes - 1) {
          float *logits =
              model.acts.output + (t - 1) * model.config.padded_vocab_size;
          cudaCheck(cudaMemcpy(cpu_logits, logits,
                               model.config.vocab_size * sizeof(float),
                               cudaMemcpyDeviceToHost));
          float coin = random_f32(&rng_state);
          int next_token =
              sample_softmax(cpu_logits, model.config.vocab_size, coin);
          gen_tokens[t] = next_token;
          if (tokenizer.init_ok) {
            const char *token_str = tokenizer_decode(&tokenizer, next_token);
            safe_printf(token_str);
          } else {
            printf("%d ", next_token);
          }
          fflush(stdout);
        }
        // Send next token from last PE to PE 0 only using NVSHMEM
        // point-to-point Only PE 0 needs the token for the next embedding
        // lookup
        if (my_pe == n_pes - 1) {
          cudaCheck(cudaMemcpy(model.nvshmem_token_buffer, &gen_tokens[t],
                               sizeof(int), cudaMemcpyHostToDevice));
          nvshmem_int_put(model.nvshmem_token_buffer,
                          model.nvshmem_token_buffer, 1, 0);
          nvshmem_quiet();
        }
        nvshmem_barrier_all();
        // Only PE 0 copies the token from GPU to CPU
        if (my_pe == 0) {
          cudaCheck(cudaMemcpy(&gen_tokens[t], model.nvshmem_token_buffer,
                               sizeof(int), cudaMemcpyDeviceToHost));
        }
        nvshmem_barrier_all();
      }
      if (my_pe == n_pes - 1)
        printf("\n---\n");
    }

    if (last_step)
      break;

    // Training step
    clock_gettime(CLOCK_MONOTONIC, &start);
    dataloader_next_batch(&train_loader);
    gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
    gpt2_zero_grad(&model);
    gpt2_backward(&model);
    gpt2_update(&model, learning_rate, 0.9f, 0.999f, 1e-8f, 0.0f, step + 1);
    cudaCheck(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_elapsed_s =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    total_sum_iteration_time_s += time_elapsed_s;
    int tokens_per_second = (B * T) / time_elapsed_s;

    if (my_pe == n_pes - 1) {
      printf("step %4d/%d: train loss %f (%f ms, %d tok/s)\n", step + 1,
             train_num_batches, model.mean_loss, time_elapsed_s * 1000,
             tokens_per_second);
    }
    logger_log_train(&logger, step, model.mean_loss);
  }

  if (my_pe == 0) {
    printf("total average iteration time: %f ms\n",
           total_sum_iteration_time_s / train_num_batches * 1000);
  }

  // Cleanup
  dataloader_free(&train_loader);
  dataloader_free(&val_loader);
  tokenizer_free(&tokenizer);
  gpt2_free(&model);
  free(cpu_logits);
  free(gen_tokens);
  cublasCheck(cublasDestroy(cublas_handle));
  logger_free(&logger);

  nvshmem_finalize();
  MPI_Finalize();

  return 0;
}
#endif