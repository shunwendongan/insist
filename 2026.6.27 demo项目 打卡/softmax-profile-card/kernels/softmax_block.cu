// Block-per-row softmax kernel.
// This is a compact demo kernel: each block computes one row and uses shared
// memory reductions for max and sum. It is closer to a real CUDA softmax kernel
// than softmax_naive.cu, but still not a production implementation.

#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" __global__
void softmax_block_kernel(const float* __restrict__ x,
                          float* __restrict__ y,
                          int rows,
                          int cols) {
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    if (row >= rows) return;

    const float* row_x = x + row * cols;
    float* row_y = y + row * cols;

    float local_max = -CUDART_INF_F;
    for (int c = tid; c < cols; c += block_size) {
        local_max = fmaxf(local_max, row_x[c]);
    }
    smem[tid] = local_max;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = smem[0];

    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += block_size) {
        float e = expf(row_x[c] - max_val);
        row_y[c] = e;
        local_sum += e;
    }
    smem[tid] = local_sum;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = smem[0];

    for (int c = tid; c < cols; c += block_size) {
        row_y[c] = row_y[c] / sum_val;
    }
}
