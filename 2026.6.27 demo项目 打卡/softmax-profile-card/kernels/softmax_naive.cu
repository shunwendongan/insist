// Naive softmax kernel for learning/correctness intuition.
// One CUDA block handles one row, but only thread 0 does the work.
// This is intentionally slow and should not be used as a performance baseline.

#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" __global__
void softmax_naive_kernel(const float* __restrict__ x,
                          float* __restrict__ y,
                          int rows,
                          int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    if (threadIdx.x == 0) {
        const float* row_x = x + row * cols;
        float* row_y = y + row * cols;

        float max_val = -CUDART_INF_F;
        for (int c = 0; c < cols; ++c) {
            max_val = fmaxf(max_val, row_x[c]);
        }

        float sum_val = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float e = expf(row_x[c] - max_val);
            row_y[c] = e;
            sum_val += e;
        }

        for (int c = 0; c < cols; ++c) {
            row_y[c] = row_y[c] / sum_val;
        }
    }
}
