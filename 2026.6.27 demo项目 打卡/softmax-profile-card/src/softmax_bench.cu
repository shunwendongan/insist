// Minimal CUDA benchmark binary for future GPU profiling with Nsight Compute.
// Build:
//   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
//   cmake --build build -j
// Profile:
//   ncu --set full ./build/softmax_bench --rows 4096 --cols 1024

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

__global__
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

int get_int_arg(int argc, char** argv, const char* name, int default_value) {
    std::string key = std::string("--") + name;
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == key) return std::atoi(argv[i + 1]);
    }
    return default_value;
}

int main(int argc, char** argv) {
    int rows = get_int_arg(argc, argv, "rows", 4096);
    int cols = get_int_arg(argc, argv, "cols", 1024);
    int iters = get_int_arg(argc, argv, "iters", 100);
    int threads = get_int_arg(argc, argv, "threads", 256);

    if ((threads & (threads - 1)) != 0) {
        std::fprintf(stderr, "--threads must be a power of two for this demo.\n");
        return 1;
    }

    size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    size_t bytes = n * sizeof(float);

    std::vector<float> h_x(n);
    std::mt19937 rng(0);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) h_x[i] = dist(rng);

    float* d_x = nullptr;
    float* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

    dim3 grid(rows);
    dim3 block(threads);
    size_t smem_bytes = static_cast<size_t>(threads) * sizeof(float);

    for (int i = 0; i < 10; ++i) {
        softmax_block_kernel<<<grid, block, smem_bytes>>>(d_x, d_y, rows, cols);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        softmax_block_kernel<<<grid, block, smem_bytes>>>(d_x, d_y, rows, cols);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / static_cast<float>(iters);

    // Rough effective traffic estimate, matching tests/test_softmax.py.
    double effective_bytes = static_cast<double>(bytes) * 5.0;
    double effective_gbs = effective_bytes / (avg_ms / 1000.0) / 1e9;

    std::printf("rows=%d cols=%d threads=%d iters=%d\n", rows, cols, threads, iters);
    std::printf("softmax_block avg_ms=%.6f estimated_effective_GB/s=%.3f\n", avg_ms, effective_gbs);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
