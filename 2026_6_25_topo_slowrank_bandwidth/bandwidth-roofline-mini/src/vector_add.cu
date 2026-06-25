#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

static void check_cuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",
                     file, line, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(call) check_cuda((call), __FILE__, __LINE__)

__global__ void vector_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  std::uint64_t n) {
    std::uint64_t idx = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

static std::uint64_t parse_u64(const char* s) {
    char* end = nullptr;
    unsigned long long v = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0') {
        std::fprintf(stderr, "Invalid integer: %s\n", s);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<std::uint64_t>(v);
}

static int parse_i32(const char* s) {
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v <= 0) {
        std::fprintf(stderr, "Invalid positive integer: %s\n", s);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(v);
}

static double parse_f64(const char* s) {
    char* end = nullptr;
    double v = std::strtod(s, &end);
    if (end == s || *end != '\0' || v <= 0.0) {
        std::fprintf(stderr, "Invalid positive number: %s\n", s);
        std::exit(EXIT_FAILURE);
    }
    return v;
}

static const char* classify_roofline(double arithmetic_intensity,
                                     double peak_fp32_gflops,
                                     double peak_mem_gbs,
                                     double* roofline_knee) {
    if (peak_fp32_gflops > 0.0 && peak_mem_gbs > 0.0) {
        *roofline_knee = peak_fp32_gflops / peak_mem_gbs;
        return arithmetic_intensity < *roofline_knee ? "memory_bound" : "compute_bound";
    }

    *roofline_knee = 0.0;
    return arithmetic_intensity < 1.0 ? "memory_bound_low_ai_expected"
                                      : "needs_hardware_peaks";
}

static void print_help(const char* argv0) {
    std::printf(
        "Usage: %s [--n N] [--warmup W] [--repeat R] [--block B]\n"
        "          [--peak-fp32-gflops GFLOPS] [--peak-mem-gbs GBPS]\n"
        "\n"
        "Example:\n"
        "  %s --n 268435456 --warmup 10 --repeat 50\n"
        "\n"
        "Notes:\n"
        "  dtype is fixed to float32 in this mini demo.\n",
        argv0, argv0);
}

template <typename T>
static double percentile_ms(std::vector<T> values, double q) {
    std::sort(values.begin(), values.end());
    if (values.empty()) return 0.0;

    double pos = q * static_cast<double>(values.size() - 1);
    std::size_t lo = static_cast<std::size_t>(pos);
    std::size_t hi = std::min(lo + 1, values.size() - 1);
    double frac = pos - static_cast<double>(lo);
    return static_cast<double>(values[lo]) * (1.0 - frac) +
           static_cast<double>(values[hi]) * frac;
}

int main(int argc, char** argv) {
    std::uint64_t n = 268435456ULL;  // 256M floats. A/B/C total about 3 GiB.
    int warmup = 10;
    int repeat = 50;
    int block_size = 256;
    double peak_fp32_gflops = 0.0;
    double peak_mem_gbs = 0.0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_help(argv[0]);
            return 0;
        } else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n = parse_u64(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = parse_i32(argv[++i]);
        } else if (std::strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            repeat = parse_i32(argv[++i]);
        } else if (std::strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
            block_size = parse_i32(argv[++i]);
        } else if (std::strcmp(argv[i], "--peak-fp32-gflops") == 0 && i + 1 < argc) {
            peak_fp32_gflops = parse_f64(argv[++i]);
        } else if (std::strcmp(argv[i], "--peak-mem-gbs") == 0 && i + 1 < argc) {
            peak_mem_gbs = parse_f64(argv[++i]);
        } else {
            std::fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            print_help(argv[0]);
            return EXIT_FAILURE;
        }
    }

    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    std::printf("# kernel=vector_add\n");
    std::printf("# device=%s\n", prop.name);
    std::printf("# dtype=float32\n");
    std::printf("# N=%llu\n", static_cast<unsigned long long>(n));
    std::printf("# warmup=%d\n", warmup);
    std::printf("# repeat=%d\n", repeat);
    std::printf("# block=%d\n", block_size);

    std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::size_t bytes_per_iter = 3ULL * bytes;  // read A, read B, write C.
    double flops_per_iter = static_cast<double>(n);  // one FP32 add per element.
    double arithmetic_intensity = flops_per_iter / static_cast<double>(bytes_per_iter);
    double roofline_knee = 0.0;
    const char* roofline_region = classify_roofline(
        arithmetic_intensity, peak_fp32_gflops, peak_mem_gbs, &roofline_knee);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    // Initialization is not included in the kernel benchmark.
    CHECK_CUDA(cudaMemset(d_a, 1, bytes));
    CHECK_CUDA(cudaMemset(d_b, 2, bytes));
    CHECK_CUDA(cudaMemset(d_c, 0, bytes));

    int grid_size = static_cast<int>((n + block_size - 1) / block_size);

    // Warmup helps reduce first-iteration launch and clock-boost noise.
    for (int i = 0; i < warmup; ++i) {
        vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::vector<float> times_ms;
    std::vector<double> sync_times_ms;
    times_ms.reserve(repeat);
    sync_times_ms.reserve(repeat);

    for (int i = 0; i < repeat; ++i) {
        CHECK_CUDA(cudaDeviceSynchronize());
        auto sync_start = std::chrono::steady_clock::now();

        CHECK_CUDA(cudaEventRecord(start));
        vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaDeviceSynchronize());
        auto sync_stop = std::chrono::steady_clock::now();

        CHECK_CUDA(cudaGetLastError());

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
        sync_times_ms.push_back(
            std::chrono::duration<double, std::milli>(sync_stop - sync_start).count());
    }

    double p50_ms = percentile_ms(times_ms, 0.50);
    double p95_ms = percentile_ms(times_ms, 0.95);
    double sync_p50_ms = percentile_ms(sync_times_ms, 0.50);
    double sync_p95_ms = percentile_ms(sync_times_ms, 0.95);

    // Decimal GB/s, using 1e9 bytes/s.
    double gbs_p50 = static_cast<double>(bytes_per_iter) / (p50_ms / 1000.0) / 1e9;
    double gbs_p95_latency = static_cast<double>(bytes_per_iter) / (p95_ms / 1000.0) / 1e9;

    std::printf(
        "RESULT,kernel=vector_add,dtype=float32,N=%llu,warmup=%d,repeat=%d,"
        "read_bytes=%zu,write_bytes=%zu,bytes_per_iter=%zu,flops_per_iter=%.0f,"
        "arithmetic_intensity_flop_per_byte=%.6f,roofline_region=%s,roofline_knee_flop_per_byte=%.6f,"
        "event_p50_ms=%.6f,event_p95_ms=%.6f,sync_p50_ms=%.6f,sync_p95_ms=%.6f,"
        "GBs_p50=%.3f,GBs_at_p95_latency=%.3f\n",
        static_cast<unsigned long long>(n),
        warmup,
        repeat,
        2ULL * bytes,
        bytes,
        bytes_per_iter,
        flops_per_iter,
        arithmetic_intensity,
        roofline_region,
        roofline_knee,
        p50_ms,
        p95_ms,
        sync_p50_ms,
        sync_p95_ms,
        gbs_p50,
        gbs_p95_latency);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}
