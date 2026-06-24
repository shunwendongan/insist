#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// ============================================================
// 教学简化版 FlashAttention forward
// ============================================================
//
// 计算:
//   O = softmax(QK^T / sqrt(D)) V
//
// 输入输出布局:
//   Q/K/V/O: [B, H, S, D] contiguous float32
//
// 简化点:
//   1. 一个 CUDA block 只负责一个 query row。
//   2. 一个 block 只有一个 warp，也就是 32 个线程。
//   3. 不使用 shared memory tile。
//   4. 不生成 S x S attention 矩阵。
//   5. 仍然使用在线 softmax，所以保留 FlashAttention 最核心的数学逻辑。
//
// 代价:
//   这个版本比 tiled 版本慢，因为 K/V 会被不同 query row 重复从 global memory 读取。
//   但代码结构更适合学习。

#define WARP_SIZE 32
#define MAX_D 128
#define MAX_D_PER_LANE ((MAX_D + WARP_SIZE - 1) / WARP_SIZE)

__forceinline__ __device__ float warp_reduce_sum(float x) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffffu, x, offset);
    }
    return __shfl_sync(0xffffffffu, x, 0);
}


__global__ void flash_attention_simple_kernel(const float* __restrict__ Q,
                                              const float* __restrict__ K,
                                              const float* __restrict__ V,
                                              float* __restrict__ O,
                                              int B,
                                              int H,
                                              int S,
                                              int D,
                                              int causal) {
    const int lane = threadIdx.x;

    // grid.x 对应 sequence row, grid.y 对应 head, grid.z 对应 batch。
    const int q_row = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    const int bh_base = ((b * H + h) * S) * D;

    // 每个 lane 负责输出向量 O[q_row, :] 中若干个 d。
    float acc[MAX_D_PER_LANE];

    #pragma unroll
    for (int t = 0; t < MAX_D_PER_LANE; ++t) {
        acc[t] = 0.0f;
    }

    // 在线 softmax 状态:
    // m = 当前已经见过的最大 score
    // l = sum exp(score - m)
    // acc = sum exp(score - m) * V
    float m = -FLT_MAX;
    float l = 0.0f;

    // causal = 0: 当前 query 可以看所有 key: 0..S-1
    // causal = 1: 当前 query 只能看过去和自己: 0..q_row
    const int kv_end = causal ? (q_row + 1) : S;

    for (int k_row = 0; k_row < kv_end; ++k_row) {
        float dot_part = 0.0f;

        // 一个 warp 分摊 D 维 dot product。
        #pragma unroll
        for (int t = 0; t < MAX_D_PER_LANE; ++t) {
            const int d = lane + t * WARP_SIZE;

            if (d < D) {
                const float q = Q[bh_base + q_row * D + d];
                const float k = K[bh_base + k_row * D + d];
                dot_part += q * k;
            }
        }

        const float dot = warp_reduce_sum(dot_part);
        const float score = dot * rsqrtf((float)D);

        // 在线 softmax:
        // 新基准 m_new = max(m_old, score)
        // 旧累计量需要乘 exp(m_old - m_new) 对齐到新基准。
        const float m_new = fmaxf(m, score);
        const float old_scale = __expf(m - m_new);
        const float p = __expf(score - m_new);

        #pragma unroll
        for (int t = 0; t < MAX_D_PER_LANE; ++t) {
            const int d = lane + t * WARP_SIZE;

            if (d < D) {
                const float v = V[bh_base + k_row * D + d];
                acc[t] = acc[t] * old_scale + p * v;
            }
        }

        l = l * old_scale + p;
        m = m_new;
    }

    // O = acc / l
    #pragma unroll
    for (int t = 0; t < MAX_D_PER_LANE; ++t) {
        const int d = lane + t * WARP_SIZE;

        if (d < D) {
            O[bh_base + q_row * D + d] = acc[t] / l;
        }
    }
}

extern "C" void solve(const float* Q,
                      const float* K,
                      const float* V,
                      float* O,
                      int B,
                      int H,
                      int S,
                      int D,
                      int causal) {
    if (B <= 0 || H <= 0 || S <= 0 || D <= 0 || D > MAX_D) {
        return;
    }

    dim3 block(WARP_SIZE);
    dim3 grid(S, H, B);

    flash_attention_simple_kernel<<<grid, block>>>(
        Q, K, V, O,
        B, H, S, D,
        causal
    );
}

