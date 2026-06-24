#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// ==========================
// 手撕版 tiled FlashAttention forward
// ==========================
//
// 计算目标：
// O = softmax(QK^T / sqrt(D)) V
//
// 输入输出布局：
// Q/K/V/O 都是 [B, H, S, D] 的 contiguous float32 数组
//
// B = batch size
// H = attention head 数量
// S = sequence length
// D = head dimension
//
// 核心思想：
// 1. 一个 CUDA block 负责同一个 (b, h) 下的 BLOCK_M 个 query row。
// 2. 一个 warp 负责一个 query row。
// 3. 每次把 BLOCK_M x D 的 Q tile 放入 shared memory。
// 4. 每次把 BLOCK_N x D 的 K/V tile 放入 shared memory。
// 5. 对 K/V tile 做流式扫描。
// 6. 使用在线 softmax 更新 m/l/acc。
// 7. 不显式生成 S x S 的 attention 矩阵，节省显存。
//8. 核心数学公式 O = softmax(QK^T / sqrt(D)) V,

#define WARP_SIZE 32


#define BLOCK_M 8

// 每轮从 K/V 中加载 32 个 key/value row。
#define BLOCK_N 32

#define MAX_D 128

#define MAX_D_PER_LANE ((MAX_D + WARP_SIZE - 1) / WARP_SIZE)


// ==========================
// warp 内求和
// ==========================
//
// 输入：
// 每个 lane 持有一个 x。
//
// 输出：
// 返回整个 warp 内所有 x 的和。
// 并且返回值会广播给 warp 内所有 lane。
//
// 用途：
// 计算 Q_i 和 K_j 的 dot product 时，
// 每个 lane 只算一部分维度，
// 最后需要 warp reduce 得到完整 dot。
__forceinline__ __device__ float warp_reduce_sum(float x) {
    // 使用 shuffle-down 做树形规约。
  
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffffu, x, offset);
    }

    // 上面的 reduction 后，只有 lane 0 一定拿到了完整 sum。
    // 但是后续每个 lane 都要用完整 dot 值更新 softmax 和 acc，
    // 所以这里把 lane 0 的 x 广播给整个 warp。
    return __shfl_sync(0xffffffffu, x, 0);
}


__global__ void flash_attention_tiled_kernel(const float* __restrict__ Q,
                                             const float* __restrict__ K,
                                             const float* __restrict__ V,
                                             float* __restrict__ O,
                                             int B,
                                             int H,
                                             int S,
                                             int D,
                                             int causal) {

    __shared__ float sQ[BLOCK_M][MAX_D];
    __shared__ float sK[BLOCK_N][MAX_D];
    __shared__ float sV[BLOCK_N][MAX_D];

    const int lane = threadIdx.x & (WARP_SIZE - 1);

    const int warp = threadIdx.x >> 5;

    // grid.z 对应 batch 维度。
    const int b = blockIdx.z;

    // grid.y 对应 head 维度。
    const int h = blockIdx.y; //不同级别粒度对应关系, 约定俗成

    // 当前 block 处理的 query tile 起始行。
 
    const int q_block = blockIdx.x * BLOCK_M;

    // 当前 warp 负责的 query row。
    
    const int q_row = q_block + warp;

    // 当前 (b, h) 对应的起始 offset。
    //
    // 原始布局：
    // Q[b][h][s][d]
    //
    // 线性下标：
    // ((b * H + h) * S + s) * D + d
    //
    // bh_base = ((b * H + h) * S) * D
    // 后面访问某个 row 的 d：
    // bh_base + row * D + d
    const int bh_base = ((b * H + h) * S) * D;

    // 判断当前 warp 对应的 query row 是否有效。
    //
    // 最后一个 query block 可能越界。
  
    const bool valid_q = (warp < BLOCK_M) && (q_row < S);


    // =====================================================
    // Step 1: 把当前 block 的 Q tile 加载到 shared memory
   
    for (int idx = threadIdx.x; idx < BLOCK_M * D; idx += blockDim.x) {//网格步长循环
        // idx 映射到 tile 内二维坐标：
        // r = query tile 内第几行
        // d = head_dim 的第几维
        const int r = idx / D;
        const int d = idx - r * D;

        // 当前 tile 内第 r 行，对应全局 query row。
        const int global_q = q_block + r;

        // 如果 global_q 没越界，就从 global memory 加载 Q。
        // 如果越界，就填 0，避免非法访问。
        sQ[r][d] = (global_q < S)
                     ? Q[bh_base + global_q * D + d]
                     : 0.0f;
    }

    __syncthreads();


    // =====================================================
    // Step 2: 初始化当前 query row 的输出累加器 acc

    float acc[MAX_D_PER_LANE];//线程步长循环,导致多个acc,后面需要累加

    #pragma unroll
    for (int t = 0; t < MAX_D_PER_LANE; ++t) {
        acc[t] = 0.0f;
    }


    // =====================================================
    // Step 3: 初始化在线 softmax 状态
  
    // m = 已经扫描过的 score 最大值
    // l = 当前 softmax denominator，即 sum exp(score - m)
    // acc = 当前未归一化的输出累计值
    //
    // 最后输出：
    // O = acc / l
    float m = -FLT_MAX;
    float l = 0.0f;


    // =====================================================
    // Step 4: causal mask 下减少 K/V 扫描范围
    // =====================================================
    //
    // 如果 causal = 0：
    // 每个 query 可以看所有 key，kv_end = S。
    //
    // 如果 causal = 1：
    // query row q 只能看 k_row <= q_row。
    //
    // 当前 block 处理 q_block ~ q_block + BLOCK_M - 1。
    // 这个 block 内最大的 q_row 是 q_block + BLOCK_M - 1。
    //
    // 所以对于整个 block 来说，超过 q_block + BLOCK_M 的 K tile
    // 肯定所有 query row 都看不到，可以不扫描。
    const int causal_end = q_block + BLOCK_M;//end token

    const int kv_end = causal
                         ? ((S < causal_end) ? S : causal_end)
                         : S;


    // =====================================================
   // Step 5.扫描kv tile
    // =====================================================
    //
    // 每次处理 BLOCK_N 行 K/V。
    //
    // kv_block = 当前 K/V tile 的起始 row。
    for (int kv_block = 0; kv_block < kv_end; kv_block += BLOCK_N) {

        // -------------------------------------------------
        // Step 5.1: 加载当前 K/V tile 到 shared memory
 
        for (int idx = threadIdx.x; idx < BLOCK_N * D; idx += blockDim.x) {
            // r 是 tile 内第几个 key/value row。
            // d 是 head_dim 维度。
            const int r = idx / D;
            const int d = idx - r * D;

            // 当前 K/V tile 内第 r 行对应的全局 row。
            const int k_row = kv_block + r;

            if (k_row < S) {
                const int g = bh_base + k_row * D + d;

                // 从 global memory 加载 K 和 V 到 shared memory。
                sK[r][d] = K[g];
                sV[r][d] = V[g];
            } else {
                // 最后一个 K/V tile 可能越界。
              
                sK[r][d] = 0.0f;
                sV[r][d] = 0.0f;
            }
        }

        __syncthreads();


        // -------------------------------------------------
        // Step 5.2: 当前 warp 扫描 K/V tile 内每一个 key
        // -------------------------------------------------
        //
        // 当前 warp 固定负责一个 q_row。
        // j 从 0 到 BLOCK_N-1，依次计算：
        //
        // score = dot(Q[q_row], K[kv_block + j]) / sqrt(D)
        //
        // 然后用这个 score 在线更新 softmax 状态。
        for (int j = 0; j < BLOCK_N; ++j) {
            const int k_row = kv_block + j;

            // 跳过无效情况：
            //
            // 1. 当前 query row 越界。
            // 2. 当前 key row 越界。
            // 3. causal attention 中，key 在 query 未来位置：
            //    k_row > q_row。
            if (!valid_q || k_row >= S || (causal && k_row > q_row)) {
                continue;
            }


            // ---------------------------------------------
            // Step 5.2.1: 计算 dot(Q_i, K_j)
      
            float dot_part = 0.0f;

            #pragma unroll
            for (int t = 0; t < MAX_D_PER_LANE; ++t) {
                const int d = lane + t * WARP_SIZE;

                if (d < D) {
                    dot_part += sQ[warp][d] * sK[j][d];
                }
            }

            // warp 内规约所有 lane 的 dot_part。
            // 返回后，每个 lane 都拿到完整 dot。
            const float dot = warp_reduce_sum(dot_part);

            // attention score = dot / sqrt(D)。
            //
            // rsqrtf(D) = 1 / sqrt(D)
            const float score = dot * rsqrtf((float)D);


            // ---------------------------------------------
            // Step 5.2.2: 在线 softmax 更新
         
            const float m_new = fmaxf(m, score);

            // 如果 m_new 变大，旧的 acc/l 都需要缩放到新的基准。
            const float old_scale = __expf(m - m_new);

            // 当前 score 对应的未归一化概率。
            const float p = __expf(score - m_new);


            // ---------------------------------------------
            // Step 5.2.3: 更新 acc
            // ---------------------------------------------
            //
            // 每个 lane 只更新自己负责的输出维度。
            //
            // acc[d] = acc[d] * old_scale + p * V_j[d]
            #pragma unroll
            for (int t = 0; t < MAX_D_PER_LANE; ++t) {
                const int d = lane + t * WARP_SIZE;

                if (d < D) {
                    acc[t] = acc[t] * old_scale + p * sV[j][d];//old_scale作用???
                }
            }

            // 更新 softmax denominator。
            l = l * old_scale + p;

            // 更新最大 score。
            m = m_new;
        }

        // 当前 K/V tile 处理完。
        //
        // 下一轮循环会复用 sK/sV 这块 shared memory，
        // 所以要确保所有 warp 都已经用完当前 tile，
        // 再让线程去加载下一块 K/V tile。
        __syncthreads();
    }


    // =====================================================
    // Step 6: 写回最终输出 O
    // =====================================================
    //
    // 在线 softmax 累计结束后：
    //
    // acc 保存的是：
    // sum_j exp(score_j - m) * V_j
    //
    // l 保存的是：
    // sum_j exp(score_j - m)
    //
    // 所以最终：
    // O = acc / l
    if (valid_q) {
        #pragma unroll
        for (int t = 0; t < MAX_D_PER_LANE; ++t) {
            const int d = lane + t * WARP_SIZE;

            if (d < D) {
                O[bh_base + q_row * D + d] = acc[t] / l;
            }
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
                      int causal) {//casual attention 自回归,如果是casual 1,代表token看不到后面的token(S限制的后面的token)
                      // casual 0,正常attention,可以看到全部的`k token
    // 参数检查：
    // B/H/S/D 必须为正。
    // D 不能超过 MAX_D。
    if (B <= 0 || H <= 0 || S <= 0 || D <= 0 || D > MAX_D) {
        return;
    }

    // 一个 block 里面有 BLOCK_M 个 warp。
    //
    // BLOCK_M = 8
    // WARP_SIZE = 32
    //
    // 所以 blockDim.x = 256 线程。
    dim3 block(BLOCK_M * WARP_SIZE);

    // grid.x：query row 方向，每个 block 处理 BLOCK_M 行。
    // grid.y：head 方向。
    // grid.z：batch 方向。
    //
    // grid.x = ceil(S / BLOCK_M)
    dim3 grid((S + BLOCK_M - 1) / BLOCK_M, H, B);

    // 启动 kernel。
    flash_attention_tiled_kernel<<<grid, block>>>(
        Q, K, V, O,
        B, H, S, D,
        causal
    );
}



