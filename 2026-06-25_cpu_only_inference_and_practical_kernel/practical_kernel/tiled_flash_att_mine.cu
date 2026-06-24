#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#define warp_size 32
#define block_m 8
#define max_d 128
#define block_n 32
#define max_dim_per_lane ((max_d + warp_size - 1) / warp_size)
// inline 函数: 处理将 warp 内每个 lane 的部分结果累加起来.
// 这里最后广播到所有 lane, 这样后续每个 lane 都能拿到完整 dot 结果.
__forceinline__ __device__ float warp_reduce_sum(float x) {
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffffu, x, offset);
    }
    return __shfl_sync(0xffffffffu, x, 0);
}
__global__ void flash_att_kernel(const float *q, const float *k, const float *v,
                                 float *output, int batch, int heads,
                                 int sequency, int dim, int casual) {
    // 当前版本固定用 max_d 控制 shared memory 尺寸, 因此 dim 不能超过 max_d.
    if (dim > max_d) {
        return;
    }
    // 首先把 CUDA block/grid 和底层 attention 维度关系映射起来.
    // grid.z -> batch, grid.y -> head, grid.x -> query block.
    int batchid = blockIdx.z;
    int headsid = blockIdx.y;
    int row_base = blockIdx.x * block_m;
    // 一个 warp 处理一个 query row.
    // lane 是 warp 内线程编号, warp 是当前 block 内第几个 warp.
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int q_row = row_base + warp;
    // (B,H) 确定之后, q/k/v/output 在全局内存里的起始偏移.
    int bh_base_dim = (((batchid * heads) + headsid) * sequency) * dim;
    const bool valid_q = (warp < block_m && q_row < sequency);
    // q_tile 是当前 block_m 个 query 向量.
    // k_tile/v_tile 是当前 block_n 个 key/value 向量.
    __shared__ float q_tile[block_m][max_d];
    __shared__ float k_tile[block_n][max_d];
    __shared__ float v_tile[block_n][max_d];
    // Step 1: 把当前 block 的 Q tile 加载到 shared memory.
    // 这里用 stride loop, 让整个 block 的线程一起搬运 block_m * dim 个元素.
    for (int idx = threadIdx.x; idx < block_m * dim; idx += blockDim.x) {
        const int r = idx / dim;
        const int c = idx - r * dim;
        // r/c 是 q_tile 内的行列, global_row 才是序列里的真实 query 行号.
        const int global_row = r + row_base;
        q_tile[r][c] = (global_row < sequency) ? q[bh_base_dim + global_row * dim + c] : 0.0f;
    }
    __syncthreads();
    // Step 2: 初始化当前 query row 的输出累加器.
    // 每个 lane 负责 dim 里的若干个元素, 所以 acc 也按 lane 分片保存.
    float acc[max_dim_per_lane];
    for (int i = 0; i < max_dim_per_lane; ++i) {
        acc[i] = 0.0f;
    }
    // Step 3: 初始化 online softmax 状态.
    // m 是已经扫描过的 score 最大值.
    // l 是当前 softmax denominator, 即 sum exp(score - m).
    float m = -FLT_MAX;
    float l = 0.0f;
    // Step 4: causal mask 下减少 K/V 扫描范围.
    // casual 为 0 表示普通 attention, 可以看到完整序列.
    // casual 非 0 表示 causal attention, query row 只能看见自己之前和自己的 key.
    int casual_end = row_base + block_m;
    int kv_end = casual ? ((sequency < casual_end) ? sequency : casual_end) : sequency;
    // Step 5: 分块扫描 K/V tile.
    // 每次把 block_n 行 K/V 搬到 shared memory, 然后让每个 warp 扫描这个 tile.
    for (int k_row = 0; k_row < kv_end; k_row += block_n) {
        // Step 5.1: 加载当前 K/V tile 到 shared memory.
        for (int t = threadIdx.x; t < block_n * dim; t += blockDim.x) {
            int row = t / dim;
            int col = t - row * dim;
            int global_k_row = k_row + row;
            if (global_k_row < sequency) {
                const int global_idx = bh_base_dim + global_k_row * dim + col;
                k_tile[row][col] = k[global_idx];
                v_tile[row][col] = v[global_idx];
            } else {
                k_tile[row][col] = 0.0f;
                v_tile[row][col] = 0.0f;
            }
        }
        __syncthreads();