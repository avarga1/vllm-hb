// Fused RMSNorm kernel.
//
// Equivalent to: out = x * weight / sqrt(mean(x^2) + eps)
//
// One block per row. Uses warp-level reduction for the mean-square,
// then broadcasts it back to all threads to apply the scale.
//
// Template params:
//   T         – element type (float / __half)
//   BLOCK     – threads per block (must be a power of 2, ≤ 1024)
//
// Grid:  (num_rows,)
// Block: (BLOCK,)

#include <cuda_fp16.h>
#include <stdint.h>

// ── Warp reduce sum ───────────────────────────────────────────────────────────

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ── Block reduce sum (across all warps in shared memory) ─────────────────────
//
// Returns the correct total to ALL threads in the block.
//
// Bug note: the naive pattern (warp-reduce → smem → final warp-reduce) only
// gives thread 0 the correct total; threads 32–(BLOCK-1) get 0 because they
// read smem[threadIdx.x >= BLOCK/32] which is always 0.  The fix is to write
// the final result back to smem[0] and broadcast it via a second __syncthreads.

template <int BLOCK>
__device__ float block_reduce_sum(float val, float* smem) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;

    // Step 1: reduce within each warp; lane 0 has the warp total.
    val = warp_reduce_sum(val);

    // Step 2: collect warp totals in shared memory.
    if (lane == 0) smem[wid] = val;
    __syncthreads();

    // Step 3: first warp reduces all warp totals → thread 0 has the block total.
    val = (threadIdx.x < (BLOCK / 32)) ? smem[lane] : 0.f;
    val = warp_reduce_sum(val);

    // Step 4: broadcast the block total to ALL threads via smem[0].
    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

// ── Device-side implementation (called from __global__ wrappers) ──────────────

template <int BLOCK>
__device__ void rms_norm_f32_impl(
    float*       __restrict__ out,
    const float* __restrict__ inp,
    const float* __restrict__ weight,
    int   hidden,
    float eps
) {
    extern __shared__ float smem[];

    const int row   = blockIdx.x;
    const float* x  = inp + row * hidden;
    float*       y  = out + row * hidden;

    float ss = 0.f;
    for (int i = threadIdx.x; i < hidden; i += BLOCK)
        ss += x[i] * x[i];
    ss = block_reduce_sum<BLOCK>(ss, smem);

    const float rms_scale = rsqrtf(ss / (float)hidden + eps);

    for (int i = threadIdx.x; i < hidden; i += BLOCK)
        y[i] = x[i] * rms_scale * weight[i];
}

template <int BLOCK>
__device__ void rms_norm_f16_impl(
    __half*       __restrict__ out,
    const __half* __restrict__ inp,
    const __half* __restrict__ weight,
    int   hidden,
    float eps
) {
    extern __shared__ float smem[];

    const int row      = blockIdx.x;
    const __half* x    = inp + row * hidden;
    __half*       y    = out + row * hidden;

    float ss = 0.f;
    for (int i = threadIdx.x; i < hidden; i += BLOCK)
        ss += __half2float(x[i]) * __half2float(x[i]);
    ss = block_reduce_sum<BLOCK>(ss, smem);

    const float rms_scale = rsqrtf(ss / (float)hidden + eps);

    for (int i = threadIdx.x; i < hidden; i += BLOCK) {
        float xi = __half2float(x[i]);
        float wi = __half2float(weight[i]);
        y[i] = __float2half(xi * rms_scale * wi);
    }
}

// ── Exported __global__ entry points (extern "C" so Rust can find them) ──────

extern "C" {

__global__ void rms_norm_f32_256(
    float* out, const float* inp, const float* weight, int hidden, float eps)
{ rms_norm_f32_impl<256>(out, inp, weight, hidden, eps); }

__global__ void rms_norm_f32_512(
    float* out, const float* inp, const float* weight, int hidden, float eps)
{ rms_norm_f32_impl<512>(out, inp, weight, hidden, eps); }

__global__ void rms_norm_f16_256(
    __half* out, const __half* inp, const __half* weight, int hidden, float eps)
{ rms_norm_f16_impl<256>(out, inp, weight, hidden, eps); }

__global__ void rms_norm_f16_512(
    __half* out, const __half* inp, const __half* weight, int hidden, float eps)
{ rms_norm_f16_impl<512>(out, inp, weight, hidden, eps); }

} // extern "C"
