// Fused RMSNorm kernel.
//
// Equivalent to: x = x * weight / sqrt(mean(x^2) + eps)
//
// One warp (or block) per row. Uses warp-level reduction for the mean-square,
// then broadcasts it back to all threads to apply the scale.
//
// Template params:
//   T         – element type (float / __half)
//   BLOCK     – threads per block (must be a power of 2, ≤ 1024)
//
// Grid:  (num_rows,)
// Block: (BLOCK,)
//
// Each block handles exactly one row of length `hidden`.

#include <cuda_fp16.h>
#include <stdint.h>

// ── Warp reduce sum ───────────────────────────────────────────────────────────

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ── Block reduce sum (across all warps) ──────────────────────────────────────

template <int BLOCK>
__device__ float block_reduce_sum(float val, float* smem) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) smem[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (BLOCK / 32)) ? smem[lane] : 0.f;
    val = warp_reduce_sum(val);

    return val;
}

// ── rms_norm_f32 ─────────────────────────────────────────────────────────────

template <int BLOCK>
__global__ void rms_norm_f32(
    float*       __restrict__ out,
    const float* __restrict__ inp,
    const float* __restrict__ weight,
    int          hidden,
    float        eps
) {
    __shared__ float smem[32];

    const int row = blockIdx.x;
    const float* x = inp    + row * hidden;
    float*       y = out    + row * hidden;

    // Sum of squares.
    float ss = 0.f;
    for (int i = threadIdx.x; i < hidden; i += BLOCK)
        ss += x[i] * x[i];
    ss = block_reduce_sum<BLOCK>(ss, smem);

    // RMS scale factor.
    const float rms_scale = rsqrtf(ss / (float)hidden + eps);

    // Apply: y = x * weight * rms_scale.
    for (int i = threadIdx.x; i < hidden; i += BLOCK)
        y[i] = x[i] * rms_scale * weight[i];
}

// ── rms_norm_f16 ─────────────────────────────────────────────────────────────

template <int BLOCK>
__global__ void rms_norm_f16(
    __half*       __restrict__ out,
    const __half* __restrict__ inp,
    const __half* __restrict__ weight,
    int           hidden,
    float         eps
) {
    __shared__ float smem[32];

    const int row = blockIdx.x;
    const __half* x = inp    + row * hidden;
    __half*       y = out    + row * hidden;

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

// ── Exported entry points (extern "C" so Rust can find them) ─────────────────

extern "C" {

__global__ void rms_norm_f32_256(
    float* out, const float* inp, const float* weight, int hidden, float eps)
{ rms_norm_f32<256>(out, inp, weight, hidden, eps); }

__global__ void rms_norm_f32_512(
    float* out, const float* inp, const float* weight, int hidden, float eps)
{ rms_norm_f32<512>(out, inp, weight, hidden, eps); }

__global__ void rms_norm_f16_256(
    __half* out, const __half* inp, const __half* weight, int hidden, float eps)
{ rms_norm_f16<256>(out, inp, weight, hidden, eps); }

__global__ void rms_norm_f16_512(
    __half* out, const __half* inp, const __half* weight, int hidden, float eps)
{ rms_norm_f16<512>(out, inp, weight, hidden, eps); }

} // extern "C"
