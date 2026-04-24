// Fused RoPE (Rotary Position Embedding) kernel.
//
// Layout expectations (same as candle-transformers Qwen2/Llama convention):
//   q, k:    [batch, num_heads, seq_len, head_dim]  (contiguous)
//   cos/sin: [seq_len, head_dim/2]
//
// Grid:  (batch * num_heads * seq_len,)
// Block: (head_dim / 2,)  — each thread handles one (x0, x1) pair.

#include <cuda_fp16.h>
#include <stdint.h>

// ── Device helpers ────────────────────────────────────────────────────────────
//
// All implementation functions are __device__ so they can be called from
// __global__ entry points without triggering "must be configured" errors.

// Single-tensor RoPE: applies rotation to one Q or K tensor.
__device__ void rope_single_f32_impl(
    float*       __restrict__ out,
    const float* __restrict__ inp,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int seq_len,
    int head_dim
) {
    const int half_dim  = head_dim >> 1;
    const int token_idx = blockIdx.x;
    const int d         = threadIdx.x;
    if (d >= half_dim) return;

    const int t    = token_idx % seq_len;
    const int base = token_idx * head_dim;
    const float x0 = inp[base + d];
    const float x1 = inp[base + d + half_dim];
    const float c  = cos_table[t * half_dim + d];
    const float s  = sin_table[t * half_dim + d];
    out[base + d]            = x0 * c - x1 * s;
    out[base + d + half_dim] = x0 * s + x1 * c;
}

__device__ void rope_single_f16_impl(
    __half*       __restrict__ out,
    const __half* __restrict__ inp,
    const __half* __restrict__ cos_table,
    const __half* __restrict__ sin_table,
    int seq_len,
    int head_dim
) {
    const int half_dim  = head_dim >> 1;
    const int token_idx = blockIdx.x;
    const int d         = threadIdx.x;
    if (d >= half_dim) return;

    const int t    = token_idx % seq_len;
    const int base = token_idx * head_dim;
    const float x0 = __half2float(inp[base + d]);
    const float x1 = __half2float(inp[base + d + half_dim]);
    const float c  = __half2float(cos_table[t * half_dim + d]);
    const float s  = __half2float(sin_table[t * half_dim + d]);
    out[base + d]            = __float2half(x0 * c - x1 * s);
    out[base + d + half_dim] = __float2half(x0 * s + x1 * c);
}

// Fused Q+K RoPE: reads both q and k, writes out_q and out_k in one pass.
__device__ void rope_fused_f32_impl(
    float*       __restrict__ out_q,
    float*       __restrict__ out_k,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int num_heads,
    int seq_len,
    int head_dim
) {
    const int half_dim  = head_dim >> 1;
    const int token_idx = blockIdx.x;
    const int d         = threadIdx.x;
    if (d >= half_dim) return;

    const int t       = token_idx % seq_len;
    const int qk_base = token_idx * head_dim;
    const float c = cos_table[t * half_dim + d];
    const float s = sin_table[t * half_dim + d];

    const float q0 = q[qk_base + d];
    const float q1 = q[qk_base + d + half_dim];
    out_q[qk_base + d]            = q0 * c - q1 * s;
    out_q[qk_base + d + half_dim] = q0 * s + q1 * c;

    const float k0 = k[qk_base + d];
    const float k1 = k[qk_base + d + half_dim];
    out_k[qk_base + d]            = k0 * c - k1 * s;
    out_k[qk_base + d + half_dim] = k0 * s + k1 * c;
}

__device__ void rope_fused_f16_impl(
    __half*       __restrict__ out_q,
    __half*       __restrict__ out_k,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ cos_table,
    const __half* __restrict__ sin_table,
    int num_heads,
    int seq_len,
    int head_dim
) {
    const int half_dim  = head_dim >> 1;
    const int token_idx = blockIdx.x;
    const int d         = threadIdx.x;
    if (d >= half_dim) return;

    const int t       = token_idx % seq_len;
    const int qk_base = token_idx * head_dim;
    const float c = __half2float(cos_table[t * half_dim + d]);
    const float s = __half2float(sin_table[t * half_dim + d]);

    const float q0 = __half2float(q[qk_base + d]);
    const float q1 = __half2float(q[qk_base + d + half_dim]);
    out_q[qk_base + d]            = __float2half(q0 * c - q1 * s);
    out_q[qk_base + d + half_dim] = __float2half(q0 * s + q1 * c);

    const float k0 = __half2float(k[qk_base + d]);
    const float k1 = __half2float(k[qk_base + d + half_dim]);
    out_k[qk_base + d]            = __float2half(k0 * c - k1 * s);
    out_k[qk_base + d + half_dim] = __float2half(k0 * s + k1 * c);
}

// ── Extern-C __global__ entry points ─────────────────────────────────────────

extern "C" {

// Single-tensor variants — used by the Rust CustomOp2 wrapper.
__global__ void rope_single_f32(
    float* out, const float* inp,
    const float* cos_table, const float* sin_table,
    int seq_len, int head_dim)
{ rope_single_f32_impl(out, inp, cos_table, sin_table, seq_len, head_dim); }

__global__ void rope_single_f16(
    __half* out, const __half* inp,
    const __half* cos_table, const __half* sin_table,
    int seq_len, int head_dim)
{ rope_single_f16_impl(out, inp, cos_table, sin_table, seq_len, head_dim); }

// Fused Q+K variants — for future use when both tensors can be dispatched together.
__global__ void rope_inplace_f32(
    float* out_q, float* out_k,
    const float* q, const float* k,
    const float* cos_table, const float* sin_table,
    int num_heads, int seq_len, int head_dim)
{ rope_fused_f32_impl(out_q, out_k, q, k, cos_table, sin_table, num_heads, seq_len, head_dim); }

__global__ void rope_inplace_f16(
    __half* out_q, __half* out_k,
    const __half* q, const __half* k,
    const __half* cos_table, const __half* sin_table,
    int num_heads, int seq_len, int head_dim)
{ rope_fused_f16_impl(out_q, out_k, q, k, cos_table, sin_table, num_heads, seq_len, head_dim); }

} // extern "C"
