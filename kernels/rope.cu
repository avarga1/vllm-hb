// Fused RoPE (Rotary Position Embedding) kernel.
//
// Applies pre-computed cos/sin tables to Q and K tensors in a single pass,
// replacing the 3-op candle sequence: narrow → rope_kernel (in-place) →
// contiguous copy. This fused version reads Q/K once and writes once.
//
// Layout expectations (same as candle-transformers Qwen2/Llama convention):
//   q, k:   [batch, num_heads, seq_len, head_dim]  (contiguous)
//   cos/sin: [seq_len, head_dim/2]                  (contiguous, pre-sliced to
//                                                     the correct position range)
//   out_q, out_k: same shape as q, k
//
// The kernel applies the standard RoPE rotation:
//   y0 = x0 * cos - x1 * sin
//   y1 = x0 * sin + x1 * cos
// where x0 = first half of head_dim, x1 = second half.
//
// Grid:  (batch * num_heads * seq_len,)
// Block: (head_dim / 2,)  — each thread handles one (x0, x1) pair.

#include <cuda_fp16.h>
#include <stdint.h>

// ── F32 ───────────────────────────────────────────────────────────────────────

__global__ void rope_f32(
    float*       __restrict__ out_q,
    float*       __restrict__ out_k,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ cos_table,  // [seq_len, head_dim/2]
    const float* __restrict__ sin_table,  // [seq_len, head_dim/2]
    int  num_heads,
    int  seq_len,
    int  head_dim           // must be even
) {
    const int half_dim = head_dim >> 1;

    // Global token index within (batch * num_heads * seq_len).
    const int token_idx = blockIdx.x;      // one block per (b, h, t) tuple
    const int d         = threadIdx.x;     // 0 .. half_dim-1

    if (d >= half_dim) return;

    // Sequence position within the current block.
    const int t = token_idx % seq_len;

    // Offset into the flat q/k tensors for this (b, h, t).
    const int qk_base = token_idx * head_dim;

    const float x0 = q[qk_base + d];
    const float x1 = q[qk_base + d + half_dim];
    const float c  = cos_table[t * half_dim + d];
    const float s  = sin_table[t * half_dim + d];
    out_q[qk_base + d]           = x0 * c - x1 * s;
    out_q[qk_base + d + half_dim] = x0 * s + x1 * c;

    const float k0 = k[qk_base + d];
    const float k1 = k[qk_base + d + half_dim];
    out_k[qk_base + d]           = k0 * c - k1 * s;
    out_k[qk_base + d + half_dim] = k0 * s + k1 * c;
}

// ── F16 ───────────────────────────────────────────────────────────────────────

__global__ void rope_f16(
    __half*       __restrict__ out_q,
    __half*       __restrict__ out_k,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ cos_table,
    const __half* __restrict__ sin_table,
    int  num_heads,
    int  seq_len,
    int  head_dim
) {
    const int half_dim  = head_dim >> 1;
    const int token_idx = blockIdx.x;
    const int d         = threadIdx.x;

    if (d >= half_dim) return;

    const int t       = token_idx % seq_len;
    const int qk_base = token_idx * head_dim;

    const float x0 = __half2float(q[qk_base + d]);
    const float x1 = __half2float(q[qk_base + d + half_dim]);
    const float c  = __half2float(cos_table[t * half_dim + d]);
    const float s  = __half2float(sin_table[t * half_dim + d]);
    out_q[qk_base + d]            = __float2half(x0 * c - x1 * s);
    out_q[qk_base + d + half_dim] = __float2half(x0 * s + x1 * c);

    const float k0 = __half2float(k[qk_base + d]);
    const float k1 = __half2float(k[qk_base + d + half_dim]);
    out_k[qk_base + d]            = __float2half(k0 * c - k1 * s);
    out_k[qk_base + d + half_dim] = __float2half(k0 * s + k1 * c);
}

// ── Single-tensor variants ────────────────────────────────────────────────────
//
// Applies RoPE to one tensor (q OR k) only.  Used by the Rust CustomOp2 path
// where candle's op framework produces one output tensor per call.

template <int UNUSED>
__global__ void rope_single_f32_impl(
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

    const int t       = token_idx % seq_len;
    const int base    = token_idx * head_dim;
    const float x0 = inp[base + d];
    const float x1 = inp[base + d + half_dim];
    const float c  = cos_table[t * half_dim + d];
    const float s  = sin_table[t * half_dim + d];
    out[base + d]           = x0 * c - x1 * s;
    out[base + d + half_dim] = x0 * s + x1 * c;
}

template <int UNUSED>
__global__ void rope_single_f16_impl(
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

    const int t     = token_idx % seq_len;
    const int base  = token_idx * head_dim;
    const float x0 = __half2float(inp[base + d]);
    const float x1 = __half2float(inp[base + d + half_dim]);
    const float c  = __half2float(cos_table[t * half_dim + d]);
    const float s  = __half2float(sin_table[t * half_dim + d]);
    out[base + d]            = __float2half(x0 * c - x1 * s);
    out[base + d + half_dim] = __float2half(x0 * s + x1 * c);
}

// ── Extern-C exports ─────────────────────────────────────────────────────────

extern "C" {
    // Fused Q+K variant (legacy / future use).
    __global__ void rope_inplace_f32(
        float* out_q, float* out_k,
        const float* q, const float* k,
        const float* cos_table, const float* sin_table,
        int num_heads, int seq_len, int head_dim)
    { rope_f32(out_q, out_k, q, k, cos_table, sin_table, num_heads, seq_len, head_dim); }

    __global__ void rope_inplace_f16(
        __half* out_q, __half* out_k,
        const __half* q, const __half* k,
        const __half* cos_table, const __half* sin_table,
        int num_heads, int seq_len, int head_dim)
    { rope_f16(out_q, out_k, q, k, cos_table, sin_table, num_heads, seq_len, head_dim); }

    // Single-tensor variants used by the Rust CustomOp2 wrapper.
    __global__ void rope_single_f32(
        float* out, const float* inp,
        const float* cos_table, const float* sin_table,
        int seq_len, int head_dim)
    { rope_single_f32_impl<0>(out, inp, cos_table, sin_table, seq_len, head_dim); }

    __global__ void rope_single_f16(
        __half* out, const __half* inp,
        const __half* cos_table, const __half* sin_table,
        int seq_len, int head_dim)
    { rope_single_f16_impl<0>(out, inp, cos_table, sin_table, seq_len, head_dim); }
}
