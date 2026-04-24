// kv_assign.cu — in-place KV slot write kernel
//
// Writes one token's K or V vector into a pre-allocated KV buffer at a specific
// sequence position without allocating a new tensor or growing the buffer shape.
//
// Buffer layout (matching candle's qwen2 attention tensors):
//   buf : [nkv, max_seq, head_dim]   (batch=1 implicit; pointer to element 0)
//   src : [nkv, head_dim]            (batch=1, seq=1 implicit)
//
// Grid  : ceil(nkv * head_dim / 128) blocks
// Block : 128 threads
// Each thread writes one (head, elem) pair.

#include <cuda_fp16.h>

template <typename T>
__device__ __forceinline__ void kv_slot_assign_impl(
    T* __restrict__ buf, const T* __restrict__ src,
    int nkv, int max_seq, int head_dim, int offset
) {
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nkv * head_dim;
    if (tid >= total) return;
    int head = tid / head_dim;
    int elem = tid % head_dim;
    buf[head * max_seq * head_dim + offset * head_dim + elem] =
        src[head * head_dim + elem];
}

extern "C" {

__global__ void kv_slot_assign_f16(
    __half* __restrict__ buf, const __half* __restrict__ src,
    int nkv, int max_seq, int head_dim, int offset
) {
    kv_slot_assign_impl<__half>(buf, src, nkv, max_seq, head_dim, offset);
}

__global__ void kv_slot_assign_f32(
    float* __restrict__ buf, const float* __restrict__ src,
    int nkv, int max_seq, int head_dim, int offset
) {
    kv_slot_assign_impl<float>(buf, src, nkv, max_seq, head_dim, offset);
}

} // extern "C"
