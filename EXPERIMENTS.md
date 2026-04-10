# vllm-hb Experiments Log

A running record of what we tried, what we measured, and what we learned.
Each experiment has a hypothesis, methodology, result, and takeaway.

---

## Setup

**Hardware**: Tesla V100-SXM2-32GB (HBM2, 900 GB/s bandwidth, 125 TFLOPS FP16)
**Model**: Qwen2.5-7B-Instruct — F16, 28 layers, hidden=3584, 8.2B params, 4 safetensor shards
**Comparison baseline**: Python vLLM 0.4.x on the same machine, adjacent GPU

**Theoretical ceiling** (batch=1 decode, memory-bandwidth-bound):
```
weights ≈ 14 GB in F16
V100 HBM bandwidth = 900 GB/s
max throughput = 14e9 / 900e9 ≈ 15.5 ms/token = ~64 tok/s
```
Neither us nor vLLM ever exceeds this. Everything is a fight for efficiency within it.

---

## Experiment 1 — Baseline: Rust vs Python vLLM

**Date**: 2026-04-09
**Branch**: `main` (v0.10.0)
**Hypothesis**: Pure Rust inference with candle should be at least competitive with Python vLLM, since Python's bytecode overhead doesn't matter for GPU-bound workloads.

### Methodology
Sequential single-client benchmarks, concurrency=1, no interference between runs.

```
vllm-hb bench --n 50 --max-tokens 128  # short decode
vllm-hb bench --n 20 --max-tokens 512  # long decode
```

### Results

| System | 128tok × 50req | 512tok × 20req | GPU util |
|--------|---------------|----------------|----------|
| vllm-hb (candle eager) | ~36 tok/s | ~35 tok/s | 7.1% |
| Python vLLM | ~43 tok/s | ~43 tok/s | 7.1% |
| **gap** | **-16%** | **-19%** | **same** |

### What We Learned

1. **SM utilization was identical (7.1%) on both systems.** The GPU is doing the same amount of math. The gap is entirely CPU-side — not GPU compute.

2. **Python's 200-400 cycle per-op overhead is irrelevant here.** At 43 tok/s, one decode step takes ~23 ms. Python dispatch cost is nanoseconds against that.

3. **The real CPU cost is kernel launch overhead.** Each `cuLaunchKernel` call costs ~5-10 µs. Our decode step launches ~150-200 kernels individually = ~1 ms of pure CPU tax per token.

4. **vLLM uses CUDA graphs for decode.** It pre-captures ~35 graphs at startup (one per batch size). Graph replay = 1 CPU dispatch for the entire forward pass. ~0.05 ms overhead vs our ~1 ms. That's the gap.

5. **CUDA graphs require static tensor shapes.** Our KV cache grows each decode step — shape is not static — so we can't capture graphs without pre-allocating KV cache to max length first (i.e., PagedAttention).

**Root cause of the gap**: CUDA graph replay vs 150+ individual kernel launches per decode step.

---

## Experiment 2 — Fused CUDA Kernels: RMSNorm + RoPE

**Date**: 2026-04-09 → 2026-04-10
**Branch**: `feat/fused-kernels` (v0.11.0)
**Hypothesis**: Replacing candle's multi-op RMSNorm sequence with a single warp-reduction kernel will reduce kernel launch count by ~224 per forward pass and measurably improve throughput.

### Background

`candle_nn::RmsNorm` with `remove_mean=false` skips its own fused path and falls through to:
```
x.sqr() → sum_keepdim() → div(hidden) → add(eps) → sqrt() → broadcast_div() → broadcast_mul()
```
That's **4-6 kernel launches per RMSNorm call**.

A 28-layer model runs 2 RMSNorm per layer (pre-attention, pre-FFN) + 1 final norm = **57 RMSNorm calls per forward pass**.

```
Before: 57 × 5 ops = ~285 kernel launches just for normalization
After:  57 × 1 kernel = 57 launches
Saved:  ~228 kernel launches per decode step
```

### Implementation

- `kernels/rms_norm.cu`: warp-level block reduction (256/512 block size × F32/F16)
- `kernels/rope.cu`: single-pass rotation kernel (one thread per head_dim/2 element)
- `src/kernels/rms_norm.rs`: `CustomOp2` impl via `apply_op2_no_bwd`
- `src/engine/arch/models/`: vendored Qwen2/Qwen3 with `with_tracing::RmsNorm` swapped out
- Mixtral + LlamaTp backends: same swap, 1-line change each

### Build Bugs Encountered

1. `nvcc -ptx` rejects multiple `--generate-code` arches simultaneously — use single `-arch=compute_70`
2. `__global__` kernels cannot call other `__global__` kernels — impl functions must be `__device__`
3. `CudaDevice::alloc()` returns `candle_core::Result`, not `cudarc::Result` — `.w()` call is wrong, use `?` directly
4. `RwLockReadGuard` doesn't impl `AsRef` — dereference with `&*guard` to get `&Storage`

### Results

| System | 128tok × 50req | 512tok × 20req |
|--------|---------------|----------------|
| vllm-hb (candle eager, Exp 1) | ~36 tok/s | ~35 tok/s |
| **vllm-hb (fused kernels)** | **39.2 tok/s** | **38.2 tok/s** |
| Python vLLM | 43.2 tok/s | 43.3 tok/s |
| **remaining gap** | **-9.3%** | **-11.6%** |

**Improvement: +3.2 tok/s (+9%) on 128-token, +3.2 tok/s (+9%) on 512-token.**

We are now at **61% of the V100 memory bandwidth ceiling**. vLLM is at **67%**.

### What We Learned

1. **The kernel wins exactly as predicted.** Reducing ~228 kernel launches per decode step translated directly to ~9% throughput gain. The math matched the result.

2. **candle's own RMSNorm doesn't use its fused path for the standard RMSNorm case** (`remove_mean=false`). This was a free win hiding in plain sight — we only found it by reading the source.

3. **Vendoring is the right call for hot-path model code.** Qwen2/Qwen3 use `candle_transformers::models::qwen2` which wraps `candle_nn::RmsNorm`. There's no hook to replace that without owning the model files. ~400 lines of vendored code, one line changed per file.

4. **The remaining ~10% gap is still CUDA graphs.** SM utilization is presumably still identical. We reduced CPU-side launch overhead but there's still ~100+ non-normalization kernels per decode step that each pay the launch tax.

5. **Under concurrent load we degrade worse than vLLM.** When two benchmark clients hit our server simultaneously, our throughput halved (39→19 tok/s). vLLM barely moved (43→43 tok/s). Their continuous batching merges concurrent requests into the same decode step — ours queues them. This is a separate problem from single-request throughput.

---

---

## Experiment 3 — Flash Attention 2

**Date**: 2026-04-10
**Branch**: `feat/fused-kernels`
**Hypothesis**: Building with `--features flash-attn` would replace our SDPA attention with a 2-3x memory-bandwidth-efficient fused attention kernel, yielding +2-4 tok/s on long sequences.

### Result: **Blocked — hardware**

```
Server GPU: Tesla V100-SXM2-32GB
Compute capability: sm_70 (Volta)
Flash Attention 2 requirement: sm_80+ (Ampere)
```

This experiment cannot run on this hardware. Need an A100, A40, RTX 3090, or H100 to test it.

### What We Learned

The gap between us and vLLM on long sequences (11.6% at 512 tokens vs 9.3% at 128 tokens) is likely partly explained by FA2 — vLLM uses it on sm_80+ hardware. On V100 specifically, vLLM also falls back to SDPA, so this experiment would be most meaningful on Ampere hardware where both can use their respective fused attention.

---

## Experiment 4 — RoPE Kernel: Already Fused

**Date**: 2026-04-10
**Hypothesis**: Our `rope_single_f32/f16` CUDA kernel would save kernel launches over candle's `rotary_emb::rope` implementation.

### Result: **No savings — candle already fuses RoPE**

Reading `candle-nn-0.10.2/src/rotary_emb.rs` line 133:
```rust
let func = dev.get_or_load_func(&kernel_name::<T>("rope_i"), &kernels::REDUCE)?;
let dst = unsafe { dev.alloc::<T>(el)? };
// ... single kernel launch
```

`candle_nn::rotary_emb::rope` is already a single fused CUDA kernel (`rope_i`). One launch per Q, one per K — same as our wrapper would do. **No launch count improvement possible here.**

### What We Learned

Not every candle operation is multi-op. The operators that were multi-op were the ones with no kernel in `kernels::REDUCE` (like RMSNorm). Before assuming a win, check whether candle already has a fused path by reading the CUDA branch of the relevant source.

Our `rope.cu` / `rope.rs` are correct implementations but redundant versus what candle provides. They can be removed or kept for future architectural use (e.g., a true Q+K fused single-dispatch variant if we ever want to combine them).

---

## Experiment 5 — Kernel Launch Profiling with nsys

**Date**: 2026-04-10
**Status**: In progress
**Hypothesis**: We estimate ~100 kernel launches remain per decode step after RMSNorm fusion. Profiling will tell us exactly what they are and where the next target is.

### Methodology

```bash
nsys profile --trace=cuda --output=/tmp/decode_profile \
  ./vllm-hb serve --model ... &
# send one warmup + one timed decode request
# nsys stats to count kernel launches by name
```

### Results

_To be filled._

---

## Open Questions

- **Kernel count after RMSNorm fusion**: We estimate ~100 launches/decode. Need nsys to confirm and identify the breakdown (attention matmuls, softmax, residual adds, etc.).

- **Pre-allocated KV cache → CUDA graphs**: If we allocate KV cache to `max_seq_len` at load time (instead of growing it), the decode step tensor shapes become static and we can capture a CUDA graph. This is the path to closing the remaining ~10% gap.

- **Concurrent throughput**: The 2× degradation under concurrent load is a real production problem. True continuous batching (merging in-flight requests into the same GPU dispatch) is a larger architectural change.

- **FA2 on sm_80+ hardware**: Valid experiment, just needs a different box.

---

## Theoretical Roadmap to Beat vLLM

```
Current:   39 tok/s  (61% of ceiling)
+ Graphs: ~44 tok/s  (69%)     ← needs pre-alloc KV cache, biggest tractable win
+ FA2:    ~46 tok/s  (72%)     ← needs sm_80+ hardware
+ Batch:  ~60+ tok/s           ← continuous batching, larger work
Ceiling:   64 tok/s  (100%)
```

vLLM today sits at 43 tok/s on this V100. Getting to 44+ with CUDA graphs would be the first time we beat them at single-request throughput on identical hardware.

Flash Attention 2 is the experiment we can't run yet — it needs the right GPU. CUDA graphs are the experiment we *can* run — and they represent the larger remaining win.
