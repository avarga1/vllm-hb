# Changelog

All notable changes to vllm-hb are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [0.5.0] — 2026-04-07

### Added
- **Paged KV-cache scheduler** — full continuous-batching infrastructure
  (`src/scheduler/`)
  - `sequence.rs` — `Sequence` / `SequenceGroup` state machine
    (Waiting → Running → Swapped → Finished)
  - `block_manager.rs` — paged GPU/CPU block allocator; CoW `append_slot`,
    `swap_out` / `swap_in`
  - `policy.rs` — `Policy` trait + `FcfsPolicy` (oldest-first, same default
    as Python vLLM)
  - `mod.rs` — `Scheduler` with three-queue design (waiting / running /
    swapped), `schedule()` / `update()` loop with greedy admission control
  - 23 unit tests covering admission, FCFS ordering, preemption, block
    accounting, swap round-trips
- **Scheduler wired into inference worker** (`src/worker/mod.rs`)
  - `Worker` now holds a `Scheduler`; requests are admitted as
    `SequenceGroup`s and served in FCFS order with memory-aware block limits
  - Step loop: drain inbox → `scheduler.schedule()` → execute →
    `scheduler.update()` — scheduler drives the hot path
- **NCCL all_reduce** (`--features nccl`) — device-to-device TP collective
  (`src/parallel/`)
  - `TpWorld` initialises one `ncclComm_t` per rank via
    `Comm::from_devices`; comms stored alongside device list
  - `group_start` / `group_end` batches sequential single-process calls
    into one NCCL collective — no thread-per-rank required
  - DType dispatch macro: F32, F16, BF16 all pass through `ncclAllReduce`
  - `TpLlamaBackend::forward` routes through `world.all_reduce(partials)`;
    NCCL or CPU selected at compile time via feature flag
  - CPU fallback unchanged; default (`cuda` only) build unaffected

### Changed
- `Worker::new` accepts no block-count arguments; uses compile-time
  constants `NUM_GPU_BLOCKS` / `NUM_CPU_BLOCKS` (tunable in `worker/mod.rs`)
- `all_reduce` call sites in `llama_tp.rs` updated from
  `comm::all_reduce(&partials, device)` → `world.all_reduce(partials)`

---

## [0.4.0] — 2026-04-07

### Added
- **Tensor-parallel Llama forward pass** — `--tensor-parallel-size N`
  genuinely splits model weights across N GPUs (`src/engine/arch/llama_tp.rs`)
- Custom TP-aware transformer layers written from scratch so `all_reduce`
  can be injected at sync points: RMSNorm (F32 accumulation), RoPE
  (seq-pos-offset aware), causal mask, sharded MHA (GQA-compatible),
  SwiGLU FFN
- Per-rank weight sharding at load time: column-parallel Q/K/V/gate/up,
  row-parallel O/down; weights streamed CPU → each rank's CUDA device
- `rayon` par_iter for parallel GPU kernel dispatch across ranks
- Validation at load time: `num_attention_heads`, `num_kv_heads`,
  `intermediate_size` must all be divisible by `tensor_parallel_size`
- Public GitHub Projects roadmap — issues #6–#12 track remaining work

### Changed
- `Engine::load` branches on `world_size`: tp=1 → `LlamaBackend`
  (candle_transformers, unchanged); tp>1 → `TpLlamaBackend`
- `--tensor-parallel-size 1` is identical to before — zero regression

---

## [0.3.0] — 2026-04-07

### Added
- Multi-architecture dispatch — Mixtral, Qwen2, Phi3 backend stubs
  (architecture detected from `config.json`; forward-pass implementations
  pending issues #8 and #9)
- Flash Attention 2 + SDPA attention backend stubs (`attention/`)
- Tensor-parallel stubs — `comm.rs`, `shard.rs` (`parallel/`)
- Speculative decoding skeleton (`speculative.rs`)
- Concurrent benchmark mode (`bench/concurrent.rs`) alongside sequential bench
- Real V100-SXM2 benchmark results in README
- Security audit CI job (`cargo audit`) — fixes RUSTSEC-2026-0049

### Changed
- Full module restructure — flat `src/` split into `engine/`, `server/`,
  `scheduler/`, `attention/`, `parallel/`, `bench/`, `tokenize/`, `types/`,
  `sampling/`, `worker/`
- Server split into `handlers.rs`, `sse.rs`, `metrics.rs`
- Types split into `openai.rs`, `pipeline.rs`

### Fixed
- CUDA feature gating — `candle-*` crates now optional behind `cuda` feature
  flag; `--no-default-features` builds clean without the CUDA toolkit
- All `clippy -D warnings` errors resolved for CPU build

---

## [0.2.0] — 2026-04-07

### Added
- OpenAI-compatible `/v1/chat/completions` — streaming and non-streaming
- `/v1/models` and `/health` endpoints
- Server-sent events (SSE) token streaming
- Temperature + top-p nucleus sampling (`sampling.rs`)
- Auto chat-template detection from `tokenizer_config.json`
  (Llama-3, ChatML, Mistral-v1)
- Auto EOS token loading from `config.json`
- Built-in throughput benchmark — `vllm-hb bench` reports TTFT, throughput,
  p50/p99 latency
- Llama model support via `candle-transformers`
- V100 / sm_70 compatibility (FP16 fallback — V100 has no native BF16)
- CORS middleware via `tower-http`
- Structured JSON logging via `tracing-subscriber`
- GitHub Actions CI — lint, CPU unit tests, security audit on every PR
- GitHub Actions release workflow — binary attached to version tags
- `[features]` in `Cargo.toml` — CUDA optional so CPU builds work without
  the toolkit

### Architecture
- HTTP: `axum 0.8` + `tokio`
- GPU compute: `candle-core / candle-nn / candle-transformers`
  (no libtorch, no C++ FFI)
- Tokenizer: `tokenizers 0.21` (HuggingFace canonical Rust implementation)
- CLI: `clap 4` with `serve` and `bench` subcommands

---

## [0.1.0] — 2025-03-01

### Added
- Initial proof of concept — Llama forward pass via candle on CUDA
- Basic HTTP server returning generated text (non-streaming)

---

[Unreleased]: https://github.com/avarga1/vllm-hb/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/avarga1/vllm-hb/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/avarga1/vllm-hb/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/avarga1/vllm-hb/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/avarga1/vllm-hb/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/avarga1/vllm-hb/releases/tag/v0.1.0
