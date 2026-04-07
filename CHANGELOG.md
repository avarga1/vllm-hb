# Changelog

All notable changes to vllm-hb are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- NCCL all_reduce (`--features nccl`) — replaces host-mediated CPU reduction
  with device-to-device `ncclAllReduce` (SUM) via cudarc's safe bindings
  - `TpWorld` initialises one `ncclComm_t` per rank via `Comm::from_devices`
  - Single-process multi-GPU pattern: `group_start` / `group_end` batches
    sequential per-rank calls into one collective — no thread-per-rank needed
  - DType dispatch: F32, F16, BF16 all supported via macro
  - CPU fallback preserved when `nccl` feature is off (zero-change default)
  - `TpLlamaBackend::forward` now calls `world.all_reduce(partials)` which
    routes to NCCL or CPU based on the active feature set
- Scheduler wired into inference worker (`worker/mod.rs`)
  - `Worker` now holds a `Scheduler`; incoming `WorkItem`s are converted to
    `SequenceGroup`s and admitted via `scheduler.add_sequence_group()`
  - Worker runs a step loop: drain inbox → `scheduler.schedule()` → execute →
    `scheduler.update()` — scheduler controls admission and FCFS ordering
  - Monotonic sequence IDs for block-table lookups
  - Pending issue #12: engine still processes one group at a time (single shared
    KV cache); batched `forward_batch()` will remove that constraint

---

## [0.4.0] — 2026-04-07

### Added
- Tensor-parallel Llama forward pass — `--tensor-parallel-size N` now genuinely
  splits model weights across N GPUs (`src/engine/arch/llama_tp.rs`)
- Custom TP-aware transformer layers: RMSNorm, RoPE, sharded MHA (GQA-compatible),
  SwiGLU FFN — written from scratch so all_reduce can be injected at sync points
- Per-rank weight sharding at load time: column-parallel Q/K/V/gate/up,
  row-parallel O/down; weights distributed from CPU → each GPU's device
- rayon par_iter for parallel GPU kernel dispatch across ranks
- Validation at load time: num_attention_heads, num_kv_heads, intermediate_size
  must all be divisible by tensor_parallel_size
- GitHub Projects roadmap (public) — issues #6–#12 track remaining work

### Changed
- `Engine::load` branches on `world_size`: tp=1 → existing `LlamaBackend`
  (candle_transformers, unchanged); tp>1 → new `TpLlamaBackend`
- `--tensor-parallel-size 1` behaviour is identical to before — zero regression

---

## [0.3.0] — 2026-04-07

### Added
- Multi-architecture support — Mixtral, Qwen2, Phi3 backends (architecture detection
  from `config.json`; forward pass stubs ready for wiring)
- Flash attention + SDPA attention backends (`attention/flash.rs`, `attention/sdpa.rs`)
- Scheduler groundwork — `BlockManager`, `SequenceGroup`, FCFS policy (`scheduler/`)
- Tensor parallel stubs — `comm.rs`, `shard.rs` (`parallel/`)
- Speculative decoding skeleton (`speculative.rs`)
- Concurrent benchmark mode (`bench/concurrent.rs`) alongside existing sequential bench
- Real V100 benchmark results in README
- Security audit CI job (`cargo audit`) — fixes RUSTSEC-2026-0049

### Changed
- Full module restructure — flat `src/` split into `engine/`, `server/`, `scheduler/`,
  `attention/`, `parallel/`, `bench/`, `tokenize/`, `types/`, `sampling/`, `worker/`
- Server split into `handlers.rs`, `sse.rs`, `metrics.rs`
- Types split into `openai.rs`, `pipeline.rs`

### Fixed
- CUDA feature gating — `candle-*` crates now optional behind `cuda` feature flag;
  `--no-default-features` builds clean on any machine without CUDA toolkit
- All `clippy -D warnings` errors resolved for CPU build

---

## [0.2.0] — 2026-04-07

### Added
- OpenAI-compatible `/v1/chat/completions` endpoint — streaming and non-streaming
- `/v1/models` and `/health` endpoints
- Server-sent events (SSE) token streaming
- Temperature + top-p nucleus sampling (`sampling.rs`)
- Auto chat template detection from `tokenizer_config.json` (Llama-3, ChatML, Mistral-v1)
- Auto EOS token loading from `config.json`
- Built-in throughput benchmark — `vllm-hb bench` reports TTFT, throughput, p50/p99
- Llama model support via `candle-transformers`
- V100 / sm_70 compatibility (FP16 fallback — V100 does not support BF16 natively)
- Channel-based inference worker — serialises GPU work, streams events per-request
- CORS middleware via `tower-http`
- Structured JSON logging via `tracing-subscriber`
- GitHub Actions CI — lint, CPU unit tests, security audit on every PR
- GitHub Actions release workflow — binary attached to version tags
- `[features]` in `Cargo.toml` — CUDA optional so CPU builds work without toolkit

### Architecture
- HTTP: `axum 0.8` + `tokio`
- GPU compute: `candle-core / candle-nn / candle-transformers` (no libtorch, no C++ FFI)
- Tokenizer: `tokenizers 0.21` (HuggingFace canonical Rust implementation)
- CLI: `clap 4` with `Serve` and `Bench` subcommands

---

## [0.1.0] — 2025-03-01

### Added
- Initial proof of concept — Llama forward pass via candle on CUDA
- Basic HTTP server returning generated text (non-streaming)

---

[Unreleased]: https://github.com/avarga1/vllm-hb/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/avarga1/vllm-hb/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/avarga1/vllm-hb/releases/tag/v0.1.0
