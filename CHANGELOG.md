# Changelog

All notable changes to vllm-hb are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [0.11.0] — 2026-04-09

### Added
- **GGUF quantization support** (`src/engine/arch/gguf_llama.rs`, issue #28)
  - `GgufLlamaBackend` — wraps `candle_transformers::models::quantized_llama::ModelWeights`
  - Accepts any GGUF quantization format: Q2_K, Q3_K_*, Q4_0, Q4_K_*, Q5_K_*, Q6_K, Q8_0, F16, F32
  - Template + clone memory model: one `Arc<ModelWeights>` loaded at startup; each sequence receives a cheap clone (O(n_layers) Arc ref-count bumps, zero weight data copied); KV state grows independently per clone
  - `GgufLlamaBackend::load(path, device)` — reads GGUF header, extracts `llama.vocab_size`, `llama.embedding_length`, `llama.block_count` metadata, builds `ModelWeights` via `candle_core::quantized::gguf_file`
  - `GgufLlamaBackend::create_seq_model()` — O(n_layers) clone of the template for one sequence
  - `GgufLlamaBackend::forward(model, token_ids, seq_pos, device)` — single forward step updating the sequence's private KV cache
  - `PerSeqCache::GgufLlama(ModelWeights)` variant added; `try_clone_external()` handles it (ModelWeights is Clone)
  - `Backend::GgufLlama` variant and all dispatch arms added in `src/engine/arch/mod.rs`
  - `Engine::load_gguf()` — detects `.gguf` extension, bypasses safetensors/config.json path, returns `Engine` with populated metadata
  - Auto-detection in `Engine::load()`: GGUF files are identified by `.gguf` extension before any other loading logic

### Changed
- `Engine::load()` now detects `.gguf` files by extension and dispatches to `load_gguf()` automatically — no API change required
- `embed_tokens` is `None` for GGUF models (quantized embedding matrix is internal to `ModelWeights`; static mean-pool embeddings are not exposed)
- Version bumped 0.10.0 → 0.11.0

---

## [0.10.0] — 2026-04-09

### Added
- **Automatic prefix caching** (`src/scheduler/prefix_cache.rs`, issue #27)
  - `PrefixCache` — LRU-approximated block-hash cache mapping FNV-1a token-block hashes to physical block IDs; capacity-bounded (default 128 entries) with FIFO eviction
  - `hash_block(tokens: &[u32]) -> u64` — FNV-1a 64-bit hash, deterministic and order-sensitive
  - `BlockManager::allocate()` — checks complete leading prompt blocks against the prefix cache before allocating fresh blocks; cache hits re-use the existing physical block (ref-count incremented)
  - `AllocResult { hit_blocks: usize }` — returned by `allocate()` so the worker knows how many prefix blocks were matched
  - `SchedulerOutputs::prefix_hit_blocks` — parallel vec to `to_prefill`, carries per-group hit counts from scheduler to worker
  - Worker `step_prefill()` receives hit count and logs prefix cache hits; structured for KV restore once `PerSeqCache::try_restore_prefix` is wired per architecture
  - `BlockManager::num_prefix_cached_blocks()` for observability
  - 9 unit tests: hash determinism, hash collision resistance, order sensitivity, miss on empty, insert+lookup, eviction, remove by block ID, len tracking, same-hash update
- **`POST /v1/embeddings`** (`src/server/handlers.rs`, issue #29)
  - Full OpenAI Embeddings API compatibility — single string or batch array input, echoed model name, `object: "list"` response
  - `EmbeddingRequest` / `EmbeddingResponse` / `EmbeddingObject` / `EmbeddingUsage` / `EmbeddingInput` types added to `src/types/openai.rs`; `EmbeddingInput` is untagged — accepts `"string"` or `["a","b"]` transparently
  - `encoding_format` field — only `"float"` supported; `"base64"` returns HTTP 400
  - Returns HTTP 501 when the model's embedding matrix was not found at load time (clear error message directs users to a dedicated embedding model)
  - 8 integration tests: 200 status, response shape, vector dimension, L2 norm ≈ 1, batch input, usage accounting, bad encoding format (400), no embed table (501)
- **`Engine::embed_tokens_clone()`** — cheap Arc ref-bump clone of the embedding matrix
- **`Engine::hidden_size()`** — expose hidden dimension for AppState
- **`Engine` embeds on load** (`src/engine/loader.rs`)
  - Probes 4 known safetensors key names (`model.embed_tokens.weight`, `transformer.wte.weight`, `model.wte.weight`, `embeddings.word_embeddings.weight`) across all architecture families
  - Loaded as F32 for CPU-side mean-pool; stored as `Option<Tensor>` in `Engine`
- **`AppState`** gains `embed_tokens: Option<Tensor>` and `hidden_size: usize`; populated from engine before the worker takes ownership
- **`POST /v1/files`** — upload a JSONL file for batch processing (issue #26)
  - Accepts raw JSONL body; stores in-memory; returns an OpenAI-compatible `FileObject`
  - File IDs are stable UUID-based identifiers reusable across batch/file endpoints
- **`POST /v1/batches`** — submit an async batch job (issue #26)
  - References an uploaded file by `input_file_id`; spawns a background Tokio task
  - Status machine: `validating` → `in_progress` → `completed` | `failed` | `cancelled`
  - `request_counts.{total, completed, failed}` updated in real time
- **`GET /v1/batches/{id}`** — poll batch status; returns live `BatchObject`
- **`GET /v1/files/{id}/content`** — retrieve output JSONL (`application/x-ndjson`)
  - Each output line: `{id, custom_id, response: {status_code, body}, error}`
- **`POST /v1/batches/{id}/cancel`** — cancel `validating` or `in_progress` batches
- **`BatchStore`** (`src/batch/mod.rs`) — in-memory shared store for files + batches;
  `Arc<Mutex<_>>` in `AppState`; zero external dependencies
- 8 integration tests covering upload, create, poll, cancel, and error paths

### Algorithm (mean-pool + L2-norm)
Input token IDs → index `embed_tokens [vocab, hidden]` → `[seq_len, hidden]` → mean across seq → `[hidden]` → L2-normalise → `Vec<f32>`. Fast (no transformer forward pass), deterministic, O(seq_len · hidden) — suitable for shared system-prompt caching and similarity search.

### Changed
- `BlockManager::allocate()` signature changed from `-> Result<()>` to `-> Result<AllocResult>`
- `SchedulerOutputs` gains `prefix_hit_blocks: Vec<usize>` field
- `BlockManager::free()` registers completed prompt blocks in prefix cache on every free (content-hash preserved for hit detection on re-admission)
- `AppState` gains `embed_tokens: Option<Tensor>`, `hidden_size: usize`, and `batch_store: Arc<Mutex<BatchStore>>`
- Version bumped 0.9.0 → 0.10.0

### Notes
- Full KV-tensor reuse (skipping the GPU forward pass for matched prefix positions) requires architecture-specific `PerSeqCache::try_restore_prefix` implementations. The infrastructure is in place; tensor restore is tracked separately.

---

## [0.9.0] — 2026-04-08

### Added
- **Presence / frequency penalties** (`src/sampling/penalty.rs`, issue #24)
  - `apply_penalties(logits, token_counts, presence_penalty, frequency_penalty)` —
    applied before temperature scaling so penalties interact naturally with nucleus
    filtering
  - `count_tokens(output_ids, vocab_size)` — O(n) histogram from output token IDs
  - Formula matches OpenAI spec:
    `logit[t] -= presence * (count > 0) + frequency * count / total_tokens`
  - Fast path: both penalties are skipped when both are within `1e-6` of zero
  - Applied only after the first output token (no-op at prefill)
  - 7 unit tests: zero no-op, presence reduces seen tokens, frequency scales with
    count, both combined, unseen tokens untouched, count_tokens correctness, out-of-range
    token ignored
- **`POST /v1/completions`** — legacy text-completion endpoint (issue #25)
  - Accepts raw `prompt: String` instead of `messages`; tokenized and forwarded
    directly to the inference worker without chat-template rendering
  - Supports all sampling parameters: `max_tokens`, `temperature`, `top_p`, `stop`,
    `seed`, `presence_penalty`, `frequency_penalty`
  - Non-streaming: returns `CompletionResponse` (`object: "text_completion"`, id
    prefix `"cmpl-"`, single `CompletionChoice` with `text` and `finish_reason`)
  - Streaming: reuses SSE machinery from chat completions
  - 7 integration tests covering 200 status, response shape, text content,
    finish reason, usage, streaming status, and SSE content-type

### Changed
- `SamplingParams` gains `presence_penalty: f32` and `frequency_penalty: f32`
  (both default `0.0`)
- `handlers::chat_completions` passes penalty fields through to `SamplingParams`
- Version bumped 0.8.0 → 0.9.0

---

## [0.8.0] — 2026-04-09

### Added
- **Stop sequences** (`src/sampling/stop.rs`, issue #21)
  - `StopChecker` — rolling suffix buffer, O(|stop_strings|) per token, char-boundary safe
  - Up to 4 stop strings per request; matched suffix stripped from final output
  - `finish_reason` is `"stop"` on stop-string match
  - 8 unit tests: empty list, blank filter, no match, exact match, split-token match,
    strip, buffer trimming, multi-stop
- **Seed + logprobs** (`src/sampling/logprobs.rs`, issue #23)
  - `seed: u64` — reproducible sampling via `SmallRng` seeded per-token (seed mixed
    with step counter so successive tokens are independently distributed)
  - `logprobs: bool` + `top_logprobs: u8` — `LogprobCollector` records chosen token
    log-prob + top-N alternatives sorted descending; bytes field populated; attached
    to response `Choice`
  - `sampling::logits_to_probs()` — shared prob computation for sampling + collection
  - 6 unit tests: chosen token, top-N excludes chosen, sorted descending, clamp,
    multiple records, bytes field
- **Tool calls — JSON + XML** (`src/tools/`, issue #22)
  - `src/tools/format.rs` — `inject_tools()` renders tool definitions as JSON schema
    block (Llama/Qwen/Mistral) or XML `<tools>` block (Hermes/Claude-style);
    `detect_format()` auto-selects from chat template string; 8 unit tests
  - `src/tools/parser.rs` — `ToolCallParser` detects + parses both formats from raw
    model output; JSON-block (bare object or ` ```json ` fence, balanced brace matching);
    XML (`<function_calls><invoke name="..."><parameter ...>`) with multi-call support;
    parameters auto-typed (numeric stays numeric); 10 unit tests
  - `FinishReason::ToolCalls` — finish reason is `"tool_calls"` when a call is detected
  - Tool definitions injected into system prompt before generation; markup stripped
    from visible assistant text in response
  - `tokenize::load_chat_template()` for format auto-detection

### Changed
- `SamplingParams` gains `stop`, `seed`, `logprobs`, `top_logprobs`, `has_tools`
- `GenerationEvent::Finished` gains `logprobs` and `tool_calls` fields
- `rand` dependency gains `small_rng` feature for `SmallRng`
- Version bumped 0.7.0 → 0.8.0

---

## [0.7.0] — 2026-04-08

### Added
- **Speculative decoding** (`src/speculative.rs`, issue #10)
  - `SpeculativeDecoder` holds a draft `Engine` and per-sequence draft KV
    caches; lives inside `Worker` behind an `Option`
  - **Draft phase**: K autoregressive forward passes on the small draft model
    with full probability distributions captured for rejection sampling
  - **Verify phase**: K+1 sequential target passes — stops at the first
    rejection so the target's KV cache is always correct, no post-step fixup
  - **Rejection sampling** (Leviathan et al. 2023): accept token i with
    probability `min(1, q_i / p_i)`; on rejection sample from `(q − p)⁺`
    correction distribution; sample a bonus token when all K are accepted
  - **Cache reconciliation**: Mixtral external caches use a pre-draft snapshot
    (cheap Arc clone) + O(j) replay; Llama/Qwen fall back to rebuilding from
    the last `DRAFT_FALLBACK_WINDOW = 128` context tokens
  - CLI flags: `--draft-model PATH`, `--speculative-steps N` (default 5)
  - Worker logs accepted-per-step at `DEBUG` level for acceptance-rate tuning
  - 7 unit tests: probability distribution, 2-D logit handling, correction
    sampling, degenerate fallback, `try_clone_external` variants
- **Library crate** (`src/lib.rs`) — all modules now exported as `vllm_hb::*`;
  enables integration tests and downstream embedding without the binary
- **HTTP integration test suite** (`tests/http_api.rs`) — 18 tests covering
  every public endpoint via `tower::ServiceExt::oneshot` with a mock worker
  (no GPU, no model weights required):
  - Health endpoint: 200 + `{"status":"ok"}`
  - Models endpoint: 200, `"object":"list"`, correct model id
  - Non-streaming chat: status, response shape, joined content, finish reason,
    role, usage arithmetic
  - Streaming chat: 200, `text/event-stream` content-type, `[DONE]` sentinel,
    role chunk, token content chunks
  - Error handling: 400 on malformed JSON, 404 on unknown route
- **`WorkerHandle::for_test()`** — injects a caller-owned sender so integration
  tests can drive a mock worker without loading weights
- **`server::router()`** — extracted from `serve()` so tests can call
  `.oneshot()` without binding a TCP port
- **OpenAI wire type doc comments** — all 11 public structs in
  `src/types/openai.rs` fully documented with field-level doc comments;
  8 unit tests added
- **Chat-template unit tests** (`src/tokenize/template.rs`) — 10 tests using
  `tempfile` fixtures covering detection and rendering for ChatML, Llama-3,
  and Mistral dialects plus edge cases
- **Test fixtures** (`tests/fixtures/`) — minimal `tokenizer.json` (WordLevel,
  12-token vocab) and three `tokenizer_config.json` dialect fixtures so the
  test suite runs fully offline
- **CI coverage job** — `cargo-llvm-cov` generates an lcov report; uploaded to
  Codecov on every push/PR (`fail_ci_if_error: false`)

### Changed
- `Worker::new` gains an `Option<SpeculativeDecoder>` parameter; standard
  decoding path unchanged when `None`
- `step_decode` splits into `step_decode_standard` (existing) +
  `step_decode_speculative` (new speculative path)
- `PerSeqCache` gains `try_clone_external()` — cheap snapshot for Mixtral
  external caches; returns `None` for Llama / LlamaTp (no-clone fallback)
- `Cargo.toml` gains a `[lib]` section (`name = "vllm_hb"`, `path = "src/lib.rs"`)
  and `[dev-dependencies]` for `axum`, `tower`, `tokio`, `serde_json`,
  `tempfile`, `bytes`
- `src/main.rs` refactored to import via `vllm_hb::*` rather than `crate::*`
- Bench doc comment code block marked `text` to prevent doctest compilation

---

## [0.6.0] — 2026-04-08

### Added
- **Qwen2 + Qwen3 forward pass** (`src/engine/arch/qwen2.rs`, `qwen3.rs`)
  - `Qwen2Backend` — wires `candle_transformers::models::qwen2`; supports
    Qwen2.5-7B-Instruct, Qwen2.5-32B-Instruct
  - `Qwen3Backend` — wires `candle_transformers::models::qwen3`; per-head
    QK RMSNorm handled internally by candle
  - Both detected from `model_type` in `config.json` (`"qwen2"` / `"qwen3"`)
- **Flash Attention 2** (`--features flash-attn`, sm_80+ only)
  - `LlamaBackend`: one-liner via `candle-transformers` built-in
  - `TpLlamaBackend` + `MixtralBackend`: `candle_flash_attn::flash_attn`
    behind `#[cfg(feature = "flash-attn")]`; SDPA path unchanged on all
    other hardware (V100, CPU)
  - Startup log confirms which path is active
- **True continuous batching** (`src/worker/mod.rs`, `src/engine/kv_cache.rs`)
  - Each `step()` runs one decode token per active sequence — no sequence
    blocks another mid-generation
  - `PerSeqCache` enum owns per-sequence KV state; worker holds
    `HashMap<seq_id, PerSeqCache>` alongside the scheduler
  - `Engine::create_kv_cache()` + `forward_with_cache()` — callers own the
    cache, no global `reset_cache()` between requests
- **Mixtral 8×7B sparse MoE** (`src/engine/arch/mixtral.rs`)
  - Written from scratch: `SparseMoeBlock` (top-K routing, renormalized
    weights, per-expert SwiGLU, scatter-add), sliding window attention,
    GQA with `repeat_kv`, external resettable KV cache
  - 10 unit tests

### Changed
- `candle-core` / `candle-nn` / `candle-transformers` bumped **0.9 → 0.10**
  (required for Qwen3 support)
- `candle-flash-attn` added as optional dep (`version = "0.10"`)
- Worker doc comment updated to reflect true continuous-batching architecture

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

[Unreleased]: https://github.com/avarga1/vllm-hb/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/avarga1/vllm-hb/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/avarga1/vllm-hb/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/avarga1/vllm-hb/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/avarga1/vllm-hb/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/avarga1/vllm-hb/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/avarga1/vllm-hb/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/avarga1/vllm-hb/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/avarga1/vllm-hb/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/avarga1/vllm-hb/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/avarga1/vllm-hb/releases/tag/v0.1.0
