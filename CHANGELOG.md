# Changelog

All notable changes to vllm-hb are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [0.2.0] — 2025-04-07

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
