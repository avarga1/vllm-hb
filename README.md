# vllm-hb

**vLLM-compatible inference runtime in pure Rust.**  
No Python. No libtorch. No C++ interop. CUDA via [cudarc](https://github.com/coreylowman/cudarc).

```
vllm-hb serve --model /path/to/Llama-3.1-8B-Instruct --port 8000
```

Drop-in replacement for the vLLM OpenAI-compatible server. Any client that
talks to vLLM talks to vllm-hb.

---

## Why

Python vLLM's scheduler runs under the GIL. At high concurrency the
scheduling loop becomes the bottleneck — GC pauses, dynamic dispatch, and
Python object overhead add up.

vllm-hb rewrites the hot path in Rust:

| Component | vLLM | vllm-hb |
|---|---|---|
| Scheduler | Python (GIL) | Rust (`parking_lot::Mutex`, zero-alloc hot path) |
| Tokenizer | Python wrapper | Rust (`tokenizers` crate — the canonical HuggingFace impl) |
| Model forward | Python → libtorch C++ | Rust → CUDA directly via `candle` |
| HTTP server | Python (uvicorn) | Rust (`axum` + `tokio`) |
| Runtime | CPython + pip | Single static binary |

---

## Features

### API
- **`POST /v1/chat/completions`** — streaming (SSE) + non-streaming, OpenAI wire format
- **`POST /v1/completions`** — legacy text completions
- **`POST /v1/embeddings`** — mean-pool + L2-normalised token embeddings
- **`POST /v1/files`** + **`POST /v1/batches`** — async offline batch inference (OpenAI Batch API)
- **`GET /v1/models`**, **`GET /health`**, **`GET /metrics`**

### Inference
- **True continuous batching** — per-sequence paged KV cache; new requests join mid-generation
- **Speculative decoding** — draft model + rejection sampling; pass `--draft-model` for 2–3× speedup
- **Flash Attention 2** — enabled with `--features flash-attn` (sm_80+ / Ampere)
- **Tensor parallelism** — `--tensor-parallel-size N` shards attention heads across N GPUs via NCCL all_reduce
- **Automatic prefix caching** — block-hash deduplication of shared KV prefix blocks (system prompts, few-shot)

### Sampling
- Temperature, top-p nucleus filtering, seed (reproducible outputs)
- Presence & frequency penalties (OpenAI spec)
- Stop sequences, logprobs + top-logprobs
- Tool calls (JSON and XML format auto-detection)

---

## Supported models

Any model in HuggingFace safetensors format with a supported architecture:

| Family | Status |
|---|---|
| Meta Llama 3 / 3.1 / 3.2 | ✅ |
| Mistral 7B / 7B-Instruct | ✅ |
| Mixtral 8×7B / 8×22B (sparse MoE) | ✅ |
| Qwen 2.5 | ✅ |
| Qwen 3 | ✅ |
| Microsoft Phi-3 | ✅ |

> Additional architectures are gated only by candle-transformers model
> implementations — contributions welcome.

---

## Getting started

### Requirements

- Rust 1.85+ (`rustup install stable`)
- CUDA 12.x toolkit (`nvcc` in PATH, or `CUDA_ROOT` set)
- NVIDIA GPU (tested on V100, A100, RTX 4090)

### Build

```bash
git clone https://github.com/avarga1/vllm-hb
cd vllm-hb
cargo build --release          # GPU (CUDA, default)
cargo build --release --no-default-features   # CPU-only (testing / CI)
```

Optional features:

```bash
# Flash Attention 2 — requires sm_80+ (Ampere/Hopper), ~2× attention speedup
cargo build --release --features flash-attn

# NCCL all_reduce for tensor parallelism — requires libnccl.so
cargo build --release --features nccl
```

If `nvcc` is not in your default PATH:

```bash
export CUDA_ROOT=/path/to/cuda
export PATH=$CUDA_ROOT/bin:$PATH
cargo build --release
```

### Run

```bash
# Single GPU
./target/release/vllm-hb serve \
  --model /models/Meta-Llama-3.1-8B-Instruct \
  --port 8000

# Multi-GPU tensor parallelism (e.g. 4 GPUs)
./target/release/vllm-hb serve \
  --model /models/Meta-Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --port 8000

# Speculative decoding with a draft model
./target/release/vllm-hb serve \
  --model /models/Meta-Llama-3.1-8B-Instruct \
  --draft-model /models/Llama-3.2-1B-Instruct \
  --speculative-steps 5 \
  --port 8000
```

### Test it

```bash
# Chat completions
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }'

# Embeddings
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "llama", "input": "Hello world"}'

# Batch inference
FILE_ID=$(curl -s http://localhost:8000/v1/files \
  -H "Content-Type: text/plain" \
  --data-raw '{"custom_id":"r1","method":"POST","url":"/v1/chat/completions","body":{"model":"llama","messages":[{"role":"user","content":"Hi"}]}}' \
  | jq -r .id)

curl http://localhost:8000/v1/batches \
  -H "Content-Type: application/json" \
  -d "{\"input_file_id\":\"$FILE_ID\",\"endpoint\":\"/v1/chat/completions\"}"
```

### Benchmark

```bash
# Start the server first, then in another terminal:
./target/release/vllm-hb bench \
  --base-url http://localhost:8000 \
  --n 100 \
  --max-tokens 128
```

Example output (NousResearch/Hermes-3-Llama-3.1-8B, NVIDIA V100 32GB, 50 requests, 128 tokens):

```
  vllm-hb benchmark
  ─────────────────────────────────────
  server     : http://localhost:8000
  model      : Hermes-3-Llama-3.1-8B
  requests   : 50
  max_tokens : 128

  Warmup complete. Running benchmark…

  Results
  ─────────────────────────────────────
  throughput   : 37.0 tok/s
  total tokens : 6400  (50 requests)

  TTFT (time to first token)
    mean       :  3459 ms
    p50        :  3458 ms
    p99        :  3468 ms

  End-to-end latency
    p50        :  3458 ms
    p99        :  3468 ms
```

> 37 tok/s ≈ 66 % of V100 HBM2 peak bandwidth utilised during single-sequence
> decode — no batching, no flash-attn. TTFT reflects full generation time in
> non-streaming mode; streaming TTFT (time to first token) is typically < 200 ms.

---

## Architecture

```
HTTP request
    │
    ▼
axum handler                    (src/server/)
    │  apply chat template
    │  tokenize → WorkItem
    ▼
WorkerHandle ──────────────────► Worker task         (src/worker/)
  unbounded channel               │
                                  │  Scheduler         (src/scheduler/)
                                  │  ├─ FCFS admission
                                  │  ├─ Paged KV block manager
                                  │  └─ Prefix cache (block-hash dedup)
                                  │
                                  │  per step: prefill or decode
                                  │  ├─ Engine::forward_with_cache  (src/engine/)
                                  │  │   └─ Backend dispatch: Llama / Mixtral /
                                  │  │      Qwen2 / Qwen3 / Phi3 / TpLlama
                                  │  ├─ Sampling (temperature, top-p, penalties)
                                  │  └─ Speculative decoder (optional)
                                  │
                                  ◄── GenerationEvent (token stream)
    │
    ▼
SSE stream / JSON response
```

---

## Roadmap

- [ ] Quantization (GPTQ / AWQ / INT8) — 70B on a single 4090
- [ ] MCP server (Model Context Protocol)
- [ ] Docker image
- [ ] Prometheus metrics (currently a stub)

---

## License

Apache 2.0
