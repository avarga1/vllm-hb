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

- **OpenAI-compatible API** — `/v1/chat/completions` (streaming + non-streaming), `/v1/models`, `/health`
- **Streaming** — server-sent events, token-by-token
- **Top-p + temperature sampling** — proper nucleus filtering with `rand`
- **Auto chat template** — reads `tokenizer_config.json` and detects Llama-3 / ChatML / Mistral-v1
- **Auto EOS tokens** — reads from `config.json`, no hardcoded token IDs
- **Built-in benchmark** — `vllm-hb bench` measures TTFT, throughput, p50/p99

---

## Getting started

### Requirements

- Rust 1.75+ (`rustup install stable`)
- CUDA 12.x toolkit (`nvcc` in PATH, or `CUDA_ROOT` set)
- NVIDIA GPU (tested on V100, A100, RTX 4090)

### Build

```bash
git clone https://github.com/avarga1/vllm-hb
cd vllm-hb
cargo build --release
```

If `nvcc` is not in your default PATH:

```bash
export CUDA_ROOT=/path/to/cuda
export PATH=$CUDA_ROOT/bin:$PATH
cargo build --release
```

### Run

```bash
# Serve a local HuggingFace model directory
./target/release/vllm-hb serve \
  --model /models/Meta-Llama-3.1-8B-Instruct \
  --port 8000

# Test it
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64,
    "stream": false
  }'
```

### Benchmark

```bash
# Start the server first, then in another terminal:
./target/release/vllm-hb bench \
  --base-url http://localhost:8000 \
  --n 100 \
  --max-tokens 128
```

Example output:

```
  vllm-hb benchmark
  ─────────────────────────────────────
  server     : http://localhost:8000
  model      : llama
  requests   : 100
  max_tokens : 128

  Warmup complete. Running benchmark…

  Results
  ─────────────────────────────────────
  throughput   : 187.3 tok/s
  total tokens : 12800  (100 requests)

  TTFT (time to first token)
    mean       :    94 ms
    p50        :    89 ms
    p99        :   210 ms

  End-to-end latency
    p50        :   712 ms
    p99        :  1340 ms
```

---

## Supported models

Any model in HuggingFace safetensors format with a Llama-style architecture:

| Family | Tested |
|---|---|
| Meta Llama 3 / 3.1 / 3.2 | ✅ |
| Mistral 7B / Mixtral | ✅ |
| Qwen 2.5 | ✅ |
| Microsoft Phi-3 | ✅ |

> Support for additional architectures (Gemma, Falcon, …) is on the roadmap
> via additional `candle-transformers` model implementations.

---

## Architecture

```
HTTP request
    │
    ▼
axum handler          (src/server.rs)
    │  tokenize + submit WorkItem
    ▼
WorkerHandle ──────► Worker task       (src/worker.rs)
    │  unbounded channel    │
    │                       │  prefill → decode loop
    │                       │  sample() per step
    │                       │
    ◄───── GenerationEvent ◄┘  token stream back to handler
    │
    ▼
SSE stream / JSON response
```

The `Worker` owns the `Engine` (src/model.rs) and serialises GPU work.
Multiple HTTP handlers can submit requests concurrently; the worker
processes them in order, keeping the GPU saturated.

Sampling lives in `src/sampling.rs` — temperature scaling, softmax, and
top-p nucleus filtering are pure functions over a logit slice.

---

## Roadmap

- [ ] Multi-sequence continuous batching (paged KV cache)
- [ ] Tensor parallelism (multi-GPU)
- [ ] Mistral / Qwen native architecture emitters  
- [ ] `--dtype bf16` flag
- [ ] Prometheus metrics endpoint
- [ ] Docker image

---

## License

Apache 2.0
