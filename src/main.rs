//! vllm-hb — Hammingbound inference runtime.
//!
//! A vLLM-compatible OpenAI API server written in pure Rust.
//! No Python. No libtorch. No C++ interop. CUDA via cudarc.
//!
//! Library code lives in `src/lib.rs`; this file is the CLI entry point only.

use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::{EnvFilter, fmt};

use vllm_hb::bench;
use vllm_hb::engine::{self, ModelConfig};
use vllm_hb::server;
use vllm_hb::tokenize;
use vllm_hb::worker;

#[allow(unused_imports)]
use vllm_hb::worker::WorkerHandle;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name    = "vllm-hb",
    version,
    about   = "Hammingbound — vLLM-compatible inference in pure Rust",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the OpenAI-compatible inference server.
    Serve(ServeArgs),
    /// Run a throughput benchmark against a running server.
    Bench(bench::BenchArgs),
}

#[derive(clap::Args, Debug)]
struct ServeArgs {
    /// Path to the model directory (safetensors + config.json).
    #[arg(long)]
    model: String,

    /// Path to the tokenizer directory (defaults to --model).
    #[arg(long)]
    tokenizer: Option<String>,

    /// Host to bind to.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to bind to.
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Maximum sequence length (prompt + completion).
    #[arg(long, default_value_t = 4096)]
    max_seq_len: usize,

    /// GPU memory utilisation fraction (0.0–1.0).
    #[arg(long, default_value_t = 0.90)]
    gpu_memory_utilization: f64,

    /// Use BF16 weights (requires sm_80+ / Ampere GPU; falls back to F16).
    #[arg(long)]
    bf16: bool,

    /// Number of GPUs for tensor parallelism (default: 1 = single GPU).
    #[arg(long, default_value_t = 1)]
    tensor_parallel_size: usize,
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("vllm_hb=info".parse()?))
        .json()
        .init();

    match Cli::parse().command {
        Command::Serve(args) => serve(args).await,
        Command::Bench(args) => bench::run(args).await,
    }
}

// ── Serve ─────────────────────────────────────────────────────────────────────

async fn serve(args: ServeArgs) -> Result<()> {
    let tokenizer_path = args.tokenizer.as_deref().unwrap_or(&args.model);

    tracing::info!(model = %args.model, "Loading tokenizer");
    let tokenizer = tokenize::load(tokenizer_path)?;
    let eos_tokens = tokenize::load_eos_tokens(&args.model)?;
    tracing::info!(
        vocab_size = tokenizer.get_vocab_size(true),
        eos_tokens = ?eos_tokens,
        "Tokenizer ready"
    );

    tracing::info!(model = %args.model, "Loading model weights");
    let engine = engine::Engine::load(ModelConfig {
        model_path: args.model.clone(),
        max_seq_len: args.max_seq_len,
        gpu_memory_utilization: args.gpu_memory_utilization,
        bf16: args.bf16,
        tensor_parallel_size: args.tensor_parallel_size,
    })?;
    tracing::info!(
        params = engine.param_count(),
        layers = engine.num_layers(),
        vocab  = engine.vocab_size(),
        device = ?engine.device,
        "Model ready"
    );

    let (worker, handle) = worker::Worker::new(engine, tokenizer.clone(), eos_tokens);
    tokio::spawn(worker.run());

    let model_name = args
        .model
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or("unknown")
        .to_string();

    let state = Arc::new(server::AppState {
        worker: handle,
        tokenizer,
        model_name,
        model_path: args.model,
    });

    let addr = format!("{}:{}", args.host, args.port);
    server::serve(state, &addr).await
}
