//! Built-in benchmark tool.
//!
//! # Modules
//! - `sequential`  — N sequential requests, TTFT + throughput summary (✅)
//! - `concurrent`  — N parallel requests, scheduler overhead test (🔜 stub)
//!
//! # Usage
//! ```
//! vllm-hb bench --base-url http://localhost:8000 --n 50 --max-tokens 128
//! ```

pub mod concurrent;
pub mod sequential;

use anyhow::Result;
use clap::Args;

// ── CLI args ──────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct BenchArgs {
    /// Base URL of the running vllm-hb server.
    #[arg(long, default_value = "http://localhost:8000")]
    pub base_url: String,

    /// Model name to pass in each request.
    #[arg(long, default_value = "vllm-hb")]
    pub model: String,

    /// Number of requests to issue.
    #[arg(long, default_value_t = 50)]
    pub n: usize,

    /// Tokens to generate per request.
    #[arg(long, default_value_t = 128)]
    pub max_tokens: usize,

    /// Prompt to send (same prompt for all requests).
    #[arg(
        long,
        default_value = "Explain the theory of general relativity in simple terms."
    )]
    pub prompt: String,

    /// Number of concurrent requests (1 = sequential; >1 requires continuous
    /// batching support in the server — see bench/concurrent.rs).
    #[arg(long, default_value_t = 1)]
    pub concurrency: usize,
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub async fn run(args: BenchArgs) -> Result<()> {
    println!();
    println!("  vllm-hb benchmark");
    println!("  ─────────────────────────────────────");
    println!("  server     : {}", args.base_url);
    println!("  model      : {}", args.model);
    println!("  requests   : {}", args.n);
    println!("  max_tokens : {}", args.max_tokens);
    println!("  concurrency: {}", args.concurrency);
    println!();

    if args.concurrency > 1 {
        concurrent::run().await
    } else {
        sequential::run(&args).await
    }
}
