//! Throughput benchmark.
//!
//! Sends N sequential generation requests to a running vllm-hb server and
//! prints a latency + throughput summary.
//!
//! Usage:
//!   vllm-hb bench --base-url http://localhost:8000 \
//!                 --model mymodel \
//!                 --n 50 \
//!                 --max-tokens 128

use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Args;
use serde_json::json;

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

    /// Prompt to send (same prompt repeated for all requests).
    #[arg(long, default_value = "Explain the theory of general relativity in simple terms.")]
    pub prompt: String,
}

pub async fn run(args: BenchArgs) -> Result<()> {
    let client   = reqwest::Client::new();
    let endpoint = format!("{}/v1/chat/completions", args.base_url);

    println!();
    println!("  vllm-hb benchmark");
    println!("  ─────────────────────────────────────");
    println!("  server     : {}", args.base_url);
    println!("  model      : {}", args.model);
    println!("  requests   : {}", args.n);
    println!("  max_tokens : {}", args.max_tokens);
    println!();

    // Warmup — one request not counted in stats.
    let _ = send(&client, &endpoint, &args, true).await;
    println!("  Warmup complete. Running benchmark…");
    println!();

    let mut ttfts:    Vec<Duration> = Vec::with_capacity(args.n);
    let mut totals:   Vec<Duration> = Vec::with_capacity(args.n);
    let mut tok_counts: Vec<usize>  = Vec::with_capacity(args.n);

    for i in 0..args.n {
        match send(&client, &endpoint, &args, false).await {
            Ok(r) => {
                ttfts.push(r.ttft);
                totals.push(r.total);
                tok_counts.push(r.completion_tokens);
                eprint!("\r  [{:>4}/{:>4}] {:.1} tok/s", i + 1, args.n,
                    r.completion_tokens as f64 / r.total.as_secs_f64());
            }
            Err(e) => {
                eprintln!("\r  [{:>4}/{:>4}] ERROR: {e}", i + 1, args.n);
            }
        }
    }
    eprintln!();

    print_summary(&ttfts, &totals, &tok_counts);
    Ok(())
}

// ── HTTP request ──────────────────────────────────────────────────────────────

struct RequestResult {
    ttft:               Duration,
    total:              Duration,
    completion_tokens:  usize,
}

async fn send(
    client:   &reqwest::Client,
    endpoint: &str,
    args:     &BenchArgs,
    warmup:   bool,
) -> Result<RequestResult> {
    let body = json!({
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": false,
    });

    let t_start = Instant::now();
    let resp    = client.post(endpoint).json(&body).send().await?;
    let ttft    = t_start.elapsed();

    if !resp.status().is_success() {
        let status = resp.status();
        let text   = resp.text().await?;
        anyhow::bail!("HTTP {status}: {text}");
    }

    let json: serde_json::Value = resp.json().await?;
    let total = t_start.elapsed();

    let completion_tokens = json["usage"]["completion_tokens"]
        .as_u64()
        .unwrap_or(0) as usize;

    if warmup {
        return Ok(RequestResult { ttft: Duration::ZERO, total: Duration::ZERO, completion_tokens });
    }

    Ok(RequestResult { ttft, total, completion_tokens })
}

// ── Summary table ─────────────────────────────────────────────────────────────

fn print_summary(ttfts: &[Duration], totals: &[Duration], tok_counts: &[usize]) {
    if totals.is_empty() {
        println!("  No successful requests.");
        return;
    }

    let total_tokens: usize = tok_counts.iter().sum();
    let total_time: f64     = totals.iter().map(|d| d.as_secs_f64()).sum();
    let throughput          = total_tokens as f64 / total_time;

    let mut sorted_ttft  = ttfts.to_vec();
    let mut sorted_total = totals.to_vec();
    sorted_ttft.sort();
    sorted_total.sort();

    let p50_ttft  = percentile(&sorted_ttft,  50);
    let p99_ttft  = percentile(&sorted_ttft,  99);
    let p50_total = percentile(&sorted_total, 50);
    let p99_total = percentile(&sorted_total, 99);
    let mean_ttft = mean(ttfts);

    println!("  Results");
    println!("  ─────────────────────────────────────");
    println!("  throughput   : {throughput:.1} tok/s");
    println!("  total tokens : {total_tokens}  ({} requests)", totals.len());
    println!();
    println!("  TTFT (time to first token)");
    println!("    mean       : {:>6.0} ms", mean_ttft.as_secs_f64() * 1000.0);
    println!("    p50        : {:>6.0} ms", p50_ttft.as_secs_f64()  * 1000.0);
    println!("    p99        : {:>6.0} ms", p99_ttft.as_secs_f64()  * 1000.0);
    println!();
    println!("  End-to-end latency");
    println!("    p50        : {:>6.0} ms", p50_total.as_secs_f64() * 1000.0);
    println!("    p99        : {:>6.0} ms", p99_total.as_secs_f64() * 1000.0);
    println!();
}

fn percentile(sorted: &[Duration], p: usize) -> Duration {
    if sorted.is_empty() { return Duration::ZERO; }
    let idx = ((p * sorted.len()) / 100).min(sorted.len() - 1);
    sorted[idx]
}

fn mean(v: &[Duration]) -> Duration {
    if v.is_empty() { return Duration::ZERO; }
    v.iter().sum::<Duration>() / v.len() as u32
}
