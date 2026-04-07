//! Sequential throughput benchmark.
//!
//! Sends N requests one at a time to a running vllm-hb server and prints
//! a latency + throughput summary.  One warmup request is discarded.

use std::time::{Duration, Instant};

use anyhow::Result;
use reqwest::Client;
use serde_json::json;

use super::BenchArgs;

// ── Entry point ───────────────────────────────────────────────────────────────

pub async fn run(args: &BenchArgs) -> Result<()> {
    let client = Client::new();
    let endpoint = format!("{}/v1/chat/completions", args.base_url);

    let _ = send(&client, &endpoint, args, true).await;
    println!("  Warmup complete. Running benchmark…");
    println!();

    let mut ttfts: Vec<Duration> = Vec::with_capacity(args.n);
    let mut totals: Vec<Duration> = Vec::with_capacity(args.n);
    let mut tok_counts: Vec<usize> = Vec::with_capacity(args.n);

    for i in 0..args.n {
        match send(&client, &endpoint, args, false).await {
            Ok(r) => {
                ttfts.push(r.ttft);
                totals.push(r.total);
                tok_counts.push(r.completion_tokens);
                eprint!(
                    "\r  [{:>4}/{:>4}] {:.1} tok/s",
                    i + 1,
                    args.n,
                    r.completion_tokens as f64 / r.total.as_secs_f64()
                );
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
    ttft: Duration,
    total: Duration,
    completion_tokens: usize,
}

async fn send(
    client: &Client,
    endpoint: &str,
    args: &BenchArgs,
    warmup: bool,
) -> Result<RequestResult> {
    let body = json!({
        "model":       args.model,
        "messages":    [{"role": "user", "content": args.prompt}],
        "max_tokens":  args.max_tokens,
        "temperature": 0.0,
        "stream":      false,
    });

    let t_start = Instant::now();
    let resp = client.post(endpoint).json(&body).send().await?;
    let ttft = t_start.elapsed();

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await?;
        anyhow::bail!("HTTP {status}: {text}");
    }

    let json: serde_json::Value = resp.json().await?;
    let total = t_start.elapsed();

    let completion_tokens = json["usage"]["completion_tokens"].as_u64().unwrap_or(0) as usize;

    if warmup {
        return Ok(RequestResult {
            ttft: Duration::ZERO,
            total: Duration::ZERO,
            completion_tokens,
        });
    }

    Ok(RequestResult {
        ttft,
        total,
        completion_tokens,
    })
}

// ── Summary ───────────────────────────────────────────────────────────────────

fn print_summary(ttfts: &[Duration], totals: &[Duration], tok_counts: &[usize]) {
    if totals.is_empty() {
        println!("  No successful requests.");
        return;
    }

    let total_tokens: usize = tok_counts.iter().sum();
    let total_time: f64 = totals.iter().map(|d| d.as_secs_f64()).sum();
    let throughput = total_tokens as f64 / total_time;

    let mut sorted_ttft = ttfts.to_vec();
    let mut sorted_total = totals.to_vec();
    sorted_ttft.sort();
    sorted_total.sort();

    println!("  Results");
    println!("  ─────────────────────────────────────");
    println!("  throughput   : {throughput:.1} tok/s");
    println!(
        "  total tokens : {total_tokens}  ({} requests)",
        totals.len()
    );
    println!();
    println!("  TTFT (time to first token)");
    println!(
        "    mean       : {:>6.0} ms",
        mean(ttfts).as_secs_f64() * 1000.0
    );
    println!(
        "    p50        : {:>6.0} ms",
        percentile(&sorted_ttft, 50).as_secs_f64() * 1000.0
    );
    println!(
        "    p99        : {:>6.0} ms",
        percentile(&sorted_ttft, 99).as_secs_f64() * 1000.0
    );
    println!();
    println!("  End-to-end latency");
    println!(
        "    p50        : {:>6.0} ms",
        percentile(&sorted_total, 50).as_secs_f64() * 1000.0
    );
    println!(
        "    p99        : {:>6.0} ms",
        percentile(&sorted_total, 99).as_secs_f64() * 1000.0
    );
    println!();
}

fn percentile(sorted: &[Duration], p: usize) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    sorted[((p * sorted.len()) / 100).min(sorted.len() - 1)]
}

fn mean(v: &[Duration]) -> Duration {
    if v.is_empty() {
        return Duration::ZERO;
    }
    v.iter().sum::<Duration>() / v.len() as u32
}
