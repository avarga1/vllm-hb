//! Concurrent load benchmark.
//!
//! # Status: STUB
//!
//! Issues `--concurrency N` requests in parallel using `tokio::spawn`,
//! then aggregates results into the same summary table used by the
//! sequential benchmark.
//!
//! ## Planned CLI changes
//!
//! `BenchArgs` gains a `--concurrency` flag (default = 1, which preserves
//! today's sequential behaviour).  When concurrency > 1, `run()` dispatches
//! to this module instead of `sequential::run()`.
//!
//! ## Why this matters
//!
//! The core thesis of vllm-hb is that Rust's scheduler has lower overhead
//! than Python-under-GIL at high concurrency.  The sequential benchmark
//! cannot show this because the worker processes one request at a time.
//! At N=50 concurrent requests the Python GIL starts costing measurably;
//! the Rust worker queue has no such penalty.
//!
//! ## Expected output additions
//!
//! ```text
//!   Concurrency  : 32
//!   Requests     : 200
//!   ─────────────────────────────────────
//!   throughput   : 1240.7 tok/s
//!   queue_depth  : 28.4 (mean)
//!   scheduler_overhead_p99 : 0.3 ms
//! ```

#![allow(dead_code)]

use anyhow::Result;

/// Run a concurrent load test.  Not yet implemented.
pub async fn run() -> Result<()> {
    anyhow::bail!(
        "Concurrent benchmark not yet implemented. \
         Track progress: https://github.com/avarga1/vllm-hb/issues/3"
    )
}
