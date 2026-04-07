//! Inference worker — serialises GPU work and streams tokens back to callers.
//!
//! The worker owns the `Engine` and runs in a dedicated Tokio task.
//! HTTP handlers submit `WorkItem`s through a `WorkerHandle` and receive
//! `GenerationEvent`s on a per-request unbounded channel.
//!
//! # Concurrency model
//!
//! ```text
//!  HTTP handler 1  ─── WorkItem ──►┐
//!  HTTP handler 2  ─── WorkItem ──►│  worker task
//!  HTTP handler N  ─── WorkItem ──►┘  (processes one request at a time;
//!                                       GPU is always saturated)
//! ```
//!
//! # Roadmap
//!
//! When `scheduler/` is real, `process()` is replaced by a batch-step loop
//! that pulls from the scheduler's ready queue rather than the raw channel.
//! `WorkerHandle::submit` will enqueue into the scheduler instead of the
//! direct channel.

use std::time::Instant;

use anyhow::Result;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

use crate::engine::Engine;
use crate::sampling;
use crate::tokenize;
use crate::types::pipeline::{FinishReason, GenerationEvent, GenerationStats, TokenEvent, WorkItem};

// ── Handle ────────────────────────────────────────────────────────────────────

/// A cheap, cloneable handle for submitting work to the inference worker.
#[derive(Clone)]
pub struct WorkerHandle {
    tx: mpsc::UnboundedSender<WorkItem>,
}

impl WorkerHandle {
    pub fn submit(&self, item: WorkItem) -> Result<()> {
        self.tx
            .send(item)
            .map_err(|_| anyhow::anyhow!("Inference worker has shut down"))
    }
}

// ── Worker ────────────────────────────────────────────────────────────────────

pub struct Worker {
    rx:         mpsc::UnboundedReceiver<WorkItem>,
    engine:     Engine,
    tokenizer:  Tokenizer,
    eos_tokens: Vec<u32>,
}

impl Worker {
    pub fn new(engine: Engine, tokenizer: Tokenizer, eos_tokens: Vec<u32>) -> (Self, WorkerHandle) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { rx, engine, tokenizer, eos_tokens }, WorkerHandle { tx })
    }

    pub async fn run(mut self) {
        tracing::info!("Inference worker ready");
        while let Some(item) = self.rx.recv().await {
            let id = item.id.clone();
            if let Err(e) = self.process(item).await {
                tracing::error!(request_id = %id, error = %e, "Generation failed");
            }
        }
        tracing::info!("Inference worker stopped");
    }

    // ── Generation loop ───────────────────────────────────────────────────────

    async fn process(&self, item: WorkItem) -> Result<()> {
        let WorkItem { id, token_ids, params, result_tx } = item;
        let prompt_len = token_ids.len();
        let t_start    = Instant::now();

        if let Err(e) = self.engine.reset_cache() {
            let _ = result_tx.send(GenerationEvent::Error(e.to_string()));
            return Err(e);
        }

        // Prefill.
        let logits = match self.engine.forward(&token_ids, 0) {
            Ok(l)  => l,
            Err(e) => {
                let _ = result_tx.send(GenerationEvent::Error(e.to_string()));
                return Err(e);
            }
        };
        let ttft_ms = t_start.elapsed().as_millis() as u64;

        // Decode.
        let mut next_token = sampling::sample(&logits, params.temperature, params.top_p)?;
        let mut all_tokens = token_ids;
        let mut gen_count  = 0usize;
        let mut finish     = FinishReason::Length;

        loop {
            all_tokens.push(next_token);
            gen_count += 1;

            let text = tokenize::decode(&self.tokenizer, &[next_token]).unwrap_or_default();
            let _ = result_tx.send(GenerationEvent::Token(TokenEvent { id: next_token, text }));

            if self.eos_tokens.contains(&next_token) {
                finish = FinishReason::Stop;
                break;
            }
            if gen_count >= params.max_tokens {
                break;
            }

            let pos = all_tokens.len() - 1;
            let logits = match self.engine.forward(&[next_token], pos) {
                Ok(l)  => l,
                Err(e) => {
                    let _ = result_tx.send(GenerationEvent::Error(e.to_string()));
                    return Err(e);
                }
            };
            next_token = sampling::sample(&logits, params.temperature, params.top_p)?;
        }

        let total_ms       = t_start.elapsed().as_millis() as u64;
        let tokens_per_sec = gen_count as f64 / (total_ms as f64 / 1000.0);

        tracing::info!(
            request_id        = %id,
            prompt_tokens     = prompt_len,
            completion_tokens = gen_count,
            ttft_ms,
            total_ms,
            tokens_per_sec    = format!("{tokens_per_sec:.1}"),
            finish_reason     = finish.as_str(),
            "Request complete"
        );

        let _ = result_tx.send(GenerationEvent::Finished {
            finish_reason: finish,
            stats: GenerationStats {
                prompt_tokens:     prompt_len,
                completion_tokens: gen_count,
                ttft_ms,
                total_ms,
                tokens_per_sec,
            },
        });

        Ok(())
    }
}
