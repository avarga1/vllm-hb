//! Inference worker — scheduler-driven continuous batching loop.
//!
//! # Architecture
//!
//! ```text
//!  HTTP handler 1  ─── WorkItem ──►┐
//!  HTTP handler 2  ─── WorkItem ──►│  inbox channel
//!  HTTP handler N  ─── WorkItem ──►┘
//!                                   │
//!                               Worker::run()
//!                                   │
//!                          ┌────────▼────────┐
//!                          │    Scheduler    │ ← admission / FCFS ordering
//!                          └────────┬────────┘
//!                                   │  to_prefill / to_decode
//!                          ┌────────▼────────┐
//!                          │     Engine      │ ← GPU forward pass
//!                          └────────┬────────┘
//!                                   │  token stream
//!                          per-request result_tx channels
//! ```
//!
//! # Batching limitation (tracked in issue #12)
//!
//! The engine's KV cache is a single shared state (one physical cache per GPU).
//! Until the engine exposes per-sequence block tables (see `scheduler/block_manager.rs`),
//! each sequence group is processed to completion before the next is started.
//! The scheduler still provides:
//!
//! - FCFS ordering across concurrent HTTP requests
//! - Memory-aware admission (blocks are allocated/freed via `BlockManager`)
//! - Correct preemption + swap infrastructure, ready when the engine gains
//!   a `forward_batch()` API that accepts per-block KV tensors.

use std::time::Instant;

use anyhow::Result;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

use crate::engine::Engine;
use crate::sampling;
use crate::scheduler::sequence::{Sequence, SequenceGroup, SequenceStatus};
use crate::scheduler::Scheduler;
use crate::tokenize;
use crate::types::pipeline::{
    FinishReason, GenerationEvent, GenerationStats, TokenEvent, WorkItem,
};

// ── Tuning constants ──────────────────────────────────────────────────────────

/// Number of GPU blocks available to the scheduler.
/// Each block holds `BLOCK_SIZE` (16) tokens of KV cache.
const NUM_GPU_BLOCKS: usize = 256;
const NUM_CPU_BLOCKS: usize = 512;

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
    rx: mpsc::UnboundedReceiver<WorkItem>,
    engine: Engine,
    tokenizer: Tokenizer,
    eos_tokens: Vec<u32>,
    scheduler: Scheduler,
    /// Monotonic counter for assigning unique sequence IDs.
    next_seq_id: u64,
}

impl Worker {
    pub fn new(engine: Engine, tokenizer: Tokenizer, eos_tokens: Vec<u32>) -> (Self, WorkerHandle) {
        let (tx, rx) = mpsc::unbounded_channel();
        (
            Self {
                rx,
                engine,
                tokenizer,
                eos_tokens,
                scheduler: Scheduler::new(NUM_GPU_BLOCKS, NUM_CPU_BLOCKS),
                next_seq_id: 0,
            },
            WorkerHandle { tx },
        )
    }

    // ── Main loop ─────────────────────────────────────────────────────────────

    pub async fn run(mut self) {
        tracing::info!(
            gpu_blocks = NUM_GPU_BLOCKS,
            cpu_blocks = NUM_CPU_BLOCKS,
            "Scheduler-driven inference worker ready"
        );

        loop {
            // Block until at least one WorkItem arrives.
            let Some(item) = self.rx.recv().await else {
                tracing::info!("Inbox closed — worker stopping");
                break;
            };
            self.admit(item);

            // Drain any requests that arrived while we were waiting.
            while let Ok(item) = self.rx.try_recv() {
                self.admit(item);
            }

            // Keep stepping until all queues are empty.
            loop {
                // Absorb requests that arrived while the GPU was busy.
                while let Ok(item) = self.rx.try_recv() {
                    self.admit(item);
                }

                if self.scheduler.num_waiting() == 0
                    && self.scheduler.num_running() == 0
                    && self.scheduler.num_swapped() == 0
                {
                    break;
                }

                self.step();
            }
        }

        tracing::info!("Inference worker stopped");
    }

    // ── Admission ─────────────────────────────────────────────────────────────

    fn admit(&mut self, item: WorkItem) {
        let id = self.next_seq_id;
        self.next_seq_id += 1;

        let seq =
            Sequence::new(id, item.token_ids, item.params, item.result_tx);
        let group = SequenceGroup::new(item.id.clone(), vec![seq]);
        self.scheduler.add_sequence_group(group);

        tracing::debug!(
            request_id = %item.id,
            waiting    = self.scheduler.num_waiting(),
            "Request admitted to scheduler"
        );
    }

    // ── Scheduler step ────────────────────────────────────────────────────────

    fn step(&mut self) {
        let outputs = self.scheduler.schedule();

        if outputs.is_empty() {
            return;
        }

        let mut done: Vec<SequenceGroup> = Vec::new();
        let still_running: Vec<SequenceGroup> = Vec::new();

        for mut group in outputs
            .to_prefill
            .into_iter()
            .chain(outputs.to_decode)
        {
            match self.run_group_to_completion(&mut group) {
                Ok(()) => done.push(group),
                Err(e) => {
                    tracing::error!(
                        request_id = %group.request_id,
                        error      = %e,
                        "Generation failed"
                    );
                    for seq in &mut group.seqs {
                        let _ = seq.result_tx.send(GenerationEvent::Error(e.to_string()));
                        seq.status = SequenceStatus::Finished(FinishReason::Length);
                    }
                    done.push(group);
                }
            }
        }

        // Return groups to the scheduler so it can free their GPU blocks.
        let return_outputs = crate::scheduler::SchedulerOutputs {
            to_prefill: done,
            to_decode: still_running,
            blocks_to_swap_in: Vec::new(),
            blocks_to_swap_out: Vec::new(),
            blocks_to_copy: Vec::new(),
        };
        self.scheduler.update(return_outputs);
    }

    // ── Generation ────────────────────────────────────────────────────────────

    fn run_group_to_completion(&self, group: &mut SequenceGroup) -> Result<()> {
        // Single shared KV cache — reset before each new group.
        self.engine.reset_cache()?;

        let seq = group
            .seqs
            .iter_mut()
            .find(|s| s.status == SequenceStatus::Running)
            .ok_or_else(|| anyhow::anyhow!("no running sequence in group {}", group.request_id))?;

        let prompt_len = seq.prompt_ids.len();
        let params = seq.params.clone();
        let t_start = Instant::now();

        // Prefill.
        let logits = self.engine.forward(&seq.prompt_ids, 0)?;
        let ttft_ms = t_start.elapsed().as_millis() as u64;

        // Decode.
        let mut next_token = sampling::sample(&logits, params.temperature, params.top_p)?;
        let mut gen_count = 0usize;
        let mut finish = FinishReason::Length;

        loop {
            seq.output_ids.push(next_token);
            gen_count += 1;

            let text = tokenize::decode(&self.tokenizer, &[next_token]).unwrap_or_default();
            let _ = seq.result_tx.send(GenerationEvent::Token(TokenEvent {
                id: next_token,
                text,
            }));

            if self.eos_tokens.contains(&next_token) {
                finish = FinishReason::Stop;
                break;
            }
            if gen_count >= params.max_tokens {
                break;
            }

            let pos = prompt_len + gen_count - 1;
            let logits = self.engine.forward(&[next_token], pos)?;
            next_token = sampling::sample(&logits, params.temperature, params.top_p)?;
        }

        seq.status = SequenceStatus::Finished(finish);

        let total_ms = t_start.elapsed().as_millis() as u64;
        let tokens_per_sec = gen_count as f64 / (total_ms.max(1) as f64 / 1000.0);

        tracing::info!(
            request_id        = %group.request_id,
            prompt_tokens     = prompt_len,
            completion_tokens = gen_count,
            ttft_ms,
            total_ms,
            tokens_per_sec    = format!("{tokens_per_sec:.1}"),
            finish_reason     = finish.as_str(),
            "Request complete"
        );

        let _ = seq.result_tx.send(GenerationEvent::Finished {
            finish_reason: finish,
            stats: GenerationStats {
                prompt_tokens: prompt_len,
                completion_tokens: gen_count,
                ttft_ms,
                total_ms,
                tokens_per_sec,
            },
        });

        Ok(())
    }
}
