//! Inference worker — true continuous-batching step loop.
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
//!                                   │  one token per sequence per step
//!                          per-request result_tx channels
//! ```
//!
//! # Continuous batching
//!
//! Each `step()` processes every active sequence for exactly one token:
//!
//! - **Prefill** (`to_prefill`): feed the full prompt through the engine,
//!   sample the first output token, store the sequence's KV cache.
//! - **Decode** (`to_decode`): feed the last output token at the correct
//!   position, sample the next token, update the KV cache in place.
//!
//! Sequences are not run to completion before the next is started.  New
//! requests that arrive while the GPU is busy are admitted on the next
//! `schedule()` call and join the existing running batch.
//!
//! # Per-sequence KV cache
//!
//! Every admitted sequence owns a `PerSeqCache` (see `engine::kv_cache`).
//! The cache is created in `step_prefill`, mutated in-place by
//! `Engine::forward_with_cache`, and dropped when the sequence finishes.
//! No global `reset_cache()` call is needed — each sequence manages its
//! own state independently.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

use crate::engine::{Engine, PerSeqCache};
use crate::sampling;
use crate::scheduler::sequence::{Sequence, SequenceGroup, SequenceStatus};
use crate::scheduler::{Scheduler, SchedulerOutputs};
use crate::tokenize;
use crate::types::pipeline::{
    FinishReason, GenerationEvent, GenerationStats, TokenEvent, WorkItem,
};

// ── Tuning constants ──────────────────────────────────────────────────────────

/// Number of GPU KV-cache blocks available to the scheduler.
/// Each block holds `BLOCK_SIZE` (16) tokens of KV state.
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

    /// Construct a handle backed by a caller-owned sender.
    ///
    /// Intended for integration tests that inject a mock worker without
    /// loading model weights.  Not part of the production API.
    #[doc(hidden)]
    pub fn for_test(tx: mpsc::UnboundedSender<WorkItem>) -> Self {
        Self { tx }
    }
}

// ── Worker ────────────────────────────────────────────────────────────────────

pub struct Worker {
    rx: mpsc::UnboundedReceiver<WorkItem>,
    engine: Engine,
    tokenizer: Tokenizer,
    eos_tokens: Vec<u32>,
    scheduler: Scheduler,
    /// Per-sequence KV cache: seq_id → cache.
    kv_caches: HashMap<u64, PerSeqCache>,
    /// Per-sequence start timestamp for TTFT and throughput stats.
    seq_start: HashMap<u64, Instant>,
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
                kv_caches: HashMap::new(),
                seq_start: HashMap::new(),
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
            "Continuous-batching inference worker ready"
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

            // Step until all queues are empty.
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

        self.seq_start.insert(id, Instant::now());

        let seq = Sequence::new(id, item.token_ids, item.params, item.result_tx);
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

        if outputs.to_prefill.is_empty() && outputs.to_decode.is_empty() {
            return;
        }

        let mut return_groups: Vec<SequenceGroup> = Vec::new();

        // ── Prefill ───────────────────────────────────────────────────────────
        for mut group in outputs.to_prefill {
            match self.step_prefill(&mut group) {
                Ok(()) => {}
                Err(e) => {
                    tracing::error!(
                        request_id = %group.request_id,
                        error      = %e,
                        "Prefill failed"
                    );
                    self.fail_group(&mut group, e.to_string());
                }
            }
            return_groups.push(group);
        }

        // ── Decode ────────────────────────────────────────────────────────────
        for mut group in outputs.to_decode {
            match self.step_decode(&mut group) {
                Ok(()) => {}
                Err(e) => {
                    tracing::error!(
                        request_id = %group.request_id,
                        error      = %e,
                        "Decode step failed"
                    );
                    self.fail_group(&mut group, e.to_string());
                }
            }
            return_groups.push(group);
        }

        // Return all groups to the scheduler.  Finished ones will have their
        // blocks freed; running ones go back to self.scheduler.running.
        self.scheduler.update(SchedulerOutputs {
            to_prefill: return_groups,
            to_decode: Vec::new(),
            blocks_to_swap_in: Vec::new(),
            blocks_to_swap_out: Vec::new(),
            blocks_to_copy: Vec::new(),
        });
    }

    // ── Prefill one sequence group ────────────────────────────────────────────

    /// Run the prompt through the engine, emit the first output token.
    fn step_prefill(&mut self, group: &mut SequenceGroup) -> Result<()> {
        let seq = group
            .seqs
            .iter_mut()
            .find(|s| s.status == SequenceStatus::Running)
            .ok_or_else(|| anyhow::anyhow!("no running sequence in group {}", group.request_id))?;

        let mut cache = self.engine.create_kv_cache()?;

        // Prefill: process all prompt tokens in one forward pass.
        let logits = self
            .engine
            .forward_with_cache(&seq.prompt_ids, 0, &mut cache)?;
        let first_token = sampling::sample(&logits, seq.params.temperature, seq.params.top_p)?;

        seq.output_ids.push(first_token);
        self.emit_token(seq, first_token);

        // Store cache for future decode steps.
        self.kv_caches.insert(seq.id, cache);

        if self.is_done(seq) {
            self.finish_seq(seq);
        }

        Ok(())
    }

    // ── Decode one step for one sequence group ────────────────────────────────

    /// Feed the last generated token, emit the next output token.
    fn step_decode(&mut self, group: &mut SequenceGroup) -> Result<()> {
        let seq = group
            .seqs
            .iter_mut()
            .find(|s| s.status == SequenceStatus::Running)
            .ok_or_else(|| anyhow::anyhow!("no running sequence in group {}", group.request_id))?;

        let cache = self
            .kv_caches
            .get_mut(&seq.id)
            .ok_or_else(|| anyhow::anyhow!("missing KV cache for seq {}", seq.id))?;

        // Position of the token we're about to generate.
        // After prefill: prompt positions 0..P-1 are in cache; first decode
        // token was at P (= prompt_len); output_ids already has that token.
        // So next decode position = prompt_len + output_ids.len() - 1.
        let last_token = *seq.output_ids.last().unwrap();
        let seq_pos = seq.prompt_ids.len() + seq.output_ids.len() - 1;

        let logits = self
            .engine
            .forward_with_cache(&[last_token], seq_pos, cache)?;
        let next_token = sampling::sample(&logits, seq.params.temperature, seq.params.top_p)?;

        seq.output_ids.push(next_token);
        self.emit_token(seq, next_token);

        if self.is_done(seq) {
            self.finish_seq(seq);
            self.kv_caches.remove(&seq.id);
        }

        Ok(())
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn is_done(&self, seq: &Sequence) -> bool {
        self.eos_tokens
            .contains(seq.output_ids.last().unwrap_or(&0))
            || seq.output_ids.len() >= seq.params.max_tokens
    }

    fn emit_token(&self, seq: &Sequence, token_id: u32) {
        let text = tokenize::decode(&self.tokenizer, &[token_id]).unwrap_or_default();
        let _ = seq
            .result_tx
            .send(GenerationEvent::Token(TokenEvent { id: token_id, text }));
    }

    fn finish_seq(&mut self, seq: &mut Sequence) {
        let finish = if self
            .eos_tokens
            .contains(seq.output_ids.last().unwrap_or(&0))
        {
            FinishReason::Stop
        } else {
            FinishReason::Length
        };

        seq.status = SequenceStatus::Finished(finish);

        let t_start = self.seq_start.remove(&seq.id).unwrap_or_else(Instant::now);
        let total_ms = t_start.elapsed().as_millis() as u64;
        let prompt_len = seq.prompt_ids.len();
        let gen_count = seq.output_ids.len();
        let tokens_per_sec = gen_count as f64 / (total_ms.max(1) as f64 / 1000.0);

        // TTFT is not directly observable per-step; approximate as time to
        // complete the prefill, which is total_ms for single-token outputs.
        // A future improvement: record the prefill timestamp separately.
        let ttft_ms = total_ms;

        tracing::info!(
            seq_id = seq.id,
            prompt_tokens = prompt_len,
            completion_tokens = gen_count,
            total_ms,
            tokens_per_sec = format!("{tokens_per_sec:.1}"),
            finish_reason = finish.as_str(),
            "Sequence complete"
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
    }

    fn fail_group(&self, group: &mut SequenceGroup, msg: String) {
        for seq in &mut group.seqs {
            let _ = seq.result_tx.send(GenerationEvent::Error(msg.clone()));
            seq.status = SequenceStatus::Finished(FinishReason::Length);
        }
    }
}
