//! GGUF / GGML quantized Llama backend.
//!
//! Wraps `candle_transformers::models::quantized_llama::ModelWeights` to give
//! it the same `forward_with_cache` interface as the other architecture backends.
//!
//! # Memory model
//!
//! `candle_transformers::quantized_llama::ModelWeights` is `Clone` and uses
//! Arc-backed `Tensor`s internally, so cloning the template model is O(n_layers)
//! ref-count bumps — no weight data is copied.  Each active sequence receives
//! its own clone, giving it an independent KV cache while sharing the quantised
//! weight tensors.  This means multi-sequence concurrent generation works
//! correctly: sequence A's KV cache does not bleed into sequence B's.
//!
//! # Quantization formats
//!
//! Any GGUF file that `candle_transformers::quantized_llama::ModelWeights::from_gguf`
//! accepts works here: Q2_K, Q3_K_*, Q4_0, Q4_K_*, Q5_K_*, Q6_K, Q8_0, F16, F32.
//! GGML (`.bin`) files are not supported — use GGUF.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;

// ── Backend ───────────────────────────────────────────────────────────────────

/// GGUF-quantized Llama backend.
///
/// Holds a "template" `ModelWeights` (freshly loaded, no KV state).  Each call
/// to `create_seq_model()` clones the template cheaply and returns an
/// independent instance suitable for one sequence's prefill + decode loop.
pub struct GgufLlamaBackend {
    /// Arc so that `clone` across threads is safe.
    model_template: Arc<ModelWeights>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub device: Device,
}

impl GgufLlamaBackend {
    /// Load a `.gguf` file from `path` (file, not directory).
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let mut file =
            fs::File::open(path).with_context(|| format!("Opening {}", path.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .with_context(|| format!("Parsing GGUF header from {}", path.display()))?;

        // Extract key metadata before consuming `content`.
        let md = &content.metadata;
        let vocab_size = md
            .get("llama.vocab_size")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(0) as usize;
        let hidden_size = md
            .get("llama.embedding_length")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(0) as usize;
        let num_layers = md
            .get("llama.block_count")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(0) as usize;

        tracing::info!(
            path = %path.display(),
            vocab  = vocab_size,
            hidden = hidden_size,
            layers = num_layers,
            "Loading GGUF model"
        );

        let model_template = ModelWeights::from_gguf(content, &mut file, device)
            .with_context(|| format!("Building ModelWeights from {}", path.display()))?;

        tracing::info!(path = %path.display(), "GGUF model loaded");

        Ok(Self {
            model_template: Arc::new(model_template),
            vocab_size,
            hidden_size,
            num_layers,
            device: device.clone(),
        })
    }

    /// Clone the template model to get a fresh, KV-cache-free instance for
    /// one sequence.  Arc-backed weights are shared; only the KV tensors that
    /// accumulate during generation are private to the clone.
    pub fn create_seq_model(&self) -> ModelWeights {
        self.model_template.as_ref().clone()
    }

    /// Run a forward pass using the per-sequence `ModelWeights`.
    ///
    /// `token_ids` is a non-empty slice of token IDs.
    /// `seq_pos`   is the position of the first token in the sequence
    ///             (0 for a fresh prefill, `prompt_len + output_so_far` for decode).
    pub fn forward(
        model: &mut ModelWeights,
        token_ids: &[u32],
        seq_pos: usize,
        device: &Device,
    ) -> Result<Tensor> {
        if token_ids.is_empty() {
            bail!("forward called with empty token_ids");
        }
        // Build [1, seq_len] u32 tensor.
        let ids = Tensor::new(token_ids, device)?.unsqueeze(0)?;
        // quantized_llama::ModelWeights::forward returns logits [vocab_size].
        model.forward(&ids, seq_pos).map_err(|e| anyhow::anyhow!(e))
    }
}
