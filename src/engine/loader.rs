//! Engine — loads model weights and dispatches forward/reset to the
//! correct architecture backend.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::arch::{
    Backend, GgufLlamaBackend, LlamaBackend, MixtralBackend, Phi3Backend, Qwen2Backend,
    Qwen3Backend, TpLlamaBackend,
};
use super::config::{HfMeta, ModelConfig};
use super::dtype;
use super::kv_cache::PerSeqCache;
use crate::parallel::TpWorld;

// ── Known embedding-weight keys (by model family) ────────────────────────────

/// Safetensors key names tried in order when loading the token embedding matrix.
///
/// Different HuggingFace model families use different key names for the same
/// `embed_tokens` weight.  We try each in sequence and use the first hit.
const EMBED_WEIGHT_KEYS: &[&str] = &[
    "model.embed_tokens.weight",         // Llama, Mistral, Qwen2
    "transformer.wte.weight",            // GPT-2, Falcon
    "model.wte.weight",                  // older HF GPT-style
    "embeddings.word_embeddings.weight", // BERT
];

// ── Engine ────────────────────────────────────────────────────────────────────

/// Owns the loaded model weights and KV cache.
///
/// The public API (`forward`, `reset_cache`, metadata accessors) is
/// identical regardless of architecture — callers never see candle types.
pub struct Engine {
    #[allow(dead_code)] // stored for future scheduler use (max_seq_len, gpu_memory_utilization)
    pub config: ModelConfig,
    backend: Backend,
    pub device: Device,
    // Shared metadata for external introspection.
    vocab_size: usize,
    num_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    /// Token embedding matrix `[vocab_size, hidden_size]` loaded once from
    /// the safetensors shards.  Used by `embed()` to produce mean-pooled,
    /// L2-normalised token embeddings without a full transformer forward pass.
    /// `None` when no known embedding weight key was found in the shards.
    embed_tokens: Option<Tensor>,
}

impl Engine {
    /// Load weights from `config.model_path`.
    ///
    /// Accepts two layouts:
    ///
    /// 1. **HuggingFace safetensors directory** — `model_path` is a directory
    ///    containing `*.safetensors` shards and a `config.json`.
    /// 2. **GGUF file** — `model_path` is a path directly to a `.gguf` file.
    ///    The architecture is inferred from GGUF metadata; `config.json` is not
    ///    required.  All GGUF quantization types supported by
    ///    `candle_transformers` are accepted (Q4_K_M, Q8_0, F16, …).
    pub fn load(config: ModelConfig) -> Result<Self> {
        let model_path_str = config.model_path.clone();
        let model_path = Path::new(&model_path_str);

        // ── Fast path: GGUF file ──────────────────────────────────────────────
        if model_path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            return Self::load_gguf(config, model_path);
        }

        // Build tensor-parallel world.  world_size=1 is always a no-op.
        let world = TpWorld::new(config.tensor_parallel_size)?;
        if world.is_single() {
            tracing::info!(
                tensor_parallel_size = 1,
                "Tensor parallelism disabled (single GPU)"
            );
        } else {
            tracing::info!(
                tensor_parallel_size = world.world_size(),
                "Tensor parallelism enabled"
            );
        }

        let device = world.device(0).clone();
        tracing::info!(device = ?device, "Compute device");

        if cfg!(feature = "flash-attn") {
            tracing::info!("Flash Attention 2 enabled (sm_80+)");
        } else {
            tracing::info!(
                "Using SDPA attention (build with --features flash-attn for FA2 on sm_80+)"
            );
        }

        let dtype = dtype::resolve(&device, config.bf16);

        let config_path = model_path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Reading {}", config_path.display()))?;

        let meta: HfMeta = serde_json::from_str(&config_str)
            .context("Parsing config.json for architecture metadata")?;

        tracing::info!(
            model_type = %meta.model_type,
            layers     = meta.num_hidden_layers,
            hidden     = meta.hidden_size,
            heads      = meta.num_attention_heads,
            vocab      = meta.vocab_size,
            "Model architecture"
        );

        let mut shards: Vec<_> = std::fs::read_dir(model_path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        shards.sort();

        if shards.is_empty() {
            bail!("No .safetensors files found in {}", model_path.display());
        }
        tracing::info!(shards = shards.len(), dtype = ?dtype, "Loading weights");

        let backend = match (meta.model_type.as_str(), world.is_single()) {
            // TP path: multi-GPU Llama with sharded weights
            ("llama" | "mistral", false) => {
                Backend::LlamaTp(TpLlamaBackend::load(&config_str, &shards, dtype, world)?)
            }
            // Single-GPU Llama (uses candle_transformers for full model)
            ("llama" | "mistral", true) => {
                Backend::Llama(LlamaBackend::load(&config_str, &shards, dtype, &device)?)
            }
            ("mixtral", _) => {
                Backend::Mixtral(MixtralBackend::load(&config_str, &shards, dtype, &device)?)
            }
            ("qwen2", _) => {
                Backend::Qwen2(Qwen2Backend::load(&config_str, &shards, dtype, &device)?)
            }
            ("qwen3", _) => {
                Backend::Qwen3(Qwen3Backend::load(&config_str, &shards, dtype, &device)?)
            }
            ("phi3", _) => Backend::Phi3(Phi3Backend::load(&config_str, &shards, dtype, &device)?),
            (other, _) => bail!(
                "Unsupported model_type: {other:?}. \
                 Supported: llama, mistral (TP-aware), mixtral, qwen2, qwen3."
            ),
        };

        // ── Load token embedding matrix for embed() ───────────────────────────
        // Try each known key name; use the first shard that contains it.
        let embed_tokens = (|| -> Option<Tensor> {
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&shards, DType::F32, &device).ok()? };
            for &key in EMBED_WEIGHT_KEYS {
                if let Ok(t) = vb.get_with_hints_dtype(
                    (meta.vocab_size, meta.hidden_size),
                    key,
                    candle_nn::Init::Const(0.0),
                    DType::F32,
                ) {
                    tracing::debug!(key, "Token embedding matrix loaded for /v1/embeddings");
                    return Some(t);
                }
            }
            tracing::debug!("No token embedding key found — /v1/embeddings will be unavailable");
            None
        })();

        tracing::info!("Weights loaded");
        Ok(Self {
            config,
            backend,
            device,
            vocab_size: meta.vocab_size,
            num_layers: meta.num_hidden_layers,
            hidden_size: meta.hidden_size,
            intermediate_size: meta.intermediate_size,
            embed_tokens,
        })
    }

    // ── GGUF fast path ───────────────────────────────────────────────────────

    fn load_gguf(config: ModelConfig, path: &Path) -> Result<Self> {
        let device = Device::Cpu;
        tracing::info!(path = %path.display(), "GGUF model detected — using quantized backend");

        let gguf = GgufLlamaBackend::load(path, &device)?;

        let vocab_size = gguf.vocab_size;
        let hidden_size = gguf.hidden_size;
        let num_layers = gguf.num_layers;
        // GGUF metadata exposes feed_forward_length; use hidden_size as fallback.
        // param_count() uses this; inaccuracy is acceptable for GGUF (it's an estimate).
        let intermediate_size = hidden_size; // conservative fallback

        tracing::info!(
            vocab  = vocab_size,
            hidden = hidden_size,
            layers = num_layers,
            "GGUF architecture"
        );

        Ok(Self {
            config,
            backend: Backend::GgufLlama(gguf),
            device,
            vocab_size,
            num_layers,
            hidden_size,
            intermediate_size,
            // Quantized embedding matrix is baked into ModelWeights;
            // mean-pooled static embeddings are not exposed via /v1/embeddings.
            embed_tokens: None,
        })
    }

    // ── Compute ───────────────────────────────────────────────────────────────

    #[allow(dead_code)]
    pub fn forward(&self, token_ids: &[u32], seq_pos: usize) -> Result<Tensor> {
        self.backend.forward(token_ids, seq_pos)
    }

    #[allow(dead_code)]
    pub fn reset_cache(&self) -> Result<()> {
        self.backend.reset_cache()
    }

    // ── Per-sequence cache API ────────────────────────────────────────────────

    /// Allocate a fresh KV cache for one sequence.
    ///
    /// Call once when a sequence is admitted; pass the returned cache to
    /// every subsequent `forward_with_cache` call for that sequence.
    pub fn create_kv_cache(&self) -> Result<PerSeqCache> {
        self.backend.create_kv_cache()
    }

    /// Run one forward step with an externally-owned per-sequence cache.
    ///
    /// Replaces `forward` + `reset_cache` in the continuous-batching worker.
    /// The cache is updated in place; pass the same instance on every step.
    pub fn forward_with_cache(
        &self,
        token_ids: &[u32],
        seq_pos: usize,
        cache: &mut PerSeqCache,
    ) -> Result<Tensor> {
        self.backend.forward_with_cache(token_ids, seq_pos, cache)
    }

    // ── Metadata ──────────────────────────────────────────────────────────────

    // ── Embeddings ────────────────────────────────────────────────────────────

    /// Compute a mean-pooled, L2-normalised embedding vector for `token_ids`.
    ///
    /// # Algorithm
    ///
    /// 1. Index `embed_tokens` with `token_ids` → `[seq_len, hidden_size]`
    /// 2. Mean-pool over the sequence dimension → `[hidden_size]`
    /// 3. L2-normalise so ‖embedding‖₂ = 1
    ///
    /// This is a *static* embedding (bag-of-tokens mean pooling of the
    /// input embedding table, no transformer forward pass).  It is fast and
    /// deterministic.  For contextual embeddings use a dedicated
    /// embedding model (E5, BGE, etc.) loaded into this server.
    ///
    /// Returns `Err` when the embedding matrix was not found during load.
    pub fn embed(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let embed_table = self
            .embed_tokens
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("embedding matrix not available for this model"))?;

        if token_ids.is_empty() {
            return Ok(vec![0.0_f32; self.hidden_size]);
        }

        // Index into [vocab_size, hidden_size] → [seq_len, hidden_size].
        let ids = Tensor::new(token_ids, &self.device)?;
        let token_embeds = embed_table.index_select(&ids, 0)?;

        // Mean-pool over seq_len → [hidden_size].
        let mean = token_embeds.mean(0)?;
        let mean_vec: Vec<f32> = mean.to_vec1()?;

        // L2-normalise.
        let norm: f32 = mean_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm = norm.max(1e-9); // avoid div-by-zero for all-zero inputs
        Ok(mean_vec.iter().map(|x| x / norm).collect())
    }

    /// Whether this engine can produce embeddings (embed_tokens was loaded).
    pub fn supports_embeddings(&self) -> bool {
        self.embed_tokens.is_some()
    }

    /// Return a cheap clone of the token embedding matrix.
    ///
    /// `candle_core::Tensor` is Arc-backed — cloning bumps a ref count, no
    /// data is copied.  Used by the server to populate `AppState` before the
    /// engine is moved into the inference worker.
    pub fn embed_tokens_clone(&self) -> Option<Tensor> {
        self.embed_tokens.clone()
    }

    // ── Metadata ──────────────────────────────────────────────────────────────

    pub fn param_count(&self) -> usize {
        let attn = 4 * self.hidden_size * self.hidden_size;
        let ffn = 3 * self.hidden_size * self.intermediate_size;
        self.vocab_size * self.hidden_size
            + self.num_layers * (attn + ffn)
            + self.vocab_size * self.hidden_size
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}
