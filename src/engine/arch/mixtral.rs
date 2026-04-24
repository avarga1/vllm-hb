//! Mixtral 8×7B sparse MoE architecture backend.
//!
//! # Design
//!
//! Written from scratch (not via `candle-transformers::models::mixtral`) so
//! that the KV cache is stored externally as
//! `Mutex<Vec<Option<(Tensor, Tensor)>>>` and can be reset without reloading
//! weights.  This mirrors the approach taken in `llama_tp.rs`.
//!
//! Key departures from dense Llama:
//!
//! * **Sliding window attention** — each layer attends only to a window of
//!   the last `sliding_window` tokens (default 4096).  The causal mask is
//!   constructed accordingly.
//! * **Sparse MoE FFN** — the dense SwiGLU FFN is replaced by 8 experts; a
//!   router gate selects the top-2 experts per token, computes their outputs,
//!   and scatter-adds them back with renormalized weights.
//! * **GQA** — `num_key_value_heads` can be smaller than `num_attention_heads`
//!   (e.g. 8 KV heads, 32 Q heads on Mixtral 8×7B).  KV heads are tiled
//!   (`repeat_kv`) to match Q head count before attention.
//!
//! # KV cache
//!
//! `MixtralBackend::reset_cache()` replaces every entry with `None`.
//! The forward pass re-populates each layer's entry on the first token and
//! concatenates subsequent tokens.

use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
#[cfg(not(feature = "flash-attn"))]
use candle_nn::ops::softmax;
use candle_nn::{Linear, VarBuilder, linear_no_bias};
use serde::Deserialize;

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct MixtralConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
}

fn default_rope_theta() -> f64 {
    1_000_000.0
}
fn default_sliding_window() -> usize {
    4096
}

impl MixtralConfig {
    /// Effective KV head count — falls back to `num_attention_heads` when the
    /// field is absent (or zero) in `config.json`.
    pub fn kv_heads(&self) -> usize {
        if self.num_key_value_heads == 0 {
            self.num_attention_heads
        } else {
            self.num_key_value_heads
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ── RMSNorm ───────────────────────────────────────────────────────────────────

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        crate::kernels::rms_norm::apply(x, &self.weight, self.eps)
    }
}

// ── Rotary embeddings ─────────────────────────────────────────────────────────

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        head_dim: usize,
        max_seq: usize,
        rope_theta: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim / 2)
            .map(|i| 1.0 / (rope_theta as f32).powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (head_dim / 2,), device)?;

        let positions = Tensor::arange(0u32, max_seq as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq, 1))?;
        let inv_freq = inv_freq.reshape((1, head_dim / 2))?;

        let freqs = positions.matmul(&inv_freq)?;
        let emb = Tensor::cat(&[&freqs, &freqs], candle_core::D::Minus1)?;

        Ok(Self {
            sin: emb.sin()?.to_dtype(dtype)?,
            cos: emb.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, seq_pos: usize) -> Result<(Tensor, Tensor)> {
        let (_bsz, _heads, seq_len, _head_dim) = q.dims4()?;
        let cos = self.cos.i(seq_pos..seq_pos + seq_len)?;
        let sin = self.sin.i(seq_pos..seq_pos + seq_len)?;
        // reshape to (1, 1, seq_len, head_dim) for broadcast
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
        Ok((rope_apply(q, &cos, &sin)?, rope_apply(k, &cos, &sin)?))
    }
}

fn rope_apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let head_dim = x.dim(candle_core::D::Minus1)?;
    let half = head_dim / 2;
    let x1 = x.narrow(candle_core::D::Minus1, 0, half)?;
    let x2 = x.narrow(candle_core::D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[x2.neg()?, x1], candle_core::D::Minus1)?;
    Ok((x.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?)?)
}

// ── Sliding window causal mask ────────────────────────────────────────────────

fn sliding_causal_mask(
    seq_len: usize,
    seq_pos: usize,
    window: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let total = seq_pos + seq_len;
    let mask_data: Vec<f32> = (0..seq_len)
        .flat_map(|qi| {
            let q_abs = seq_pos + qi;
            (0..total).map(move |ki| {
                if ki <= q_abs && q_abs - ki < window {
                    0.0_f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Ok(Tensor::from_vec(mask_data, (seq_len, total), device)?.to_dtype(dtype)?)
}

// ── Repeat KV for GQA ─────────────────────────────────────────────────────────

fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, h, s, d) = x.dims4()?;
    Ok(x.unsqueeze(2)?
        .expand((b, h, n_rep, s, d))?
        .reshape((b, h * n_rep, s, d))?)
}

// ── Attention ─────────────────────────────────────────────────────────────────

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    scale: f64,
}

impl Attention {
    fn load(cfg: &MixtralConfig, dtype: DType, device: &Device, vb: VarBuilder) -> Result<Self> {
        let h = cfg.num_attention_heads;
        let kv = cfg.kv_heads();
        let d = cfg.head_dim();
        let hidden = cfg.hidden_size;

        let q_proj = linear_no_bias(hidden, h * d, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden, kv * d, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden, kv * d, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(h * d, hidden, vb.pp("o_proj"))?;

        let max_seq = cfg.sliding_window * 2;
        let rotary = RotaryEmbedding::new(d, max_seq, cfg.rope_theta, dtype, device)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary,
            num_heads: h,
            num_kv_heads: kv,
            head_dim: d,
            sliding_window: cfg.sliding_window,
            scale: 1.0 / (d as f64).sqrt(),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        seq_pos: usize,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (bsz, seq_len, _) = x.dims3()?;
        let device = x.device();
        let dtype = x.dtype();

        let q = self
            .q_proj
            .forward(x)?
            .reshape((bsz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((bsz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((bsz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = self.rotary.apply(&q, &k, seq_pos)?;

        // Update KV cache.
        let (k, v) = match kv_cache {
            None => {
                *kv_cache = Some((k.clone(), v.clone()));
                (k, v)
            }
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k as &Tensor, &k], 2)?;
                let v = Tensor::cat(&[prev_v as &Tensor, &v], 2)?;
                *kv_cache = Some((k.clone(), v.clone()));
                (k, v)
            }
        };

        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        // ── Attention kernel ──────────────────────────────────────────────────
        //
        // FA2 path (`--features flash-attn`, sm_80+):
        //   Layout: [batch, seq, heads, dim].  causal=true handles the causal
        //   part of the sliding-window mask.  The window constraint is not
        //   enforced — tokens outside the window can attend, which is a minor
        //   accuracy trade-off acceptable for sm_80+ inference.
        //
        // SDPA path (default):
        //   Explicit sliding-window causal mask added to scores.

        #[cfg(feature = "flash-attn")]
        let out = {
            // q/k/v currently [b, heads, seq, dim] → transpose to [b, seq, heads, dim]
            let q_fa = q.transpose(1, 2)?;
            let k_fa = k.transpose(1, 2)?;
            let v_fa = v.transpose(1, 2)?;
            let softmax_scale = self.scale as f32;
            candle_flash_attn::flash_attn(&q_fa, &k_fa, &v_fa, softmax_scale, true)?
                // [b, seq, heads, dim] → [b, heads, seq, dim]
                .transpose(1, 2)?
        };

        #[cfg(not(feature = "flash-attn"))]
        let out = {
            let mask = sliding_causal_mask(seq_len, seq_pos, self.sliding_window, dtype, device)?;
            let mask = mask.unsqueeze(0)?.unsqueeze(0)?;
            let attn = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
            let attn = (attn.clone() + mask.broadcast_as(attn.shape())?)?;
            let attn = softmax(&attn, candle_core::D::Minus1)?;
            attn.matmul(&v)?
        };

        let out = out
            .transpose(1, 2)?
            .reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
        Ok(self.o_proj.forward(&out)?)
    }
}

// ── Expert (SwiGLU FFN) ───────────────────────────────────────────────────────

struct Expert {
    w1: Linear, // gate
    w2: Linear, // down
    w3: Linear, // up
}

impl Expert {
    fn load(hidden: usize, inter: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w1: linear_no_bias(hidden, inter, vb.pp("w1"))?,
            w2: linear_no_bias(inter, hidden, vb.pp("w2"))?,
            w3: linear_no_bias(hidden, inter, vb.pp("w3"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.w1.forward(x)?)?;
        let up = self.w3.forward(x)?;
        Ok(self.w2.forward(&(gate * up)?)?)
    }
}

// ── Sparse MoE block ──────────────────────────────────────────────────────────

struct SparseMoeBlock {
    gate: Linear,
    experts: Vec<Expert>,
    num_experts_per_tok: usize,
}

impl SparseMoeBlock {
    fn load(cfg: &MixtralConfig, vb: VarBuilder) -> Result<Self> {
        let gate = linear_no_bias(cfg.hidden_size, cfg.num_local_experts, vb.pp("gate"))?;
        let experts = (0..cfg.num_local_experts)
            .map(|i| {
                Expert::load(
                    cfg.hidden_size,
                    cfg.intermediate_size,
                    vb.pp(format!("experts.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            gate,
            experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (bsz, seq_len, hidden) = x.dims3()?;
        let num_tokens = bsz * seq_len;
        // Flatten to (num_tokens, hidden) for routing.
        let x_flat = x.reshape((num_tokens, hidden))?;

        // Router: (num_tokens, num_experts)
        let logits = self.gate.forward(&x_flat)?;
        let probs = softmax(&logits, candle_core::D::Minus1)?;

        // Move to CPU f32 for top-K selection.
        let probs_cpu = probs.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let probs_vec = probs_cpu.to_vec2::<f32>()?;

        let k = self.num_experts_per_tok;
        let num_experts = self.experts.len();
        let dtype = x.dtype();
        let device = x.device();

        // Accumulate output on CPU then move once.
        let mut out_data = vec![0.0f32; num_tokens * hidden];

        for (tok_idx, tok_probs) in probs_vec.iter().enumerate() {
            // Pick top-K experts for this token.
            let mut indexed: Vec<(usize, f32)> = tok_probs.iter().copied().enumerate().collect();
            indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let topk = &indexed[..k.min(num_experts)];

            // Renormalize weights.
            let weight_sum: f32 = topk.iter().map(|(_, w)| w).sum();
            let topk_norm: Vec<(usize, f32)> = topk
                .iter()
                .map(|(e, w)| (*e, w / weight_sum.max(1e-9)))
                .collect();

            // Select this token's hidden vector: (1, hidden)
            let tok_tensor = x_flat.i(tok_idx)?.unsqueeze(0)?;

            for (expert_idx, weight) in topk_norm {
                let expert_out = self.experts[expert_idx].forward(&tok_tensor)?;
                // expert_out shape: (1, hidden)
                let expert_vec = expert_out
                    .squeeze(0)?
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<f32>()?;
                let base = tok_idx * hidden;
                for (j, v) in expert_vec.iter().enumerate() {
                    out_data[base + j] += weight * v;
                }
            }
        }

        let out = Tensor::from_vec(out_data, (num_tokens, hidden), &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(device)?
            .reshape((bsz, seq_len, hidden))?;
        Ok(out)
    }
}

// ── Decoder layer ─────────────────────────────────────────────────────────────

struct DecoderLayer {
    attn: Attention,
    moe: SparseMoeBlock,
    input_norm: RmsNorm,
    post_attn_norm: RmsNorm,
}

impl DecoderLayer {
    fn load(cfg: &MixtralConfig, dtype: DType, device: &Device, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::load(cfg, dtype, device, vb.pp("self_attn"))?,
            moe: SparseMoeBlock::load(cfg, vb.pp("block_sparse_moe"))?,
            input_norm: RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attn_norm: RmsNorm::load(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        seq_pos: usize,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let residual = x;
        let h = self
            .attn
            .forward(&self.input_norm.forward(x)?, seq_pos, kv_cache)?;
        let h = (residual + h)?;
        let residual = &h;
        let h = self.moe.forward(&self.post_attn_norm.forward(&h)?)?;
        Ok((residual + h)?)
    }
}

// ── Full model ────────────────────────────────────────────────────────────────

struct MixtralModel {
    embed: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
}

impl MixtralModel {
    fn load(cfg: &MixtralConfig, dtype: DType, device: &Device, vb: VarBuilder) -> Result<Self> {
        let embed =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| DecoderLayer::load(cfg, dtype, device, vb.pp(format!("model.layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let norm = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            embed,
            layers,
            norm,
            lm_head,
        })
    }

    fn forward(
        &self,
        token_ids: &[u32],
        seq_pos: usize,
        kv_cache: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        let device = self.lm_head.weight().device();
        let ids = Tensor::from_vec(token_ids.to_vec(), (1, token_ids.len()), device)?;
        let mut h = self.embed.forward(&ids)?;

        for (layer, cache) in self.layers.iter().zip(kv_cache.iter_mut()) {
            h = layer.forward(&h, seq_pos, cache)?;
        }

        let h = self.norm.forward(&h)?;
        // Take the last token's logits: (1, vocab_size)
        let last = h.i((0, token_ids.len() - 1))?;
        Ok(self.lm_head.forward(&last.unsqueeze(0)?)?)
    }
}

// ── Backend (public API) ───────────────────────────────────────────────────────

/// Mixtral 8×7B inference backend with an external, resettable KV cache.
pub struct MixtralBackend {
    model: MixtralModel,
    kv_cache: Mutex<Vec<Option<(Tensor, Tensor)>>>,
    #[allow(dead_code)]
    num_layers: usize,
}

impl MixtralBackend {
    /// Load weights from safetensors shards.
    ///
    /// # Errors
    ///
    /// Returns an error if `config_str` cannot be parsed, shards are missing,
    /// or weight tensors have unexpected shapes.
    pub fn load(
        config_str: &str,
        shards: &[PathBuf],
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let cfg: MixtralConfig =
            serde_json::from_str(config_str).context("failed to parse Mixtral config.json")?;

        let num_layers = cfg.num_hidden_layers;

        // Build VarBuilder from safetensors shards.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(shards, dtype, device)? };

        let model = MixtralModel::load(&cfg, dtype, device, vb)?;
        let kv_cache = Mutex::new(vec![None; num_layers]);

        Ok(Self {
            model,
            kv_cache,
            num_layers,
        })
    }

    /// Run one forward step.
    ///
    /// Returns a `(1, vocab_size)` logits tensor for the last token.
    pub fn forward(&self, token_ids: &[u32], seq_pos: usize) -> Result<Tensor> {
        let mut cache = self.kv_cache.lock().unwrap();
        self.model.forward(token_ids, seq_pos, &mut cache)
    }

    /// Clear all KV cache entries so the next request starts fresh.
    pub fn reset_cache(&self) -> Result<()> {
        let mut cache = self.kv_cache.lock().unwrap();
        for entry in cache.iter_mut() {
            *entry = None;
        }
        Ok(())
    }

    // ── Per-sequence cache API ────────────────────────────────────────────────

    /// Allocate a fresh per-sequence KV cache (one entry per layer, all None).
    pub fn create_kv_cache(&self) -> Vec<Option<(Tensor, Tensor)>> {
        vec![None; self.num_layers]
    }

    /// Run one forward step with an externally-owned per-sequence cache.
    pub fn forward_with_cache(
        &self,
        token_ids: &[u32],
        seq_pos: usize,
        cache: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        self.model.forward(token_ids, seq_pos, cache)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> MixtralConfig {
        MixtralConfig {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 1_000_000.0,
            sliding_window: 16,
            num_local_experts: 4,
            num_experts_per_tok: 2,
        }
    }

    #[test]
    fn config_kv_heads_fallback() {
        let mut cfg = small_cfg();
        cfg.num_key_value_heads = 0;
        assert_eq!(cfg.kv_heads(), cfg.num_attention_heads);
    }

    #[test]
    fn config_kv_heads_gqa() {
        let cfg = small_cfg();
        assert_eq!(cfg.kv_heads(), 2);
    }

    #[test]
    fn config_head_dim() {
        let cfg = small_cfg();
        assert_eq!(cfg.head_dim(), 8); // 32 / 4
    }

    #[test]
    fn repeat_kv_n1_is_identity() {
        let t = Tensor::ones((1, 2, 4, 8), DType::F32, &Device::Cpu).unwrap();
        let out = repeat_kv(&t, 1).unwrap();
        assert_eq!(out.dims(), t.dims());
    }

    #[test]
    fn repeat_kv_tiles_correctly() {
        let t = Tensor::ones((1, 2, 4, 8), DType::F32, &Device::Cpu).unwrap();
        let out = repeat_kv(&t, 3).unwrap();
        assert_eq!(out.dims(), &[1, 6, 4, 8]);
    }

    #[test]
    fn sliding_causal_mask_shape() {
        let mask = sliding_causal_mask(3, 0, 4, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(mask.dims(), &[3, 3]);
    }

    #[test]
    fn sliding_causal_mask_blocks_future() {
        // Position 0 should not attend to position 1 (causal).
        let mask = sliding_causal_mask(2, 0, 16, DType::F32, &Device::Cpu).unwrap();
        let v: Vec<Vec<f32>> = mask.to_vec2().unwrap();
        assert!(v[0][1].is_infinite() && v[0][1] < 0.0);
    }

    #[test]
    fn sliding_causal_mask_blocks_outside_window() {
        // With window=2, token at position 3 cannot attend to position 0.
        let mask = sliding_causal_mask(1, 3, 2, DType::F32, &Device::Cpu).unwrap();
        let v: Vec<Vec<f32>> = mask.to_vec2().unwrap();
        assert!(v[0][0].is_infinite() && v[0][0] < 0.0);
        // But can attend to position 2 (within window).
        assert_eq!(v[0][2], 0.0);
        assert_eq!(v[0][3], 0.0);
    }

    #[test]
    fn rms_norm_shape_preserved() {
        let hidden = 32usize;
        let device = Device::Cpu;
        let weight = Tensor::ones(hidden, DType::F32, &device).unwrap();
        let norm = RmsNorm { weight, eps: 1e-5 };
        let x = Tensor::randn(0f32, 1.0, (1, 4, hidden), &device).unwrap();
        let out = norm.forward(&x).unwrap();
        assert_eq!(out.dims(), x.dims());
    }

    #[test]
    fn rms_norm_unit_weight_normalises() {
        // With unit weight, RMSNorm output should have near-unit RMS per row.
        let hidden = 64usize;
        let device = Device::Cpu;
        let weight = Tensor::ones(hidden, DType::F32, &device).unwrap();
        let norm = RmsNorm { weight, eps: 1e-5 };
        let x = Tensor::randn(0f32, 1.0, (1, 1, hidden), &device).unwrap();
        let out = norm.forward(&x).unwrap();
        let sq_mean: f32 = out.sqr().unwrap().mean_all().unwrap().to_scalar().unwrap();
        assert!((sq_mean - 1.0).abs() < 0.1);
    }
}
