//! Qwen2 model — vendored from candle-transformers 0.10.2 with fused RMSNorm
//! and pre-allocated KV cache for constant-shape decode (Exp 6B Phase 1).
//!
//! Changes from upstream:
//!   1. `with_tracing::RmsNorm` → `super::RmsNorm` (calls our CUDA kernel)
//!   2. Attention struct carries pre-allocated k_buf/v_buf tensors.
//!      During decode (q_len == 1) on CUDA, slot-assign replaces Tensor::cat,
//!      keeping buffer shape constant across all decode steps.
//!      This is the prerequisite for CUDA graph capture (Phase 2).

/// Maximum sequence length backed by the pre-allocated KV buffer.
/// Sequences longer than this fall back to the original cat-based path.
/// Size per layer: nkv × MAX_KV_BUF × head_dim × dtype_bytes
/// For Qwen2.5-7B (nkv=8, hd=128, f16): 8 × 2048 × 128 × 2 = 4 MB/layer.
const MAX_KV_BUF: usize = 2048;

use super::RmsNorm;
use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, Linear, VarBuilder, linear, linear_no_bias};
use candle_transformers::utils::repeat_kv;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub hidden_act: Activation,
}

// ── Rotary Embeddings ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ── MLP ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            gate_proj: linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(h, i, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(i, h, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

// ── Attention ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
    // Pre-allocated decode buffers (Exp 6B Phase 1).
    // Fixed shape [1, nkv, MAX_KV_BUF, head_dim] — never grows.
    k_buf: Tensor,
    v_buf: Tensor,
    k_buf_initialized: bool, // true once prefill cache has been copied into k_buf
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = h / nh;
        let dev = vb.device();
        let dtype = vb.dtype();
        // Pre-allocate fixed-shape KV buffers for decode.  Shape is constant regardless
        // of sequence position, which allows CUDA graph capture in Phase 2.
        let k_buf = Tensor::zeros((1, nkv, MAX_KV_BUF, hd), dtype, dev)?;
        let v_buf = Tensor::zeros((1, nkv, MAX_KV_BUF, hd), dtype, dev)?;
        Ok(Self {
            q_proj: linear(h, nh * hd, vb.pp("q_proj"))?,
            k_proj: linear(h, nkv * hd, vb.pp("k_proj"))?,
            v_proj: linear(h, nkv * hd, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(nh * hd, h, vb.pp("o_proj"))?,
            num_heads: nh,
            num_kv_heads: nkv,
            num_kv_groups: nh / nkv,
            head_dim: hd,
            hidden_size: h,
            rotary_emb,
            kv_cache: None,
            k_buf,
            v_buf,
            k_buf_initialized: false,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self
            .q_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = self
            .k_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = self
            .v_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        // ── Decode fast-path: pre-alloc slot-assign (Exp 6B Phase 1) ─────────
        // Conditions: decode step (q_len == 1), on CUDA, within buffer bounds.
        let use_preallloc = false && q_len == 1 // disabled: enable only with CUDA graph capture (Phase 2)
            && seqlen_offset + 1 <= MAX_KV_BUF
            && matches!(xs.device(), Device::Cuda(_));

        if use_preallloc {
            // One-time init: copy prefill keys into the pre-alloc buffer.
            // This allocates once per sequence (not per decode step).
            if !self.k_buf_initialized {
                if let Some((pk, pv)) = &self.kv_cache {
                    let prefill_len = pk.dim(2)?;
                    if prefill_len <= MAX_KV_BUF {
                        let rest = MAX_KV_BUF - prefill_len;
                        let zk = Tensor::zeros(
                            (1, self.num_kv_heads, rest, self.head_dim),
                            pk.dtype(),
                            pk.device(),
                        )?;
                        let zv = Tensor::zeros(
                            (1, self.num_kv_heads, rest, self.head_dim),
                            pv.dtype(),
                            pv.device(),
                        )?;
                        self.k_buf = Tensor::cat(&[pk, &zk], 2)?;
                        self.v_buf = Tensor::cat(&[pv, &zv], 2)?;
                    }
                    // If prefill > MAX_KV_BUF the buffer stays zeroed but we still
                    // set initialized=true; the bounds check above will route through
                    // the cat fallback for the rest of the sequence.
                }
                self.k_buf_initialized = true;
            }

            // Slot-assign: constant-shape output every step.
            self.k_buf =
                crate::kernels::kv_assign::assign_slot(&self.k_buf, &key_states, seqlen_offset)
                    .map_err(candle_core::Error::wrap)?;
            self.v_buf =
                crate::kernels::kv_assign::assign_slot(&self.v_buf, &value_states, seqlen_offset)
                    .map_err(candle_core::Error::wrap)?;

            let kv_len = seqlen_offset + 1;
            let k_cache = self.k_buf.narrow(2, 0, kv_len)?;
            let v_cache = self.v_buf.narrow(2, 0, kv_len)?;

            let k_cache = repeat_kv(k_cache, self.num_kv_groups)?.contiguous()?;
            let v_cache = repeat_kv(v_cache, self.num_kv_groups)?.contiguous()?;

            let orig_dtype = query_states.dtype();
            let q_f32 = query_states.to_dtype(DType::F32)?;
            let k_f32 = k_cache.to_dtype(DType::F32)?;
            let v_f32 = v_cache.to_dtype(DType::F32)?;
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (q_f32.matmul(&k_f32.transpose(2, 3)?)? * scale)?;
            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(&mask.to_dtype(DType::F32)?)?,
            };
            return candle_nn::ops::softmax_last_dim(&attn_weights)?
                .matmul(&v_f32)?
                .to_dtype(orig_dtype)?
                .transpose(1, 2)?
                .reshape((b_sz, q_len, self.hidden_size))?
                .apply(&self.o_proj);
        }

        // ── Original cat-based path (prefill or CPU or sequence > MAX_KV_BUF) ─
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((pk, pv)) => (
                Tensor::cat(&[pk, &key_states], 2)?,
                Tensor::cat(&[pv, &value_states], 2)?,
            ),
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states = repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        // Compute attention in F32 for numerical stability.
        // In F16, Q@K^T can overflow (e.g. Qwen2.5 k_proj biases up to ~414 cause
        // dot-products > 65504 → +Inf stored to F16 → NaN in softmax).
        // Casting Q, K, V to F32 before the matmul avoids this at minimal overhead.
        let orig_dtype = query_states.dtype();
        let q_f32 = query_states.to_dtype(DType::F32)?;
        let k_f32 = key_states.to_dtype(DType::F32)?;
        let v_f32 = value_states.to_dtype(DType::F32)?;

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_weights = (q_f32.matmul(&k_f32.transpose(2, 3)?)? * scale)?;
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(&mask.to_dtype(DType::F32)?)?,
        };
        candle_nn::ops::softmax_last_dim(&attn_weights)?
            .matmul(&v_f32)?
            .to_dtype(orig_dtype)?
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
        self.k_buf_initialized = false;
        // k_buf/v_buf content doesn't need zeroing — k_buf_initialized=false
        // ensures they're re-populated on the next sequence's first decode step.
    }
}

// ── Decoder Layer ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?,
            mlp: MLP::new(cfg, vb.pp("mlp"))?,
            input_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let post_ln = xs.apply(&self.post_attention_layernorm)?;
        let mlp_out = self.mlp.forward(&post_ln)?;
        residual + mlp_out
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    sliding_window: usize,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for idx in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(idx))?);
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + self.sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), self.dtype, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?;
        }
        xs.apply(&self.norm)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

// ── ModelForCausalLM ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    base_model: Model,
    lm_head: Linear,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let base_model = Model::new(cfg, vb.clone())?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            // Tied embeddings: share the embedding weight as the LM head.
            Linear::new(base_model.embed_tokens.embeddings().clone(), None)
        };
        Ok(Self {
            base_model,
            lm_head,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        self.base_model
            .forward(input_ids, seqlen_offset)?
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base_model.clear_kv_cache();
    }
}
