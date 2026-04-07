//! Tensor-parallel Llama backend.
//!
//! Replaces the single-GPU `LlamaBackend` when `--tensor-parallel-size N > 1`.
//! Implements the full Llama forward pass with column/row-parallel linears so
//! each GPU holds only `1/N` of the weight parameters.
//!
//! # Weight sharding
//!
//! ```text
//! Layer weights        Strategy          Each rank holds
//! ─────────────────    ──────────────    ────────────────────────────────
//! q/k/v_proj           column-parallel   [heads/N * head_dim, hidden]
//! o_proj               row-parallel      [hidden, heads/N * head_dim]
//! gate_proj / up_proj  column-parallel   [intermediate/N, hidden]
//! down_proj            row-parallel      [hidden, intermediate/N]
//! embed_tokens         replicated        [vocab, hidden]
//! input/post norms     replicated        [hidden]
//! lm_head              replicated        [vocab, hidden]
//! ```
//!
//! # Sync points per layer
//!
//! ```text
//! hidden (same on all ranks)
//!   → attn_partial()   [GPU-parallel matmuls, local attention on head shard]
//!   → all_reduce       [host-mediated sum → identical hidden on all ranks]
//!   → ffn_partial()    [GPU-parallel gate/up/down matmuls]
//!   → all_reduce       [host-mediated sum → identical hidden on all ranks]
//! ```
//!
//! Because CUDA kernel launches are async, both `attn_partial()` calls hit
//! their GPUs before the host blocks on the first `to_device(cpu)` in
//! `all_reduce`.  Both GPUs compute in parallel; transfers serialise only
//! the host-side reduce step.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::ops::softmax;
use parking_lot::Mutex;
use rayon::prelude::*;
use serde::Deserialize;

use crate::parallel::{TpWorld, column_shard, row_shard};

// ── Config ────────────────────────────────────────────────────────────────────

/// Subset of HuggingFace `config.json` needed for TP Llama.
#[derive(Deserialize, Debug, Clone)]
pub struct LlamaTpConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: usize, // 0 → same as num_attention_heads (MHA)
    #[allow(dead_code)]
    pub vocab_size: usize,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
}

fn default_eps() -> f64 {
    1e-5
}
fn default_rope_theta() -> f64 {
    10_000.0
}

impl LlamaTpConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    /// Effective KV head count (handles models that omit the field = full MHA).
    pub fn kv_heads(&self) -> usize {
        if self.num_key_value_heads == 0 {
            self.num_attention_heads
        } else {
            self.num_key_value_heads
        }
    }
}

// ── RMSNorm ───────────────────────────────────────────────────────────────────

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let rms = (x_f32.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?
            .sqrt()?
            .recip()?;
        let normed = x_f32.broadcast_mul(&rms)?;
        let out = normed
            .to_dtype(orig_dtype)?
            .broadcast_mul(&self.weight.to_dtype(orig_dtype)?)?;
        Ok(out)
    }
}

// ── Rotary Embeddings ─────────────────────────────────────────────────────────

/// Apply RoPE to query and key tensors.
///
/// `q`, `k`: `[seq, n_heads, head_dim]`
/// `seq_pos`: absolute position of the first token in this forward call.
fn apply_rope(q: &Tensor, k: &Tensor, seq_pos: usize, rope_theta: f64) -> Result<(Tensor, Tensor)> {
    let (seq, _nq, head_dim) = q.dims3()?;
    let half = head_dim / 2;
    let device = q.device();
    let dtype = q.dtype();

    // inv_freq: [half]
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0f32 / (rope_theta as f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, half, &Device::Cpu)?.to_device(device)?;

    // positions: [seq]
    let positions: Vec<f32> = (seq_pos..seq_pos + seq).map(|p| p as f32).collect();
    let positions = Tensor::from_vec(positions, seq, &Device::Cpu)?.to_device(device)?;

    // freqs: [seq, half]  (outer product via broadcast)
    let freqs = positions
        .unsqueeze(1)?
        .broadcast_mul(&inv_freq.unsqueeze(0)?)?;
    let cos = freqs.cos()?.to_dtype(dtype)?; // [seq, half]
    let sin = freqs.sin()?.to_dtype(dtype)?;

    let rotate = |x: &Tensor| -> Result<Tensor> {
        let (s, nh, hd) = x.dims3()?;
        let x1 = x.narrow(2, 0, hd / 2)?; // [s, nh, half]
        let x2 = x.narrow(2, hd / 2, hd / 2)?;
        // Expand cos/sin from [s, half] → [s, 1, half] → broadcast [s, nh, half]
        let c = cos.unsqueeze(1)?.broadcast_as((s, nh, hd / 2))?;
        let si = sin.unsqueeze(1)?.broadcast_as((s, nh, hd / 2))?;
        Ok(Tensor::cat(
            &[
                (&x1.broadcast_mul(&c)? - &x2.broadcast_mul(&si)?)?,
                (&x1.broadcast_mul(&si)? + &x2.broadcast_mul(&c)?)?,
            ],
            2,
        )?)
    };

    Ok((rotate(q)?, rotate(k)?))
}

// ── Causal mask ───────────────────────────────────────────────────────────────

/// Build an additive causal mask `[sq, sk]`.
///
/// Entries are `0` where the query can attend (j ≤ absolute query position)
/// and `-inf` for future positions.  Handles KV-cache offsets correctly.
fn causal_mask(
    sq: usize,
    sk: usize,
    seq_pos: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    // q_abs[i] = seq_pos + i,  k_abs[j] = j
    let q_abs = Tensor::arange(seq_pos as u32, (seq_pos + sq) as u32, &Device::Cpu)?
        .to_device(device)?
        .reshape((sq, 1))?
        .broadcast_as((sq, sk))?;
    let k_abs = Tensor::arange(0u32, sk as u32, &Device::Cpu)?
        .to_device(device)?
        .reshape((1, sk))?
        .broadcast_as((sq, sk))?;
    let neg_inf = Tensor::full(f32::NEG_INFINITY, (sq, sk), device)?.to_dtype(dtype)?;
    let zeros = Tensor::zeros((sq, sk), dtype, device)?;
    Ok(k_abs.gt(&q_abs)?.where_cond(&neg_inf, &zeros)?)
}

// ── Layer weights (per rank) ──────────────────────────────────────────────────

struct TpLayerWeights {
    // Column-parallel (each rank: output_dim/N rows)
    q_proj: Tensor,    // [local_heads * head_dim, hidden]
    k_proj: Tensor,    // [local_kv_heads * head_dim, hidden]
    v_proj: Tensor,    // [local_kv_heads * head_dim, hidden]
    gate_proj: Tensor, // [inter/N, hidden]
    up_proj: Tensor,   // [inter/N, hidden]
    // Row-parallel (each rank: input_dim/N columns)
    o_proj: Tensor,    // [hidden, local_heads * head_dim]
    down_proj: Tensor, // [hidden, inter/N]
    // Replicated norms
    input_norm: RmsNorm,
    post_attn_norm: RmsNorm,
}

// ── Per-rank state ────────────────────────────────────────────────────────────

struct RankState {
    device: Device,
    layers: Vec<TpLayerWeights>,
    embed_tokens: Tensor, // [vocab, hidden]
    final_norm: RmsNorm,
    lm_head: Tensor, // [vocab, hidden]
    /// KV cache per layer: (keys, values) each `[seq, local_kv_heads, head_dim]`
    kv_cache: Mutex<Vec<(Option<Tensor>, Option<Tensor>)>>,
    local_heads: usize,
    local_kv_heads: usize,
}

impl RankState {
    fn embed(&self, token_ids: &[u32]) -> Result<Tensor> {
        let ids = Tensor::from_vec(token_ids.to_vec(), token_ids.len(), &self.device)?;
        Ok(self.embed_tokens.index_select(&ids, 0)?) // [seq, hidden]
    }

    /// Compute the row-parallel attention partial for this rank.
    /// Returns `[seq, hidden]` (partial — must be all-reduced with other ranks).
    fn attn_partial(
        &self,
        x: &Tensor, // [seq, hidden]  (output of input_layernorm)
        layer: usize,
        seq_pos: usize,
        cfg: &LlamaTpConfig,
    ) -> Result<Tensor> {
        let w = &self.layers[layer];
        let (seq, _) = x.dims2()?;
        let head_dim = cfg.head_dim();

        // Sharded projections
        let q = x.matmul(&w.q_proj.t()?)?; // [seq, local_heads * head_dim]
        let k = x.matmul(&w.k_proj.t()?)?; // [seq, local_kv_heads * head_dim]
        let v = x.matmul(&w.v_proj.t()?)?;

        // [seq, heads, head_dim]
        let q = q.reshape((seq, self.local_heads, head_dim))?;
        let k = k.reshape((seq, self.local_kv_heads, head_dim))?;
        let v = v.reshape((seq, self.local_kv_heads, head_dim))?;

        // RoPE
        let (q, k) = apply_rope(&q, &k, seq_pos, cfg.rope_theta)?;

        // KV cache append
        let (k, v) = {
            let mut cache = self.kv_cache.lock();
            let (k_cache, v_cache) = &mut cache[layer];
            let k = match k_cache {
                Some(prev) => Tensor::cat(&[prev.clone(), k], 0)?,
                None => k,
            };
            let v = match v_cache {
                Some(prev) => Tensor::cat(&[prev.clone(), v], 0)?,
                None => v,
            };
            *k_cache = Some(k.clone());
            *v_cache = Some(v.clone());
            (k, v)
        };

        let sk = k.dim(0)?; // total sequence length (with cache)

        // GQA: expand KV heads if local_kv_heads < local_heads
        let k = if self.local_kv_heads < self.local_heads {
            let n_rep = self.local_heads / self.local_kv_heads;
            k.unsqueeze(2)? // [sk, kv, 1, dim]
                .expand((sk, self.local_kv_heads, n_rep, head_dim))?
                .reshape((sk, self.local_heads, head_dim))?
        } else {
            k
        };
        let v = if self.local_kv_heads < self.local_heads {
            let n_rep = self.local_heads / self.local_kv_heads;
            v.unsqueeze(2)?
                .expand((sk, self.local_kv_heads, n_rep, head_dim))?
                .reshape((sk, self.local_heads, head_dim))?
        } else {
            v
        };

        // Attention scores — work in [heads, seq, dim] layout for matmul
        // q: [sq, heads, dim] → [heads, sq, dim]
        // k: [sk, heads, dim] → [heads, dim, sk]  (transposed for scores)
        let q_t = q.transpose(0, 1)?; // [heads, sq, dim]
        let k_t = k.transpose(0, 1)?.transpose(1, 2)?; // [heads, dim, sk]
        let v_t = v.transpose(0, 1)?; // [heads, sk, dim]

        let scale = (head_dim as f64).sqrt();
        let mut scores = (q_t.matmul(&k_t)? / scale)?; // [heads, sq, sk]

        // Causal mask (prefill only — decode has sq=1, no masking needed)
        if seq > 1 {
            let mask = causal_mask(seq, sk, seq_pos, &self.device, scores.dtype())?;
            // mask [sq, sk] → broadcast to [heads, sq, sk]
            scores = scores.broadcast_add(&mask.unsqueeze(0)?)?;
        }

        let probs = softmax(&scores, 2)?; // [heads, sq, sk]
        let attn_out = probs.matmul(&v_t)?; // [heads, sq, dim]
        let attn_out = attn_out
            .transpose(0, 1)? // [sq, heads, dim]
            .contiguous()?
            .reshape((seq, self.local_heads * head_dim))?; // [sq, heads*dim]

        // Row-parallel output projection (partial result)
        let partial = attn_out.matmul(&w.o_proj.t()?)?; // [sq, hidden]
        Ok(partial)
    }

    /// Compute the row-parallel FFN partial for this rank.
    /// Returns `[seq, hidden]` (partial — must be all-reduced with other ranks).
    fn ffn_partial(&self, x: &Tensor, layer: usize) -> Result<Tensor> {
        let w = &self.layers[layer];
        let gate = candle_nn::ops::silu(&x.matmul(&w.gate_proj.t()?)?)?;
        let up = x.matmul(&w.up_proj.t()?)?;
        let partial = (gate * up)?.matmul(&w.down_proj.t()?)?; // [seq, hidden]
        Ok(partial)
    }

    fn input_norm(&self, x: &Tensor, layer: usize) -> Result<Tensor> {
        self.layers[layer].input_norm.forward(x)
    }

    fn post_attn_norm(&self, x: &Tensor, layer: usize) -> Result<Tensor> {
        self.layers[layer].post_attn_norm.forward(x)
    }

    fn final_forward(&self, x: &Tensor) -> Result<Tensor> {
        let (seq, _) = x.dims2()?;
        let normed = self.final_norm.forward(x)?;
        // Take only the last token position for logits
        let last = normed.narrow(0, seq - 1, 1)?; // [1, hidden]
        let logits = last.matmul(&self.lm_head.t()?)?; // [1, vocab]
        Ok(logits.squeeze(0)?) // [vocab]
    }

    fn reset_kv_cache(&self, num_layers: usize) {
        let mut cache = self.kv_cache.lock();
        *cache = vec![(None, None); num_layers];
    }
}

// ── Weight loading ────────────────────────────────────────────────────────────

/// Merge all safetensors shards into one CPU tensor map.
fn load_all_weights(shards: &[PathBuf], dtype: DType) -> Result<HashMap<String, Tensor>> {
    let mut out: HashMap<String, Tensor> = HashMap::new();
    for shard in shards {
        let loaded = candle_core::safetensors::load(shard, &Device::Cpu)
            .with_context(|| format!("Loading {}", shard.display()))?;
        for (name, tensor) in loaded {
            out.insert(name, tensor.to_dtype(dtype)?);
        }
    }
    Ok(out)
}

fn get(weights: &HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    weights
        .get(name)
        .cloned()
        .with_context(|| format!("Weight not found: {name}"))
}

// ── TpLlamaBackend ────────────────────────────────────────────────────────────

pub struct TpLlamaBackend {
    ranks: Vec<RankState>,
    world: TpWorld,
    cfg: LlamaTpConfig,
}

impl TpLlamaBackend {
    pub fn load(
        config_str: &str,
        shards: &[PathBuf],
        dtype: DType,
        world: TpWorld,
    ) -> Result<Self> {
        let cfg: LlamaTpConfig =
            serde_json::from_str(config_str).context("Parsing LlamaTpConfig")?;

        let n = world.world_size();

        // Validate divisibility
        if !cfg.num_attention_heads.is_multiple_of(n) {
            bail!(
                "num_attention_heads ({}) must be divisible by tensor_parallel_size ({})",
                cfg.num_attention_heads,
                n
            );
        }
        if !cfg.kv_heads().is_multiple_of(n) {
            bail!(
                "num_key_value_heads ({}) must be divisible by tensor_parallel_size ({})",
                cfg.kv_heads(),
                n
            );
        }
        if !cfg.intermediate_size.is_multiple_of(n) {
            bail!(
                "intermediate_size ({}) must be divisible by tensor_parallel_size ({})",
                cfg.intermediate_size,
                n
            );
        }

        let local_heads = cfg.num_attention_heads / n;
        let local_kv_heads = cfg.kv_heads() / n;

        tracing::info!(
            world_size = n,
            local_heads,
            local_kv_heads,
            "Loading TP Llama weights (all shards → CPU)"
        );

        // Load all weights on CPU once, then shard
        let all_weights = load_all_weights(shards, dtype)?;

        let ranks: Result<Vec<RankState>> = (0..n)
            .map(|rank| {
                let device = world.device(rank).clone();

                // Build per-layer weights for this rank
                let layers: Result<Vec<TpLayerWeights>> = (0..cfg.num_hidden_layers)
                    .map(|l| {
                        let pf = format!("model.layers.{l}");

                        let q_full = get(&all_weights, &format!("{pf}.self_attn.q_proj.weight"))?;
                        let k_full = get(&all_weights, &format!("{pf}.self_attn.k_proj.weight"))?;
                        let v_full = get(&all_weights, &format!("{pf}.self_attn.v_proj.weight"))?;
                        let o_full = get(&all_weights, &format!("{pf}.self_attn.o_proj.weight"))?;
                        let gate_full = get(&all_weights, &format!("{pf}.mlp.gate_proj.weight"))?;
                        let up_full = get(&all_weights, &format!("{pf}.mlp.up_proj.weight"))?;
                        let down_full = get(&all_weights, &format!("{pf}.mlp.down_proj.weight"))?;
                        let in_norm = get(&all_weights, &format!("{pf}.input_layernorm.weight"))?;
                        let post_norm = get(
                            &all_weights,
                            &format!("{pf}.post_attention_layernorm.weight"),
                        )?;

                        Ok(TpLayerWeights {
                            q_proj: column_shard(&q_full, rank, n)?.to_device(&device)?,
                            k_proj: column_shard(&k_full, rank, n)?.to_device(&device)?,
                            v_proj: column_shard(&v_full, rank, n)?.to_device(&device)?,
                            o_proj: row_shard(&o_full, rank, n)?.to_device(&device)?,
                            gate_proj: column_shard(&gate_full, rank, n)?.to_device(&device)?,
                            up_proj: column_shard(&up_full, rank, n)?.to_device(&device)?,
                            down_proj: row_shard(&down_full, rank, n)?.to_device(&device)?,
                            input_norm: RmsNorm::new(in_norm.to_device(&device)?, cfg.rms_norm_eps),
                            post_attn_norm: RmsNorm::new(
                                post_norm.to_device(&device)?,
                                cfg.rms_norm_eps,
                            ),
                        })
                    })
                    .collect();

                Ok(RankState {
                    embed_tokens: get(&all_weights, "model.embed_tokens.weight")?
                        .to_device(&device)?,
                    final_norm: RmsNorm::new(
                        get(&all_weights, "model.norm.weight")?.to_device(&device)?,
                        cfg.rms_norm_eps,
                    ),
                    lm_head: get(&all_weights, "lm_head.weight")?.to_device(&device)?,
                    kv_cache: Mutex::new(vec![(None, None); cfg.num_hidden_layers]),
                    device,
                    layers: layers?,
                    local_heads,
                    local_kv_heads,
                })
            })
            .collect();

        tracing::info!("TP weights distributed across {} GPU(s)", n);
        Ok(Self {
            ranks: ranks?,
            world,
            cfg,
        })
    }

    pub fn forward(&self, token_ids: &[u32], seq_pos: usize) -> Result<Tensor> {
        // ── Embedding (replicated weight, same result on all ranks) ───────────
        let mut hidden: Vec<Tensor> = self
            .ranks
            .iter()
            .map(|r| r.embed(token_ids))
            .collect::<Result<_>>()?;

        // ── Transformer layers ────────────────────────────────────────────────
        for layer in 0..self.cfg.num_hidden_layers {
            // Input norm (replicated — same result on all ranks)
            let normed: Vec<Tensor> = self
                .ranks
                .iter()
                .zip(hidden.iter())
                .map(|(r, h)| r.input_norm(h, layer))
                .collect::<Result<_>>()?;

            // Attention partials — issued to all GPUs; CUDA runs them in parallel
            let attn_partials: Vec<Tensor> = self
                .ranks
                .par_iter()
                .zip(normed.par_iter())
                .map(|(r, x)| r.attn_partial(x, layer, seq_pos, &self.cfg))
                .collect::<Result<_>>()?;

            // All-reduce attention → identical on all ranks
            let attn_out = self.world.all_reduce(attn_partials)?;

            // Residual + post-attn norm
            let post_normed: Vec<Tensor> = self
                .ranks
                .iter()
                .zip(hidden.iter())
                .map(|(r, h)| {
                    let x = (h + attn_out.to_device(&r.device)?)?;
                    r.post_attn_norm(&x, layer)
                })
                .collect::<Result<_>>()?;

            // Update hidden with residual (before FFN)
            let after_attn: Vec<Tensor> = self
                .ranks
                .iter()
                .zip(hidden.iter())
                .map(|(r, h)| -> Result<Tensor> { Ok((h + attn_out.to_device(&r.device)?)?) })
                .collect::<Result<_>>()?;

            // FFN partials — parallel GPU compute
            let ffn_partials: Vec<Tensor> = self
                .ranks
                .par_iter()
                .zip(post_normed.par_iter())
                .map(|(r, x)| r.ffn_partial(x, layer))
                .collect::<Result<_>>()?;

            // All-reduce FFN → identical on all ranks
            let ffn_out = self.world.all_reduce(ffn_partials)?;

            // Residual
            hidden = self
                .ranks
                .iter()
                .zip(after_attn.iter())
                .map(|(r, h)| -> Result<Tensor> { Ok((h + ffn_out.to_device(&r.device)?)?) })
                .collect::<Result<_>>()?;
        }

        // ── Final norm + LM head (rank 0 has replicated weights) ─────────────
        self.ranks[0].final_forward(&hidden[0])
    }

    pub fn reset_cache(&self) -> Result<()> {
        let n = self.cfg.num_hidden_layers;
        for r in &self.ranks {
            r.reset_kv_cache(n);
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn world_size(&self) -> usize {
        self.world.world_size()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RMSNorm ───────────────────────────────────────────────────────────────

    #[test]
    fn rmsnorm_ones_weight_normalises() {
        let weight = Tensor::ones(4, DType::F32, &Device::Cpu).unwrap();
        let norm = RmsNorm::new(weight, 1e-6);
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &Device::Cpu).unwrap();
        let out = norm.forward(&x).unwrap();
        // All outputs should be unit-normalised: sum(out^2) / 4 ≈ 1
        let sq_mean: f32 = out.sqr().unwrap().mean_all().unwrap().to_scalar().unwrap();
        assert!(
            (sq_mean - 1.0).abs() < 0.01,
            "expected sq_mean≈1, got {sq_mean}"
        );
    }

    #[test]
    fn rmsnorm_zero_input_stays_zero() {
        let weight = Tensor::ones(4, DType::F32, &Device::Cpu).unwrap();
        let norm = RmsNorm::new(weight, 1e-6);
        let x = Tensor::zeros((2, 4), DType::F32, &Device::Cpu).unwrap();
        let out = norm.forward(&x).unwrap();
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|v| v.abs() < 1e-5), "expected zeros");
    }

    // ── RoPE ──────────────────────────────────────────────────────────────────

    #[test]
    fn rope_preserves_shape() {
        let q = Tensor::randn(0f32, 1f32, (3, 4, 8), &Device::Cpu).unwrap();
        let k = Tensor::randn(0f32, 1f32, (3, 4, 8), &Device::Cpu).unwrap();
        let (q2, k2) = apply_rope(&q, &k, 0, 10_000.0).unwrap();
        assert_eq!(q2.dims(), q.dims());
        assert_eq!(k2.dims(), k.dims());
    }

    #[test]
    fn rope_at_pos0_and_pos1_differ() {
        let q = Tensor::ones((1, 1, 8), DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones((1, 1, 8), DType::F32, &Device::Cpu).unwrap();
        let (q0, _) = apply_rope(&q, &k, 0, 10_000.0).unwrap();
        let (q1, _) = apply_rope(&q, &k, 1, 10_000.0).unwrap();
        let d0: Vec<f32> = q0.flatten_all().unwrap().to_vec1().unwrap();
        let d1: Vec<f32> = q1.flatten_all().unwrap().to_vec1().unwrap();
        assert_ne!(
            d0, d1,
            "RoPE at pos 0 and 1 should produce different vectors"
        );
    }

    // ── Causal mask ───────────────────────────────────────────────────────────

    #[test]
    fn causal_mask_shape() {
        let mask = causal_mask(3, 3, 0, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(mask.dims(), &[3, 3]);
    }

    #[test]
    fn causal_mask_lower_tri_is_zero() {
        let mask = causal_mask(4, 4, 0, &Device::Cpu, DType::F32).unwrap();
        let data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // Lower triangular (including diagonal) should be 0
        // Upper triangular should be -inf
        for i in 0..4 {
            for j in 0..4 {
                let v = data[i * 4 + j];
                if j <= i {
                    assert_eq!(v, 0.0, "mask[{i}][{j}] should be 0");
                } else {
                    assert!(v.is_infinite() && v < 0.0, "mask[{i}][{j}] should be -inf");
                }
            }
        }
    }

    #[test]
    fn causal_mask_with_kv_offset_decode() {
        // Decode step: sq=1, sk=5 (4 cached + 1 new), seq_pos=4
        // Query at abs position 4, all keys 0..4 are visible
        let mask = causal_mask(1, 5, 4, &Device::Cpu, DType::F32).unwrap();
        let data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            data.iter().all(|v| *v == 0.0),
            "all keys visible during decode: {data:?}"
        );
    }

    // ── Config parsing ────────────────────────────────────────────────────────

    #[test]
    fn config_kv_heads_defaults_to_attention_heads() {
        let json = r#"{"hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,
                       "num_attention_heads":4,"vocab_size":100}"#;
        let cfg: LlamaTpConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.kv_heads(), 4, "should default to num_attention_heads");
    }

    #[test]
    fn config_head_dim() {
        let json = r#"{"hidden_size":128,"intermediate_size":256,"num_hidden_layers":2,
                       "num_attention_heads":4,"vocab_size":100}"#;
        let cfg: LlamaTpConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.head_dim(), 32);
    }
}
