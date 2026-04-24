//! Compressed-tensors pack-quantized format support.
//!
//! Parses the `quantization_config` block of an HF `config.json` produced
//! by Neural Magic / Red Hat's `compressed-tensors` library (used by many
//! HF checkpoints including `cyankiwi/gemma-4-*-AWQ-4bit`).
//!
//! # Weight layout on disk
//!
//! For a Linear of shape `[out_features, in_features]` the safetensors
//! shard stores:
//!
//! - `<prefix>.weight_packed` — `[out_features, in_features / 8]` as
//!   `int32`.  Each int32 holds 8 consecutive signed int4 values along
//!   the `in_features` axis.  Slot 0 occupies bits 0-3, slot 7 bits 28-31.
//! - `<prefix>.weight_scale` — `[out_features, in_features / group_size]`
//!   in the model's native float dtype (fp16 or bf16).  Each scale
//!   applies to `group_size` consecutive int4 values in `in_features`.
//!
//! `symmetric: true` (the only format we support here) means there is
//! no zero-point — dequantized value = int4 × scale[group].  Asymmetric
//! variants would have an additional `weight_zero_point` tensor.
//!
//! # Int4 unpack math
//!
//! For input index `i` along in_features:
//! ```text
//! int32_col = i / 8                  // which int32 word
//! slot      = i % 8                  // which nibble within the word
//! packed    = weight_packed[row, int32_col] as u32
//! nibble    = (packed >> (slot * 4)) & 0xF
//! int4      = if (nibble & 0x8) != 0 { nibble as i8 - 16 } else { nibble as i8 }
//! group_idx = i / group_size
//! dequant   = (int4 as f32) * (weight_scale[row, group_idx] as f32)
//! ```

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use std::collections::HashMap;

// ── Config types ──────────────────────────────────────────────────────────────

/// Top-level `quantization_config` block from HF `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct CompressedTensorsConfig {
    #[serde(default = "default_format")]
    pub format: String,

    pub config_groups: HashMap<String, CompressedTensorsGroup>,

    /// Modules that are NOT quantized (loaded as regular fp16/bf16 tensors).
    /// Common entries: `lm_head`, `router.proj`, first-N-layers mlp, vision tower.
    #[serde(default)]
    pub ignore: Vec<String>,

    #[serde(default)]
    pub quant_method: Option<String>,

    #[serde(default)]
    pub quantization_status: Option<String>,

    #[serde(default)]
    pub kv_cache_scheme: Option<serde_json::Value>,
}

fn default_format() -> String {
    "pack-quantized".to_string()
}

/// One entry in `config_groups` — specifies quant args for a set of
/// modules matching `targets`.
#[derive(Debug, Clone, Deserialize)]
pub struct CompressedTensorsGroup {
    #[serde(default = "default_format")]
    pub format: String,
    pub targets: Vec<String>,
    pub weights: CompressedTensorsQuantArgs,
    #[serde(default)]
    pub input_activations: Option<serde_json::Value>,
    #[serde(default)]
    pub output_activations: Option<serde_json::Value>,
}

/// Per-group weight quantization args.  The subset we care about for
/// pack-quantized W4A16 symmetric: `num_bits=4, group_size=*, symmetric=true, type="int"`.
#[derive(Debug, Clone, Deserialize)]
pub struct CompressedTensorsQuantArgs {
    pub num_bits: u32,
    pub group_size: usize,
    pub symmetric: bool,
    #[serde(default = "default_type")]
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(default)]
    pub strategy: Option<String>,
    #[serde(default)]
    pub observer: Option<String>,
    #[serde(default)]
    pub dynamic: bool,
    #[serde(default)]
    pub actorder: Option<serde_json::Value>,
    #[serde(default)]
    pub block_structure: Option<serde_json::Value>,
    #[serde(default)]
    pub scale_dtype: Option<String>,
    #[serde(default)]
    pub zp_dtype: Option<String>,
}

fn default_type() -> String {
    "int".to_string()
}

impl CompressedTensorsConfig {
    /// Extract the `quantization_config` sub-object from a raw config.json.
    /// Returns `Ok(None)` if the config has no quantization metadata.
    pub fn from_config_json(s: &str) -> Result<Option<Self>> {
        #[derive(Deserialize)]
        struct Outer {
            #[serde(default)]
            quantization_config: Option<CompressedTensorsConfig>,
        }
        let outer: Outer = serde_json::from_str(s)
            .context("Parsing config.json to extract quantization_config")?;
        Ok(outer.quantization_config)
    }

    /// Return the (single, for now) default group's quant args.  Real
    /// checkpoints we've seen have exactly one group named `group_0`.
    pub fn default_group(&self) -> Result<&CompressedTensorsGroup> {
        // Prefer "group_0" if present, otherwise the first group.
        if let Some(g) = self.config_groups.get("group_0") {
            return Ok(g);
        }
        self.config_groups
            .values()
            .next()
            .ok_or_else(|| anyhow::anyhow!("CompressedTensorsConfig has no config_groups"))
    }

    /// Check whether a given Linear prefix (e.g.
    /// `"model.language_model.layers.0.mlp.gate_proj"`) is listed in
    /// the `ignore` list, meaning it was NOT quantized.
    pub fn is_ignored(&self, prefix: &str) -> bool {
        // Exact match or prefix match with trailing dot.
        self.ignore
            .iter()
            .any(|p| p == prefix || prefix.starts_with(&format!("{p}.")))
    }
}

impl CompressedTensorsQuantArgs {
    /// Validate that we support this combination of quant args.
    pub fn validate_supported(&self) -> Result<()> {
        if self.num_bits != 4 {
            bail!(
                "compressed-tensors: only num_bits=4 is supported, got {}",
                self.num_bits
            );
        }
        if !self.symmetric {
            bail!(
                "compressed-tensors: only symmetric=true is supported \
                 (W4A16 symmetric). Asymmetric weight quant needs a \
                 weight_zero_point tensor and different dequant math."
            );
        }
        if self.ty != "int" {
            bail!(
                "compressed-tensors: only type=\"int\" is supported, got {:?}",
                self.ty
            );
        }
        if self.group_size == 0 || !self.group_size.is_multiple_of(8) {
            bail!(
                "compressed-tensors: group_size must be a nonzero multiple \
                 of 8 (8 int4 values per int32), got {}",
                self.group_size
            );
        }
        Ok(())
    }
}

// ── Int4 unpack + dequant ─────────────────────────────────────────────────────

/// Dequantize a pack-quantized weight tensor pair to full-precision fp16/bf16/f32.
///
/// # Arguments
///
/// - `packed`   : `[out_features, in_features / 8]` with dtype I32 or U32
/// - `scales`   : `[out_features, in_features / group_size]` with dtype F16 or BF16 or F32
/// - `group_size` : number of int4 values per scale (must be multiple of 8)
///
/// # Returns
///
/// `[out_features, in_features]` tensor with the same dtype as `scales`.
///
/// # Performance
///
/// This is a CPU-side dequant-on-load path.  For large weights it is slow
/// (seconds per GB).  Phase 2 (#41) will replace this with a CUDA kernel
/// that keeps weights packed in VRAM and dequants in-register during GEMM.
pub fn dequantize_pack_quantized(
    packed: &Tensor,
    scales: &Tensor,
    group_size: usize,
) -> Result<Tensor> {
    if group_size == 0 || !group_size.is_multiple_of(8) {
        bail!(
            "dequantize_pack_quantized: group_size must be a nonzero multiple of 8, got {group_size}"
        );
    }

    let packed_shape = packed.dims();
    let scales_shape = scales.dims();
    if packed_shape.len() != 2 {
        bail!(
            "dequantize_pack_quantized: packed must be 2D, got shape {:?}",
            packed_shape
        );
    }
    if scales_shape.len() != 2 {
        bail!(
            "dequantize_pack_quantized: scales must be 2D, got shape {:?}",
            scales_shape
        );
    }

    let out_features = packed_shape[0];
    let in_features_per_int32 = packed_shape[1];
    let in_features = in_features_per_int32 * 8;

    if scales_shape[0] != out_features {
        bail!(
            "dequantize_pack_quantized: scales rows ({}) must match packed rows ({})",
            scales_shape[0],
            out_features
        );
    }
    let groups_per_row = scales_shape[1];
    if in_features != groups_per_row * group_size {
        bail!(
            "dequantize_pack_quantized: in_features ({}) != groups_per_row ({}) * group_size ({})",
            in_features,
            groups_per_row,
            group_size
        );
    }

    // Move tensors to CPU and materialize to host arrays for the loop.
    // (We're running a pure-Rust dequant — CUDA path lives in Phase 2.)
    let packed_cpu = packed.to_device(&Device::Cpu)?;
    let scales_cpu = scales.to_device(&Device::Cpu)?;

    // Packed can be stored as I32 (HF's PyTorch default) or U32; read as u32 via bitcast.
    let packed_u32: Vec<u32> = match packed_cpu.dtype() {
        DType::I32 => {
            let v: Vec<i32> = packed_cpu.flatten_all()?.to_vec1()?;
            v.into_iter().map(|x| x as u32).collect()
        }
        DType::U32 => packed_cpu.flatten_all()?.to_vec1()?,
        dt => bail!(
            "dequantize_pack_quantized: packed dtype must be I32 or U32, got {:?}",
            dt
        ),
    };

    // Scales keep their dtype; we dequant in f32 internally then cast back.
    let scales_f32: Vec<f32> = scales_cpu.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let out_dtype = scales.dtype();

    // Unpack into a dense f32 buffer.
    let mut out = vec![0f32; out_features * in_features];
    for row in 0..out_features {
        let packed_base = row * in_features_per_int32;
        let scales_base = row * groups_per_row;
        let out_base = row * in_features;
        for col_int32 in 0..in_features_per_int32 {
            let word = packed_u32[packed_base + col_int32];
            for slot in 0..8 {
                let i = col_int32 * 8 + slot;
                let group_idx = i / group_size;
                let scale = scales_f32[scales_base + group_idx];
                let nibble = ((word >> (slot * 4)) & 0xF) as i8;
                // Sign-extend int4 → i8: if top bit set, subtract 16.
                let int4 = if nibble & 0x8 != 0 {
                    nibble - 16
                } else {
                    nibble
                };
                out[out_base + i] = (int4 as f32) * scale;
            }
        }
    }

    // Materialize as a candle tensor with the original scale dtype.
    let t = Tensor::from_vec(out, (out_features, in_features), &Device::Cpu)?;
    Ok(t.to_dtype(out_dtype)?.to_device(packed.device())?)
}

// ── VarBuilder integration ────────────────────────────────────────────────────

/// Load a pack-quantized Linear weight from a VarBuilder, dequantizing to
/// full precision at load time.
///
/// Returns the dequantized weight tensor with shape
/// `[out_features, in_features]` and the VarBuilder's active dtype.
///
/// # Usage pattern (Phase 1 dequant-on-load path)
///
/// ```ignore
/// let w = load_packed_weight(&vb.pp("gate_proj"), out, inp, 32)?;
/// let bias = vb.pp("gate_proj").get(out, "bias").ok();
/// let linear = candle_nn::Linear::new(w, bias);
/// ```
///
/// # Expected safetensors keys (under the `vb` prefix)
///
/// - `weight_packed` — int32, shape `[out_features, in_features / 8]`
/// - `weight_scale`  — fp16/bf16, shape `[out_features, in_features / group_size]`
///
/// Phase 2 (#41) will replace this with a path that keeps `packed` and
/// `scales` resident on GPU and calls a fused W4A16 GEMM kernel instead
/// of pre-dequantizing.
pub fn load_packed_weight(
    vb: &VarBuilder,
    out_features: usize,
    in_features: usize,
    group_size: usize,
) -> Result<Tensor> {
    if !in_features.is_multiple_of(8) {
        bail!(
            "load_packed_weight: in_features ({}) must be a multiple of 8 \
             (8 int4 values per int32)",
            in_features
        );
    }
    if !in_features.is_multiple_of(group_size) {
        bail!(
            "load_packed_weight: in_features ({}) must be a multiple of \
             group_size ({})",
            in_features,
            group_size
        );
    }

    let packed_shape = (out_features, in_features / 8);
    let scales_shape = (out_features, in_features / group_size);

    // compressed-tensors stores weight_packed as torch.int32. candle's
    // VarBuilder.get() respects the dtype stored in the safetensors file
    // — so we call get_with_hints_dtype to request I32 exactly.
    let packed = vb
        .get_with_hints_dtype(
            packed_shape,
            "weight_packed",
            candle_nn::Init::Const(0.0),
            DType::I32,
        )
        .with_context(|| {
            format!(
                "Loading weight_packed shape {:?} from safetensors",
                packed_shape
            )
        })?;

    // Scales are stored in the model's native float dtype. Use the
    // VarBuilder's default dtype (fp16 on V100, bf16 on Ampere+).
    let scales = vb.get(scales_shape, "weight_scale").with_context(|| {
        format!(
            "Loading weight_scale shape {:?} from safetensors",
            scales_shape
        )
    })?;

    dequantize_pack_quantized(&packed, &scales, group_size)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // ── Config parse ──────────────────────────────────────────────────────────

    const CYANKIWI_GEMMA4_CONFIG_SNIPPET: &str = r#"{
        "model_type": "gemma4",
        "quantization_config": {
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "format": "pack-quantized",
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "group_size": 32,
                        "symmetric": true,
                        "type": "int",
                        "strategy": "group",
                        "observer": "mse",
                        "dynamic": false,
                        "actorder": null,
                        "block_structure": null,
                        "scale_dtype": null,
                        "zp_dtype": null
                    },
                    "input_activations": null,
                    "output_activations": null
                }
            },
            "ignore": [
                "model.language_model.layers.0.mlp.gate_proj",
                "lm_head"
            ],
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed"
        }
    }"#;

    #[test]
    fn parses_cyankiwi_gemma4_awq_config() {
        let cfg = CompressedTensorsConfig::from_config_json(CYANKIWI_GEMMA4_CONFIG_SNIPPET)
            .unwrap()
            .expect("quantization_config must be present");
        assert_eq!(cfg.format, "pack-quantized");
        assert_eq!(cfg.quant_method.as_deref(), Some("compressed-tensors"));
        assert_eq!(cfg.ignore.len(), 2);
        assert!(cfg.is_ignored("lm_head"));
        assert!(cfg.is_ignored("model.language_model.layers.0.mlp.gate_proj"));
        // Prefix-with-dot match
        assert!(cfg.is_ignored("model.language_model.layers.0.mlp.gate_proj.weight"));
        assert!(!cfg.is_ignored("model.language_model.layers.1.mlp.gate_proj"));

        let g = cfg.default_group().unwrap();
        assert_eq!(g.weights.num_bits, 4);
        assert_eq!(g.weights.group_size, 32);
        assert!(g.weights.symmetric);
        g.weights.validate_supported().unwrap();
    }

    #[test]
    fn rejects_unsupported_num_bits() {
        let args = CompressedTensorsQuantArgs {
            num_bits: 8,
            group_size: 32,
            symmetric: true,
            ty: "int".into(),
            strategy: None,
            observer: None,
            dynamic: false,
            actorder: None,
            block_structure: None,
            scale_dtype: None,
            zp_dtype: None,
        };
        let err = args.validate_supported().unwrap_err();
        assert!(err.to_string().contains("num_bits=4"));
    }

    #[test]
    fn rejects_asymmetric() {
        let args = CompressedTensorsQuantArgs {
            num_bits: 4,
            group_size: 32,
            symmetric: false,
            ty: "int".into(),
            strategy: None,
            observer: None,
            dynamic: false,
            actorder: None,
            block_structure: None,
            scale_dtype: None,
            zp_dtype: None,
        };
        assert!(
            args.validate_supported()
                .unwrap_err()
                .to_string()
                .contains("symmetric=true")
        );
    }

    // ── Int4 unpack ───────────────────────────────────────────────────────────

    /// Hand-computed test case.
    ///
    /// Weight row [1, 2, 3, 4, -1, -2, -3, -4] packed into one int32:
    ///   slot 0: 1  → bits 0-3  = 0x1
    ///   slot 1: 2  → bits 4-7  = 0x2
    ///   slot 2: 3  → bits 8-11 = 0x3
    ///   slot 3: 4  → bits 12-15= 0x4
    ///   slot 4: -1 → bits 16-19= 0xF (two's complement int4)
    ///   slot 5: -2 → bits 20-23= 0xE
    ///   slot 6: -3 → bits 24-27= 0xD
    ///   slot 7: -4 → bits 28-31= 0xC
    /// packed u32 = 0xCDEF_4321
    ///
    /// Scale = 0.5 (group_size=8 so one scale for the whole row).
    /// Expected dequant: [0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0]
    #[test]
    fn dequant_hand_packed_row_is_bit_exact() {
        let dev = Device::Cpu;
        let packed_u32: u32 = 0xCDEF_4321;
        let packed = Tensor::from_vec(vec![packed_u32], (1, 1), &dev).unwrap();
        let scales = Tensor::from_vec(vec![0.5_f32], (1, 1), &dev).unwrap();

        let out = dequantize_pack_quantized(&packed, &scales, 8).unwrap();
        assert_eq!(out.dims(), &[1, 8]);
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        let expected = [0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0];
        for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "slot {i}: got {got}, expected {exp}"
            );
        }
    }

    /// Boundary values: int4 range is [-8, 7].  Verify sign extension
    /// at both ends.
    #[test]
    fn dequant_int4_boundary_values() {
        let dev = Device::Cpu;
        // Row of: 7, -8, 0, 1, 2, 3, 4, 5 all × scale=1.0
        //   7 = 0x7, -8 = 0x8, 0 = 0x0, 1 = 0x1, ..., 5 = 0x5
        let packed_u32: u32 = (0x7 << 0)
            | (0x8 << 4)
            | (0x0 << 8)
            | (0x1 << 12)
            | (0x2 << 16)
            | (0x3 << 20)
            | (0x4 << 24)
            | (0x5 << 28);
        let packed = Tensor::from_vec(vec![packed_u32], (1, 1), &dev).unwrap();
        let scales = Tensor::from_vec(vec![1.0_f32], (1, 1), &dev).unwrap();

        let out = dequantize_pack_quantized(&packed, &scales, 8).unwrap();
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v, vec![7.0, -8.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    /// Multi-row, multi-group: verify that each row uses its own scales
    /// and each group of `group_size` values maps to the right scale.
    #[test]
    fn dequant_multi_row_multi_group() {
        let dev = Device::Cpu;
        // 2 rows, 16 cols, group_size=8 → packed [2, 2] u32, scales [2, 2].
        // Row 0: group 0 = all 1s (× scale 2.0), group 1 = all 2s (× scale 3.0)
        // Row 1: group 0 = all 3s (× scale 0.1), group 1 = all -1s (× scale 10.0)
        let row0_g0: u32 = 0x11111111; // eight 1s packed
        let row0_g1: u32 = 0x22222222; // eight 2s packed
        let row1_g0: u32 = 0x33333333; // eight 3s packed
        let row1_g1: u32 = 0xFFFFFFFF; // eight -1s packed (two's complement)

        let packed =
            Tensor::from_vec(vec![row0_g0, row0_g1, row1_g0, row1_g1], (2, 2), &dev).unwrap();
        let scales = Tensor::from_vec(vec![2.0_f32, 3.0, 0.1, 10.0], (2, 2), &dev).unwrap();

        let out = dequantize_pack_quantized(&packed, &scales, 8).unwrap();
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();

        let mut expected = vec![0f32; 32];
        for i in 0..8 {
            expected[i] = 1.0 * 2.0;
        } // row 0, group 0
        for i in 8..16 {
            expected[i] = 2.0 * 3.0;
        } // row 0, group 1
        for i in 16..24 {
            expected[i] = 3.0 * 0.1;
        } // row 1, group 0
        for i in 24..32 {
            expected[i] = -1.0 * 10.0;
        } // row 1, group 1

        for i in 0..32 {
            assert!(
                (v[i] - expected[i]).abs() < 1e-5,
                "slot {i}: got {}, expected {}",
                v[i],
                expected[i]
            );
        }
    }

    /// Using i32 storage (safetensors default) should work the same as u32.
    #[test]
    fn dequant_accepts_i32_packed() {
        let dev = Device::Cpu;
        // Same test as the first one but stored as i32.  Note that
        // 0xCDEF4321 as u32 = -839915743 as i32 (since top bit set).
        let packed_i32: i32 = 0xCDEF_4321_u32 as i32;
        let packed = Tensor::from_vec(vec![packed_i32], (1, 1), &dev).unwrap();
        let scales = Tensor::from_vec(vec![0.5_f32], (1, 1), &dev).unwrap();

        let out = dequantize_pack_quantized(&packed, &scales, 8).unwrap();
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        let expected = [0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0];
        for (got, exp) in v.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-6);
        }
    }

    /// Invalid group_size is rejected cleanly.
    #[test]
    fn dequant_rejects_bad_group_size() {
        let dev = Device::Cpu;
        let packed = Tensor::from_vec(vec![0u32], (1, 1), &dev).unwrap();
        let scales = Tensor::from_vec(vec![1.0f32], (1, 1), &dev).unwrap();
        assert!(dequantize_pack_quantized(&packed, &scales, 0).is_err());
        assert!(dequantize_pack_quantized(&packed, &scales, 7).is_err()); // not multiple of 8
    }

    // ── VarBuilder integration ────────────────────────────────────────────────

    /// End-to-end: build an in-memory VarBuilder with synthetic packed +
    /// scale tensors, load via `load_packed_weight`, assert dequantized
    /// output matches hand-computed expected values.
    #[test]
    fn load_packed_weight_roundtrip() {
        use candle_core::DType;
        use candle_nn::VarMap;

        let dev = Device::Cpu;
        let vm = VarMap::new();
        // Use f32 scales for the test (simpler numerics; real checkpoints use f16/bf16).
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

        // Build the same hand-packed row as dequant_hand_packed_row_is_bit_exact,
        // but inside a VarBuilder's VarMap so `load_packed_weight` can fetch them.
        // Row of [1, 2, 3, 4, -1, -2, -3, -4] × scale 0.5.
        let packed_u32: u32 = 0xCDEF_4321;
        let packed_i32: i32 = packed_u32 as i32;
        let packed = Tensor::from_vec(vec![packed_i32], (1, 1), &dev).unwrap();
        let scales = Tensor::from_vec(vec![0.5_f32], (1, 1), &dev).unwrap();

        {
            // VarMap's set_one inserts a variable at the given name.
            let mut data = vm.data().lock().unwrap();
            data.insert(
                "gate_proj.weight_packed".to_string(),
                candle_core::Var::from_tensor(&packed).unwrap(),
            );
            data.insert(
                "gate_proj.weight_scale".to_string(),
                candle_core::Var::from_tensor(&scales).unwrap(),
            );
        }

        let w = load_packed_weight(
            &vb.pp("gate_proj"),
            /*out*/ 1,
            /*in*/ 8,
            /*group*/ 8,
        )
        .unwrap();
        assert_eq!(w.dims(), &[1, 8]);
        let v: Vec<f32> = w.flatten_all().unwrap().to_vec1().unwrap();
        let expected = [0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0];
        for (got, exp) in v.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-6, "got {}, expected {}", got, exp);
        }
    }

    /// Shape mismatch (in_features not divisible by 8) is rejected early
    /// with a clear message instead of panicking inside dequantize.
    #[test]
    fn load_packed_weight_rejects_bad_shapes() {
        use candle_nn::VarMap;
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        // in_features=7 is not a multiple of 8.
        let err = load_packed_weight(&vb.pp("x"), 1, 7, 7).unwrap_err();
        assert!(err.to_string().contains("multiple of 8"));
        // in_features=16 but group_size=10 (not dividing)
        let err = load_packed_weight(&vb.pp("x"), 1, 16, 10).unwrap_err();
        assert!(err.to_string().contains("group_size"));
    }
}
