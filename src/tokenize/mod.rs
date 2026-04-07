//! Tokenizer loading, encoding/decoding, and chat-template rendering.
//!
//! # Modules
//! - `template` — dialect detection + ChatML / Llama3 / Mistral-v1 renderers

pub mod template;

use std::path::Path;

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

use crate::types::openai::ChatMessage;

// ── Public API ────────────────────────────────────────────────────────────────

/// Load a tokenizer from a local directory or a HuggingFace model ID.
pub fn load(model_path: &str) -> Result<Tokenizer> {
    let local = Path::new(model_path).join("tokenizer.json");
    if local.exists() {
        Tokenizer::from_file(&local)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {}: {e}", local.display()))
    } else {
        Tokenizer::from_pretrained(model_path, None)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer '{model_path}': {e}"))
    }
}

/// Load EOS token IDs from `config.json` (`eos_token_id` may be int or array).
pub fn load_eos_tokens(model_path: &str) -> Result<Vec<u32>> {
    let config_path = Path::new(model_path).join("config.json");
    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("Reading {}", config_path.display()))?;

    let v: serde_json::Value = serde_json::from_str(&raw)?;

    let ids = match &v["eos_token_id"] {
        serde_json::Value::Number(n) => {
            vec![n.as_u64().context("eos_token_id out of range")? as u32]
        }
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|x| x.as_u64())
            .map(|x| x as u32)
            .collect(),
        _ => {
            tracing::warn!("No eos_token_id in config.json — falling back to token id 2");
            vec![2]
        }
    };

    tracing::debug!(eos_tokens = ?ids, "EOS tokens loaded");
    Ok(ids)
}

/// Encode `text` into token IDs.
pub fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("Encode error: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

/// Decode token IDs back to a UTF-8 string.
pub fn decode(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String> {
    tokenizer
        .decode(ids, true)
        .map_err(|e| anyhow::anyhow!("Decode error: {e}"))
}

/// Apply the model's chat template to a list of messages.
pub fn apply_chat_template(model_path: &str, messages: &[ChatMessage]) -> Result<String> {
    let dialect = template::detect(model_path).unwrap_or(template::TemplateDialect::ChatML);
    tracing::debug!(dialect = ?dialect, "Chat template");
    Ok(template::render(dialect, messages))
}
