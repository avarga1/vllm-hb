//! Tokenizer loading, encoding/decoding, and chat-template rendering.
//!
//! EOS token IDs and the chat template are read directly from the model's
//! HuggingFace config files — no hardcoded token IDs, no hardcoded templates.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::types::ChatMessage;

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
///
/// Reads `tokenizer_config.json` and detects the template dialect:
///   - **Llama 3** — `<|begin_of_text|><|start_header_id|>…<|end_header_id|>`
///   - **ChatML** — `<|im_start|>…<|im_end|>`  (Mistral, Qwen, Hermes, …)
///   - **Mistral v1** — `[INST]…[/INST]`
///
/// Falls back to ChatML if the template cannot be identified.
pub fn apply_chat_template(model_path: &str, messages: &[ChatMessage]) -> Result<String> {
    let dialect = detect_template(model_path).unwrap_or(TemplateDialect::ChatML);
    tracing::debug!(dialect = ?dialect, "Chat template");
    Ok(render(dialect, messages))
}

// ── Template detection ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
enum TemplateDialect {
    Llama3,
    ChatML,
    MistralV1,
}

#[derive(Deserialize)]
struct TokenizerConfig {
    chat_template: Option<String>,
}

fn detect_template(model_path: &str) -> Option<TemplateDialect> {
    let path = Path::new(model_path).join("tokenizer_config.json");
    let raw  = std::fs::read_to_string(path).ok()?;
    let cfg: TokenizerConfig = serde_json::from_str(&raw).ok()?;
    let tmpl = cfg.chat_template?;

    if tmpl.contains("<|start_header_id|>") {
        Some(TemplateDialect::Llama3)
    } else if tmpl.contains("<|im_start|>") {
        Some(TemplateDialect::ChatML)
    } else if tmpl.contains("[INST]") {
        Some(TemplateDialect::MistralV1)
    } else {
        None
    }
}

// ── Template renderers ────────────────────────────────────────────────────────

fn render(dialect: TemplateDialect, messages: &[ChatMessage]) -> String {
    match dialect {
        TemplateDialect::Llama3    => render_llama3(messages),
        TemplateDialect::ChatML    => render_chatml(messages),
        TemplateDialect::MistralV1 => render_mistral_v1(messages),
    }
}

fn render_llama3(messages: &[ChatMessage]) -> String {
    let mut out = String::from("<|begin_of_text|>");
    for msg in messages {
        out.push_str(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
            msg.role, msg.content
        ));
    }
    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    out
}

fn render_chatml(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, msg.content
        ));
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

fn render_mistral_v1(messages: &[ChatMessage]) -> String {
    let mut out = String::from("<s>");
    let mut i = 0;
    while i < messages.len() {
        // Optionally absorb a leading system turn into the first user turn.
        let sys_prefix = if i == 0 && messages[i].role == "system" {
            let s = format!("<<SYS>>\n{}\n<</SYS>>\n\n", messages[i].content);
            i += 1;
            s
        } else {
            String::new()
        };

        if i < messages.len() && messages[i].role == "user" {
            out.push_str(&format!("[INST] {}{} [/INST]", sys_prefix, messages[i].content));
            i += 1;
        }

        if i < messages.len() && messages[i].role == "assistant" {
            out.push_str(&format!(" {} </s><s>", messages[i].content));
            i += 1;
        }
    }
    out
}
