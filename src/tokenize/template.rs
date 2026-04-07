//! Chat template detection and rendering.
//!
//! Reads `tokenizer_config.json` and auto-detects the dialect:
//!
//! | Dialect    | Signal in `chat_template`          | Examples                    |
//! |------------|------------------------------------|-----------------------------|
//! | Llama 3    | `<\|start_header_id\|>`            | Meta-Llama-3, Hermes-3      |
//! | ChatML     | `<\|im_start\|>`                   | Mistral, Qwen, Hermes-2     |
//! | Mistral v1 | `[INST]`                           | Mistral-7B-Instruct-v0.1/2  |
//!
//! Falls back to ChatML when the template cannot be identified.

use serde::Deserialize;
use std::path::Path;

use crate::types::openai::ChatMessage;

// ── Dialect enum ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub enum TemplateDialect {
    Llama3,
    ChatML,
    MistralV1,
}

// ── Detection ─────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct TokenizerConfig {
    chat_template: Option<String>,
}

pub fn detect(model_path: &str) -> Option<TemplateDialect> {
    let path = Path::new(model_path).join("tokenizer_config.json");
    let raw = std::fs::read_to_string(path).ok()?;
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

// ── Renderers ─────────────────────────────────────────────────────────────────

pub fn render(dialect: TemplateDialect, messages: &[ChatMessage]) -> String {
    match dialect {
        TemplateDialect::Llama3 => render_llama3(messages),
        TemplateDialect::ChatML => render_chatml(messages),
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
        let sys_prefix = if i == 0 && messages[i].role == "system" {
            let s = format!("<<SYS>>\n{}\n<</SYS>>\n\n", messages[i].content);
            i += 1;
            s
        } else {
            String::new()
        };

        if i < messages.len() && messages[i].role == "user" {
            out.push_str(&format!(
                "[INST] {}{} [/INST]",
                sys_prefix, messages[i].content
            ));
            i += 1;
        }
        if i < messages.len() && messages[i].role == "assistant" {
            out.push_str(&format!(" {} </s><s>", messages[i].content));
            i += 1;
        }
    }
    out
}
