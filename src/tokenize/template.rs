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

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
        }
    }

    // ── Detection ─────────────────────────────────────────────────────────────

    fn write_cfg(dir: &TempDir, template: &str) {
        let json = format!(r#"{{"chat_template": "{template}"}}"#);
        fs::write(dir.path().join("tokenizer_config.json"), json).unwrap();
    }

    #[test]
    fn detect_llama3() {
        let dir = TempDir::new().unwrap();
        write_cfg(
            &dir,
            "<|start_header_id|>user<|end_header_id|>\\n{q}<|eot_id|>",
        );
        let dialect = detect(dir.path().to_str().unwrap()).unwrap();
        assert!(matches!(dialect, TemplateDialect::Llama3));
    }

    #[test]
    fn detect_chatml() {
        let dir = TempDir::new().unwrap();
        write_cfg(&dir, "<|im_start|>user\\n{q}<|im_end|>");
        let dialect = detect(dir.path().to_str().unwrap()).unwrap();
        assert!(matches!(dialect, TemplateDialect::ChatML));
    }

    #[test]
    fn detect_mistral_v1() {
        let dir = TempDir::new().unwrap();
        write_cfg(&dir, "[INST] {q} [/INST]");
        let dialect = detect(dir.path().to_str().unwrap()).unwrap();
        assert!(matches!(dialect, TemplateDialect::MistralV1));
    }

    #[test]
    fn detect_unknown_template_returns_none() {
        let dir = TempDir::new().unwrap();
        write_cfg(&dir, "some_other_format {q}");
        assert!(detect(dir.path().to_str().unwrap()).is_none());
    }

    #[test]
    fn detect_missing_file_returns_none() {
        assert!(detect("/nonexistent/path").is_none());
    }

    #[test]
    fn detect_missing_chat_template_key() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"other_key": "val"}"#,
        )
        .unwrap();
        assert!(detect(dir.path().to_str().unwrap()).is_none());
    }

    // ── Rendering ─────────────────────────────────────────────────────────────

    #[test]
    fn render_chatml_single_user() {
        let out = render_chatml(&[msg("user", "hello")]);
        assert!(out.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn render_chatml_system_and_user() {
        let out = render_chatml(&[msg("system", "be nice"), msg("user", "hi")]);
        assert!(out.contains("<|im_start|>system\nbe nice<|im_end|>"));
        assert!(out.contains("<|im_start|>user\nhi<|im_end|>"));
    }

    #[test]
    fn render_llama3_has_header_tags() {
        let out = render_llama3(&[msg("user", "test")]);
        assert!(out.starts_with("<|begin_of_text|>"));
        assert!(out.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(out.contains("test<|eot_id|>"));
        assert!(out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn render_mistral_v1_user_assistant_turn() {
        let msgs = vec![msg("user", "hi"), msg("assistant", "hello")];
        let out = render_mistral_v1(&msgs);
        assert!(out.contains("[INST] hi [/INST]"));
        assert!(out.contains("hello </s>"));
    }

    #[test]
    fn render_mistral_v1_system_prefix_merged_with_first_user() {
        let msgs = vec![msg("system", "be helpful"), msg("user", "hi")];
        let out = render_mistral_v1(&msgs);
        // System content is merged into the [INST] block.
        assert!(out.contains("<<SYS>>\nbe helpful\n<</SYS>>"));
        assert!(out.contains("[INST]"));
        assert!(out.contains("hi [/INST]"));
    }

    #[test]
    fn render_chatml_empty_messages() {
        let out = render_chatml(&[]);
        assert_eq!(out, "<|im_start|>assistant\n");
    }
}
