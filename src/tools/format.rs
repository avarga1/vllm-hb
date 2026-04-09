//! Tool definition formatting — inject available functions into the system prompt.
//!
//! Different model families expect tool definitions in different formats.
//! We detect the model family from the tokenizer config and render accordingly:
//!
//! | Family                        | Format injected            |
//! |-------------------------------|----------------------------|
//! | Llama-3, Qwen, Mistral        | JSON schema block           |
//! | Hermes / Claude-style         | XML `<tools>` block         |
//!
//! Both are injected as a system message prepended to the conversation.

use crate::types::openai::Tool;

/// Rendering dialect for tool definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFormat {
    /// JSON schema block — standard OpenAI / Llama-3 / Qwen style.
    ///
    /// ```text
    /// You have access to the following tools:
    /// ```json
    /// [{"type":"function","function":{"name":"...","parameters":{...}}}]
    /// ```
    /// Respond with a JSON object when calling a tool.
    /// ```
    Json,

    /// XML block — Claude / Hermes style.
    ///
    /// ```text
    /// <tools>
    /// <tool_description>
    ///   <tool_name>get_weather</tool_name>
    ///   <description>...</description>
    ///   <parameters>...</parameters>
    /// </tool_description>
    /// </tools>
    /// ```
    Xml,
}

/// Render `tools` as a system-prompt prefix using the given format.
///
/// Returns an empty string when `tools` is empty (no-op fast path).
pub fn inject_tools(tools: &[Tool], format: ToolFormat) -> String {
    if tools.is_empty() {
        return String::new();
    }

    match format {
        ToolFormat::Json => render_json(tools),
        ToolFormat::Xml => render_xml(tools),
    }
}

/// Detect which format a model is likely to expect based on its chat template
/// string.  Falls back to JSON (the OpenAI standard) when unknown.
pub fn detect_format(chat_template: &str) -> ToolFormat {
    // Hermes / Claude-style templates explicitly reference <tool_description>
    // or <function_calls> in their template strings.
    if chat_template.contains("<tool_description>")
        || chat_template.contains("<function_calls>")
        || chat_template.contains("<tools>")
    {
        ToolFormat::Xml
    } else {
        ToolFormat::Json
    }
}

// ── Renderers ─────────────────────────────────────────────────────────────────

fn render_json(tools: &[Tool]) -> String {
    let schema = serde_json::to_string_pretty(tools).unwrap_or_default();
    format!(
        "You have access to the following tools:\n```json\n{schema}\n```\n\
         When you want to call a tool, respond ONLY with a JSON object of this \
         form and nothing else:\n\
         {{\"name\": \"<tool_name>\", \"arguments\": {{...}}}}"
    )
}

fn render_xml(tools: &[Tool]) -> String {
    let mut out = String::from("<tools>\n");

    for tool in tools {
        let f = &tool.function;
        out.push_str("<tool_description>\n");
        out.push_str(&format!("  <tool_name>{}</tool_name>\n", f.name));

        if let Some(desc) = &f.description {
            out.push_str(&format!("  <description>{desc}</description>\n"));
        }

        if let Some(params) = &f.parameters {
            let params_str = serde_json::to_string_pretty(params).unwrap_or_default();
            out.push_str(&format!("  <parameters>{params_str}</parameters>\n"));
        }

        out.push_str("</tool_description>\n");
    }

    out.push_str("</tools>\n\n");
    out.push_str(
        "When you want to call a tool, respond with XML:\n\
         <function_calls>\n\
         <invoke name=\"<tool_name>\">\n\
         <parameter name=\"<param_name>\"><value></parameter>\n\
         </invoke>\n\
         </function_calls>",
    );

    out
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::openai::{FunctionDef, Tool};

    fn weather_tool() -> Tool {
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "get_weather".into(),
                description: Some("Get the weather for a city".into()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": { "type": "string" }
                    }
                })),
            },
        }
    }

    #[test]
    fn empty_tools_returns_empty_string() {
        assert!(inject_tools(&[], ToolFormat::Json).is_empty());
        assert!(inject_tools(&[], ToolFormat::Xml).is_empty());
    }

    #[test]
    fn json_format_contains_tool_name() {
        let out = inject_tools(&[weather_tool()], ToolFormat::Json);
        assert!(out.contains("get_weather"), "output: {out}");
        assert!(out.contains("```json"), "output: {out}");
    }

    #[test]
    fn json_format_contains_description() {
        let out = inject_tools(&[weather_tool()], ToolFormat::Json);
        assert!(out.contains("Get the weather for a city"), "output: {out}");
    }

    #[test]
    fn xml_format_contains_tool_name() {
        let out = inject_tools(&[weather_tool()], ToolFormat::Xml);
        assert!(
            out.contains("<tool_name>get_weather</tool_name>"),
            "output: {out}"
        );
    }

    #[test]
    fn xml_format_contains_function_calls_template() {
        let out = inject_tools(&[weather_tool()], ToolFormat::Xml);
        assert!(out.contains("<function_calls>"), "output: {out}");
    }

    #[test]
    fn detect_format_json_for_llama() {
        // Llama-3 template has no tool_description / function_calls tags
        assert_eq!(
            detect_format("<|start_header_id|>system<|end_header_id|>"),
            ToolFormat::Json
        );
    }

    #[test]
    fn detect_format_xml_for_hermes() {
        assert_eq!(
            detect_format("{% if tool_description %}<tool_description>{% endif %}"),
            ToolFormat::Xml
        );
    }

    #[test]
    fn detect_format_xml_for_function_calls() {
        assert_eq!(
            detect_format("respond with <function_calls>...</function_calls>"),
            ToolFormat::Xml
        );
    }
}
