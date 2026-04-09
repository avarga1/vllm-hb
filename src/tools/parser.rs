//! Tool-call detection and parsing.
//!
//! Scans raw model output for tool-call syntax in two formats:
//!
//! ## JSON-block (Llama-3, Qwen, Mistral function-calling)
//!
//! ```text
//! {"name": "get_weather", "arguments": {"location": "San Francisco"}}
//! ```
//!
//! The model emits a bare JSON object (sometimes wrapped in a ```json fence).
//! We try to detect this by looking for `"name"` and `"arguments"` keys at the
//! top level of a JSON object that covers most of the output.
//!
//! ## XML (Claude / Hermes)
//!
//! ```xml
//! <function_calls>
//! <invoke name="get_weather">
//! <parameter name="location">San Francisco</parameter>
//! </invoke>
//! </function_calls>
//! ```
//!
//! ## Output
//!
//! When a call is detected `ToolCallParser::parse` returns a `ParsedOutput`
//! with:
//! - `tool_calls` — the extracted calls
//! - `visible_text` — the output with the tool-call markup stripped out
//!   (may be empty — many models emit nothing but the call)
//!
//! When no call is detected `visible_text` is the full output unchanged.

use crate::types::openai::{FunctionCall, ToolCall};
use uuid::Uuid;

/// A single parsed tool call extracted from model output.
#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    /// Unique call id, e.g. `"call_abc123"`.
    pub id: String,
    /// Name of the function to invoke.
    pub name: String,
    /// JSON string of the arguments object.
    pub arguments: String,
}

impl ParsedToolCall {
    /// Convert into the wire type used in responses.
    pub fn into_tool_call(self) -> ToolCall {
        ToolCall {
            id: self.id,
            tool_type: "function".into(),
            function: FunctionCall {
                name: self.name,
                arguments: self.arguments,
            },
        }
    }
}

/// Result of parsing model output for tool calls.
#[derive(Debug)]
pub struct ParsedOutput {
    /// Extracted tool calls (empty = no call detected).
    pub tool_calls: Vec<ParsedToolCall>,
    /// Assistant text with tool-call markup removed.
    pub visible_text: String,
}

/// Stateless parser — call [`ToolCallParser::parse`] on each completed output.
pub struct ToolCallParser;

impl ToolCallParser {
    /// Parse `raw` model output for JSON-block or XML tool calls.
    ///
    /// Tries JSON detection first (cheaper), then XML.  Returns the stripped
    /// visible text and any extracted calls.
    pub fn parse(raw: &str) -> ParsedOutput {
        let trimmed = raw.trim();

        // Try JSON block first.
        if let Some(out) = parse_json_block(trimmed) {
            return out;
        }

        // Try XML.
        if let Some(out) = parse_xml(trimmed) {
            return out;
        }

        // No tool call found — return full output unchanged.
        ParsedOutput {
            tool_calls: Vec::new(),
            visible_text: raw.to_string(),
        }
    }
}

// ── JSON-block parser ─────────────────────────────────────────────────────────

/// Detect and parse a JSON-style tool call.
///
/// Accepts:
/// - Bare JSON object: `{"name":"fn","arguments":{...}}`
/// - Fenced: ` ```json\n{...}\n``` `
/// - With optional leading text before the JSON object
fn parse_json_block(text: &str) -> Option<ParsedOutput> {
    // Strip optional ```json ... ``` fence.
    let inner = if let Some(s) = strip_json_fence(text) {
        s
    } else {
        text
    };

    // Find the outermost `{` that looks like a tool call object.
    let start = inner.find('{')?;
    let json_candidate = find_balanced_object(&inner[start..])?;

    let v: serde_json::Value = serde_json::from_str(json_candidate).ok()?;
    let obj = v.as_object()?;

    let name = obj.get("name")?.as_str()?.to_string();
    let args = obj.get("arguments")?;

    // `arguments` may already be a string (pre-serialised) or an object.
    let arguments = match args {
        serde_json::Value::String(s) => s.clone(),
        _ => serde_json::to_string(args).ok()?,
    };

    let call = ParsedToolCall {
        id: format!("call_{}", &Uuid::new_v4().to_string()[..8]),
        name,
        arguments,
    };

    // Visible text = everything before the JSON block (usually empty).
    let visible_text = inner[..start].trim().to_string();

    Some(ParsedOutput {
        tool_calls: vec![call],
        visible_text,
    })
}

fn strip_json_fence(text: &str) -> Option<&str> {
    let text = text.trim();
    let body = text
        .strip_prefix("```json")
        .or_else(|| text.strip_prefix("```"))?
        .trim_start_matches('\n');
    let end = body.rfind("```")?;
    Some(body[..end].trim())
}

/// Walk the string character-by-character to find the matching closing `}` for
/// the opening `{` at the start.  Returns the balanced substring or `None` if
/// the braces are unbalanced.
fn find_balanced_object(s: &str) -> Option<&str> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    let mut end = None;

    for (i, ch) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        match ch {
            '\\' if in_string => escape = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i + 1);
                    break;
                }
            }
            _ => {}
        }
    }

    end.map(|e| &s[..e])
}

// ── XML parser ────────────────────────────────────────────────────────────────

/// Detect and parse XML-style function calls.
///
/// Looks for `<function_calls>...<invoke name="...">...</invoke>...</function_calls>`.
fn parse_xml(text: &str) -> Option<ParsedOutput> {
    let fc_start = text.find("<function_calls>")?;
    let fc_end = text.find("</function_calls>")? + "</function_calls>".len();

    let fc_block = &text[fc_start..fc_end];

    let mut calls = Vec::new();
    let mut search = fc_block;

    while let Some(inv_start) = search.find("<invoke") {
        let rest = &search[inv_start..];

        // Extract the `name` attribute from <invoke name="...">
        let name = extract_xml_attr(rest, "name")?;

        // Find the closing </invoke>
        let inv_end = rest.find("</invoke>")? + "</invoke>".len();
        let invoke_body = &rest[..inv_end];

        // Extract all <parameter name="...">value</parameter> pairs → JSON object.
        let arguments = extract_parameters_as_json(invoke_body);

        calls.push(ParsedToolCall {
            id: format!("call_{}", &Uuid::new_v4().to_string()[..8]),
            name,
            arguments,
        });

        search = &rest[inv_end..];
    }

    if calls.is_empty() {
        return None;
    }

    // Visible text = everything outside the <function_calls> block.
    let visible_text = format!("{}{}", &text[..fc_start], &text[fc_end..])
        .trim()
        .to_string();

    Some(ParsedOutput {
        tool_calls: calls,
        visible_text,
    })
}

/// Extract the value of a named attribute from an XML tag string.
/// Handles both `name="value"` and `name='value'`.
fn extract_xml_attr(tag: &str, attr: &str) -> Option<String> {
    // Find `attr="` or `attr='`
    let search_dq = format!("{attr}=\"");
    let search_sq = format!("{attr}='");

    if let Some(pos) = tag.find(&search_dq) {
        let rest = &tag[pos + search_dq.len()..];
        let end = rest.find('"')?;
        return Some(rest[..end].to_string());
    }
    if let Some(pos) = tag.find(&search_sq) {
        let rest = &tag[pos + search_sq.len()..];
        let end = rest.find('\'')?;
        return Some(rest[..end].to_string());
    }
    None
}

/// Extract `<parameter name="key">value</parameter>` pairs from an invoke
/// block and serialise them as a JSON object string.
fn extract_parameters_as_json(invoke_body: &str) -> String {
    let mut map = serde_json::Map::new();
    let mut search = invoke_body;

    while let Some(p_start) = search.find("<parameter") {
        let rest = &search[p_start..];
        let Some(tag_end) = rest.find('>') else {
            break;
        };
        let tag = &rest[..tag_end + 1];

        let Some(name) = extract_xml_attr(tag, "name") else {
            search = &rest[tag_end + 1..];
            continue;
        };

        let after_tag = &rest[tag_end + 1..];
        let Some(close) = after_tag.find("</parameter>") else {
            break;
        };
        let value = after_tag[..close].trim();

        // Try to parse value as JSON; fall back to string.
        let json_value = serde_json::from_str::<serde_json::Value>(value)
            .unwrap_or_else(|_| serde_json::Value::String(value.to_string()));

        map.insert(name, json_value);
        search = &after_tag[close + "</parameter>".len()..];
    }

    serde_json::to_string(&serde_json::Value::Object(map)).unwrap_or_else(|_| "{}".into())
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── JSON-block ────────────────────────────────────────────────────────────

    #[test]
    fn json_bare_object() {
        let raw = r#"{"name": "get_weather", "arguments": {"location": "SF"}}"#;
        let out = ToolCallParser::parse(raw);
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "get_weather");
        assert!(out.tool_calls[0].arguments.contains("SF"));
        assert!(out.visible_text.is_empty());
    }

    #[test]
    fn json_fenced_block() {
        let raw = "```json\n{\"name\": \"fn\", \"arguments\": {\"x\": 1}}\n```";
        let out = ToolCallParser::parse(raw);
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "fn");
    }

    #[test]
    fn json_arguments_already_string() {
        let raw = r#"{"name": "fn", "arguments": "{\"x\": 1}"}"#;
        let out = ToolCallParser::parse(raw);
        assert_eq!(out.tool_calls.len(), 1);
        assert!(out.tool_calls[0].arguments.contains("x"));
    }

    #[test]
    fn json_with_leading_text() {
        let raw = "Sure, I'll call the function.\n{\"name\": \"fn\", \"arguments\": {}}";
        let out = ToolCallParser::parse(raw);
        assert_eq!(out.tool_calls.len(), 1);
        assert!(out.visible_text.contains("Sure"));
    }

    #[test]
    fn no_tool_call_returns_full_text() {
        let raw = "The weather in SF is sunny.";
        let out = ToolCallParser::parse(raw);
        assert!(out.tool_calls.is_empty());
        assert_eq!(out.visible_text, raw);
    }

    // ── XML ───────────────────────────────────────────────────────────────────

    #[test]
    fn xml_single_call() {
        let raw = "<function_calls>\n\
                   <invoke name=\"get_weather\">\n\
                   <parameter name=\"location\">San Francisco</parameter>\n\
                   </invoke>\n\
                   </function_calls>";
        let out = ToolCallParser::parse(raw);
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "get_weather");
        assert!(out.tool_calls[0].arguments.contains("San Francisco"));
        assert!(out.visible_text.is_empty());
    }

    #[test]
    fn xml_multiple_calls() {
        let raw = "<function_calls>\n\
                   <invoke name=\"fn_a\"><parameter name=\"x\">1</parameter></invoke>\n\
                   <invoke name=\"fn_b\"><parameter name=\"y\">2</parameter></invoke>\n\
                   </function_calls>";
        let out = ToolCallParser::parse(raw);
        assert_eq!(out.tool_calls.len(), 2);
        assert_eq!(out.tool_calls[0].name, "fn_a");
        assert_eq!(out.tool_calls[1].name, "fn_b");
    }

    #[test]
    fn xml_numeric_parameter_stays_numeric() {
        let raw = "<function_calls>\
                   <invoke name=\"fn\">\
                   <parameter name=\"count\">42</parameter>\
                   </invoke>\
                   </function_calls>";
        let out = ToolCallParser::parse(raw);
        let args: serde_json::Value = serde_json::from_str(&out.tool_calls[0].arguments).unwrap();
        assert_eq!(args["count"], 42);
    }

    #[test]
    fn xml_with_surrounding_text() {
        let raw = "I'll look that up.\n\
                   <function_calls>\
                   <invoke name=\"fn\"><parameter name=\"q\">hello</parameter></invoke>\
                   </function_calls>\n\
                   Let me know if you need more.";
        let out = ToolCallParser::parse(raw);
        assert_eq!(out.tool_calls.len(), 1);
        assert!(out.visible_text.contains("I'll look that up"));
        assert!(out.visible_text.contains("Let me know"));
        assert!(!out.visible_text.contains("<function_calls>"));
    }

    #[test]
    fn call_id_is_unique() {
        let raw = r#"{"name": "fn", "arguments": {}}"#;
        let a = ToolCallParser::parse(raw);
        let b = ToolCallParser::parse(raw);
        assert_ne!(a.tool_calls[0].id, b.tool_calls[0].id);
    }

    #[test]
    fn into_tool_call_sets_type() {
        let raw = r#"{"name": "fn", "arguments": {}}"#;
        let out = ToolCallParser::parse(raw);
        let tc = out.tool_calls.into_iter().next().unwrap().into_tool_call();
        assert_eq!(tc.tool_type, "function");
        assert_eq!(tc.function.name, "fn");
    }
}
