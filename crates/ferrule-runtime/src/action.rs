use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentActionKind {
    RespondText,
    CallTool,
    Finish,
    Invalid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedAction {
    pub kind: AgentActionKind,
    pub raw_text: String,
    pub response_text: Option<String>,
    pub tool_name: Option<String>,
    pub tool_input: Option<String>,
    pub finish_message: Option<String>,
    pub error: Option<String>,
}

impl ParsedAction {
    pub fn invalid(raw_text: String, error: impl Into<String>) -> Self {
        Self {
            kind: AgentActionKind::Invalid,
            raw_text,
            response_text: None,
            tool_name: None,
            tool_input: None,
            finish_message: None,
            error: Some(error.into()),
        }
    }
}

pub fn parse_action(raw: &str) -> ParsedAction {
    let candidate = extract_action_block(raw).unwrap_or_default();

    if candidate.is_empty() {
        return ParsedAction::invalid(raw.trim().to_string(), "no valid ACTION block found");
    }

    parse_action_core(&candidate, raw)
}

fn extract_action_block(raw: &str) -> Option<String> {
    let start = raw.find("<ACTION>")?;
    let end = raw.find("</ACTION>")?;
    if end <= start {
        return None;
    }

    let inner = &raw[start + "<ACTION>".len()..end];
    Some(inner.trim().to_string())
}

fn parse_action_core(candidate: &str, raw_text: &str) -> ParsedAction {
    let text = candidate.trim().to_string();

    if text.is_empty() {
        return ParsedAction::invalid(raw_text.to_string(), "empty ACTION block");
    }

    if let Some(rest) = text.strip_prefix("FINAL:") {
        return ParsedAction {
            kind: AgentActionKind::Finish,
            raw_text: raw_text.to_string(),
            response_text: None,
            tool_name: None,
            tool_input: None,
            finish_message: Some(rest.trim().to_string()),
            error: None,
        };
    }

    if text.starts_with("TOOL:") {
        let mut lines = text.lines();
        let first = lines.next().unwrap_or_default();
        let tool_name = first.trim_start_matches("TOOL:").trim().to_string();

        let mut tool_input = String::new();
        for line in lines {
            if let Some(rest) = line.strip_prefix("INPUT:") {
                tool_input = rest.trim().to_string();
                break;
            }
        }

        if tool_name.is_empty() {
            return ParsedAction::invalid(raw_text.to_string(), "missing tool name");
        }

        return ParsedAction {
            kind: AgentActionKind::CallTool,
            raw_text: raw_text.to_string(),
            response_text: None,
            tool_name: Some(tool_name),
            tool_input: Some(tool_input),
            finish_message: None,
            error: None,
        };
    }

    ParsedAction::invalid(
        raw_text.to_string(),
        "ACTION block must contain either TOOL: or FINAL:",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_final_action_block() {
        let out = parse_action("<ACTION>\nFINAL: 4\n</ACTION>");
        assert!(matches!(out.kind, AgentActionKind::Finish));
        assert_eq!(out.finish_message.as_deref(), Some("4"));
    }

    #[test]
    fn parses_tool_action_block() {
        let out = parse_action("<ACTION>\nTOOL: echo\nINPUT: 2+2\n</ACTION>");
        assert!(matches!(out.kind, AgentActionKind::CallTool));
        assert_eq!(out.tool_name.as_deref(), Some("echo"));
        assert_eq!(out.tool_input.as_deref(), Some("2+2"));
    }

    #[test]
    fn extracts_action_block_from_surrounding_text() {
        let out = parse_action(
            "I should use a tool.\n\n<ACTION>\nTOOL: echo\nINPUT: 2+2\n</ACTION>\nThanks.",
        );
        assert!(matches!(out.kind, AgentActionKind::CallTool));
        assert_eq!(out.tool_name.as_deref(), Some("echo"));
    }

    #[test]
    fn rejects_text_without_action_block() {
        let out = parse_action("The answer is 4");
        assert!(matches!(out.kind, AgentActionKind::Invalid));
    }
}
