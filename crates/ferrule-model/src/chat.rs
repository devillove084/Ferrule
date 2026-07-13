use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A model-neutral role in a chat conversation.
///
/// Keeping roles as an enum makes unsupported roles unrepresentable instead of
/// passing arbitrary protocol strings into model templates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// A complete, model-neutral chat message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(ChatRole::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(ChatRole::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(ChatRole::Assistant, content)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ChatFormatError {
    #[error("chat input must contain at least one message")]
    EmptyMessages,
    #[error("chat message {index} ({role:?}) has empty content")]
    EmptyContent { index: usize, role: ChatRole },
    #[error("system message at index {index} must be the first message")]
    SystemMessageNotFirst { index: usize },
    #[error("chat message {index} has role {actual:?}; expected {expected:?}")]
    UnexpectedRole {
        index: usize,
        expected: ChatRole,
        actual: ChatRole,
    },
    #[error("chat input must end with a user message, found {actual:?}")]
    MustEndWithUser { actual: ChatRole },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    ChatML,
    Llama3,
    Qwen,
    DeepSeekV4,
    Plain,
}

impl ChatTemplate {
    pub fn name(self) -> &'static str {
        match self {
            Self::ChatML => "chatml",
            Self::Llama3 => "llama3",
            Self::Qwen => "qwen",
            Self::DeepSeekV4 => "deepseek-v4",
            Self::Plain => "plain",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "chatml" => Some(Self::ChatML),
            "llama3" | "llama-3" => Some(Self::Llama3),
            "qwen" => Some(Self::Qwen),
            "deepseek-v4" | "deepseekv4" | "dsv4" | "deepseek" => Some(Self::DeepSeekV4),
            "plain" | "none" => Some(Self::Plain),
            _ => None,
        }
    }

    /// Format a complete conversation as a generation prompt.
    ///
    /// Input is an optional leading system message followed by alternating user
    /// and assistant messages, ending with a user message. The returned prompt
    /// includes the template's assistant generation marker.
    pub fn format_messages(self, messages: &[ChatMessage]) -> Result<String, ChatFormatError> {
        validate_messages(messages)?;

        Ok(match self {
            Self::ChatML | Self::Qwen => format_chatml_messages(messages),
            Self::Llama3 => format_llama3_messages(messages),
            Self::DeepSeekV4 => format_deepseek_v4_messages(messages),
            Self::Plain => format_plain_messages(messages),
        })
    }

    pub fn format_turn(self, turn: &str, first_turn: bool) -> String {
        match self {
            Self::ChatML => {
                if first_turn {
                    format!(
                        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{turn}<|im_end|>\n<|im_start|>assistant\n"
                    )
                } else {
                    format!("\n<|im_start|>user\n{turn}<|im_end|>\n<|im_start|>assistant\n")
                }
            }
            Self::Llama3 => {
                if first_turn {
                    format!(
                        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{turn}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    )
                } else {
                    format!(
                        "<|start_header_id|>user<|end_header_id|>\n\n{turn}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    )
                }
            }
            Self::Qwen => {
                if first_turn {
                    format!(
                        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{turn}<|im_end|>\n<|im_start|>assistant\n"
                    )
                } else {
                    format!("\n<|im_start|>user\n{turn}<|im_end|>\n<|im_start|>assistant\n")
                }
            }
            Self::DeepSeekV4 => {
                if first_turn {
                    format!("<｜begin▁of▁sentence｜><｜User｜>{turn}<｜Assistant｜></think>")
                } else {
                    format!("<｜User｜>{turn}<｜Assistant｜></think>")
                }
            }
            Self::Plain => {
                if first_turn {
                    format!("User: {turn}\nAssistant:")
                } else {
                    format!("\nUser: {turn}\nAssistant:")
                }
            }
        }
    }
}

fn validate_messages(messages: &[ChatMessage]) -> Result<(), ChatFormatError> {
    if messages.is_empty() {
        return Err(ChatFormatError::EmptyMessages);
    }

    for (index, message) in messages.iter().enumerate() {
        if message.content.trim().is_empty() {
            return Err(ChatFormatError::EmptyContent {
                index,
                role: message.role,
            });
        }
        if message.role == ChatRole::System && index != 0 {
            return Err(ChatFormatError::SystemMessageNotFirst { index });
        }
    }

    let first_conversation_index = usize::from(messages[0].role == ChatRole::System);
    if first_conversation_index == messages.len() {
        return Err(ChatFormatError::UnexpectedRole {
            index: 0,
            expected: ChatRole::User,
            actual: ChatRole::System,
        });
    }

    for (offset, message) in messages[first_conversation_index..].iter().enumerate() {
        let index = first_conversation_index + offset;
        let expected = if offset % 2 == 0 {
            ChatRole::User
        } else {
            ChatRole::Assistant
        };
        if message.role != expected {
            return Err(ChatFormatError::UnexpectedRole {
                index,
                expected,
                actual: message.role,
            });
        }
    }

    let last_role = messages
        .last()
        .expect("non-empty messages validated above")
        .role;
    if last_role != ChatRole::User {
        return Err(ChatFormatError::MustEndWithUser { actual: last_role });
    }

    Ok(())
}

fn format_chatml_messages(messages: &[ChatMessage]) -> String {
    let has_system = messages[0].role == ChatRole::System;
    let mut output = String::new();
    if !has_system {
        output.push_str("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");
    }
    for message in messages {
        output.push_str("<|im_start|>");
        output.push_str(message.role.as_str());
        output.push('\n');
        output.push_str(&message.content);
        output.push_str("<|im_end|>\n");
    }
    output.push_str("<|im_start|>assistant\n");
    output
}

fn format_llama3_messages(messages: &[ChatMessage]) -> String {
    let mut output = String::from("<|begin_of_text|>");
    for message in messages {
        output.push_str("<|start_header_id|>");
        output.push_str(message.role.as_str());
        output.push_str("<|end_header_id|>\n\n");
        output.push_str(&message.content);
        output.push_str("<|eot_id|>");
    }
    output.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    output
}

fn format_deepseek_v4_messages(messages: &[ChatMessage]) -> String {
    let mut output = String::from("<｜begin▁of▁sentence｜>");
    for message in messages {
        match message.role {
            // DeepSeek's official template places the system prompt directly
            // after BOS; it has no dedicated system-role control token.
            ChatRole::System => output.push_str(&message.content),
            ChatRole::User => {
                output.push_str("<｜User｜>");
                output.push_str(&message.content);
            }
            ChatRole::Assistant => {
                output.push_str("<｜Assistant｜></think>");
                output.push_str(&message.content);
                output.push_str("<｜end▁of▁sentence｜>");
            }
        }
    }
    output.push_str("<｜Assistant｜></think>");
    output
}

fn format_plain_messages(messages: &[ChatMessage]) -> String {
    let mut output = String::new();
    for (index, message) in messages.iter().enumerate() {
        if index != 0 {
            output.push('\n');
        }
        match message.role {
            ChatRole::System => output.push_str("System: "),
            ChatRole::User => output.push_str("User: "),
            ChatRole::Assistant => output.push_str("Assistant: "),
        }
        output.push_str(&message.content);
    }
    output.push_str("\nAssistant:");
    output
}

/// Auto-detect chat template from tokenizer_config.json.
pub fn detect_chat_template(model_dir: &Path) -> ChatTemplate {
    let path = model_dir.join("tokenizer_config.json");
    let Ok(text) = std::fs::read_to_string(path) else {
        return ChatTemplate::Plain;
    };

    // DeepSeek-V4 official encoding uses full-width special-token brackets.
    if text.contains("<｜begin▁of▁sentence｜>") && text.contains("<｜end▁of▁sentence｜>")
    {
        return ChatTemplate::DeepSeekV4;
    }

    // Qwen uses ChatML markers and often carries Qwen-specific tokenizer metadata.
    let lower = text.to_ascii_lowercase();
    if text.contains("<|im_start|>") && text.contains("<|im_end|>") && lower.contains("qwen") {
        return ChatTemplate::Qwen;
    }

    // Generic ChatML
    if text.contains("<|im_start|>") && text.contains("<|im_end|>") {
        return ChatTemplate::ChatML;
    }

    // Llama 3
    if text.contains("<|begin_of_text|>") || text.contains("<|eot_id|>") {
        return ChatTemplate::Llama3;
    }

    ChatTemplate::Plain
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek_v4_chat_template_matches_official_single_turn() {
        assert_eq!(
            ChatTemplate::DeepSeekV4.format_turn("Hello", true),
            "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>"
        );
    }

    #[test]
    fn deepseek_v4_chat_template_matches_official_followup_turn() {
        assert_eq!(
            ChatTemplate::DeepSeekV4.format_turn("How are you?", false),
            "<｜User｜>How are you?<｜Assistant｜></think>"
        );
    }

    #[test]
    fn complete_single_turn_golden_for_every_template() {
        let messages = [ChatMessage::user("Hello")];
        let cases = [
            (
                ChatTemplate::ChatML,
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
            ),
            (
                ChatTemplate::Qwen,
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
            ),
            (
                ChatTemplate::Llama3,
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            ),
            (
                ChatTemplate::DeepSeekV4,
                "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>",
            ),
            (ChatTemplate::Plain, "User: Hello\nAssistant:"),
        ];

        for (template, expected) in cases {
            assert_eq!(template.format_messages(&messages).unwrap(), expected);
        }
    }

    #[test]
    fn complete_system_multi_turn_golden_for_every_template() {
        let messages = [
            ChatMessage::system("Be concise."),
            ChatMessage::user("One?"),
            ChatMessage::assistant("1."),
            ChatMessage::user("Two?"),
        ];
        let cases = [
            (
                ChatTemplate::ChatML,
                "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nOne?<|im_end|>\n<|im_start|>assistant\n1.<|im_end|>\n<|im_start|>user\nTwo?<|im_end|>\n<|im_start|>assistant\n",
            ),
            (
                ChatTemplate::Qwen,
                "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nOne?<|im_end|>\n<|im_start|>assistant\n1.<|im_end|>\n<|im_start|>user\nTwo?<|im_end|>\n<|im_start|>assistant\n",
            ),
            (
                ChatTemplate::Llama3,
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe concise.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nOne?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n1.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nTwo?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            ),
            (
                ChatTemplate::DeepSeekV4,
                "<｜begin▁of▁sentence｜>Be concise.<｜User｜>One?<｜Assistant｜></think>1.<｜end▁of▁sentence｜><｜User｜>Two?<｜Assistant｜></think>",
            ),
            (
                ChatTemplate::Plain,
                "System: Be concise.\nUser: One?\nAssistant: 1.\nUser: Two?\nAssistant:",
            ),
        ];

        for (template, expected) in cases {
            assert_eq!(template.format_messages(&messages).unwrap(), expected);
        }
    }

    #[test]
    fn complete_messages_reject_empty_and_invalid_inputs() {
        assert_eq!(
            ChatTemplate::Plain.format_messages(&[]),
            Err(ChatFormatError::EmptyMessages)
        );
        assert_eq!(
            ChatTemplate::Plain.format_messages(&[ChatMessage::user("  ")]),
            Err(ChatFormatError::EmptyContent {
                index: 0,
                role: ChatRole::User,
            })
        );
        assert_eq!(
            ChatTemplate::Plain
                .format_messages(&[ChatMessage::user("Hello"), ChatMessage::system("late"),]),
            Err(ChatFormatError::SystemMessageNotFirst { index: 1 })
        );
        assert!(
            ChatTemplate::Plain
                .format_messages(&[ChatMessage::assistant("Hello")])
                .is_err()
        );
        assert!(
            ChatTemplate::Plain
                .format_messages(&[ChatMessage::user("Hello"), ChatMessage::assistant("Hi"),])
                .is_err()
        );
    }

    #[test]
    fn format_turn_remains_compatible_with_single_user_messages() {
        for template in [
            ChatTemplate::ChatML,
            ChatTemplate::Llama3,
            ChatTemplate::Qwen,
            ChatTemplate::DeepSeekV4,
            ChatTemplate::Plain,
        ] {
            assert_eq!(
                template
                    .format_messages(&[ChatMessage::user("Hello")])
                    .unwrap(),
                template.format_turn("Hello", true)
            );
        }
    }

    #[test]
    fn deepseek_v4_template_aliases_parse() {
        assert_eq!(
            ChatTemplate::from_name("dsv4"),
            Some(ChatTemplate::DeepSeekV4)
        );
        assert_eq!(
            ChatTemplate::from_name("deepseek-v4"),
            Some(ChatTemplate::DeepSeekV4)
        );
    }
}
