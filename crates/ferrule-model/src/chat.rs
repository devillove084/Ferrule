use std::path::Path;

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
