use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    OlmoeInstruct,
    ChatML,
    Llama3,
    Qwen,
    Plain,
}

impl ChatTemplate {
    pub fn name(self) -> &'static str {
        match self {
            Self::OlmoeInstruct => "olmoe-instruct",
            Self::ChatML => "chatml",
            Self::Llama3 => "llama3",
            Self::Qwen => "qwen",
            Self::Plain => "plain",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "olmoe" | "olmoe-instruct" => Some(Self::OlmoeInstruct),
            "chatml" => Some(Self::ChatML),
            "llama3" | "llama-3" => Some(Self::Llama3),
            "qwen" => Some(Self::Qwen),
            "plain" | "none" => Some(Self::Plain),
            _ => None,
        }
    }

    pub fn format_turn(self, turn: &str, first_turn: bool) -> String {
        match self {
            Self::OlmoeInstruct => {
                if first_turn {
                    format!("<|endoftext|>\n<|user|>\n{turn}\n<|assistant|>")
                } else {
                    format!("\n<|user|>\n{turn}\n<|assistant|>")
                }
            }
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

    // OLMoE: <|user|> + <|assistant|>
    if text.contains("<|user|>") && text.contains("<|assistant|>") {
        return ChatTemplate::OlmoeInstruct;
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
