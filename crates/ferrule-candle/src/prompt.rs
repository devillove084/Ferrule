#[derive(Debug, Clone)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

pub fn render_chat_prompt(template: &str, messages: &[ChatMessage]) -> String {
    match template {
        "plain" => render_plain(messages),
        "instruct_v1" => render_instruct_v1(messages),
        _ => render_plain(messages),
    }
}

fn render_plain(messages: &[ChatMessage]) -> String {
    let mut out = String::new();

    for msg in messages {
        let role = match msg.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Tool => "tool",
        };

        out.push_str(role);
        out.push_str(": ");
        out.push_str(&msg.content);
        out.push('\n');
    }

    out.push_str("assistant: ");
    out
}

fn render_instruct_v1(messages: &[ChatMessage]) -> String {
    let mut out = String::new();

    for msg in messages {
        match msg.role {
            ChatRole::System => {
                out.push_str("<|system|>\n");
                out.push_str(&msg.content);
                out.push('\n');
            }
            ChatRole::User => {
                out.push_str("<|user|>\n");
                out.push_str(&msg.content);
                out.push('\n');
            }
            ChatRole::Assistant => {
                out.push_str("<|assistant|>\n");
                out.push_str(&msg.content);
                out.push('\n');
            }
            ChatRole::Tool => {
                out.push_str("<|tool|>\n");
                out.push_str(&msg.content);
                out.push('\n');
            }
        }
    }

    out.push_str("<|assistant|>\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_plain_prompt() {
        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: "You are helpful.".to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: "Hello".to_string(),
            },
        ];

        let out = render_chat_prompt("plain", &messages);
        assert!(out.contains("system: You are helpful."));
        assert!(out.contains("user: Hello"));
        assert!(out.ends_with("assistant: "));
    }
}
