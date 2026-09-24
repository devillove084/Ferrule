//! Goldens rendered by Jinja2 from the actual NAS Qwen3.5-0.8B template.
use ferrule_model::chat::ChatTemplateOptions;
use ferrule_model::{ChatMessage, ChatTemplate};

#[test]
fn qwen35_text_template_matches_actual_jinja_goldens() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/qwen35_chat.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let messages: Vec<ChatMessage> = serde_json::from_value(case["messages"].clone()).unwrap();
        let generation = case["add_generation_prompt"].as_bool().unwrap();
        let actual = if case["enable_thinking"].is_null() && generation {
            ChatTemplate::Qwen35.format_messages(&messages).unwrap()
        } else {
            ChatTemplate::Qwen35
                .format_messages_with_options(
                    &messages,
                    ChatTemplateOptions {
                        add_generation_prompt: generation,
                        enable_thinking: case["enable_thinking"].as_bool().unwrap_or(false),
                    },
                )
                .unwrap()
        };
        assert_eq!(actual, case["expected"].as_str().unwrap());
    }
    assert_eq!(
        ChatTemplate::from_name("qwen3.5"),
        Some(ChatTemplate::Qwen35)
    );
    assert_eq!(
        ChatTemplate::Qwen35.format_turn("  Hi \n", true),
        ChatTemplate::Qwen35
            .format_messages(&[ChatMessage::user("Hi")])
            .unwrap()
    );
}

#[test]
fn retained_qwen35_turn_closes_the_uncommitted_eos_marker() {
    let next = ChatTemplate::Qwen35.format_turn(" next ", false);
    assert_eq!(
        next,
        "<|im_end|>\n<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}
