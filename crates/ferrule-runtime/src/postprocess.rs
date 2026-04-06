pub fn truncate_on_role_marker(text: &str) -> String {
    let markers = [
        "\nassistant",
        "\nuser",
        "\nsystem",
        "<|assistant|>",
        "<|user|>",
        "<|system|>",
    ];
    let mut end = text.len();

    for marker in markers {
        if let Some(idx) = text.find(marker) {
            end = end.min(idx);
        }
    }

    text[..end].trim().to_string()
}

pub fn normalize_answer(text: &str) -> String {
    truncate_on_role_marker(text)
        .trim_matches('"')
        .trim()
        .to_string()
}
