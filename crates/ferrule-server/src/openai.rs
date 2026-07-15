use ferrule_model::{ChatMessage, ChatRole, ChatTemplate};
use ferrule_runtime::SequenceFinishReason;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<OpenAiChatMessage>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default = "default_one")]
    pub n: usize,
    #[serde(default)]
    pub stop: Option<StopInput>,
    #[serde(default)]
    pub ignore_eos: bool,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<usize>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    #[serde(default)]
    pub user: Option<String>,
}

const fn default_one() -> usize {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiChatMessage {
    pub role: ChatRole,
    pub content: OpenAiChatContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAiChatContent {
    Text(String),
    Parts(Vec<OpenAiChatContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiChatContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
}

impl OpenAiChatMessage {
    fn into_model_message(self) -> Result<ChatMessage, String> {
        let content = match self.content {
            OpenAiChatContent::Text(text) => text,
            OpenAiChatContent::Parts(parts) => {
                if parts.is_empty() {
                    return Err("chat content parts must not be empty".into());
                }
                let mut text = String::new();
                for (index, part) in parts.into_iter().enumerate() {
                    if part.kind != "text" {
                        return Err(format!(
                            "chat content part {index} has unsupported type '{}'",
                            part.kind
                        ));
                    }
                    let part_text = part.text.ok_or_else(|| {
                        format!("chat text content part {index} is missing its text field")
                    })?;
                    text.push_str(&part_text);
                }
                text
            }
        };
        Ok(ChatMessage::new(self.role, content))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopInput {
    One(String),
    Many(Vec<String>),
}

impl StopInput {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug)]
pub(crate) struct ValidatedGenerationRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub stream: bool,
    pub include_usage: bool,
    pub stop: Vec<String>,
    pub ignore_eos: bool,
}

impl ChatCompletionRequest {
    pub(crate) fn validate(
        self,
        expected_model: &str,
        template: ChatTemplate,
    ) -> Result<ValidatedGenerationRequest, String> {
        validate_greedy_parameters(
            GreedyParameters {
                model: &self.model,
                n: self.n,
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                min_p: self.min_p,
                repetition_penalty: self.repetition_penalty,
            },
            expected_model,
        )?;
        if self.frequency_penalty.is_some_and(|value| value != 0.0)
            || self.presence_penalty.is_some_and(|value| value != 0.0)
        {
            return Err("frequency_penalty and presence_penalty are not supported".into());
        }
        if self.logprobs == Some(true) || self.top_logprobs.is_some() {
            return Err("streaming logprobs are not supported".into());
        }
        if self.tools.is_some() || self.tool_choice.is_some() {
            return Err("tool calling is not supported".into());
        }
        if self.response_format.is_some() {
            return Err("response_format is not supported".into());
        }
        if self.max_completion_tokens.is_some() && self.max_tokens.is_some() {
            return Err("provide only one of max_completion_tokens or max_tokens".into());
        }
        let max_tokens = self.max_completion_tokens.or(self.max_tokens).unwrap_or(16);
        if max_tokens == 0 {
            return Err("max_completion_tokens must be greater than zero".into());
        }
        let stop = validate_stop(self.stop)?;

        // Seed and user are accepted because greedy execution is deterministic and
        // neither field changes model semantics in that mode.
        let _ = (self.seed, self.user);
        let messages = self
            .messages
            .into_iter()
            .map(OpenAiChatMessage::into_model_message)
            .collect::<Result<Vec<_>, _>>()?;
        let prompt = template
            .format_messages(&messages)
            .map_err(|error| error.to_string())?;

        Ok(ValidatedGenerationRequest {
            prompt,
            max_tokens,
            stream: self.stream,
            include_usage: self
                .stream_options
                .is_some_and(|options| options.include_usage),
            stop,
            ignore_eos: self.ignore_eos,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: serde_json::Value,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default = "default_one")]
    pub n: usize,
    #[serde(default = "default_one")]
    pub best_of: usize,
    #[serde(default)]
    pub stop: Option<StopInput>,
    #[serde(default)]
    pub ignore_eos: bool,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    logprobs: Option<usize>,
    #[serde(default)]
    echo: bool,
}

impl CompletionRequest {
    pub(crate) fn validate(
        self,
        expected_model: &str,
    ) -> Result<ValidatedGenerationRequest, String> {
        validate_greedy_parameters(
            GreedyParameters {
                model: &self.model,
                n: self.n,
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                min_p: self.min_p,
                repetition_penalty: self.repetition_penalty,
            },
            expected_model,
        )?;
        if self.best_of != 1 {
            return Err("Ferrule currently supports best_of = 1 only".into());
        }
        if self.logprobs.is_some() {
            return Err("logprobs is not supported".into());
        }
        if self.echo {
            return Err("echo is not supported".into());
        }
        if self.stream_options.is_some() && !self.stream {
            return Err("stream_options is only supported when stream = true".into());
        }

        let prompt = match self.prompt {
            serde_json::Value::String(prompt) => prompt,
            serde_json::Value::Array(_) => {
                return Err(
                    "batch prompts are not supported; prompt must be a single string".into(),
                );
            }
            _ => return Err("prompt must be a single string".into()),
        };
        let max_tokens = self.max_tokens.unwrap_or(16);
        if max_tokens == 0 {
            return Err("max_tokens must be greater than zero".into());
        }
        let stop = validate_stop(self.stop)?;

        // Seed is accepted because greedy execution is deterministic.
        let _ = self.seed;
        Ok(ValidatedGenerationRequest {
            prompt,
            max_tokens,
            stream: self.stream,
            include_usage: self
                .stream_options
                .is_some_and(|options| options.include_usage),
            stop,
            ignore_eos: self.ignore_eos,
        })
    }
}

struct GreedyParameters<'a> {
    model: &'a str,
    n: usize,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
}

fn validate_greedy_parameters(
    parameters: GreedyParameters<'_>,
    expected_model: &str,
) -> Result<(), String> {
    if parameters.model != expected_model {
        return Err(format!(
            "model '{}' is not served; available model is '{expected_model}'",
            parameters.model
        ));
    }
    if parameters.n != 1 {
        return Err("Ferrule currently supports exactly one completion per request (n = 1)".into());
    }
    if parameters.temperature.is_some_and(|value| value != 0.0) {
        return Err(
            "Ferrule resident serving currently supports greedy temperature = 0 only".into(),
        );
    }
    if parameters.top_p.is_some_and(|value| value != 1.0) {
        return Err("Ferrule resident serving currently requires top_p = 1".into());
    }
    if parameters.top_k.is_some_and(|value| value != 1) {
        return Err("Ferrule resident serving currently requires top_k = 1".into());
    }
    if parameters.min_p.is_some_and(|value| value != 0.0) {
        return Err("Ferrule resident serving currently requires min_p = 0".into());
    }
    if parameters
        .repetition_penalty
        .is_some_and(|value| value != 1.0)
    {
        return Err("Ferrule resident serving currently requires repetition_penalty = 1".into());
    }
    Ok(())
}

fn validate_stop(stop: Option<StopInput>) -> Result<Vec<String>, String> {
    let stop = stop.map(StopInput::into_vec).unwrap_or_default();
    if stop.iter().any(String::is_empty) {
        return Err("stop strings must not be empty".into());
    }
    if stop.len() > 4 {
        return Err("at most four stop strings are supported".into());
    }
    Ok(stop)
}

#[derive(Debug, Serialize)]
pub(crate) struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChatCompletionChunk<'a> {
    pub id: &'a str,
    pub object: &'static str,
    pub created: u64,
    pub model: &'a str,
    pub choices: Vec<ChunkChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChunkChoice<'a> {
    pub index: usize,
    pub delta: ChunkDelta<'a>,
    pub finish_reason: Option<&'a str>,
}

#[derive(Debug, Default, Serialize)]
pub(crate) struct ChunkDelta<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<&'a str>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChatCompletionResponse<'a> {
    pub id: &'a str,
    pub object: &'static str,
    pub created: u64,
    pub model: &'a str,
    pub choices: Vec<ResponseChoice<'a>>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub(crate) struct ResponseChoice<'a> {
    pub index: usize,
    pub message: AssistantMessage<'a>,
    pub finish_reason: &'a str,
}

#[derive(Debug, Serialize)]
pub(crate) struct AssistantMessage<'a> {
    pub role: &'static str,
    pub content: &'a str,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    pub(crate) fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct CompletionChunk<'a> {
    pub id: &'a str,
    pub object: &'static str,
    pub created: u64,
    pub model: &'a str,
    pub choices: Vec<TextCompletionChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub(crate) struct CompletionResponse<'a> {
    pub id: &'a str,
    pub object: &'static str,
    pub created: u64,
    pub model: &'a str,
    pub choices: Vec<TextCompletionChoice<'a>>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub(crate) struct TextCompletionChoice<'a> {
    pub text: &'a str,
    pub index: usize,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<&'a str>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ErrorEnvelope<'a> {
    pub error: ErrorObject<'a>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ErrorObject<'a> {
    pub message: &'a str,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub param: Option<&'a str>,
    pub code: Option<&'a str>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizeRequest {
    #[serde(default)]
    model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_true")]
    add_special_tokens: bool,
}

const fn default_true() -> bool {
    true
}

impl TokenizeRequest {
    pub(crate) fn validate(self, expected_model: &str) -> Result<String, String> {
        if let Some(model) = &self.model
            && !model.is_empty()
            && model != expected_model
        {
            return Err(format!("model '{model}' is not served by this server"));
        }
        if !self.add_special_tokens {
            return Err("add_special_tokens=false is not supported".into());
        }
        Ok(self.prompt)
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct TokenizeResponse<'a> {
    pub id: &'a str,
    pub object: &'static str,
    pub created: u64,
    pub model: &'a str,
    pub data: Vec<TokenizeData>,
}

#[derive(Debug, Serialize)]
pub(crate) struct TokenizeData {
    pub object: &'static str,
    pub tokens: Vec<u32>,
    pub count: usize,
}

pub(crate) fn openai_finish_reason(reason: SequenceFinishReason) -> &'static str {
    match reason {
        SequenceFinishReason::MaxTokens | SequenceFinishReason::Context => "length",
        SequenceFinishReason::Eos
        | SequenceFinishReason::StopString
        | SequenceFinishReason::NoCandidate => "stop",
        SequenceFinishReason::Cancelled => "cancelled",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 8,
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "min_p": 0,
            "stream": true,
            "stream_options": {"include_usage": true},
            "ignore_eos": true
        }))
        .unwrap()
    }

    #[test]
    fn validates_official_greedy_benchmark_shape() {
        let validated = request()
            .validate("test", ChatTemplate::DeepSeekV4)
            .unwrap();
        assert!(validated.stream);
        assert!(validated.include_usage);
        assert!(validated.ignore_eos);
        assert_eq!(validated.max_tokens, 8);
        assert!(validated.prompt.contains("<｜User｜>hello"));
    }

    #[test]
    fn accepts_vllm_text_content_parts_for_chat() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "hello"}]
            }],
            "max_completion_tokens": 8,
            "stream": true,
            "stream_options": {"include_usage": true}
        }))
        .unwrap();
        let validated = request.validate("test", ChatTemplate::DeepSeekV4).unwrap();
        assert!(validated.prompt.contains("<｜User｜>hello"));
    }

    #[test]
    fn rejects_non_text_chat_content_parts() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "x"}}]
            }]
        }))
        .unwrap();
        assert!(request.validate("test", ChatTemplate::Plain).is_err());
    }

    #[test]
    fn rejects_parameters_the_driver_does_not_implement() {
        let mut request = request();
        request.temperature = Some(0.7);
        assert!(request.validate("test", ChatTemplate::Plain).is_err());
    }

    #[test]
    fn accepts_string_or_array_stop() {
        for stop in [serde_json::json!("END"), serde_json::json!(["A", "B"])] {
            let mut value = serde_json::to_value(request()).unwrap();
            value["stop"] = stop;
            let parsed: ChatCompletionRequest = serde_json::from_value(value).unwrap();
            assert!(
                !parsed
                    .validate("test", ChatTemplate::Plain)
                    .unwrap()
                    .stop
                    .is_empty()
            );
        }
    }

    fn completion(value: serde_json::Value) -> CompletionRequest {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn validates_text_completion_greedy_shape() {
        let validated = completion(serde_json::json!({
            "model": "test",
            "prompt": "hello",
            "max_tokens": 8,
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "min_p": 0,
            "repetition_penalty": 1,
            "stop": ["A", "B"],
            "ignore_eos": true,
            "seed": 42,
            "stream": true,
            "stream_options": {"include_usage": true}
        }))
        .validate("test")
        .unwrap();
        assert_eq!(validated.prompt, "hello");
        assert_eq!(validated.max_tokens, 8);
        assert_eq!(validated.stop, ["A", "B"]);
        assert!(validated.ignore_eos);
        assert!(validated.stream);
        assert!(validated.include_usage);
    }

    #[test]
    fn rejects_batch_and_unsupported_completion_parameters() {
        for (field, value) in [
            ("prompt", serde_json::json!(["one", "two"])),
            ("n", serde_json::json!(2)),
            ("best_of", serde_json::json!(2)),
            ("logprobs", serde_json::json!(1)),
            ("echo", serde_json::json!(true)),
        ] {
            let mut request = serde_json::json!({"model": "test", "prompt": "hello"});
            request[field] = value;
            assert!(completion(request).validate("test").is_err(), "{field}");
        }
    }

    #[test]
    fn accepts_vllm_completion_logprobs_null() {
        let request = completion(serde_json::json!({
            "model": "test",
            "prompt": "hello",
            "repetition_penalty": 1.0,
            "max_tokens": 8,
            "logprobs": null,
            "stream": true,
            "stream_options": {"include_usage": true}
        }));
        assert!(request.validate("test").is_ok());
    }

    #[test]
    fn rejects_unknown_completion_parameters_during_deserialization() {
        let parsed = serde_json::from_value::<CompletionRequest>(serde_json::json!({
            "model": "test",
            "prompt": "hello",
            "suffix": "not supported"
        }));
        assert!(parsed.is_err());
    }
}
