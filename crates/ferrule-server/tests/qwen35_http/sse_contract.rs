//! Strict contract for Ferrule's one-token-event-per-chunk, include_usage SSE.
//! Token events (including empty text) are not visible characters or GPU timing.
use serde_json::Value;

#[derive(Debug, Default)]
pub struct Sample {
    pending: Vec<u8>,
    data: Vec<String>,
    pub id: Option<String>,
    object: Option<String>,
    pub tokens: usize,
    pub text: String,
    pub finish_reason: Option<String>,
    pub usage: Option<(u64, u64, u64)>,
    done: bool,
}

impl Sample {
    pub fn parse(body: &str) -> Result<Self, String> {
        let mut sample = Self::default();
        sample.push(body.as_bytes())?;
        sample.finish()?;
        Ok(sample)
    }

    // Body frames need not align with SSE events, lines, or UTF-8 codepoints.
    pub fn push(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.pending.extend_from_slice(bytes);
        while let Some(end) = self.pending.iter().position(|byte| *byte == b'\n') {
            let line: Vec<_> = self.pending.drain(..=end).collect();
            let line = std::str::from_utf8(&line[..line.len() - 1])
                .map_err(|error| format!("invalid SSE UTF-8: {error}"))?;
            let line = line.strip_suffix('\r').unwrap_or(line);
            if line.is_empty() {
                if !self.data.is_empty() {
                    let data = std::mem::take(&mut self.data).join("\n");
                    self.event(&data)?;
                }
            } else if line.starts_with(':') {
                // Keepalive comments are not token events.
            } else if let Some(data) = line.strip_prefix("data:") {
                self.data
                    .push(data.strip_prefix(' ').unwrap_or(data).into());
            } else {
                return Err(format!("unexpected SSE field: {line}"));
            }
        }
        Ok(())
    }

    fn event(&mut self, data: &str) -> Result<(), String> {
        if self.done {
            return Err("event after DONE (including duplicate DONE)".into());
        }
        if data == "[DONE]" {
            if self.finish_reason.is_none() || self.usage.is_none() {
                return Err("DONE before terminal/usage".into());
            }
            self.done = true;
            return Ok(());
        }
        let event: Value = serde_json::from_str(data).map_err(|error| error.to_string())?;
        if !event.is_object() || event.get("error").is_some() {
            return Err(format!("SSE error or non-object event: {event}"));
        }
        let id = event["id"]
            .as_str()
            .filter(|id| !id.is_empty())
            .ok_or("missing id")?;
        let object = event["object"].as_str().ok_or("missing object")?;
        if !matches!(object, "text_completion" | "chat.completion.chunk") {
            return Err(format!("unexpected object: {object}"));
        }
        if self.id.as_deref().is_some_and(|previous| previous != id)
            || self
                .object
                .as_deref()
                .is_some_and(|previous| previous != object)
        {
            return Err("SSE identity changed".into());
        }
        self.id = Some(id.into());
        self.object = Some(object.into());
        let choices = event["choices"].as_array().ok_or("missing choices array")?;
        if let Some(usage) = event.get("usage").filter(|usage| !usage.is_null()) {
            if !choices.is_empty() || self.finish_reason.is_none() || self.usage.is_some() {
                return Err("duplicate or out-of-order usage".into());
            }
            let prompt = required_u64(usage, "prompt_tokens")?;
            let completion = required_u64(usage, "completion_tokens")?;
            let total = required_u64(usage, "total_tokens")?;
            if prompt.checked_add(completion) != Some(total) || completion != self.tokens as u64 {
                return Err(format!(
                    "usage/token conservation failed: {usage}, events={}",
                    self.tokens
                ));
            }
            self.usage = Some((prompt, completion, total));
            return Ok(());
        }
        if choices.len() != 1 || self.finish_reason.is_some() || self.usage.is_some() {
            return Err("duplicate terminal, late token, or invalid choices".into());
        }
        let choice = &choices[0];
        if choice["index"].as_u64() != Some(0) {
            return Err("unexpected choice index".into());
        }
        let reason = choice.get("finish_reason").ok_or("missing finish_reason")?;
        let text = if object == "text_completion" {
            &choice["text"]
        } else {
            &choice["delta"]["content"]
        };
        if reason.is_null() {
            let text = text.as_str().ok_or("token chunk missing text/content")?;
            self.tokens += 1;
            self.text.push_str(text);
        } else {
            let reason = reason
                .as_str()
                .filter(|reason| matches!(*reason, "length" | "stop"))
                .ok_or("invalid finish reason")?;
            if !(text.is_null() || text.as_str() == Some("")) {
                return Err("terminal contains uncounted token text".into());
            }
            self.finish_reason = Some(reason.into());
        }
        Ok(())
    }

    pub fn finish(&self) -> Result<(), String> {
        if !self.pending.is_empty() || !self.data.is_empty() || !self.done {
            return Err("truncated SSE or missing DONE".into());
        }
        Ok(())
    }

    pub fn require_length(
        &self,
        expected_tokens: usize,
        prompt: Option<usize>,
    ) -> Result<(), String> {
        self.finish()?;
        let (actual_prompt, completion, _) = self.usage.ok_or("missing usage")?;
        if self.finish_reason.as_deref() != Some("length")
            || self.tokens != expected_tokens
            || completion != expected_tokens as u64
            || prompt.is_some_and(|expected| actual_prompt != expected as u64)
        {
            return Err(format!("incorrect exact token/terminal contract: {self:?}"));
        }
        Ok(())
    }

    pub fn request_id(&self) -> Result<u64, String> {
        let id = self.id.as_deref().ok_or("no token identity")?;
        id.strip_prefix("cmpl-ferrule-")
            .or_else(|| id.strip_prefix("chatcmpl-ferrule-"))
            .ok_or("not a Ferrule completion id")?
            .parse()
            .map_err(|error| format!("invalid request id: {error}"))
    }
}

pub fn required_u64(value: &Value, key: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing unsigned integer {key}: {value}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn event(choice: Value, usage: Value) -> String {
        format!(
            "data: {}\n\n",
            json!({"id":"cmpl-ferrule-7", "object":"text_completion", "choices":choice,"usage":usage})
        )
    }
    fn token(text: &str) -> String {
        event(
            json!([{"index":0,"text":text,"finish_reason":null}]),
            Value::Null,
        )
    }
    fn terminal() -> String {
        event(
            json!([{"index":0,"text":"","finish_reason":"length"}]),
            Value::Null,
        )
    }
    fn usage(completion: Value, total: Value) -> String {
        event(
            json!([]),
            json!({"prompt_tokens":3,"completion_tokens":completion,"total_tokens":total}),
        )
    }
    fn valid() -> String {
        format!(
            ": keepalive\n\n{}{}{}{}data: [DONE]\n\n",
            token(""),
            token("河"),
            terminal(),
            usage(json!(2), json!(5))
        )
    }

    #[test]
    fn strict_sse_accepts_fragmented_utf8_crlf_empty_tokens_and_coalesced_frames() {
        let body = valid().replace('\n', "\r\n");
        let mut sample = Sample::default();
        for byte in body.as_bytes() {
            sample.push(&[*byte]).unwrap();
        }
        sample.require_length(2, Some(3)).unwrap();
        assert_eq!(sample.text, "河");
        assert_eq!(sample.request_id().unwrap(), 7);
        Sample::parse(&valid())
            .unwrap()
            .require_length(2, Some(3))
            .unwrap();
    }

    #[test]
    fn strict_sse_rejects_numeric_prefix_error_whitespace_and_absent_tokens() {
        for body in [
            valid().replace("\"completion_tokens\":2", "\"completion_tokens\":20"),
            format!(
                "{}{}data: [DONE]\n\n",
                terminal(),
                usage(json!(20), json!(23))
            ),
            valid().replace(
                "data: [DONE]",
                "data: {\"error\" : {\"message\":\"failed\"}}\n\ndata: [DONE]",
            ),
            valid().replace(&token(""), ""),
            valid().replace(&usage(json!(2), json!(5)), &usage(json!("2"), json!(5))),
            valid().replace(&usage(json!(2), json!(5)), &usage(json!(2), json!(6))),
        ] {
            assert!(
                Sample::parse(&body)
                    .and_then(|sample| sample.require_length(2, None))
                    .is_err(),
                "{body}"
            );
        }
    }

    #[test]
    fn strict_sse_rejects_missing_duplicate_out_of_order_and_post_done_events() {
        for body in [
            valid().replace("data: [DONE]\n\n", ""),
            format!("{}data: [DONE]\n\n", valid()),
            format!("{}{}", valid(), token("late")),
            valid().replace(&terminal(), &format!("{}{}", terminal(), terminal())),
            valid().replace(&terminal(), ""),
            valid().replace(&terminal(), &format!("{}{}", terminal(), token("late"))),
            format!("{}{}", usage(json!(0), json!(3)), valid()),
            valid().replace(&usage(json!(2), json!(5)), ""),
            valid().trim_end().to_string(),
            valid().replacen("cmpl-ferrule-7", "cmpl-ferrule-8", 1),
        ] {
            assert!(Sample::parse(&body).is_err(), "{body}");
        }
    }

    #[test]
    fn strict_sse_requires_exact_expected_usage_and_real_integer_gauges() {
        let sample = Sample::parse(&valid()).unwrap();
        assert!(sample.require_length(20, None).is_err());
        assert!(sample.require_length(2, Some(30)).is_err());
        for value in [
            json!({}),
            json!({"free":null}),
            json!({"free":"0"}),
            json!({"free":-1}),
            json!({"free":0.0}),
        ] {
            assert!(required_u64(&value, "free").is_err());
        }
        assert_eq!(required_u64(&json!({"free":0}), "free").unwrap(), 0);
    }
}
