//! Minimal OpenAI-compatible HTTP server.
//!
//! No external HTTP dependency — std::net + manual HTTP/1.1.
//! One request at a time (serialized by mutex), supports streaming via SSE.

#![cfg_attr(not(feature = "cuda"), allow(dead_code, unused_imports))]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use ferrule_runtime::ChatTemplate;

/// A generation function: (prompt, max_tokens) -> (response_text, prompt_tokens, generated_tokens).
pub type GenFn =
    Box<dyn Fn(&str, usize) -> anyhow::Result<(String, usize, usize)> + Send + 'static>;

/// A streaming generation function: calls `on_token` with each generated text piece.
/// Returns (prompt_tokens, generated_tokens).
pub type StreamingGenFn = Box<
    dyn Fn(&str, usize, Box<dyn Fn(&str) + Send>) -> anyhow::Result<(usize, usize)>
        + Send
        + 'static,
>;

/// Emit a Server-Sent Events chunk.
fn write_sse_chunk(stream: &mut std::net::TcpStream, data: &str) {
    let _ = write!(stream, "data: {data}\n\n");
    let _ = stream.flush();
}

pub fn run(
    generate: GenFn,
    generate_stream: Option<StreamingGenFn>,
    model_name: String,
    template: ChatTemplate,
    host: &str,
    port: u16,
) -> anyhow::Result<()> {
    let gen = Arc::new(Mutex::new(generate));
    let gen_stream = generate_stream.map(|g| Arc::new(Mutex::new(g)));
    let listener =
        TcpListener::bind((host, port)).map_err(|e| anyhow::anyhow!("bind {host}:{port}: {e}"))?;

    println!(
        "{} listening on http://{host}:{port}",
        console::style("Server ready.").cyan()
    );
    println!("  GET /health");
    println!("  GET /v1/models");
    println!("  POST /v1/chat/completions");

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let gen = gen.clone();
        let gen_stream = gen_stream.clone();
        let model_name = model_name.clone();

        std::thread::spawn(move || {
            handle(stream, gen, gen_stream, template, &model_name);
        });
    }

    Ok(())
}

fn handle(
    mut stream: std::net::TcpStream,
    gen: Arc<Mutex<GenFn>>,
    gen_stream: Option<Arc<Mutex<StreamingGenFn>>>,
    template: ChatTemplate,
    model_name: &str,
) {
    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    });

    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        respond(&mut stream, 400, "");
        return;
    }
    let method = parts[0];
    let path = parts[1];

    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() {
            break;
        }
        if line.trim().is_empty() {
            break;
        }
        if line.to_ascii_lowercase().starts_with("content-length:") {
            content_length = line["content-length:".len()..].trim().parse().unwrap_or(0);
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        let _ = reader.read_exact(&mut body);
    }

    match (method, path) {
        ("GET", "/health") => {
            respond_json(&mut stream, 200, r#"{"status":"ok"}"#);
        }
        ("GET", "/v1/models") => {
            let resp = serde_json::json!({
                "object": "list",
                "data": [{"id": model_name, "object": "model", "owned_by": "ferrule"}]
            });
            respond_json(&mut stream, 200, &serde_json::to_string(&resp).unwrap());
        }
        ("POST", "/v1/chat/completions") => {
            let body_str = String::from_utf8_lossy(&body);
            let req: serde_json::Value = match serde_json::from_str(&body_str) {
                Ok(v) => v,
                Err(_) => {
                    respond(&mut stream, 400, "");
                    return;
                }
            };

            let max_tokens = req["max_tokens"].as_u64().unwrap_or(256).min(4096) as usize;
            let prompt = build_prompt(&req, template);
            let stream_requested = req["stream"].as_bool().unwrap_or(false);

            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            let completion_id = format!("ferrule-{now}");

            if stream_requested {
                // ── Streaming path ──────────────────────────────────────
                let gen_stream = match gen_stream {
                    Some(ref gs) => gs,
                    None => {
                        respond(&mut stream, 500, "");
                        return;
                    }
                };

                let gs = match gen_stream.lock() {
                    Ok(g) => g,
                    Err(_) => {
                        respond(&mut stream, 500, "");
                        return;
                    }
                };

                // Write SSE headers. We intentionally omit `Transfer-Encoding: chunked`:
                // this minimal std::net server streams by flushing SSE frames and
                // closing the connection, not by emitting HTTP chunk-size records.
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
                );
                let _ = stream.flush();

                let stream = Arc::new(Mutex::new(stream));
                let model_name = model_name.to_string();

                let on_token: Box<dyn Fn(&str) + Send> = Box::new({
                    let stream = stream.clone();
                    let cid = completion_id.clone();
                    let mn = model_name.clone();
                    move |token_text: &str| {
                        let chunk = serde_json::json!({
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": now,
                            "model": mn,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": token_text},
                                "finish_reason": serde_json::Value::Null
                            }]
                        });
                        let mut s = stream.lock().unwrap();
                        write_sse_chunk(&mut s, &serde_json::to_string(&chunk).unwrap());
                    }
                });

                match gs(&prompt, max_tokens, on_token) {
                    Ok((prompt_tokens, generated_tokens)) => {
                        let finish_reason = if generated_tokens >= max_tokens {
                            "length"
                        } else {
                            "stop"
                        };

                        // Emit final chunk with finish_reason
                        let final_chunk = serde_json::json!({
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": now,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason
                            }],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": generated_tokens,
                                "total_tokens": prompt_tokens + generated_tokens
                            }
                        });
                        let mut s = stream.lock().unwrap();
                        write_sse_chunk(&mut s, &serde_json::to_string(&final_chunk).unwrap());
                        // Emit [DONE] message
                        write_sse_chunk(&mut s, "[DONE]");
                    }
                    Err(_) => {
                        // Stream error — can't change status, just close
                    }
                }
            } else {
                // ── Non-streaming path ──────────────────────────────────
                let gen = match gen.lock() {
                    Ok(g) => g,
                    Err(_) => {
                        respond(&mut stream, 500, "");
                        return;
                    }
                };

                match gen(&prompt, max_tokens) {
                    Ok((text, prompt_tokens, generated_tokens)) => {
                        let finish_reason = if generated_tokens >= max_tokens {
                            "length"
                        } else {
                            "stop"
                        };
                        let resp = serde_json::json!({
                            "id": format!("ferrule-{now}"),
                            "object": "chat.completion",
                            "created": now,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": finish_reason
                            }],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": generated_tokens,
                                "total_tokens": prompt_tokens + generated_tokens
                            }
                        });
                        respond_json(&mut stream, 200, &serde_json::to_string(&resp).unwrap());
                    }
                    Err(_) => {
                        respond(&mut stream, 500, "");
                    }
                }
            }
        }
        _ => {
            respond(&mut stream, 404, "");
        }
    }
}

fn build_prompt(req: &serde_json::Value, template: ChatTemplate) -> String {
    let messages = &req["messages"];
    if let Some(arr) = messages.as_array() {
        let mut s = String::new();
        let mut first_message = true;
        let mut last_role = "user";

        for msg in arr {
            let role = msg["role"].as_str().unwrap_or("user");
            let content = msg["content"].as_str().unwrap_or("");
            append_chat_message(&mut s, template, role, content, first_message);
            first_message = false;
            last_role = role;
        }

        if last_role != "assistant" {
            s.push_str(assistant_prefix(template));
        }
        s
    } else {
        req["prompt"].as_str().unwrap_or("").to_string()
    }
}

fn append_chat_message(
    out: &mut String,
    template: ChatTemplate,
    role: &str,
    content: &str,
    first_message: bool,
) {
    match template {
        ChatTemplate::OlmoeInstruct => {
            if first_message {
                out.push_str("<|endoftext|>\n");
            }
            match role {
                "system" => out.push_str(&format!("{content}\n")),
                "assistant" => out.push_str(&format!("<|assistant|>\n{content}\n")),
                _ => out.push_str(&format!("<|user|>\n{content}\n")),
            }
        }
        ChatTemplate::ChatML | ChatTemplate::Qwen => {
            let role = match role {
                "system" | "assistant" | "user" => role,
                _ => "user",
            };
            out.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }
        ChatTemplate::Llama3 => {
            let role = match role {
                "system" | "assistant" | "user" => role,
                _ => "user",
            };
            if first_message {
                out.push_str("<|begin_of_text|>");
            }
            out.push_str(&format!(
                "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            ));
        }
        ChatTemplate::DeepSeekV4 => {
            if first_message {
                out.push_str("<｜begin▁of▁sentence｜>");
            }
            match role {
                "system" => out.push_str(content),
                "assistant" => out.push_str(&format!(
                    "<｜Assistant｜></think>{content}<｜end▁of▁sentence｜>"
                )),
                _ => out.push_str(&format!("<｜User｜>{content}")),
            }
        }
        ChatTemplate::Plain => match role {
            "system" => out.push_str(&format!("System: {content}\n")),
            "assistant" => out.push_str(&format!("Assistant: {content}\n")),
            _ => out.push_str(&format!("User: {content}\n")),
        },
    }
}

fn assistant_prefix(template: ChatTemplate) -> &'static str {
    match template {
        ChatTemplate::OlmoeInstruct => "<|assistant|>",
        ChatTemplate::ChatML | ChatTemplate::Qwen => "<|im_start|>assistant\n",
        ChatTemplate::Llama3 => "<|start_header_id|>assistant<|end_header_id|>\n\n",
        ChatTemplate::DeepSeekV4 => "<｜Assistant｜></think>",
        ChatTemplate::Plain => "Assistant:",
    }
}

fn respond(stream: &mut std::net::TcpStream, code: u16, body: &str) {
    let status = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Unknown",
    };
    let _ = write!(
        stream,
        "HTTP/1.1 {code} {status}\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
}

fn respond_json(stream: &mut std::net::TcpStream, code: u16, body: &str) {
    let status = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Unknown",
    };
    let _ = write!(
        stream,
        "HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
}
