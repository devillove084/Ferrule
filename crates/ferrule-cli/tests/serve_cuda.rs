#![cfg(unix)]

include!("support/serve_fixture.rs");

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires NAS Qwen3.5-0.8B and one CUDA GPU; launches the real default CLI"]
fn qwen35_cuda_default_serve_sse_and_sigterm() {
    let directory = std::env::var_os("FERRULE_QWEN35_08B_DIR")
        .unwrap_or_else(|| "/mnt/nas1/hf/Qwen3.5-0.8B".into());
    let fixture = Fixture::new();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let log = std::fs::File::create(fixture.0.join("server.log")).unwrap();
    let mut server = Server(
        Command::new(env!("CARGO_BIN_EXE_ferrule"))
            .arg("serve")
            .arg(directory)
            .args(["--backend", "cuda", "--port", &address.port().to_string()])
            .stdin(Stdio::null())
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(60);
    while !health(address).unwrap_or(false) {
        assert!(server.0.try_wait().unwrap().is_none(), "{}", fixture.log());
        assert!(
            Instant::now() < deadline,
            "startup deadline: {}",
            fixture.log()
        );
        std::thread::sleep(Duration::from_millis(100));
    }
    for (path, body, expected) in [
        (
            "/v1/chat/completions",
            json!({"model":"qwen3.5-0.8b", "messages":[{"role":"user","content":"Hi"}], "max_completion_tokens":1,"stream":true,"stream_options":{"include_usage":true}}),
            "Hello",
        ),
        (
            "/v1/completions",
            json!({"model":"qwen3.5-0.8b", "prompt":"The capital of France is", "max_tokens":1,"stream":true,"stream_options":{"include_usage":true}}),
            " Paris",
        ),
    ] {
        let body = body.to_string();
        let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(2)).unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream.write_all(format!("POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(90);
        let mut output = Vec::new();
        loop {
            let remaining = deadline
                .checked_duration_since(Instant::now())
                .expect("absolute SSE deadline");
            stream.set_read_timeout(Some(remaining)).unwrap();
            let mut buffer = [0; 8192];
            let count = stream.read(&mut buffer).unwrap();
            if count == 0 {
                break;
            }
            output.extend_from_slice(&buffer[..count]);
        }
        let output = String::from_utf8(output).unwrap();
        assert!(output.starts_with("HTTP/1.1 200"), "{output}");
        assert!(output.contains(expected), "{output}");
        assert!(output.contains("\"finish_reason\":\"length\""), "{output}");
        assert!(output.contains("\"completion_tokens\":1"), "{output}");
        assert!(output.contains("data: [DONE]"), "{output}");
    }
    // SAFETY: positive PID of the child owned by this RAII guard.
    #[allow(unsafe_code)]
    let result = unsafe { libc::kill(server.0.id().try_into().unwrap(), libc::SIGTERM) };
    assert_eq!(result, 0);
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(status) = server.0.try_wait().unwrap() {
            assert!(status.success(), "{status}: {}", fixture.log());
            break;
        }
        assert!(
            Instant::now() < deadline,
            "shutdown deadline: {}",
            fixture.log()
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        TcpListener::bind(address).is_ok(),
        "listener survived SIGTERM"
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires exact Qwen3.5-35B-A3B-FP8 NAS checkpoint and one 24 GiB CUDA GPU; bounded 900s launcher"]
fn qwen35_35b_fp8_f32_default_serve_sse_and_sigterm() {
    let directory = std::env::var_os("FERRULE_NUMERIC_FP8_MODEL_DIR")
        .unwrap_or_else(|| "/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8".into());
    let fixture = Fixture::new();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let log = std::fs::File::create(fixture.0.join("server.log")).unwrap();
    let started = Instant::now();
    let deadline = started + Duration::from_secs(840);
    // No backend/precision/capacity overrides: this tests the exact profile's defaults.
    let mut server = Server(
        Command::new(env!("CARGO_BIN_EXE_ferrule"))
            .env("FERRULE_LOG_FORMAT", "json")
            .arg("serve")
            .arg(directory)
            .args(["--port", &address.port().to_string()])
            .stdin(Stdio::null())
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .spawn()
            .unwrap(),
    );
    let pid = server.0.id();
    let startup_deadline = started + Duration::from_secs(120);
    while !health(address).unwrap_or(false) {
        assert!(server.0.try_wait().unwrap().is_none(), "{}", fixture.log());
        assert!(
            Instant::now() < startup_deadline,
            "startup deadline: {}",
            fixture.log()
        );
        std::thread::sleep(Duration::from_millis(100));
    }
    eprintln!(
        "35B default CLI ready after {:?}, pid={pid}",
        started.elapsed()
    );
    for (prompt, expected, prompt_tokens, max_tokens) in [
        ("Hello", ", I am", 1, 3),
        ("The capital of France is", " Paris", 5, 1),
    ] {
        let request_started = Instant::now();
        let body = json!({"model":"qwen3.5-35b-a3b-fp8", "prompt":prompt,
            "max_tokens":max_tokens,"stream":true,"stream_options":{"include_usage":true}})
        .to_string();
        let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(2)).unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream.write_all(format!("POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).as_bytes()).unwrap();
        let mut output = Vec::new();
        let mut first_token_time = None;
        loop {
            let remaining = deadline
                .checked_duration_since(Instant::now())
                .unwrap_or_else(|| panic!("absolute SSE deadline: {}", fixture.log()));
            stream.set_read_timeout(Some(remaining)).unwrap();
            let mut buffer = [0; 8192];
            let n = stream
                .read(&mut buffer)
                .unwrap_or_else(|e| panic!("{e}: {}", fixture.log()));
            if n == 0 {
                break;
            }
            output.extend_from_slice(&buffer[..n]);
            if first_token_time.is_none() {
                // Inspect complete SSE data lines as bytes arrive, not after DONE.
                let received = String::from_utf8_lossy(&output);
                if received
                    .lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
                    .any(|event| {
                        event["choices"][0]["text"]
                            .as_str()
                            .is_some_and(|text| !text.is_empty())
                    })
                {
                    first_token_time = Some(request_started.elapsed());
                }
            }
        }
        let output = String::from_utf8(output).unwrap();
        assert!(
            output.starts_with("HTTP/1.1 200"),
            "{output}: {}",
            fixture.log()
        );
        let events: Vec<serde_json::Value> = output
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|data| *data != "[DONE]")
            .map(|data| serde_json::from_str(data).unwrap())
            .collect();
        let text: String = events
            .iter()
            .filter_map(|event| event["choices"][0]["text"].as_str())
            .collect();
        assert_eq!(text, expected, "{output}");
        assert!(
            events
                .iter()
                .any(|event| event["choices"][0]["finish_reason"] == "length"),
            "{output}"
        );
        assert!(
            events
                .iter()
                .any(|event| event["usage"]["completion_tokens"] == max_tokens
                    && event["usage"]["prompt_tokens"] == prompt_tokens),
            "{output}"
        );
        assert!(output.contains("data: [DONE]"), "{output}");
        eprintln!(
            "35B runtime SSE prompt={prompt:?} text={text:?} ttft={:?} elapsed={:?}; max_tokens={max_tokens} (not full-logits comparison)",
            first_token_time.expect("a real token must arrive before terminal"),
            request_started.elapsed()
        );
    }
    // SAFETY: only signal the positive PID of our RAII-owned child.
    #[allow(unsafe_code)]
    let result = unsafe { libc::kill(pid.try_into().unwrap(), libc::SIGTERM) };
    assert_eq!(result, 0);
    let shutdown_deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(status) = server.0.try_wait().unwrap() {
            assert!(status.success(), "{status}: {}", fixture.log());
            break;
        }
        assert!(
            Instant::now() < shutdown_deadline,
            "shutdown deadline: {}",
            fixture.log()
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        TcpListener::bind(address).is_ok(),
        "listener survived SIGTERM"
    );
    let log = fixture.log();
    eprintln!("35B CLI owner log:\n{log}");
    assert!(log.contains("numeric-fp8-f32-tf32x3"), "{log}");
    let reports: Vec<serde_json::Value> = log
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    for message in [
        "Qwen3.5-35B resident owner final runtime stats",
        "Qwen3.5-35B owner final model stats",
    ] {
        let report = reports
            .iter()
            .find(|event| event["fields"]["message"] == message)
            .unwrap_or_else(|| panic!("missing close report: {log}"));
        assert_eq!(report["fields"]["physically_closed"], true, "{log}");
    }
    // Two prompts, four output tokens, and every final KV append is still executed.
    assert!(
        log.contains("prefill_chunks: 2, prefill_tokens: 6, decode_steps: 4, emitted_tokens: 4"),
        "{log}"
    );
    assert!(
        log.contains("max_experts: 1024, max_bytes: 4294967296"),
        "{log}"
    );
    assert!(log.contains("allocated_pages: 0"), "{log}");
    assert!(!log.contains("quarantined: true"), "{log}");
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let contexts = Command::new("nvidia-smi")
            .args(["--query-compute-apps=pid", "--format=csv,noheader,nounits"])
            .output()
            .unwrap();
        assert!(contexts.status.success(), "nvidia-smi failed");
        let contexts = String::from_utf8(contexts.stdout).unwrap();
        if !contexts.lines().any(|line| line.trim() == pid.to_string()) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "child {pid} GPU context survived shutdown"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
    eprintln!(
        "35B SIGTERM drained, listener closed, child GPU contexts absent; total {:?}",
        started.elapsed()
    );
}
