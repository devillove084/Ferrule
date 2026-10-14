#![cfg(unix)]

include!("support/serve_fixture.rs");

const DEADLINE: Duration = Duration::from_secs(20);

fn shutdown_with_signal(signal: i32) {
    shutdown_with_options(signal, "pipeline", &[]);
}

fn shutdown_with_options(signal: i32, engine: &str, options: &[&str]) {
    let fixture = Fixture::new();
    let probe = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = probe.local_addr().unwrap();
    drop(probe);
    let log = std::fs::File::create(fixture.0.join("server.log")).unwrap();
    let mut server = Server(
        Command::new(env!("CARGO_BIN_EXE_ferrule"))
            .arg("serve")
            .arg(&fixture.0)
            .args([
                "--backend",
                "cpu",
                "--engine",
                engine,
                "--pipeline-parallel",
                if engine == "pipeline" { "2" } else { "1" },
                "--ctx-size",
                "64",
                "--max-active-sequences",
                "1",
                "--kv-cache-mb",
                "1",
                "--prefill-chunk-size",
                "1",
                "--max-batch-tokens",
                "1",
                "--host",
                "127.0.0.1",
                "--port",
                &address.port().to_string(),
            ])
            .args(options)
            .stdin(Stdio::null())
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + DEADLINE;
    loop {
        assert!(
            server.0.try_wait().unwrap().is_none(),
            "startup failed: {}",
            fixture.log()
        );
        if health(address).unwrap_or(false) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "startup timeout: {}",
            fixture.log()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    if !options.is_empty() {
        let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(2)).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream
            .write_all(b"GET /admission HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        let value: serde_json::Value =
            serde_json::from_str(response.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(value["requests"]["limit"], 7);
        assert_eq!(value["prompt_bytes"]["limit"], 4096);
        assert_eq!(value["max_body_bytes"], 512);
        assert_eq!(value["runtime"]["status"], "available");
        assert_eq!(value["runtime"]["waiting_requests"]["limit"], 2);
        assert_eq!(value["runtime"]["request_identities"]["limit"], 3);
        assert_eq!(value["runtime"]["session_identities"]["limit"], 1);
    }
    // SAFETY: kill takes scalar arguments; this is our live child's positive PID,
    // never the test process or a process group.
    #[allow(unsafe_code)]
    let result = unsafe { libc::kill(server.0.id().try_into().unwrap(), signal) };
    assert_eq!(result, 0);
    let deadline = Instant::now() + DEADLINE;
    loop {
        if let Some(status) = server.0.try_wait().unwrap() {
            assert!(
                status.success(),
                "non-graceful exit {status}: {}",
                fixture.log()
            );
            break;
        }
        assert!(
            Instant::now() < deadline,
            "shutdown timeout: {}",
            fixture.log()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(
        TcpListener::bind(address).is_ok(),
        "HTTP listener survived shutdown"
    );
}

#[test]
fn sigterm_drains_and_joins_the_real_cli_worker() {
    shutdown_with_signal(libc::SIGTERM);
}

#[test]
fn sigint_drains_and_joins_the_real_cli_worker() {
    shutdown_with_signal(libc::SIGINT);
}

#[test]
fn admission_cli_options_reach_live_resident_worker() {
    shutdown_with_options(
        libc::SIGTERM,
        "resident",
        &[
            "--max-inflight-requests",
            "7",
            "--max-prompt-bytes",
            "4096",
            "--max-body-bytes",
            "512",
            "--runtime-max-waiting-requests",
            "2",
            "--runtime-max-request-identities",
            "3",
            "--runtime-max-session-identities",
            "1",
        ],
    );
}

#[test]
fn unsupported_pipeline_runtime_override_is_not_silently_ignored() {
    let fixture = Fixture::new();
    let log = std::fs::File::create(fixture.0.join("server.log")).unwrap();
    let mut server = Server(
        Command::new(env!("CARGO_BIN_EXE_ferrule"))
            .arg("serve")
            .arg(&fixture.0)
            .args([
                "--backend",
                "cpu",
                "--engine",
                "pipeline",
                "--ctx-size",
                "64",
                "--kv-cache-mb",
                "1",
                "--runtime-max-session-identities",
                "1",
            ])
            .stdin(Stdio::null())
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + DEADLINE;
    loop {
        if let Some(status) = server.0.try_wait().unwrap() {
            assert!(!status.success());
            break;
        }
        assert!(
            Instant::now() < deadline,
            "unsupported override did not fail startup: {}",
            fixture.log()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(
        fixture
            .log()
            .contains("engine does not support resident admission options"),
        "{}",
        fixture.log()
    );
}

#[test]
fn cpu_process_pp2_serves_a_real_request_and_shuts_down_without_gpu() {
    let fixture = Fixture::new();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let log = std::fs::File::create(fixture.0.join("server.log")).unwrap();
    let mut server = Server(
        Command::new(env!("CARGO_BIN_EXE_ferrule"))
            .args([
                "serve",
                fixture.0.to_str().unwrap(),
                "--engine",
                "pipeline",
                "--backend",
                "cpu",
                "--rank-backend",
                "process",
                "--pipeline-parallel",
                "2",
                "--ctx-size",
                "64",
                "--kv-cache-mb",
                "1",
                "--max-active-sequences",
                "1",
                "--prefill-chunk-size",
                "8",
                "--max-batch-tokens",
                "16",
                "--rank-timeout-ms",
                "5000",
                "--host",
                "127.0.0.1",
                "--port",
                &address.port().to_string(),
            ])
            .stdin(Stdio::null())
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .spawn()
            .unwrap(),
    );
    let end = Instant::now() + DEADLINE;
    while !health(address).unwrap_or(false) {
        assert!(server.0.try_wait().unwrap().is_none(), "{}", fixture.log());
        assert!(Instant::now() < end, "startup timeout: {}", fixture.log());
        std::thread::sleep(Duration::from_millis(20));
    }
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(2)).unwrap();
    stream.set_read_timeout(Some(DEADLINE)).unwrap();
    stream.set_write_timeout(Some(DEADLINE)).unwrap();
    let body = json!({"model":"qwen3-dense","prompt":"hello a b","max_tokens":2,"stream":false})
        .to_string();
    write!(stream,"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",body.len(),body).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "{response}\n{}",
        fixture.log()
    );
    assert!(response.contains("choices"), "{response}");
    assert!(fixture.log().contains("Process"), "{}", fixture.log());
    // SAFETY: signal only this test-owned server PID, never the parent/group.
    #[allow(unsafe_code)]
    let signalled = unsafe { libc::kill(server.0.id() as i32, libc::SIGTERM) };
    assert_eq!(signalled, 0);
    let end = Instant::now() + DEADLINE;
    loop {
        if let Some(status) = server.0.try_wait().unwrap() {
            assert!(status.success(), "{}", fixture.log());
            break;
        }
        assert!(Instant::now() < end, "shutdown timeout: {}", fixture.log());
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(TcpListener::bind(address).is_ok());
}

#[test]
fn host_kv_policy_rejects_real_cpu_engines_before_ready() {
    for engine in ["resident", "pipeline"] {
        let fixture = Fixture::new();
        let log = std::fs::File::create(fixture.0.join("server.log")).unwrap();
        let mut server = Server(
            Command::new(env!("CARGO_BIN_EXE_ferrule"))
                .arg("serve")
                .arg(&fixture.0)
                .args([
                    "--backend",
                    "cpu",
                    "--engine",
                    engine,
                    "--kv-execution-policy",
                    "host-kv-decode",
                    "--prefill-chunk-size",
                    "2",
                    "--max-batch-tokens",
                    "2",
                    "--max-active-sequences",
                    "1",
                    "--ctx-size",
                    "64",
                    "--kv-cache-mb",
                    "1",
                    "--port",
                    "0",
                ])
                .stdin(Stdio::null())
                .stdout(log.try_clone().unwrap())
                .stderr(log)
                .spawn()
                .unwrap(),
        );
        let deadline = Instant::now() + DEADLINE;
        loop {
            if let Some(status) = server.0.try_wait().unwrap() {
                assert!(!status.success());
                break;
            }
            assert!(
                Instant::now() < deadline,
                "HostKV startup must reject CPU: {}",
                fixture.log()
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        let log = fixture.log();
        assert!(log.contains("HostKV decode"), "{log}");
        assert!(!log.contains("API listening"), "{log}");
    }
}
