#![cfg(unix)]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::json;

const DEADLINE: Duration = Duration::from_secs(20);

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ferrule-cli-serve-signal-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        let fixture = Self(path);
        std::fs::write(
            fixture.0.join("config.json"),
            json!({
                "architectures":["Qwen3ForCausalLM"], "model_type":"qwen3",
                "hidden_size":4, "intermediate_size":8, "num_hidden_layers":2,
                "num_attention_heads":2, "num_key_value_heads":1, "head_dim":2,
                "rms_norm_eps":0.000001, "rope_theta":10000.0, "rope_scaling":null,
                "max_position_embeddings":64, "vocab_size":8, "tie_word_embeddings":true,
                "attention_bias":false, "attention_dropout":0.0, "hidden_act":"silu",
                "torch_dtype":"bfloat16", "use_cache":true, "use_sliding_window":false,
                "sliding_window":null, "max_window_layers":2, "initializer_range":0.02,
                "bos_token_id":2, "eos_token_id":7
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(fixture.0.join("tokenizer.json"), json!({
            "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
            "normalizer":null, "pre_tokenizer":{"type":"WhitespaceSplit"},
            "post_processor":null, "decoder":null,
            "model":{"type":"WordLevel", "vocab":{"<unk>":0,"a":1,"hello":2,"b":3,"c":4,"d":5,"e":6,"<eos>":7},"unk_token":"<unk>"}
        }).to_string()).unwrap();
        let mut tensors = vec![
            ("model.embed_tokens.weight".to_owned(), vec![8, 4]),
            ("model.norm.weight".to_owned(), vec![4]),
        ];
        for layer in 0..2 {
            for (name, shape) in [
                ("input_layernorm.weight", vec![4]),
                ("post_attention_layernorm.weight", vec![4]),
                ("self_attn.q_proj.weight", vec![4, 4]),
                ("self_attn.k_proj.weight", vec![2, 4]),
                ("self_attn.v_proj.weight", vec![2, 4]),
                ("self_attn.o_proj.weight", vec![4, 4]),
                ("self_attn.q_norm.weight", vec![2]),
                ("self_attn.k_norm.weight", vec![2]),
                ("mlp.gate_proj.weight", vec![8, 4]),
                ("mlp.up_proj.weight", vec![8, 4]),
                ("mlp.down_proj.weight", vec![4, 8]),
            ] {
                tensors.push((format!("model.layers.{layer}.{name}"), shape));
            }
        }
        let mut payload = Vec::new();
        let mut header = serde_json::Map::new();
        for (name, shape) in tensors {
            let start = payload.len();
            for index in 0..shape.iter().product::<usize>() {
                let value: f32 = if name == "model.embed_tokens.weight" {
                    if index / 4 == 1 { 0.5 } else { 0.125 }
                } else if shape.len() == 1 {
                    1.0
                } else {
                    0.0
                };
                payload.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            }
            header.insert(
                name,
                json!({"dtype":"BF16", "shape":shape, "data_offsets":[start,payload.len()]}),
            );
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(payload);
        std::fs::write(fixture.0.join("model.safetensors"), file).unwrap();
        fixture
    }

    fn log(&self) -> String {
        std::fs::read_to_string(self.0.join("server.log")).unwrap_or_default()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn health(address: SocketAddr) -> std::io::Result<bool> {
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_millis(200))?;
    stream.set_read_timeout(Some(Duration::from_secs(1)))?;
    stream.set_write_timeout(Some(Duration::from_secs(1)))?;
    stream.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response.starts_with("HTTP/1.1 200") && response.contains("\"status\":\"ok\""))
}

fn shutdown_with_signal(signal: i32) {
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
                "pipeline",
                "--pipeline-parallel",
                "2",
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
