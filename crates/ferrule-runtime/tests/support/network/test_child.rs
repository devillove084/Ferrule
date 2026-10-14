//! TEST ONLY: real CPU linear child, faults and execution ledger. Never linked
//! into the production library or selected as a CUDA fallback.
#[cfg(not(unix))]
fn main() {}
#[cfg(unix)]
#[path = "test_artifact.rs"]
mod test_artifact;
#[cfg(unix)]
fn main() {
    if std::env::args().nth(1).as_deref() == Some("artifact") {
        println!("{}", test_artifact::fingerprint());
        return;
    }
    if std::env::args().nth(1).as_deref() == Some("endpoint") {
        endpoint_main();
        return;
    }
    use ferrule_runtime::parallel::process::*;
    use serde_json::{Value, json};
    use std::fs::{File, OpenOptions};
    use std::io::Write;
    use std::os::fd::AsFd;
    use std::time::Duration;

    struct CpuLinear {
        config: Value,
    }
    impl ProcessChildHandler for CpuLinear {
        type Command = Value;
        type Output = Value;
        fn execute(
            &mut self,
            request: ProcessRequest<Value>,
        ) -> Result<Value, ProcessHandlerError> {
            if let Some(path) = self.config["ledger"].as_str() {
                let mut f = OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(path)
                    .unwrap();
                writeln!(
                    f,
                    "{} {} {}",
                    request.identity.command.get(),
                    request.identity.transaction.get(),
                    request.session
                )
                .unwrap();
                f.sync_all().unwrap();
            }
            match request.command["fault"].as_str() {
                Some("hang") => loop {
                    std::thread::sleep(Duration::from_secs(1));
                },
                Some("exit") => std::process::exit(17),
                Some("unknown") => {
                    return Err(ProcessHandlerError::unknown(
                        ProcessFailureKind::Handler,
                        "test-only failed fence",
                    ));
                }
                Some("fenced") => {
                    return Err(ProcessHandlerError::fenced(
                        ProcessFailureKind::Handler,
                        "test-only fenced error",
                    ));
                }
                _ => {}
            }
            let delay = request.command["delay_ms"].as_u64().unwrap_or(0);
            std::thread::sleep(Duration::from_millis(delay));
            let values: Vec<f32> =
                serde_json::from_value(request.command["values"].clone()).unwrap_or_default();
            let rows = request.command["rows"].as_u64().unwrap_or(1) as usize;
            let input = self.config["in_features"].as_u64().unwrap_or(3) as usize;
            let output = self.config["out_features"].as_u64().unwrap_or(2) as usize;
            let world = self.config["world"].as_u64().unwrap_or(1) as usize;
            let rank = self.config["rank"].as_u64().unwrap_or(0) as usize;
            let mode = self.config["mode"].as_str().unwrap_or("dp");
            let result = if request.command["phase"] == "apply" {
                values
            } else {
                assert_eq!(values.len(), rows * input);
                let columns = if mode == "tp_column" {
                    output / world + usize::from(rank < output % world)
                } else {
                    output
                };
                let mut result = vec![0f32; rows * columns];
                for row in 0..rows {
                    for col in 0..columns {
                        let global_col = if mode == "tp_column" {
                            rank * (output / world) + rank.min(output % world) + col
                        } else {
                            col
                        };
                        if global_col >= output {
                            continue;
                        }
                        for k in 0..input {
                            if mode == "tp_row"
                                && !(rank * (input / world) + rank.min(input % world)
                                    ..(rank + 1) * (input / world) + (rank + 1).min(input % world))
                                    .contains(&k)
                            {
                                continue;
                            }
                            // Deterministic nontrivial CPU oracle weights.
                            result[row * columns + col] += values[row * input + k]
                                * ((global_col * input + k + 1) as f32 / 8.0);
                        }
                    }
                }
                result
            };
            Ok(
                json!({"values": result, "pid": std::process::id(), "transaction": request.identity.transaction.get(), "session": request.session, "command": request.identity.command.get()}),
            )
        }
        fn shutdown(&mut self) -> Result<(), ProcessHandlerError> {
            if self.config["shutdown_hang"] == true {
                loop {
                    std::thread::sleep(Duration::from_secs(1));
                }
            }
            Ok(())
        }
    }
    impl Drop for CpuLinear {
        fn drop(&mut self) {
            if self.config["drop_hang"] == true {
                loop {
                    std::thread::sleep(Duration::from_secs(1));
                }
            }
        }
    }
    let reader = File::from(std::io::stdin().as_fd().try_clone_to_owned().unwrap());
    let writer = File::from(std::io::stdout().as_fd().try_clone_to_owned().unwrap());
    let result = child_serve_with(reader, writer, ProcessChildConfig::default(), |boot| {
        if let Some(path) = boot.config["started"].as_str() {
            std::fs::write(path, std::process::id().to_string()).unwrap();
        }
        if boot.config["startup_hang"] == true {
            loop {
                std::thread::sleep(Duration::from_secs(1));
            }
        }
        Ok(CpuLinear {
            config: boot.config,
        })
    });
    if result.is_err() {
        std::process::exit(2);
    }
}

#[cfg(unix)]
fn endpoint_main() {
    use ferrule_runtime::parallel::network::*;
    use ferrule_runtime::parallel::process::*;
    use serde_json::Value;
    use std::time::Duration;
    let path = std::env::args().nth(2).expect("TEST ONLY endpoint config");
    let cfg: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let binding: NetworkBinding = serde_json::from_value(cfg["binding"].clone()).unwrap();
    let files: TlsFiles = serde_json::from_value(cfg["tls"].clone()).unwrap();
    let listener = NetworkEndpoint::<Value>::bind("127.0.0.1:0".parse().unwrap(), false).unwrap();
    // A local file, not a probe connection consuming the endpoint's one accept.
    std::fs::write(
        cfg["ready_file"].as_str().unwrap(),
        listener.local_addr().unwrap().to_string(),
    )
    .unwrap();
    let endpoint = NetworkEndpoint {
        local_identity: binding.identity,
        binding,
        config: cfg["boot"].clone(),
        options: NetworkOptions {
            startup_timeout: Duration::from_secs(5),
            command_timeout: Duration::from_millis(500),
            ..NetworkOptions::default()
        },
        tls: ServerTls::load(&files).unwrap(),
        launch: ProcessLaunch::new(std::env::current_exe().unwrap()),
        local_options: ProcessOwnerConfig {
            terminate_grace: Duration::from_millis(10),
            kill_grace: Duration::from_millis(200),
            ..ProcessOwnerConfig::default()
        },
    };
    if endpoint.serve_once(listener).is_err() {
        std::process::exit(2);
    }
}
