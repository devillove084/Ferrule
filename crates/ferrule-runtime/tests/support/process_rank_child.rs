//! Test-only process fixture, compiled as an explicit example without libtest
//! stdout. Fake work/fault injection never enters the production child API.
#[cfg(not(unix))]
fn main() {}
#[cfg(unix)]
fn main() {
    fixture::run();
}

#[cfg(unix)]
mod fixture {
    use std::fs::File;
    use std::io::{Read, Write};
    use std::os::fd::AsFd;
    use std::rc::Rc;
    use std::thread;
    use std::time::Duration;

    use ferrule_runtime::parallel::process::*;
    use serde_json::{Value, json};

    struct Handler {
        bias: i64,
        shutdown_hang: bool,
        drop_hang: bool,
        _local: Rc<()>,
    }
    impl ProcessChildHandler for Handler {
        type Command = Value;
        type Output = Value;
        fn execute(
            &mut self,
            request: ProcessRequest<Value>,
        ) -> Result<Value, ProcessHandlerError> {
            match request.command["op"].as_str().unwrap_or("echo") {
                "fail" => Err(ProcessHandlerError::fenced(
                    ProcessFailureKind::Handler,
                    "test handler failure",
                )),
                "unknown" => Err(ProcessHandlerError::unknown(
                    ProcessFailureKind::Handler,
                    "test fence failure",
                )),
                "panic" => panic!("test-only handler panic"),
                "hang" => hang(),
                "exit" => std::process::exit(17),
                _ => Ok(json!({
                    "value": request.command["value"].as_i64().unwrap_or(0) + self.bias,
                    "blob": request.command["blob"],
                    "pid": std::process::id(), "command": request.identity.command.get(),
                    "transaction": request.identity.transaction.get(), "session": request.session,
                    "epoch": request.identity.owner.epoch.get(),
                    "owner": request.identity.owner.owner_instance.get(),
                })),
            }
        }
        fn shutdown(&mut self) -> Result<(), ProcessHandlerError> {
            if self.shutdown_hang {
                hang();
            }
            Ok(())
        }
    }
    impl Drop for Handler {
        fn drop(&mut self) {
            if self.drop_hang {
                hang();
            }
        }
    }

    pub fn run() {
        let mode = std::env::args().nth(1).unwrap_or_else(|| "normal".into());
        if mode == "ignore_term" || mode == "raw_no_read" {
            ignore_term();
        }
        // Raw files avoid stdout line buffering and are distinct from stderr.
        let reader = File::from(std::io::stdin().as_fd().try_clone_to_owned().unwrap());
        let writer = File::from(std::io::stdout().as_fd().try_clone_to_owned().unwrap());
        let timeout_ms = std::env::args()
            .nth(2)
            .map(|arg| arg.parse::<u64>().unwrap())
            .unwrap_or(300000);
        let options = ProcessChildConfig {
            io_timeout: Duration::from_millis(timeout_ms),
            ..ProcessChildConfig::default()
        };
        if mode == "decoder" {
            use ferrule_runtime::parallel::process::decoder::DecoderEndpoint;
            let launch = ProcessLaunch::new(std::env::current_exe().unwrap())
                .arg("decoder")
                .arg(timeout_ms.to_string());
            if child_serve_with(reader, writer, options, |boot| {
                DecoderEndpoint::initialize(boot, launch)
            })
            .is_err()
            {
                std::process::exit(2);
            }
            return;
        }
        if mode.starts_with("raw_") {
            raw(reader, writer, &mode);
            return;
        }
        let result = child_serve_with(reader, writer, options, |boot| {
            if boot.config["startup_hang"] == true {
                hang();
            }
            if boot.config["startup_fail"] == true {
                return Err(ProcessHandlerError::fenced(
                    ProcessFailureKind::Startup,
                    "test initializer failure",
                ));
            }
            let delay = boot.config["delay_ms"].as_u64().unwrap_or(0);
            thread::sleep(Duration::from_millis(delay));
            Ok(Handler {
                bias: boot.config["bias"].as_i64().unwrap_or(0),
                shutdown_hang: mode == "shutdown_hang",
                drop_hang: mode == "drop_hang",
                _local: Rc::new(()),
            })
        });
        if result.is_err() {
            std::process::exit(2);
        }
    }
    fn hang() -> ! {
        loop {
            thread::sleep(Duration::from_secs(60));
        }
    }
    #[expect(
        unsafe_code,
        reason = "test-only child ignores TERM to exercise production KILL escalation"
    )]
    fn ignore_term() {
        unsafe {
            libc::signal(libc::SIGTERM, libc::SIG_IGN);
        }
    }

    fn read(reader: &mut File) -> Value {
        let mut prefix = [0; 4];
        reader.read_exact(&mut prefix).unwrap();
        let count = u32::from_be_bytes(prefix) as usize;
        assert!(count < 16 * 1024 * 1024);
        let mut body = vec![0; count];
        reader.read_exact(&mut body).unwrap();
        serde_json::from_slice(&body).unwrap()
    }
    fn write(writer: &mut File, message: Value, fragmented: bool) {
        let body =
            serde_json::to_vec(&json!({"version": PROCESS_PROTOCOL_VERSION, "message": message}))
                .unwrap();
        let mut frame = (body.len() as u32).to_be_bytes().to_vec();
        frame.extend(body);
        if fragmented {
            for chunk in frame.chunks(3) {
                writer.write_all(chunk).unwrap();
            }
        } else {
            writer.write_all(&frame).unwrap();
        }
    }
    fn raw(mut reader: File, mut writer: File, mode: &str) {
        let boot = read(&mut reader);
        let boot = &boot["message"];
        let mut ready =
            json!({"type":"ready", "identity":boot["identity"], "limits":boot["limits"]});
        if mode == "raw_stale_ready" {
            ready["identity"]["epoch"] = json!(999);
        }
        if mode == "raw_oversized_ready" {
            writer.write_all(&u32::MAX.to_be_bytes()).unwrap();
            return;
        }
        write(&mut writer, ready, mode == "raw_fragmented");
        if mode == "raw_no_read" {
            hang();
        }
        let command = read(&mut reader);
        let command = &command["message"];
        match mode {
            "raw_oversized" => {
                writer.write_all(&u32::MAX.to_be_bytes()).unwrap();
            }
            "raw_truncated" => {
                writer.write_all(&100u32.to_be_bytes()).unwrap();
                writer.write_all(b"{\"ver").unwrap();
            }
            "raw_partial_prefix" => {
                writer.write_all(&[0, 0]).unwrap();
                hang();
            }
            "raw_slow_body" => {
                writer.write_all(&1000u32.to_be_bytes()).unwrap();
                loop {
                    writer.write_all(b" ").unwrap();
                    thread::sleep(Duration::from_millis(10));
                }
            }
            _ => {
                let mut identity = command["identity"].clone();
                match mode {
                    "raw_stale_epoch" => identity["owner"]["epoch"] = json!(999),
                    "raw_stale_rank" => identity["owner"]["rank"] = json!(999),
                    "raw_stale_owner" => identity["owner"]["owner_instance"] = json!(999),
                    "raw_stale_command" => identity["command"] = json!(999),
                    "raw_stale_txn" => identity["transaction"] = json!(999),
                    _ => (),
                }
                write(
                    &mut writer,
                    json!({"type":"complete", "identity":identity, "session":command["session"], "payload":{"value":42}}),
                    mode == "raw_fragmented",
                );
            }
        }
    }
}
