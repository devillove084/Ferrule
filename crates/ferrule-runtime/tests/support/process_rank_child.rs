//! Test-only process fixture, compiled as an explicit example without libtest
//! stdout. Fake work/fault injection never enters the production child API.
#[cfg(all(unix, feature = "cuda"))]
#[path = "numeric_expert_fixture.rs"]
mod numeric_expert_fixture;

#[cfg(all(unix, feature = "cpu-reference"))]
#[path = "hybrid_fixture.rs"]
#[allow(dead_code)]
mod hybrid_fixture;

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

    // Synthetic execution without cpu-reference is deliberately confined to this
    // test executable, never installed by the production hidden worker.
    fn bind_decoder_fixture(
        boot: &ferrule_runtime::parallel::process::decoder::DecoderBoot,
    ) -> ferrule_common::Result<ferrule_model::transformer::BoundDecoderResources> {
        use ferrule_model::transformer::{DecoderLoadOptions, SyntheticDecoderRecipe};
        use ferrule_runtime::parallel::process::decoder::DecoderRecipeKind;
        if boot.recipe != DecoderRecipeKind::Synthetic {
            return Err(ferrule_common::Error::Execution {
                message: "real process model binding belongs to the ferrule-composition application entrypoint".into(),
            });
        }
        let file = File::open(boot.checkpoint.join("config.json"))?;
        let config: Value = serde_json::from_reader(file).unwrap();
        DecoderLoadOptions::new(&SyntheticDecoderRecipe::new(), &config).open_hf(
            &boot.checkpoint,
            ferrule_model::ModelFamily::Unknown("synthetic".into()),
        )
    }

    fn bind_numeric_fixture(
        _boot: &ferrule_runtime::parallel::expert_process::NumericBoot,
    ) -> ferrule_common::Result<ferrule_model::transformer::BoundDecoderResources> {
        Err(ferrule_common::Error::Execution {
            message: "real numeric process binding belongs to the ferrule-composition application entrypoint".into(),
        })
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
        #[cfg(feature = "cuda")]
        if mode == "numeric_gpu" {
            use ferrule_runtime::parallel::expert_process::{NumericBoot, NumericExpertChild};
            let result = child_serve_with(reader, writer, options, |boot| {
                let config: NumericBoot = serde_json::from_value(boot.config.clone()).unwrap();
                let resources =
                    crate::numeric_expert_fixture::bind(&config.checkpoint, 8, true, false);
                NumericExpertChild::initialize_bound(boot, resources)
            });
            if let Err(error) = result {
                eprintln!("real numeric CUDA child: {error}");
                std::process::exit(2);
            }
            return;
        }
        if mode == "typed_cpu" {
            use ferrule_runtime::parallel::expert_process::{NumericCommand, NumericReply};
            struct Typed;
            enum Output {
                Reply(NumericReply),
                Fail,
                Deep,
            }
            impl serde::Serialize for Output {
                fn serialize<S: serde::Serializer>(
                    &self,
                    serializer: S,
                ) -> Result<S::Ok, S::Error> {
                    match self {
                        Self::Reply(reply) => reply.serialize(serializer),
                        Self::Fail => Err(serde::ser::Error::custom(
                            "injected output serializer failure",
                        )),
                        Self::Deep => {
                            let mut value = Value::Null;
                            for _ in 0..130 {
                                value = Value::Array(vec![value]);
                            }
                            value.serialize(serializer)
                        }
                    }
                }
            }
            impl ProcessChildHandler for Typed {
                type Command = NumericCommand;
                type Output = Output;
                fn execute(
                    &mut self,
                    request: ProcessRequest<NumericCommand>,
                ) -> Result<Output, ProcessHandlerError> {
                    match request.command {
                        NumericCommand::Compute { layer: 1, .. } => Ok(Output::Fail),
                        NumericCommand::Compute { layer: 2, .. } => Ok(Output::Deep),
                        NumericCommand::Compute { tokens, .. } => {
                            Ok(Output::Reply(NumericReply::Results {
                                owner: request.identity.owner.rank.get(),
                                tokens,
                            }))
                        }
                        NumericCommand::ComputeRoutes { tokens, .. } => {
                            Ok(Output::Reply(NumericReply::RouteResults {
                                owner: request.identity.owner.rank.get(),
                                tokens,
                            }))
                        }
                        NumericCommand::Stats => Err(ProcessHandlerError::fenced(
                            ProcessFailureKind::Handler,
                            "stats unavailable",
                        )),
                    }
                }
            }
            if child_serve_with(reader, writer, options, |_| Ok(Typed)).is_err() {
                std::process::exit(2);
            }
            return;
        }
        if mode.starts_with("numeric_") {
            numeric(reader, writer, options, &mode);
            return;
        }
        #[cfg(feature = "cpu-reference")]
        if mode == "hybrid_decoder" {
            use ferrule_runtime::parallel::process::decoder::DecoderChild;
            let max = ProcessFrameLimits::MAX_FRAME_BYTES;
            let options = ProcessChildConfig {
                frame_limits: ProcessFrameLimits {
                    max_frame_bytes: max,
                    max_config_bytes: max,
                    max_command_bytes: max,
                    max_output_bytes: max,
                    max_error_bytes: 4096,
                },
                ..options
            };
            if child_serve_with(reader, writer, options, |boot| {
                DecoderChild::initialize(boot, |config| {
                    // The parent owns this checkpoint directory; child Drop must not remove it.
                    let fixture = std::mem::ManuallyDrop::new(super::hybrid_fixture::Fixture {
                        dir: config.checkpoint.clone(),
                        oracle: serde_json::from_str(include_str!(
                            "../../../ferrule-model/tests/fixtures/hybrid_cpu/oracle.json"
                        ))
                        .unwrap(),
                    });
                    DecoderChild::hybrid_factory(config, &fixture.resources())
                })
            })
            .is_err()
            {
                std::process::exit(2);
            }
            return;
        }
        if mode == "decoder" {
            use ferrule_runtime::parallel::process::decoder::DecoderEndpoint;
            let launch = ProcessLaunch::new(std::env::current_exe().unwrap())
                .arg("decoder")
                .arg(timeout_ms.to_string());
            if child_serve_with(reader, writer, options, |boot| {
                DecoderEndpoint::initialize(
                    boot,
                    launch,
                    ferrule_runtime::parallel::process::decoder::DecoderBindings {
                        decoder: bind_decoder_fixture,
                        numeric: bind_numeric_fixture,
                    },
                )
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
    // Real production framing/owner lifecycle with CPU-only fault injection.
    // These replies are protocol fixtures, never numeric CUDA evidence.
    fn numeric(reader: File, writer: File, options: ProcessChildConfig, mode: &str) {
        use ferrule_runtime::parallel::expert_process::*;
        struct NumericFixture {
            boot: NumericBoot,
            mode: String,
            calls: usize,
            tokens: usize,
        }
        impl ProcessChildHandler for NumericFixture {
            type Command = NumericCommand;
            type Output = NumericReply;
            fn execute(
                &mut self,
                request: ProcessRequest<NumericCommand>,
            ) -> Result<NumericReply, ProcessHandlerError> {
                match request.command {
                    NumericCommand::Stats => {
                        let payload = self.boot.expected.iter().map(|s| s.bytes).sum::<u64>();
                        Ok(NumericReply::Stats(NumericProcessStats {
                            pid: std::process::id(),
                            owner: self.boot.owner,
                            slot: self.boot.rank().unwrap().slot.get(),
                            device: self.boot.device().unwrap(),
                            owned_experts: self
                                .boot
                                .expected
                                .iter()
                                .map(|s| (s.layer, s.expert))
                                .collect(),
                            calls: self.calls,
                            tokens: self.tokens,
                            host_payload_bytes: payload,
                            host_read_bytes: payload,
                            host_hits: 6,
                            resident_experts: self.boot.expected.len(),
                            resident_bytes: payload as usize,
                            uploads: self.boot.expected.len() as u64,
                            evictions: 0,
                            pending_upload_bytes: 0,
                            quarantined: false,
                        }))
                    }
                    NumericCommand::Compute { mut tokens, .. } => {
                        if self.mode == "numeric_hang" {
                            hang();
                        }
                        if self.mode == "numeric_exit" {
                            std::process::exit(17);
                        }
                        if self.mode == "numeric_unknown" {
                            return Err(ProcessHandlerError::unknown(
                                ProcessFailureKind::Handler,
                                "injected unknown",
                            ));
                        }
                        if self.mode == "numeric_fenced" {
                            return Err(ProcessHandlerError::fenced(
                                ProcessFailureKind::Handler,
                                "injected fenced",
                            ));
                        }
                        if self.mode == "numeric_bad_reply" {
                            tokens[0].source += 1;
                        }
                        if self.mode == "numeric_nonfinite" {
                            tokens[0].values[0] = f32::NAN.to_bits();
                        }
                        if self.mode == "numeric_barrier" {
                            let dir = std::env::var_os("FERRULE_NUMERIC_BARRIER").unwrap();
                            let dir = std::path::Path::new(&dir);
                            std::fs::write(dir.join(format!("entered-{}", self.boot.owner)), b"1")
                                .unwrap();
                            let deadline = std::time::Instant::now() + Duration::from_secs(2);
                            while self
                                .boot
                                .placement
                                .members
                                .iter()
                                .any(|m| !dir.join(format!("entered-{m}")).exists())
                            {
                                assert!(
                                    std::time::Instant::now() < deadline,
                                    "serialized numeric batch"
                                );
                                thread::sleep(Duration::from_millis(1));
                            }
                        }
                        self.calls += 1;
                        self.tokens += tokens.len();
                        Ok(NumericReply::Results {
                            owner: self.boot.owner,
                            tokens,
                        })
                    }
                    NumericCommand::ComputeRoutes { tokens, .. } => {
                        Ok(NumericReply::RouteResults {
                            owner: self.boot.owner,
                            tokens,
                        })
                    }
                }
            }
            fn shutdown(&mut self) -> Result<(), ProcessHandlerError> {
                if self.mode.contains("shutdown_hang") {
                    hang();
                }
                Ok(())
            }
        }
        let mode = mode.to_string();
        let result = child_serve_with(reader, writer, options, move |boot| {
            let config: NumericBoot = serde_json::from_value(boot.config).unwrap();
            config.group().unwrap();
            if let Some(dir) = std::env::var_os("FERRULE_NUMERIC_BARRIER") {
                std::fs::write(
                    std::path::Path::new(&dir).join(format!("pid-{}", config.owner)),
                    std::process::id().to_string(),
                )
                .unwrap();
            }
            if config.rank().unwrap().slot.get() == 1 {
                if mode.contains("boot_hang") {
                    hang();
                }
                if mode.contains("boot_exit") {
                    std::process::exit(17);
                }
                if mode.contains("boot_unknown") {
                    return Err(ProcessHandlerError::unknown(
                        ProcessFailureKind::Startup,
                        "injected startup unknown",
                    ));
                }
            }
            Ok(NumericFixture {
                boot: config,
                mode,
                calls: 0,
                tokens: 0,
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
