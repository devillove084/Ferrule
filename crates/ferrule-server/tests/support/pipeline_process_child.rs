//! Test-only executable using the production endpoint, not a libtest stdout pipe.
//! Boot/shutdown evidence lets HTTP tests verify separate PP/EP children.
#[cfg(not(unix))]
fn main() {}

#[cfg(unix)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::os::fd::AsFd;
    use std::path::PathBuf;

    use ferrule_runtime::parallel::process::{
        ProcessChildHandler, ProcessHandlerError, ProcessLaunch, ProcessRequest, child_serve,
        decoder::DecoderEndpoint,
    };
    use serde_json::{Value, json};

    struct ObservedEndpoint {
        inner: DecoderEndpoint,
        evidence: Option<PathBuf>,
    }
    impl ProcessChildHandler for ObservedEndpoint {
        type Command = Value;
        type Output = Value;
        fn execute(
            &mut self,
            request: ProcessRequest<Value>,
        ) -> Result<Value, ProcessHandlerError> {
            self.inner.execute(request)
        }
        fn shutdown(&mut self) -> Result<(), ProcessHandlerError> {
            self.inner.shutdown()?;
            if let Some(path) = &self.evidence {
                std::fs::write(path.with_extension("stopped"), b"shutdown acknowledged").unwrap();
            }
            Ok(())
        }
    }

    let reader = File::from(std::io::stdin().as_fd().try_clone_to_owned()?);
    let writer = File::from(std::io::stdout().as_fd().try_clone_to_owned()?);
    let launch = ProcessLaunch::new(std::env::current_exe()?);
    child_serve(reader, writer, |boot| {
        let record =
            json!({"pid":std::process::id(), "identity":boot.identity, "config":boot.config});
        let inner = DecoderEndpoint::initialize(boot, launch)?;
        let evidence = std::env::var_os("FERRULE_PIPELINE_CHILD_EVIDENCE").map(|directory| {
            let path = PathBuf::from(directory).join(format!("{}.json", std::process::id()));
            std::fs::write(&path, record.to_string()).unwrap();
            path
        });
        Ok(ObservedEndpoint { inner, evidence })
    })?;
    Ok(())
}
