//! TEST ONLY CPU process fixture. Never dispatched by the production CLI.
#[cfg(not(unix))]
fn main() {}
#[cfg(unix)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use ferrule_runtime::parallel::process::*;
    use serde_json::{Value, json};
    use std::fs::File;
    use std::os::fd::AsFd;
    struct Cpu {
        boot: Value,
        fault: String,
    }
    impl ProcessChildHandler for Cpu {
        type Command = Value;
        type Output = Value;
        fn execute(
            &mut self,
            request: ProcessRequest<Value>,
        ) -> Result<Value, ProcessHandlerError> {
            if self.fault == "unknown" {
                return Err(ProcessHandlerError::unknown(
                    ProcessFailureKind::Handler,
                    "TEST unknown",
                ));
            }
            if self.fault == "exit" {
                std::process::exit(17);
            }
            let c = &request.command;
            let s = &self.boot["spec"];
            let rank = s["rank"].as_u64().unwrap() as usize;
            let world = s["ranks"].as_u64().unwrap() as usize;
            let rows = s["rows"].as_u64().unwrap() as usize;
            let input = s["in_features"].as_u64().unwrap() as usize;
            let output = s["out_features"].as_u64().unwrap() as usize;
            assert_eq!(c["transaction"], request.identity.transaction.get());
            assert_eq!(c["rank"], rank);
            assert_eq!(c["rows"], rows);
            let values: Vec<f32> = serde_json::from_value(c["values"].clone()).unwrap();
            let weight: Vec<f32> = serde_json::from_value(self.boot["weight"].clone()).unwrap();
            let span = |len: usize| {
                let start = rank * (len / world) + rank.min(len % world);
                start..start + len / world + usize::from(rank < len % world)
            };
            let result = if c["phase"] == "apply" {
                values
            } else {
                let columns = if s["mode"] == "tp-column" {
                    span(output)
                } else {
                    0..output
                };
                let inputs = if s["mode"] == "tp-row" {
                    span(input)
                } else {
                    0..input
                };
                let mut result = Vec::new();
                for row in 0..rows {
                    for col in columns.clone() {
                        result.push(
                            inputs
                                .clone()
                                .map(|k| values[row * input + k] * weight[col * input + k])
                                .sum::<f32>(),
                        );
                    }
                }
                result
            };
            Ok(
                json!({"transaction": if self.fault == "bad_reply" { 999 } else { request.identity.transaction.get() },
                "rank": rank, "phase": c["phase"], "rows": rows, "values": result}),
            )
        }
    }
    let fault = std::env::args().nth(1).unwrap_or_default();
    let input = File::from(std::io::stdin().as_fd().try_clone_to_owned()?);
    let output = File::from(std::io::stdout().as_fd().try_clone_to_owned()?);
    child_serve_with(input, output, ProcessChildConfig::default(), |boot| {
        Ok(Cpu {
            boot: boot.config,
            fault,
        })
    })?;
    Ok(())
}
