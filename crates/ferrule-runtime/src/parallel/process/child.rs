use std::io::{Read, Write};
use std::mem::ManuallyDrop;
use std::os::fd::AsRawFd;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Duration;

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::ipc::{self, Message};
use super::{
    ProcessChildConfig, ProcessCommandIdentity, ProcessError, ProcessFailureKind,
    ProcessFrameLimits, ProcessHandlerError, ProcessIdentity, ProcessQuiescence, deadline_after,
    protocol,
};

#[derive(Debug)]
pub struct ProcessBoot {
    pub identity: ProcessIdentity,
    pub limits: ProcessFrameLimits,
    pub config: Value,
}
#[derive(Debug)]
pub struct ProcessRequest<I> {
    pub identity: ProcessCommandIdentity,
    pub session: u64,
    pub command: I,
}

/// No Send/Sync bound. Construction, execute, shutdown and Drop remain local.
/// Ready/Ok/Fenced error must prove all asynchronous accesses have stopped.
/// Unknown retains the handler without shutdown/Drop. As with ReplicaWorker,
/// panic unwinding must not free DMA-referenced temporaries: the handler and
/// initializer must retain such storage themselves before calling device code.
/// A handler may not write stdout or start background work after returning.
pub trait ProcessChildHandler: Sized {
    type Command: DeserializeOwned;
    type Output: Serialize;
    fn execute(
        &mut self,
        request: ProcessRequest<Self::Command>,
    ) -> Result<Self::Output, ProcessHandlerError>;
    fn shutdown(&mut self) -> Result<(), ProcessHandlerError> {
        Ok(())
    }
}

/// Read Boot, initialize a real worker, then send Ready. Use unbuffered owned
/// files for the wire; buffering a partial frame can defeat pipe readiness.
pub fn child_serve<R, W, H, F>(reader: R, writer: W, initialize: F) -> Result<(), ProcessError>
where
    R: Read + AsRawFd,
    W: Write + AsRawFd,
    H: ProcessChildHandler,
    F: FnOnce(ProcessBoot) -> Result<H, ProcessHandlerError>,
{
    child_serve_with(reader, writer, ProcessChildConfig::default(), initialize)
}

/// `io_timeout` bounds Boot, individual frames and reply writes, not healthy
/// command-pipe idleness. The independent parent still bounds each complete
/// command and shutdown, including a hung handler or destructor.
pub fn child_serve_with<R, W, H, F>(
    mut reader: R,
    mut writer: W,
    options: ProcessChildConfig,
    initialize: F,
) -> Result<(), ProcessError>
where
    R: Read + AsRawFd,
    W: Write + AsRawFd,
    H: ProcessChildHandler,
    F: FnOnce(ProcessBoot) -> Result<H, ProcessHandlerError>,
{
    options.frame_limits.validate()?;
    deadline_after(options.io_timeout)?;
    ipc::nonblocking(reader.as_raw_fd())?;
    ipc::nonblocking(writer.as_raw_fd())?;
    let Message::Boot {
        identity,
        limits,
        config,
    } = ipc::receive(
        &mut reader,
        options.frame_limits.max_frame_bytes,
        deadline_after(options.io_timeout)?,
    )?
    else {
        return Err(protocol("first message must be Boot"));
    };
    limits.validate()?;
    let ceiling = options.frame_limits;
    if limits.max_frame_bytes > ceiling.max_frame_bytes
        || limits.max_config_bytes > ceiling.max_config_bytes
        || limits.max_command_bytes > ceiling.max_command_bytes
        || limits.max_output_bytes > ceiling.max_output_bytes
        || limits.max_error_bytes > ceiling.max_error_bytes
    {
        return Err(protocol("Boot limits exceed child-local ceilings"));
    }
    ipc::serialize(&config, limits.max_config_bytes)?;
    let boot = ProcessBoot {
        identity,
        limits,
        config,
    };
    let handler = match guarded(|| initialize(boot)) {
        Ok(handler) => handler,
        Err(failure) => {
            send(
                &mut writer,
                failure_message(identity, None, None, failure.clone(), limits),
                limits,
                options.io_timeout,
            )?;
            return Err(ProcessError::RemoteFailure { failure });
        }
    };
    // Retain the allocation, including inline DMA sources, even if an unknown
    // failure or a later serialization/transport panic unwinds this function.
    let mut handler = ManuallyDrop::new(Box::new(handler));
    send(
        &mut writer,
        Message::Ready { identity, limits },
        limits,
        options.io_timeout,
    )?;
    let mut last_command = 0;
    loop {
        let message = ipc::receive_idle(&mut reader, limits.max_frame_bytes, options.io_timeout)?;
        match message {
            Message::Execute {
                identity: key,
                session,
                payload,
            } => {
                if key.owner != identity || key.command.get() <= last_command {
                    return Err(protocol("wrong owner or replayed command"));
                }
                last_command = key.command.get();
                ipc::serialize(&payload, limits.max_command_bytes)?;
                let result = match serde_json::from_value(payload) {
                    Ok(command) => guarded(|| {
                        handler.execute(ProcessRequest {
                            identity: key,
                            session,
                            command,
                        })
                    })
                    .and_then(|output| {
                        ipc::payload(&output, limits.max_output_bytes).map_err(|error| {
                            ProcessHandlerError::unknown(
                                ProcessFailureKind::OutputEncode,
                                error.to_string(),
                            )
                        })
                    }),
                    Err(error) => Err(ProcessHandlerError::fenced(
                        ProcessFailureKind::CommandDecode,
                        error.to_string(),
                    )),
                };
                match result {
                    Ok(payload) => send(
                        &mut writer,
                        Message::Complete {
                            identity: key,
                            session,
                            payload,
                        },
                        limits,
                        options.io_timeout,
                    )?,
                    Err(failure) => {
                        let unknown = failure.quiescence == ProcessQuiescence::Unknown;
                        send(
                            &mut writer,
                            failure_message(
                                identity,
                                Some(key),
                                Some(session),
                                failure.clone(),
                                limits,
                            ),
                            limits,
                            options.io_timeout,
                        )?;
                        if unknown {
                            return Err(ProcessError::RemoteFailure { failure });
                        }
                    }
                }
            }
            Message::Shutdown { owner } if owner == identity => {
                if let Err(failure) = guarded(|| handler.shutdown()) {
                    send(
                        &mut writer,
                        failure_message(identity, None, None, failure.clone(), limits),
                        limits,
                        options.io_timeout,
                    )?;
                    return Err(ProcessError::RemoteFailure { failure });
                }
                // Ack is after shutdown AND Drop. A stuck destructor is still
                // covered by the independent parent's shutdown deadline.
                drop(ManuallyDrop::into_inner(handler));
                send(
                    &mut writer,
                    Message::ShutdownAck { owner },
                    limits,
                    options.io_timeout,
                )?;
                return Ok(());
            }
            _ => return Err(protocol("unexpected child command")),
        }
    }
}

fn guarded<T>(
    operation: impl FnOnce() -> Result<T, ProcessHandlerError>,
) -> Result<T, ProcessHandlerError> {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(result) => result,
        Err(payload) => {
            // Arbitrary panic payload Drop is not trusted on this boundary.
            std::mem::forget(payload);
            Err(ProcessHandlerError::unknown(
                ProcessFailureKind::Panic,
                "child handler panicked",
            ))
        }
    }
}

fn failure_message(
    owner: ProcessIdentity,
    command: Option<ProcessCommandIdentity>,
    session: Option<u64>,
    failure: ProcessHandlerError,
    limits: ProcessFrameLimits,
) -> Message {
    Message::Failure {
        owner,
        command,
        session,
        failure: ipc::clip_error(failure, limits.max_error_bytes),
    }
}
fn send<W: Write + AsRawFd>(
    writer: &mut W,
    message: Message,
    limits: ProcessFrameLimits,
    timeout: Duration,
) -> Result<(), ProcessError> {
    let deadline = deadline_after(timeout)?;
    let frame = ipc::encode(message, limits.max_frame_bytes)?;
    ipc::send(writer, &frame, deadline)
}
