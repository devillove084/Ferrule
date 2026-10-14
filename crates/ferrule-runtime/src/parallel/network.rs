//! Authenticated, host-staged linear rank transport, not a collective or scheduler.
//!
//! One endpoint accepts one connection for one configured rank lifetime. Both
//! peers validate a CA chain AND pin the other's leaf before application traffic.
//! The pinned certificate authorizes exactly the locally configured binding;
//! received identity fields never grant authority. No reconnect, restart or replay.
//! Unknown remote custody is not a local process termination report or GPU fence.
//!
//! Call on a host control owner. Filesystem access, OS process spawn and arbitrary
//! serde implementations cannot be preempted; all network waits have absolute
//! deadlines. Endpoint process/device isolation after host loss remains a
//! deployment responsibility. See `tests/support/network/CLI_CONTRACT.md` for CLI integration.

mod tls;
mod transport;

use std::fmt;
use std::marker::PhantomData;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

use ferrule_common::execution::ExecutionTransactionId;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::data::{PanicQuiescence, ReplicaWorker, WorkRequest};
use super::process::{
    ProcessCommandId, ProcessCommandIdentity, ProcessError, ProcessFrameLimits,
    ProcessHandlerError, ProcessIdentity, ProcessLaunch, ProcessOwnerConfig, ProcessQuiescence,
    ProcessRankOwner,
};
pub use tls::{ClientTls, ServerTls, TlsFiles};
use transport::{Channel, Envelope, Message, bounded, check_deadline, deadline_after, payload};

pub const NETWORK_PROTOCOL_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinearMode {
    Dp,
    TpColumn,
    TpRow,
}

/// Trusted rank placement and geometry, checked independently on both hosts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinearTarget {
    pub mode: LinearMode,
    pub world: u32,
    pub device: u32,
    pub in_features: usize,
    pub out_features: usize,
    pub max_rows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkBinding {
    pub run_id: String,
    pub controller_id: String,
    pub node_id: String,
    pub identity: ProcessIdentity,
    pub target: LinearTarget,
}
impl NetworkBinding {
    pub fn validate(&self) -> Result<(), NetworkError> {
        for name in [&self.run_id, &self.controller_id, &self.node_id] {
            if name.is_empty() || name.len() > 256 || name.chars().any(char::is_control) {
                return Err(NetworkError::Configuration("invalid configured identity"));
            }
        }
        let t = &self.target;
        if t.world == 0
            || self.identity.rank.get() >= t.world
            || t.in_features == 0
            || t.out_features == 0
            || t.max_rows == 0
            || t.in_features.checked_mul(t.out_features).is_none()
            || t.max_rows
                .checked_mul(t.in_features.max(t.out_features))
                .is_none()
        {
            return Err(NetworkError::Configuration(
                "invalid linear placement/geometry",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NetworkOptions {
    pub limits: ProcessFrameLimits,
    /// Connect/accept, TLS, application handshake, process Boot and Ready.
    pub startup_timeout: Duration,
    /// Fixed frame/command budget, also matched in the application handshake.
    pub command_timeout: Duration,
}
impl Default for NetworkOptions {
    fn default() -> Self {
        Self {
            limits: ProcessFrameLimits::default(),
            startup_timeout: Duration::from_secs(30),
            command_timeout: Duration::from_secs(30),
        }
    }
}
impl NetworkOptions {
    pub fn validate(self) -> Result<(), NetworkError> {
        self.limits
            .validate()
            .map_err(|_| NetworkError::Configuration("invalid frame limits"))?;
        deadline_after(self.startup_timeout)?;
        deadline_after(self.command_timeout)?;
        if self.command_timeout.as_millis() == 0
            || self.command_timeout.as_millis() > u64::MAX as u128
            || !self
                .command_timeout
                .subsec_nanos()
                .is_multiple_of(1_000_000)
        {
            return Err(NetworkError::Configuration(
                "command timeout must be whole milliseconds",
            ));
        }
        Ok(())
    }
}

/// Deliberately separate from ProcessError: a lost connection proves no reap.
#[derive(Debug)]
pub enum NetworkError {
    Configuration(&'static str),
    Authentication,
    Protocol(&'static str),
    FrameTooLarge,
    Serialization,
    Deadline,
    Disconnected,
    Io(std::io::Error),
    /// Valid matching failure with an explicit child fence.
    Fenced(ProcessHandlerError),
    /// Terminal. Never inspect its nested cause to downgrade custody.
    Unknown(Box<NetworkError>),
    Unavailable,
}
impl NetworkError {
    pub fn is_command_rejection(&self) -> bool {
        matches!(
            self,
            Self::Configuration(_)
                | Self::FrameTooLarge
                | Self::Serialization
                | Self::Deadline
                | Self::Fenced(_)
        )
    }
}
impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // No remote payload, PEM, file path or private key in transport errors.
        match self {
            Self::Configuration(s) => write!(f, "network configuration: {s}"),
            Self::Protocol(s) => write!(f, "network protocol: {s}"),
            Self::Authentication => f.write_str("network authentication failed"),
            Self::FrameTooLarge => f.write_str("network frame/payload exceeds configured limit"),
            Self::Serialization => f.write_str("network JSON encoding/decoding failed"),
            Self::Deadline => f.write_str("network absolute deadline expired"),
            Self::Disconnected => f.write_str("network peer disconnected"),
            Self::Io(_) => f.write_str("network I/O failed"),
            Self::Fenced(_) => f.write_str("remote rank reported a fenced failure"),
            Self::Unknown(_) => f.write_str("network rank quiescence unknown; replay forbidden"),
            Self::Unavailable => f.write_str("network rank is permanently unavailable"),
        }
    }
}
impl std::error::Error for NetworkError {}
impl From<std::io::Error> for NetworkError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkOwnerState {
    Ready,
    Unknown,
    Shutdown,
}

pub struct NetworkRankOwner<C, I, O> {
    binding: NetworkBinding,
    options: NetworkOptions,
    channel: Option<Channel>,
    next_command: u64,
    state: NetworkOwnerState,
    types: PhantomData<fn(C, I) -> O>,
}
impl<C, I, O> fmt::Debug for NetworkRankOwner<C, I, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NetworkRankOwner")
            .field("binding", &self.binding)
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}
impl<C: Serialize, I: Serialize, O: DeserializeOwned> NetworkRankOwner<C, I, O> {
    pub fn connect(
        address: SocketAddr,
        tls: &ClientTls,
        binding: NetworkBinding,
        config: &C,
        options: NetworkOptions,
    ) -> Result<Self, NetworkError> {
        binding.validate()?;
        options.validate()?;
        let deadline = deadline_after(options.startup_timeout)?;
        let frame = Envelope::encode(
            Message::Hello {
                binding: binding.clone(),
                limits: options.limits,
                command_timeout_ms: options.command_timeout.as_millis() as u64,
                config: payload(config, options.limits.max_config_bytes)?,
            },
            options.limits.max_frame_bytes,
        )?;
        check_deadline(deadline)?;
        let socket = TcpStream::connect_timeout(
            &address,
            deadline.saturating_duration_since(Instant::now()),
        )?;
        let mut channel = Channel::client(socket, tls, deadline)?;
        // Once Hello may reach the endpoint, even startup failure can mean a
        // spawned remote owner. Never turn it into a local fenced rejection.
        let startup = (|| {
            channel.send(&frame, deadline)?;
            match channel.receive(options.limits.max_frame_bytes, deadline)? {
                Message::Ready {
                    binding: actual,
                    limits,
                } if actual == binding && limits == options.limits => Ok(()),
                _ => Err(NetworkError::Protocol("invalid Ready identity/type/limits")),
            }
        })();
        startup.map_err(|e| NetworkError::Unknown(Box::new(e)))?;
        Ok(Self {
            binding,
            options,
            channel: Some(channel),
            next_command: 1,
            state: NetworkOwnerState::Ready,
            types: PhantomData,
        })
    }

    pub fn execute(
        &mut self,
        transaction: ExecutionTransactionId,
        session: u64,
        input: &I,
    ) -> Result<O, NetworkError> {
        if self.state != NetworkOwnerState::Ready {
            return Err(NetworkError::Unavailable);
        }
        let deadline = deadline_after(self.options.command_timeout)?;
        let next = self
            .next_command
            .checked_add(1)
            .ok_or(NetworkError::Configuration("command identity exhausted"))?;
        let key = ProcessCommandIdentity {
            owner: self.binding.identity,
            command: ProcessCommandId::new(self.next_command)
                .map_err(|_| NetworkError::Configuration("zero command"))?,
            transaction: transaction.into(),
        };
        // Bounded serialization rejects locally before TLS buffering, without
        // consuming the sequence number or disturbing a healthy connection.
        let frame = Envelope::encode(
            Message::Execute {
                identity: key,
                session,
                payload: payload(input, self.options.limits.max_command_bytes)?,
            },
            self.options.limits.max_frame_bytes,
        )?;
        check_deadline(deadline)?;
        self.next_command = next;
        self.state = NetworkOwnerState::Unknown;
        // A panicking decoder also drops the socket and leaves terminal state.
        let mut channel = self.channel.take().ok_or(NetworkError::Unavailable)?;
        let result = (|| {
            channel.send(&frame, deadline)?;
            let message = channel.receive(self.options.limits.max_frame_bytes, deadline)?;
            match message {
                Message::Complete {
                    identity,
                    session: actual,
                    payload,
                } if identity == key && actual == session => {
                    bounded(&payload, self.options.limits.max_output_bytes)?;
                    let output =
                        serde_json::from_value(payload).map_err(|_| NetworkError::Serialization)?;
                    check_deadline(deadline)?;
                    Ok(output)
                }
                Message::Failure {
                    owner,
                    command: Some(command),
                    session: Some(actual),
                    failure,
                } if owner == self.binding.identity
                    && command == key
                    && actual == session
                    && failure.message.len() <= self.options.limits.max_error_bytes
                    && failure.quiescence == ProcessQuiescence::Fenced =>
                {
                    Err(NetworkError::Fenced(failure))
                }
                _ => Err(NetworkError::Protocol(
                    "invalid completion/failure correlation",
                )),
            }
        })();
        match result {
            Ok(o) => {
                self.channel = Some(channel);
                self.state = NetworkOwnerState::Ready;
                Ok(o)
            }
            Err(e @ NetworkError::Fenced(_)) => {
                self.channel = Some(channel);
                self.state = NetworkOwnerState::Ready;
                Err(e)
            }
            Err(e) => Err(self.fail(e)),
        }
    }

    /// Adapter boundary for existing DP/TP workers. Only explicit pre-dispatch
    /// rejection or a matching Fenced reply may return as an ordinary error.
    pub fn execute_or_quarantine(
        &mut self,
        transaction: ExecutionTransactionId,
        session: u64,
        input: &I,
    ) -> Result<O, NetworkError> {
        let result = self.execute(transaction, session, input);
        if result.as_ref().is_err_and(|e| !e.is_command_rejection()) {
            std::panic::panic_any(PanicQuiescence::Unknown);
        }
        result
    }
}
impl<C, I, O> NetworkRankOwner<C, I, O> {
    pub fn binding(&self) -> &NetworkBinding {
        &self.binding
    }
    pub fn state(&self) -> NetworkOwnerState {
        self.state
    }
    fn fail(&mut self, error: NetworkError) -> NetworkError {
        self.state = NetworkOwnerState::Unknown;
        self.channel.take();
        NetworkError::Unknown(Box::new(error))
    }
    pub fn shutdown(&mut self) -> Result<(), NetworkError> {
        self.shutdown_until(deadline_after(self.options.command_timeout)?)
    }
    pub fn shutdown_until(&mut self, absolute_deadline: Instant) -> Result<(), NetworkError> {
        if self.state == NetworkOwnerState::Shutdown {
            return Ok(());
        }
        if self.state != NetworkOwnerState::Ready {
            return Err(NetworkError::Unavailable);
        }
        let result = (|| {
            let deadline = absolute_deadline.min(deadline_after(self.options.command_timeout)?);
            check_deadline(deadline)?;
            let frame = Envelope::encode(
                Message::Shutdown {
                    owner: self.binding.identity,
                    budget_ms: deadline
                        .saturating_duration_since(Instant::now())
                        .as_millis() as u64,
                },
                self.options.limits.max_frame_bytes,
            )?;
            let channel = self.channel.as_mut().ok_or(NetworkError::Unavailable)?;
            channel.send(&frame, deadline)?;
            match channel.receive(self.options.limits.max_frame_bytes, deadline)? {
                Message::ShutdownAck { owner } if owner == self.binding.identity => Ok(()),
                _ => Err(NetworkError::Protocol("invalid shutdown acknowledgement")),
            }
        })();
        match result {
            Ok(()) => {
                self.state = NetworkOwnerState::Shutdown;
                self.channel.take();
                Ok(())
            }
            Err(e) => Err(self.fail(e)),
        }
    }
    pub fn shutdown_or_quarantine(&mut self) -> Result<(), NetworkError> {
        if self.shutdown().is_err() {
            std::panic::panic_any(PanicQuiescence::Unknown);
        }
        Ok(())
    }
}

/// Direct DP adapter. TP adapters must instead call execute_or_quarantine with
/// TensorWork::transaction and serialize their existing Compute/Apply DTOs.
impl<C: Serialize, I: Serialize, O: DeserializeOwned> ReplicaWorker<I>
    for NetworkRankOwner<C, I, O>
{
    type Output = O;
    type Error = NetworkError;
    fn execute(&mut self, request: WorkRequest<I>) -> Result<O, NetworkError> {
        if request.rank != self.binding.identity.rank {
            return Err(NetworkError::Configuration("DP request rank mismatch"));
        }
        self.execute_or_quarantine(request.transaction, request.session.0, &request.input)
    }
    fn shutdown(&mut self) -> Result<(), NetworkError> {
        self.shutdown_or_quarantine()
    }
}

/// Launch/Boot are trusted endpoint-local configuration, never an RPC argument.
/// Only expose a linear child initializer at a production rank endpoint.
pub struct NetworkEndpoint<C> {
    pub binding: NetworkBinding,
    pub options: NetworkOptions,
    pub tls: ServerTls,
    pub launch: ProcessLaunch,
    pub local_identity: ProcessIdentity,
    pub local_options: ProcessOwnerConfig,
    pub config: C,
}
impl<C: Serialize> NetworkEndpoint<C> {
    /// Port zero is useful for tests. Non-loopback must be explicitly authorized.
    pub fn bind(address: SocketAddr, allow_remote_bind: bool) -> Result<TcpListener, NetworkError> {
        if !address.ip().is_loopback() && !allow_remote_bind {
            return Err(NetworkError::Configuration(
                "non-loopback bind requires explicit authorization",
            ));
        }
        Ok(TcpListener::bind(address)?)
    }

    /// Consumes the listener and endpoint; a second rank connection is impossible.
    /// A successful return requires child ShutdownAck AND reap before network ACK.
    pub fn serve_once(self, listener: TcpListener) -> Result<(), NetworkError> {
        self.binding.validate()?;
        self.options.validate()?;
        self.local_options
            .validate()
            .map_err(|_| NetworkError::Configuration("invalid local process options"))?;
        let deadline = deadline_after(self.options.startup_timeout)?;
        let expected = payload(&self.config, self.options.limits.max_config_bytes)?;
        listener.set_nonblocking(true)?;
        let socket = loop {
            check_deadline(deadline)?;
            match listener.accept() {
                Ok((socket, _)) => break socket,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    transport::wait(&listener, libc::POLLIN, Some(deadline))?
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        };
        drop(listener);
        let mut channel = Channel::server(socket, &self.tls, deadline)?;
        match channel.receive(self.options.limits.max_frame_bytes, deadline)? {
            Message::Hello {
                binding,
                limits,
                command_timeout_ms,
                config,
            } if binding == self.binding
                && limits == self.options.limits
                && command_timeout_ms == self.options.command_timeout.as_millis() as u64
                && config == expected =>
            {
                bounded(&config, self.options.limits.max_config_bytes)?;
            }
            _ => return Err(NetworkError::Protocol("unauthorized binding/limits/Boot")),
        }
        check_deadline(deadline)?;
        let mut local_options = self.local_options;
        local_options.startup_timeout = local_options
            .startup_timeout
            .min(deadline.saturating_duration_since(Instant::now()));
        local_options.command_timeout = local_options
            .command_timeout
            .min(self.options.command_timeout);
        let mut child = ProcessRankOwner::<C, Value, Value>::spawn(
            self.launch,
            self.local_identity,
            &self.config,
            local_options,
        )
        .map_err(|_| {
            NetworkError::Unknown(Box::new(NetworkError::Protocol(
                "local child startup failed",
            )))
        })?;
        let result = (|| {
            let frame = Envelope::encode(
                Message::Ready {
                    binding: self.binding.clone(),
                    limits: self.options.limits,
                },
                self.options.limits.max_frame_bytes,
            )?;
            channel.send(&frame, deadline)?;
            let mut next_command = 1u64;
            loop {
                // Idle is healthy. The first readable TLS byte/plaintext starts
                // one immutable deadline covering the entire frame and reply.
                let frame_deadline = channel.wait_frame(self.options.command_timeout)?;
                let message =
                    channel.receive(self.options.limits.max_frame_bytes, frame_deadline)?;
                let reply = match message {
                    Message::Execute {
                        identity,
                        session,
                        payload,
                    } if identity.owner == self.binding.identity
                        && identity.command.get() == next_command =>
                    {
                        next_command = next_command
                            .checked_add(1)
                            .ok_or(NetworkError::Protocol("command exhausted"))?;
                        bounded(&payload, self.options.limits.max_command_bytes)?;
                        check_deadline(frame_deadline)?;
                        // Process' observer already invalidates and cleans up on
                        // unwind. A private payload enforces this frame's remaining
                        // budget without modifying/restarting the process protocol.
                        let execution =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                child.execute_observed(
                                    identity.transaction.into(),
                                    session,
                                    &payload,
                                    &mut || {
                                        if Instant::now() >= frame_deadline
                                            || channel.disconnected()
                                        {
                                            std::panic::resume_unwind(Box::new(Interrupted));
                                        }
                                    },
                                )
                            }));
                        let execution = match execution {
                            Ok(result) => result,
                            Err(e) if e.is::<Interrupted>() => return Err(NetworkError::Deadline),
                            Err(e) => std::panic::resume_unwind(e),
                        };
                        check_deadline(frame_deadline)?;
                        match execution {
                            Ok(output) => {
                                bounded(&output, self.options.limits.max_output_bytes)?;
                                Message::Complete {
                                    identity,
                                    session,
                                    payload: output,
                                }
                            }
                            Err(ProcessError::RemoteFailure { failure })
                                if failure.quiescence == ProcessQuiescence::Fenced
                                    && failure.message.len()
                                        <= self.options.limits.max_error_bytes =>
                            {
                                Message::Failure {
                                    owner: self.binding.identity,
                                    command: Some(identity),
                                    session: Some(session),
                                    failure,
                                }
                            }
                            Err(_) => {
                                return Err(NetworkError::Protocol(
                                    "local child quiescence not established",
                                ));
                            }
                        }
                    }
                    Message::Shutdown { owner, budget_ms } if owner == self.binding.identity => {
                        let frame_start = frame_deadline - self.options.command_timeout;
                        let remote_deadline = frame_start
                            .checked_add(Duration::from_millis(budget_ms))
                            .ok_or(NetworkError::Protocol("shutdown budget overflow"))?
                            .min(frame_deadline);
                        child.shutdown_until(remote_deadline).map_err(|_| {
                            NetworkError::Protocol("local shutdown quiescence not established")
                        })?;
                        let ack = Envelope::encode(
                            Message::ShutdownAck { owner },
                            self.options.limits.max_frame_bytes,
                        )?;
                        channel.send(&ack, remote_deadline)?;
                        return Ok(());
                    }
                    _ => {
                        return Err(NetworkError::Protocol(
                            "invalid owner/command/type; replay forbidden",
                        ));
                    }
                };
                let frame = Envelope::encode(reply, self.options.limits.max_frame_bytes)?;
                channel.send(&frame, frame_deadline)?;
            }
        })();
        if result.is_err() {
            // OS cleanup is deliberately not exposed as remote fence evidence.
            let _ = child.terminate();
        }
        result.map_err(|e| NetworkError::Unknown(Box::new(e)))
    }
}
struct Interrupted;

pub fn serve_once<C: Serialize>(
    endpoint: NetworkEndpoint<C>,
    listener: TcpListener,
) -> Result<(), NetworkError> {
    endpoint.serve_once(listener)
}

#[cfg(test)]
#[path = "../../tests/unit/parallel/network/mod.rs"]
mod tests;
