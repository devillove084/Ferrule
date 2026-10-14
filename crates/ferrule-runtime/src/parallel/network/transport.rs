use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

use rustls::{ClientConnection, Connection, ServerConnection};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::super::process::{
    ProcessCommandIdentity, ProcessFrameLimits, ProcessHandlerError, ProcessIdentity,
};
use super::{ClientTls, NETWORK_PROTOCOL_VERSION, NetworkBinding, NetworkError, ServerTls, tls};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Envelope {
    version: u16,
    message: Message,
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Message {
    Hello {
        binding: NetworkBinding,
        limits: ProcessFrameLimits,
        command_timeout_ms: u64,
        config: Value,
    },
    Ready {
        binding: NetworkBinding,
        limits: ProcessFrameLimits,
    },
    Execute {
        identity: ProcessCommandIdentity,
        session: u64,
        payload: Value,
    },
    Complete {
        identity: ProcessCommandIdentity,
        session: u64,
        payload: Value,
    },
    Failure {
        owner: ProcessIdentity,
        command: Option<ProcessCommandIdentity>,
        session: Option<u64>,
        failure: ProcessHandlerError,
    },
    Shutdown {
        owner: ProcessIdentity,
        budget_ms: u64,
    },
    ShutdownAck {
        owner: ProcessIdentity,
    },
}
impl Envelope {
    pub(super) fn encode(message: Message, limit: usize) -> Result<Vec<u8>, NetworkError> {
        bounded(
            &Self {
                version: NETWORK_PROTOCOL_VERSION,
                message,
            },
            limit,
        )
    }
}

pub(super) fn bounded<T: Serialize + ?Sized>(
    value: &T,
    limit: usize,
) -> Result<Vec<u8>, NetworkError> {
    use crate::parallel::codec::{BoundedJsonError, bounded_json};

    bounded_json(value, limit).map_err(|error| match error {
        BoundedJsonError::Limit => NetworkError::FrameTooLarge,
        BoundedJsonError::Serialize(_) => NetworkError::Serialization,
    })
}
pub(super) fn payload<T: Serialize>(value: &T, limit: usize) -> Result<Value, NetworkError> {
    serde_json::from_slice(&bounded(value, limit)?).map_err(|_| NetworkError::Serialization)
}
pub(super) fn deadline_after(duration: Duration) -> Result<Instant, NetworkError> {
    if duration.is_zero() {
        return Err(NetworkError::Configuration("zero deadline budget"));
    }
    Instant::now()
        .checked_add(duration)
        .ok_or(NetworkError::Configuration("deadline overflow"))
}
pub(super) fn check_deadline(deadline: Instant) -> Result<(), NetworkError> {
    if Instant::now() >= deadline {
        Err(NetworkError::Deadline)
    } else {
        Ok(())
    }
}
pub(super) fn wait(
    fd: &impl AsRawFd,
    events: i16,
    deadline: Option<Instant>,
) -> Result<(), NetworkError> {
    loop {
        let timeout = if let Some(deadline) = deadline {
            check_deadline(deadline)?;
            let remaining = deadline.saturating_duration_since(Instant::now());
            remaining
                .as_millis()
                .saturating_add(1)
                .min(i32::MAX as u128) as i32
        } else {
            -1
        };
        let mut entry = libc::pollfd {
            fd: fd.as_raw_fd(),
            events,
            revents: 0,
        };
        // The borrowed descriptor remains open; poll initializes only revents.
        #[allow(unsafe_code)]
        let result = unsafe { libc::poll(&mut entry, 1, timeout) };
        if let Some(deadline) = deadline {
            check_deadline(deadline)?;
        }
        if result > 0 {
            if entry.revents & libc::POLLNVAL != 0 {
                return Err(NetworkError::Disconnected);
            }
            return Ok(());
        }
        if result < 0 {
            let e = io::Error::last_os_error();
            if e.kind() != io::ErrorKind::Interrupted {
                return Err(e.into());
            }
        }
    }
}

pub(super) struct Channel {
    pub(super) socket: TcpStream,
    pub(super) connection: Connection,
    pending: Option<u8>,
    handshake_bytes: usize,
}
impl Channel {
    pub(super) fn client(
        socket: TcpStream,
        tls: &ClientTls,
        deadline: Instant,
    ) -> Result<Self, NetworkError> {
        let connection = Connection::Client(
            ClientConnection::new(tls.config.clone(), tls.name.clone())
                .map_err(|_| NetworkError::Authentication)?,
        );
        Self::handshake(socket, connection, &tls.pin, deadline)
    }
    pub(super) fn server(
        socket: TcpStream,
        tls: &ServerTls,
        deadline: Instant,
    ) -> Result<Self, NetworkError> {
        let connection = Connection::Server(
            ServerConnection::new(tls.config.clone()).map_err(|_| NetworkError::Authentication)?,
        );
        Self::handshake(socket, connection, &tls.pin, deadline)
    }
    fn handshake(
        socket: TcpStream,
        mut connection: Connection,
        pin: &rustls::pki_types::CertificateDer<'_>,
        deadline: Instant,
    ) -> Result<Self, NetworkError> {
        socket.set_nonblocking(true)?;
        socket.set_nodelay(true)?;
        connection.set_buffer_limit(Some(64 * 1024));
        let mut channel = Self {
            socket,
            connection,
            pending: None,
            handshake_bytes: 0,
        };
        while channel.connection.is_handshaking() {
            channel.flush(deadline)?;
            if !channel.connection.is_handshaking() {
                break;
            }
            channel.read_tls(deadline)?;
        }
        channel.flush(deadline)?;
        tls::verify_peer(&channel.connection, pin)?;
        check_deadline(deadline)?;
        Ok(channel)
    }
    fn flush(&mut self, deadline: Instant) -> Result<(), NetworkError> {
        while self.connection.wants_write() {
            check_deadline(deadline)?;
            match self.connection.write_tls(&mut self.socket) {
                Ok(0) => return Err(NetworkError::Disconnected),
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    wait(&self.socket, libc::POLLOUT, Some(deadline))?
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
        check_deadline(deadline)
    }
    fn read_tls(&mut self, deadline: Instant) -> Result<(), NetworkError> {
        loop {
            check_deadline(deadline)?;
            match self.connection.read_tls(&mut self.socket) {
                Ok(0) => return Err(NetworkError::Disconnected),
                Ok(count) => {
                    if self.connection.is_handshaking() {
                        self.handshake_bytes = self.handshake_bytes.saturating_add(count);
                        if self.handshake_bytes > 1024 * 1024 {
                            return Err(NetworkError::Authentication);
                        }
                    }
                    self.connection
                        .process_new_packets()
                        .map_err(|_| NetworkError::Authentication)?;
                    return check_deadline(deadline);
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    wait(&self.socket, libc::POLLIN, Some(deadline))?
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
    }
    pub(super) fn write_plain(
        &mut self,
        mut bytes: &[u8],
        deadline: Instant,
    ) -> Result<(), NetworkError> {
        while !bytes.is_empty() {
            check_deadline(deadline)?;
            // Once accepted by the TLS writer this is possibly dispatched,
            // even if a subsequent ciphertext flush times out.
            let count = self
                .connection
                .writer()
                .write(&bytes[..bytes.len().min(16 * 1024)])?;
            if count == 0 && !self.connection.wants_write() {
                return Err(NetworkError::Disconnected);
            }
            bytes = &bytes[count..];
            self.flush(deadline)?;
        }
        self.flush(deadline)
    }
    pub(super) fn send(&mut self, bytes: &[u8], deadline: Instant) -> Result<(), NetworkError> {
        let length = u32::try_from(bytes.len()).map_err(|_| NetworkError::FrameTooLarge)?;
        if length == 0 {
            return Err(NetworkError::Protocol("empty frame"));
        }
        self.write_plain(&length.to_be_bytes(), deadline)?;
        self.write_plain(bytes, deadline)
    }
    fn read_plain(&mut self, bytes: &mut [u8], deadline: Instant) -> Result<(), NetworkError> {
        let mut offset = 0;
        if let Some(byte) = self.pending.take() {
            bytes[0] = byte;
            offset = 1;
        }
        while offset < bytes.len() {
            check_deadline(deadline)?;
            match self.connection.reader().read(&mut bytes[offset..]) {
                Ok(0) => return Err(NetworkError::Disconnected),
                Ok(n) => offset += n,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    self.flush(deadline)?;
                    self.read_tls(deadline)?;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
        check_deadline(deadline)
    }
    pub(super) fn receive(
        &mut self,
        limit: usize,
        deadline: Instant,
    ) -> Result<Message, NetworkError> {
        let mut prefix = [0u8; 4];
        self.read_plain(&mut prefix, deadline)?;
        let count = u32::from_be_bytes(prefix) as usize;
        if count == 0 {
            return Err(NetworkError::Protocol("empty frame"));
        }
        if count > limit {
            return Err(NetworkError::FrameTooLarge);
        }
        let mut bytes = vec![0; count];
        self.read_plain(&mut bytes, deadline)?;
        let envelope: Envelope =
            serde_json::from_slice(&bytes).map_err(|_| NetworkError::Serialization)?;
        check_deadline(deadline)?;
        if envelope.version != NETWORK_PROTOCOL_VERSION {
            return Err(NetworkError::Protocol("unsupported version"));
        }
        Ok(envelope.message)
    }
    pub(super) fn wait_frame(&mut self, budget: Duration) -> Result<Instant, NetworkError> {
        if self.pending.is_none() {
            let mut first = [0];
            match self.connection.reader().read(&mut first) {
                Ok(0) => return Err(NetworkError::Disconnected),
                Ok(_) => self.pending = Some(first[0]),
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    wait(&self.socket, libc::POLLIN, None)?
                }
                Err(e) => return Err(e.into()),
            }
        }
        deadline_after(budget)
    }
    pub(super) fn disconnected(&self) -> bool {
        matches!(self.socket.peek(&mut [0]), Ok(0))
    }
}
impl Drop for Channel {
    fn drop(&mut self) {
        let _ = self.socket.shutdown(Shutdown::Both);
    }
}
