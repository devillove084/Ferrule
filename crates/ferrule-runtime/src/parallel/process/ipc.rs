use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, RawFd};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{
    PROCESS_PROTOCOL_VERSION, ProcessCommandIdentity, ProcessError, ProcessFrameLimits,
    ProcessHandlerError, ProcessIdentity, deadline_after, protocol,
};

pub(super) const HARD_MAX_FRAME_BYTES: usize = 128 * 1024 * 1024;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    version: u16,
    message: Message,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Message {
    Boot {
        identity: ProcessIdentity,
        limits: ProcessFrameLimits,
        config: Value,
    },
    Ready {
        identity: ProcessIdentity,
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
    },
    ShutdownAck {
        owner: ProcessIdentity,
    },
}

/// Unlike to_vec followed by a length check, this refuses growth as soon as a
/// serializer exceeds the limit. User-provided Serialize/Deserialize code must
/// still be cooperative; it is not executed in a preemptible sandbox.
struct LimitedBuffer {
    bytes: Vec<u8>,
    limit: usize,
    exceeded: bool,
}
impl Write for LimitedBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            self.exceeded = true;
            return Err(io::Error::other("serialization byte limit"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(super) fn serialize<T: Serialize + ?Sized>(
    value: &T,
    limit: usize,
) -> Result<Vec<u8>, ProcessError> {
    let mut writer = LimitedBuffer {
        bytes: Vec::new(),
        limit,
        exceeded: false,
    };
    let result = serde_json::to_writer(&mut writer, value);
    if writer.exceeded {
        return Err(ProcessError::FrameTooLarge { limit });
    }
    result.map_err(|source| ProcessError::Json {
        operation: "serialize process value",
        source,
    })?;
    Ok(writer.bytes)
}

pub(super) fn payload<T: Serialize + ?Sized>(
    value: &T,
    limit: usize,
) -> Result<Value, ProcessError> {
    serde_json::from_slice(&serialize(value, limit)?).map_err(|source| ProcessError::Json {
        operation: "decode bounded process value",
        source,
    })
}

pub(super) fn encode(message: Message, limit: usize) -> Result<Vec<u8>, ProcessError> {
    serialize(
        &Envelope {
            version: PROCESS_PROTOCOL_VERSION,
            message,
        },
        limit,
    )
}

pub(super) fn send<W: Write + AsRawFd>(
    writer: &mut W,
    body: &[u8],
    deadline: Instant,
) -> Result<(), ProcessError> {
    send_observed(writer, body, deadline, &mut || {}, &mut false)
}

pub(super) fn send_observed<W: Write + AsRawFd>(
    writer: &mut W,
    body: &[u8],
    deadline: Instant,
    observe: &mut dyn FnMut(),
    sent: &mut bool,
) -> Result<(), ProcessError> {
    let count = u32::try_from(body.len()).map_err(|_| ProcessError::FrameTooLarge {
        limit: HARD_MAX_FRAME_BYTES,
    })?;
    write_all(writer, &count.to_be_bytes(), deadline, observe, sent)?;
    write_all(writer, body, deadline, observe, sent)
}

/// Only the wait for a new frame is unbounded. Once the pipe becomes readable,
/// even a partial prefix/body must finish under one non-sliding I/O deadline.
/// EOF/HUP wakes the idle wait too; no heartbeat or synthetic work is sent.
pub(super) fn receive_idle<R: Read + AsRawFd>(
    reader: &mut R,
    limit: usize,
    timeout: Duration,
) -> Result<Message, ProcessError> {
    poll_fd(reader.as_raw_fd(), libc::POLLIN, None, &mut || {})?;
    receive(reader, limit, deadline_after(timeout)?)
}

pub(super) fn receive<R: Read + AsRawFd>(
    reader: &mut R,
    limit: usize,
    deadline: Instant,
) -> Result<Message, ProcessError> {
    receive_observed(reader, limit, deadline, &mut || {})
}

pub(super) fn receive_observed<R: Read + AsRawFd>(
    reader: &mut R,
    limit: usize,
    deadline: Instant,
    observe: &mut dyn FnMut(),
) -> Result<Message, ProcessError> {
    let mut prefix = [0u8; 4];
    read_all(reader, &mut prefix, deadline, false, observe)?;
    let count = u32::from_be_bytes(prefix) as usize;
    if count == 0 {
        return Err(protocol("empty frame"));
    }
    if count > limit {
        return Err(ProcessError::FrameTooLarge { limit });
    }
    let mut body = vec![0u8; count];
    read_all(reader, &mut body, deadline, true, observe)?;
    check_deadline(deadline)?;
    let envelope: Envelope =
        serde_json::from_slice(&body).map_err(|source| ProcessError::Json {
            operation: "decode process frame",
            source,
        })?;
    if envelope.version != PROCESS_PROTOCOL_VERSION {
        return Err(protocol("unsupported protocol version"));
    }
    check_deadline(deadline)?;
    Ok(envelope.message)
}

fn write_all<W: Write + AsRawFd>(
    writer: &mut W,
    mut bytes: &[u8],
    deadline: Instant,
    observe: &mut dyn FnMut(),
    sent: &mut bool,
) -> Result<(), ProcessError> {
    while !bytes.is_empty() {
        observe();
        check_deadline(deadline)?;
        match writer.write(bytes) {
            Ok(0) => return Err(ProcessError::PipeClosed),
            Ok(count) => {
                *sent = true;
                bytes = &bytes[count..];
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                poll_fd(writer.as_raw_fd(), libc::POLLOUT, Some(deadline), observe)?
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(source) => {
                return Err(ProcessError::Io {
                    operation: "write process frame",
                    source,
                });
            }
        }
    }
    check_deadline(deadline)
}

fn read_all<R: Read + AsRawFd>(
    reader: &mut R,
    bytes: &mut [u8],
    deadline: Instant,
    body: bool,
    observe: &mut dyn FnMut(),
) -> Result<(), ProcessError> {
    let mut offset = 0;
    while offset < bytes.len() {
        observe();
        check_deadline(deadline)?;
        match reader.read(&mut bytes[offset..]) {
            Ok(0) => {
                return Err(if body || offset > 0 {
                    ProcessError::TruncatedFrame
                } else {
                    ProcessError::PipeClosed
                });
            }
            Ok(count) => offset += count,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                poll_fd(reader.as_raw_fd(), libc::POLLIN, Some(deadline), observe)?
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(source) => {
                return Err(ProcessError::Io {
                    operation: "read process frame",
                    source,
                });
            }
        }
    }
    check_deadline(deadline)
}

pub(super) fn check_deadline(deadline: Instant) -> Result<(), ProcessError> {
    if Instant::now() >= deadline {
        Err(ProcessError::Deadline)
    } else {
        Ok(())
    }
}

#[expect(
    unsafe_code,
    reason = "fcntl changes only the flags of the caller-owned Unix pipe descriptor"
)]
pub(super) fn nonblocking(fd: RawFd) -> Result<(), ProcessError> {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(ProcessError::Io {
            operation: "set process pipe nonblocking",
            source: io::Error::last_os_error(),
        });
    }
    Ok(())
}

#[expect(
    unsafe_code,
    reason = "poll borrows one initialized pollfd for the duration of the syscall"
)]
fn poll_fd(
    fd: RawFd,
    events: libc::c_short,
    deadline: Option<Instant>,
    observe: &mut dyn FnMut(),
) -> Result<(), ProcessError> {
    loop {
        observe();
        let millis = if let Some(deadline) = deadline {
            check_deadline(deadline)?;
            let remaining = deadline.saturating_duration_since(Instant::now());
            // Observation must not extend the absolute command deadline.
            remaining.as_millis().saturating_add(1).min(10) as i32
        } else {
            -1
        };
        let mut descriptor = libc::pollfd {
            fd,
            events,
            revents: 0,
        };
        let result = unsafe { libc::poll(&mut descriptor, 1, millis) };
        if result > 0 {
            if let Some(deadline) = deadline {
                check_deadline(deadline)?;
            }
            if descriptor.revents & libc::POLLNVAL != 0 {
                return Err(ProcessError::Io {
                    operation: "poll process pipe",
                    source: io::Error::from_raw_os_error(libc::EBADF),
                });
            }
            // HUP may coexist with readable bytes. Let read consume them or
            // report EOF/truncation; write similarly reports EPIPE on POLLERR.
            return Ok(());
        }
        if result == 0 {
            continue;
        }
        let source = io::Error::last_os_error();
        if source.kind() != io::ErrorKind::Interrupted {
            return Err(ProcessError::Io {
                operation: "poll process pipe",
                source,
            });
        }
    }
}

pub(super) fn clip_error(mut failure: ProcessHandlerError, limit: usize) -> ProcessHandlerError {
    let mut end = failure.message.len().min(limit);
    while !failure.message.is_char_boundary(end) {
        end -= 1;
    }
    failure.message.truncate(end);
    failure
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixStream;
    use std::sync::mpsc;
    use std::thread;

    #[test]
    fn idle_eof_wakes_without_a_deadline() {
        let (mut reader, writer) = UnixStream::pair().unwrap();
        reader.set_nonblocking(true).unwrap();
        let (tx, rx) = mpsc::channel();
        let child = thread::spawn(move || {
            tx.send(receive_idle(&mut reader, 1024, Duration::from_millis(50)))
                .unwrap();
        });
        thread::sleep(Duration::from_millis(150));
        assert!(matches!(rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
        drop(writer);
        assert!(matches!(
            rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            Err(ProcessError::PipeClosed)
        ));
        child.join().unwrap();
    }

    #[test]
    fn idle_partial_prefix_and_progressing_body_share_one_absolute_deadline() {
        for progressing in [false, true] {
            let (mut reader, mut writer) = UnixStream::pair().unwrap();
            reader.set_nonblocking(true).unwrap();
            writer.write_all(&[0, 0]).unwrap();
            let (started, ready) = mpsc::channel();
            let (tx, rx) = mpsc::channel();
            let child = thread::spawn(move || {
                let start = Instant::now();
                started.send(()).unwrap();
                let result = receive_idle(&mut reader, 1024, Duration::from_millis(150));
                tx.send((result, start.elapsed())).unwrap();
            });
            ready.recv_timeout(Duration::from_secs(1)).unwrap();
            if progressing {
                thread::sleep(Duration::from_millis(60));
                writer.write_all(&[0, 100]).unwrap();
                for _ in 0..8 {
                    thread::sleep(Duration::from_millis(30));
                    if writer.write_all(b" ").is_err() {
                        break;
                    }
                }
            }
            let (result, elapsed) = rx.recv_timeout(Duration::from_secs(1)).unwrap();
            assert!(matches!(result, Err(ProcessError::Deadline)), "{result:?}");
            assert!(elapsed >= Duration::from_millis(150));
            assert!(
                elapsed < Duration::from_millis(280),
                "frame progress extended deadline: {elapsed:?}"
            );
            drop(writer);
            child.join().unwrap();
        }
    }
}
