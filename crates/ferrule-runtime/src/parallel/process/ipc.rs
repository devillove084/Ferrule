use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, RawFd};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, value::RawValue};

use super::{
    PROCESS_PROTOCOL_VERSION, ProcessCommandIdentity, ProcessError, ProcessFrameLimits,
    ProcessHandlerError, ProcessIdentity, deadline_after, protocol,
};

pub(super) const HARD_MAX_FRAME_BYTES: usize = 128 * 1024 * 1024;

/// Compile-time opt-in diagnostics (`FERRULE_PROCESS_IPC_INSTRUMENT=1`). Normal
/// builds take a constant disabled path: no runtime flag lookup, clocks, TLS
/// counters or file I/O. Instrumented builds additionally require a sink path;
/// aggregate per thread and emit once at thread exit, never inside an I/O loop.
/// Timings are inclusive wall observations, not an additive latency ledger.
mod timing {
    use std::cell::{Cell, RefCell};
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::OnceLock;
    use std::time::Instant;

    static PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
    pub(super) const INSTRUMENTED: bool = match option_env!("FERRULE_PROCESS_IPC_INSTRUMENT") {
        Some(value) => matches!(value.as_bytes(), [b'1']),
        None => false,
    };
    /// Per-chunk clocking is deliberately a separate compile-time experiment.
    /// Aggregate instrumentation never reports a fabricated copy duration.
    const DETAILED_COPY: bool = INSTRUMENTED
        && match option_env!("FERRULE_PROCESS_IPC_DETAILED") {
            Some(value) => matches!(value.as_bytes(), [b'1']),
            None => false,
        };

    #[inline]
    fn enabled() -> bool {
        INSTRUMENTED
            && PATH
                .get_or_init(|| {
                    std::env::var_os("FERRULE_PROCESS_IPC_TIMING")
                        .filter(|path| !path.is_empty())
                        .map(PathBuf::from)
                })
                .is_some()
    }
    const PHASES: [&str; 7] = [
        "encode",
        "payload_decode",
        "frame_decode",
        "frame_write",
        "frame_read",
        "pipe_wait",
        "copy",
    ];
    #[derive(Clone, Copy)]
    pub(super) enum Phase {
        Encode,
        PayloadDecode,
        FrameDecode,
        FrameWrite,
        FrameRead,
        PipeWait,
        Copy,
    }
    #[derive(Clone, Copy, serde::Serialize)]
    struct Counter {
        calls: u64,
        completed: u64,
        completed_bytes: u64,
        wall_ns: Option<u128>,
    }
    impl Default for Counter {
        fn default() -> Self {
            Self {
                calls: 0,
                completed: 0,
                completed_bytes: 0,
                wall_ns: Some(0),
            }
        }
    }
    struct Counters([Counter; 7]);
    impl Default for Counters {
        fn default() -> Self {
            let mut counters = [Counter::default(); 7];
            counters[Phase::Copy as usize].wall_ns = DETAILED_COPY.then_some(0);
            Self(counters)
        }
    }
    thread_local! { static COUNTERS: RefCell<Counters> = RefCell::new(Counters::default()); }
    impl Drop for Counters {
        fn drop(&mut self) {
            let Some(Some(path)) = PATH.get() else {
                return;
            };
            let phases = PHASES
                .into_iter()
                .zip(self.0)
                .collect::<std::collections::BTreeMap<_, _>>();
            let record = serde_json::json!({"schema":"ferrule.ipc-timing.v2",
                "pid":std::process::id(), "thread":format!("{:?}", std::thread::current().id()),
                "phases":phases});
            // Best-effort diagnostics never replace a transport error or ACK.
            // The benchmark runner requires valid, nonempty artifacts separately.
            if let Ok(mut bytes) = serde_json::to_vec(&record) {
                bytes.push(b'\n');
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                {
                    let _ = file.write_all(&bytes);
                }
            }
        }
    }
    use crate::parallel::codec::{BoundedJsonError, bounded_json, bounded_json_counted};

    /// The ordinary build and instrumented-but-disabled mode use the same
    /// uncounted writer. No always-on counters, clock reads, TLS or allocations.
    pub(super) fn encode<T: serde::Serialize + ?Sized>(
        value: &T,
        limit: usize,
    ) -> Result<Vec<u8>, BoundedJsonError> {
        if !enabled() {
            return bounded_json(value, limit);
        }
        encode_counted::<DETAILED_COPY, _>(value, limit)
    }

    fn encode_counted<const DETAILED: bool, T: serde::Serialize + ?Sized>(
        value: &T,
        limit: usize,
    ) -> Result<Vec<u8>, BoundedJsonError> {
        let copy_ns = Cell::new(0u128);
        let (result, counts) = bounded_json_counted(value, limit, |_| {
            DETAILED.then(|| CopyChunk {
                start: Instant::now(),
                wall_ns: &copy_ns,
            })
        });
        // All accepted writes are accounted for even if a later chunk fails.
        // This is the only copy TLS update for the entire bounded encode.
        COUNTERS.with(|counters| {
            let mut counters = counters.borrow_mut();
            let c = &mut counters.0[Phase::Copy as usize];
            c.calls = c.calls.saturating_add(counts.calls);
            c.completed = c.completed.saturating_add(counts.calls);
            c.completed_bytes = c.completed_bytes.saturating_add(counts.bytes as u64);
            c.wall_ns =
                DETAILED.then(|| c.wall_ns.unwrap_or_default().saturating_add(copy_ns.get()));
        });
        result
    }

    struct CopyChunk<'a> {
        start: Instant,
        wall_ns: &'a Cell<u128>,
    }
    impl Drop for CopyChunk<'_> {
        fn drop(&mut self) {
            self.wall_ns.set(
                self.wall_ns
                    .get()
                    .saturating_add(self.start.elapsed().as_nanos()),
            );
        }
    }

    pub(super) struct Span {
        phase: Phase,
        start: Option<Instant>,
        completed_bytes: Option<usize>,
    }
    impl Span {
        #[inline]
        pub(super) fn new(phase: Phase) -> Self {
            if !INSTRUMENTED {
                return Self {
                    phase,
                    start: None,
                    completed_bytes: None,
                };
            }

            Self {
                phase,
                start: enabled().then(Instant::now),
                completed_bytes: None,
            }
        }
        #[inline]
        pub(super) fn complete(&mut self, bytes: usize) {
            if INSTRUMENTED {
                self.completed_bytes = Some(bytes);
            }
        }
    }
    impl Drop for Span {
        #[inline]
        fn drop(&mut self) {
            let Some(start) = self.start else {
                return;
            };
            let ns = start.elapsed().as_nanos();
            COUNTERS.with(|counters| {
                let mut counters = counters.borrow_mut();
                let c = &mut counters.0[self.phase as usize];
                c.calls = c.calls.saturating_add(1);
                c.wall_ns = Some(c.wall_ns.unwrap_or_default().saturating_add(ns));
                if let Some(bytes) = self.completed_bytes {
                    c.completed = c.completed.saturating_add(1);
                    c.completed_bytes = c.completed_bytes.saturating_add(bytes as u64);
                }
            });
        }
    }

    #[cfg(test)]
    mod tests {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/unit/parallel/ipc_timing.rs"
        ));
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope<P = Value> {
    version: u16,
    message: Message<P>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Message<P = Value> {
    Boot {
        identity: ProcessIdentity,
        limits: ProcessFrameLimits,
        config: P,
    },
    Ready {
        identity: ProcessIdentity,
        limits: ProcessFrameLimits,
    },
    Execute {
        identity: ProcessCommandIdentity,
        session: u64,
        payload: P,
    },
    Complete {
        identity: ProcessCommandIdentity,
        session: u64,
        payload: P,
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

pub(super) fn serialize<T: Serialize + ?Sized>(
    value: &T,
    limit: usize,
) -> Result<Vec<u8>, ProcessError> {
    use crate::parallel::codec::BoundedJsonError;

    let mut measured = timing::Span::new(timing::Phase::Encode);
    let bytes = timing::encode(value, limit).map_err(|error| match error {
        BoundedJsonError::Limit => ProcessError::FrameTooLarge { limit },
        BoundedJsonError::Serialize(source) => ProcessError::Json {
            operation: "serialize process value",
            source,
        },
    })?;
    measured.complete(bytes.len());
    Ok(bytes)
}

#[cfg(test)]
pub(super) fn payload<T: Serialize + ?Sized>(
    value: &T,
    limit: usize,
) -> Result<Value, ProcessError> {
    let bytes = serialize(value, limit)?;
    let mut measured = timing::Span::new(timing::Phase::PayloadDecode);
    let result = serde_json::from_slice(&bytes).map_err(|source| ProcessError::Json {
        operation: "decode bounded process value",
        source,
    });
    if result.is_ok() {
        measured.complete(bytes.len());
    }
    result
}

/// Serialize a typed payload once, bounded before serializer chunk growth.
/// The borrow is used only while assembling the envelope; no Value tree is built.
pub(super) fn raw_payload(bytes: &[u8]) -> Result<&RawValue, ProcessError> {
    let mut measured = timing::Span::new(timing::Phase::PayloadDecode);
    let normalize = scan_json(bytes)?;
    let raw: &RawValue = serde_json::from_slice(bytes).map_err(|source| ProcessError::Json {
        operation: "decode bounded process value",
        source,
    })?;
    if normalize {
        #[cfg(test)]
        codec_tests::record_value_fallback();
        // RawValue checks syntax but, unlike Value, accepts out-of-range JSON
        // numbers. Keep producer errors on the OutputEncode negative path.
        let _: Value = serde_json::from_slice(bytes).map_err(|source| ProcessError::Json {
            operation: "decode bounded process value",
            source,
        })?;
    }
    measured.complete(bytes.len());
    Ok(raw)
}

pub(super) fn check_payload(raw: &RawValue, limit: usize) -> Result<(), ProcessError> {
    if raw.get().len() > limit {
        return Err(ProcessError::FrameTooLarge { limit });
    }
    // Value used to normalize numeric spellings before charging the limit.
    // Integer NumericToken frames need no normalization. Retain the old check
    // for floats/large integers (e.g. 1e2 expands to 100.0), so raw length alone
    // cannot admit something the old canonical-byte limit refused.
    let normalize = scan_json(raw.get().as_bytes())?;
    if normalize {
        #[cfg(test)]
        codec_tests::record_value_fallback();
        let value: Value =
            serde_json::from_str(raw.get()).map_err(|source| ProcessError::Json {
                operation: "decode bounded process value",
                source,
            })?;
        serialize(&value, limit)?;
    }
    #[cfg(test)]
    test_clock::after_payload_check();
    Ok(())
}

pub(super) fn encode(message: Message, limit: usize) -> Result<Vec<u8>, ProcessError> {
    encode_message(message, limit)
}

pub(super) fn encode_message<P: Serialize>(
    message: Message<P>,
    limit: usize,
) -> Result<Vec<u8>, ProcessError> {
    serialize(
        &Envelope {
            version: PROCESS_PROTOCOL_VERSION,
            message,
        },
        limit,
    )
}

/// Own the bounded wire buffer, not a second allocation/tree for its payload.
#[derive(Debug)]
pub(super) struct Frame {
    body: Vec<u8>,
    deadline: Instant,
}
impl Frame {
    pub(super) fn message(&self) -> Result<Message<&RawValue>, ProcessError> {
        self.check_deadline()?;
        let message = decode_frame(&self.body)?;
        self.check_deadline()?;
        Ok(message)
    }

    /// Recheck the original receive deadline after payload validation/decoding,
    /// before admitting work. Parsing a frame must not start a fresh budget.
    pub(super) fn check_deadline(&self) -> Result<(), ProcessError> {
        check_deadline(self.deadline)
    }
}

fn scan_json(bytes: &[u8]) -> Result<bool, ProcessError> {
    // RawValue skips without the usual depth/numeric-range checks. This scan
    // enforces the original depth ceiling and identifies numeric spellings
    // needing Value's normalization. Strings and their escapes are opaque.
    let mut depth = 0usize;
    let mut string = false;
    let mut escaped = false;
    let mut number_start = None;
    let mut normalize = false;
    for (index, &byte) in bytes.iter().enumerate() {
        if string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                string = false;
            }
            continue;
        }
        if byte == b'-' || byte.is_ascii_digit() {
            number_start.get_or_insert(index);
            continue;
        }
        if let Some(start) = number_start.take() {
            normalize |= integer_needs_normalization(&bytes[start..index]);
        }
        match byte {
            b'"' => string = true,
            b'.' | b'e' | b'E' => normalize = true,
            b'{' | b'[' => {
                depth += 1;
                if depth >= 128 {
                    return Err(ProcessError::Json {
                        operation: "decode process frame",
                        source: <serde_json::Error as serde::de::Error>::custom(
                            "recursion limit exceeded",
                        ),
                    });
                }
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    if let Some(start) = number_start {
        normalize |= integer_needs_normalization(&bytes[start..]);
    }
    Ok(normalize)
}

fn integer_needs_normalization(number: &[u8]) -> bool {
    // JSON syntax is checked by serde. Compare magnitudes as decimal bytes:
    // positive integers cover u64, negative ones cover i64, with -0 normalized
    // separately. This must not build a Value tree just for 19/20-digit IDs.
    let (digits, maximum) = if let Some(digits) = number.strip_prefix(b"-") {
        if digits == b"0" {
            return true;
        }
        (digits, b"9223372036854775808".as_slice())
    } else {
        (number, b"18446744073709551615".as_slice())
    };
    digits.len() > maximum.len() || (digits.len() == maximum.len() && digits > maximum)
}

fn decode_frame(body: &[u8]) -> Result<Message<&RawValue>, ProcessError> {
    let mut measured = timing::Span::new(timing::Phase::FrameDecode);
    scan_json(body)?;
    // A strict struct reads the tag and borrowed fields in one pass, unlike
    // internally tagged enums (Content buffering) or a separate tag probe.
    // Some(raw null) must remain distinct from an absent field so duplicates
    // cannot hide behind null and variant field-presence checks stay strict.
    fn present<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<&'de RawValue>, D::Error> {
        <&RawValue>::deserialize(d).map(Some)
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum Kind {
        Boot,
        Ready,
        Execute,
        Complete,
        Failure,
        Shutdown,
        ShutdownAck,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Fields<'a> {
        #[serde(rename = "type")]
        kind: Kind,
        #[serde(default, borrow, deserialize_with = "present")]
        identity: Option<&'a RawValue>,
        #[serde(default, borrow, deserialize_with = "present")]
        limits: Option<&'a RawValue>,
        #[serde(default, borrow, deserialize_with = "present")]
        config: Option<&'a RawValue>,
        #[serde(default, borrow, deserialize_with = "present")]
        session: Option<&'a RawValue>,
        #[serde(default, borrow, deserialize_with = "present")]
        payload: Option<&'a RawValue>,
        #[serde(default, borrow, deserialize_with = "present")]
        owner: Option<&'a RawValue>,
        #[serde(default, borrow, deserialize_with = "present")]
        command: Option<&'a RawValue>,
        #[serde(default, borrow, deserialize_with = "present")]
        failure: Option<&'a RawValue>,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawEnvelope<'a> {
        version: u16,
        #[serde(borrow, deserialize_with = "fields")]
        message: Fields<'a>,
    }
    fn fields<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Fields<'de>, D::Error> {
        struct Object;
        impl<'de> serde::de::Visitor<'de> for Object {
            type Value = Fields<'de>;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a process message object")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                map: M,
            ) -> Result<Self::Value, M::Error> {
                Fields::deserialize(serde::de::value::MapAccessDeserializer::new(map))
            }
        }
        d.deserialize_map(Object)
    }
    fn json(source: serde_json::Error) -> ProcessError {
        ProcessError::Json {
            operation: "decode process frame",
            source,
        }
    }
    fn required(raw: Option<&RawValue>) -> Result<&RawValue, ProcessError> {
        raw.ok_or_else(|| protocol("missing process message field"))
    }
    fn field<T: serde::de::DeserializeOwned>(raw: Option<&RawValue>) -> Result<T, ProcessError> {
        serde_json::from_str(required(raw)?.get()).map_err(json)
    }
    fn optional<T: serde::de::DeserializeOwned>(
        raw: Option<&RawValue>,
    ) -> Result<Option<T>, ProcessError> {
        raw.map(|v| serde_json::from_str(v.get()).map_err(json))
            .transpose()
            .map(Option::flatten)
    }
    let envelope: RawEnvelope<'_> = serde_json::from_slice(body).map_err(json)?;
    if envelope.version != PROCESS_PROTOCOL_VERSION {
        return Err(protocol("unsupported protocol version"));
    }
    let f = envelope.message;
    let actual = [
        f.identity, f.limits, f.config, f.session, f.payload, f.owner, f.command, f.failure,
    ]
    .map(|v| v.is_some());
    let (allowed, mandatory) = match f.kind {
        Kind::Boot => (0b00000111u8, 0b00000111),
        Kind::Ready => (0b00000011, 0b00000011),
        Kind::Execute | Kind::Complete => (0b00011001, 0b00011001),
        Kind::Failure => (0b11101000, 0b10100000),
        Kind::Shutdown | Kind::ShutdownAck => (0b00100000, 0b00100000),
    };
    let present = actual
        .into_iter()
        .enumerate()
        .fold(0u8, |mask, (i, present)| mask | (u8::from(present) << i));
    if present & !allowed != 0 || present & mandatory != mandatory {
        return Err(protocol("invalid process message fields"));
    }
    let message = match f.kind {
        Kind::Boot => Message::Boot {
            identity: field(f.identity)?,
            limits: field(f.limits)?,
            config: required(f.config)?,
        },
        Kind::Ready => Message::Ready {
            identity: field(f.identity)?,
            limits: field(f.limits)?,
        },
        Kind::Execute => Message::Execute {
            identity: field(f.identity)?,
            session: field(f.session)?,
            payload: required(f.payload)?,
        },
        Kind::Complete => Message::Complete {
            identity: field(f.identity)?,
            session: field(f.session)?,
            payload: required(f.payload)?,
        },
        Kind::Failure => Message::Failure {
            owner: field(f.owner)?,
            command: optional(f.command)?,
            session: optional(f.session)?,
            failure: field(f.failure)?,
        },
        Kind::Shutdown => Message::Shutdown {
            owner: field(f.owner)?,
        },
        Kind::ShutdownAck => Message::ShutdownAck {
            owner: field(f.owner)?,
        },
    };
    measured.complete(body.len());
    Ok(message)
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
    let mut measured = timing::Span::new(timing::Phase::FrameWrite);
    let count = u32::try_from(body.len()).map_err(|_| ProcessError::FrameTooLarge {
        limit: HARD_MAX_FRAME_BYTES,
    })?;
    write_all(writer, &count.to_be_bytes(), deadline, observe, sent)?;
    write_all(writer, body, deadline, observe, sent)?;
    measured.complete(body.len() + 4);
    Ok(())
}

/// Only the wait for a new frame is unbounded. Once the pipe becomes readable,
/// even a partial prefix/body must finish under one non-sliding I/O deadline.
/// EOF/HUP wakes the idle wait too; no heartbeat or synthetic work is sent.
pub(super) fn receive_idle<R: Read + AsRawFd>(
    reader: &mut R,
    limit: usize,
    timeout: Duration,
) -> Result<Frame, ProcessError> {
    poll_fd(reader.as_raw_fd(), libc::POLLIN, None, &mut || {})?;
    receive(reader, limit, deadline_after(timeout)?)
}

pub(super) fn receive<R: Read + AsRawFd>(
    reader: &mut R,
    limit: usize,
    deadline: Instant,
) -> Result<Frame, ProcessError> {
    receive_observed(reader, limit, deadline, &mut || {})
}

pub(super) fn receive_observed<R: Read + AsRawFd>(
    reader: &mut R,
    limit: usize,
    deadline: Instant,
    observe: &mut dyn FnMut(),
) -> Result<Frame, ProcessError> {
    let mut measured = timing::Span::new(timing::Phase::FrameRead);
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
    measured.complete(body.len() + 4);
    drop(measured);
    check_deadline(deadline)?;
    Ok(Frame { body, deadline })
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
    #[cfg(test)]
    let now = test_clock::now();
    #[cfg(not(test))]
    let now = Instant::now();
    if now >= deadline {
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
    let mut measured = deadline.map(|_| timing::Span::new(timing::Phase::PipeWait));
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
            if let Some(measured) = &mut measured {
                measured.complete(0);
            }
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
#[path = "../../../tests/support/ipc_frame_contracts.rs"]
mod pr26_contracts;

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

#[cfg(test)]
#[path = "../../../tests/unit/parallel/process_codec.rs"]
mod codec_tests;

#[cfg(test)]
#[path = "../../../tests/bench/numeric_process_cpu.rs"]
mod numeric_cpu_bench;

// Private, thread-local clock seam for deterministic decode/admission tests.
// No production clock, timer, synchronization or cancellation behavior changes.
#[cfg(test)]
pub(super) mod test_clock {
    use std::cell::Cell;
    use std::time::{Duration, Instant};
    thread_local! {
        static NOW: Cell<Option<Instant>> = const { Cell::new(None) };
        static PAYLOAD_CHECKS_UNTIL_EXPIRED: Cell<usize> = const { Cell::new(0) };
    }
    pub(crate) fn expire_after_payload_checks(count: usize) {
        PAYLOAD_CHECKS_UNTIL_EXPIRED.with(|remaining| remaining.set(count));
    }
    pub(super) fn after_payload_check() {
        PAYLOAD_CHECKS_UNTIL_EXPIRED.with(|remaining| {
            let count = remaining.get();
            if count != 0 {
                remaining.set(count - 1);
                if count == 1 {
                    expire();
                }
            }
        });
    }
    pub(super) fn now() -> Instant {
        NOW.with(|now| now.get().unwrap_or_else(Instant::now))
    }
    pub(crate) fn expire() {
        NOW.with(|now| now.set(Some(Instant::now() + Duration::from_secs(60))));
    }
    pub(crate) struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            NOW.with(|now| now.set(None));
            PAYLOAD_CHECKS_UNTIL_EXPIRED.with(|remaining| remaining.set(0));
        }
    }
}
