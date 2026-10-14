use std::io::{self, Write};

use serde::Serialize;

#[derive(Debug)]
pub(crate) enum BoundedJsonError {
    Limit,
    Serialize(serde_json::Error),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CopyCounts {
    pub calls: u64,
    pub bytes: usize,
}

struct BoundedJsonWriter<F, const COUNT: bool> {
    bytes: Vec<u8>,
    limit: usize,
    exceeded: bool,
    accepted_writes: u64,
    observe_copy: F,
}

impl<F, const COUNT: bool> BoundedJsonWriter<F, COUNT> {
    fn new(limit: usize, observe_copy: F) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
            exceeded: false,
            accepted_writes: 0,
            observe_copy,
        }
    }
}

impl<F: FnMut(usize) -> G, G, const COUNT: bool> Write for BoundedJsonWriter<F, COUNT> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            self.exceeded = true;
            return Err(io::Error::other("bounded JSON limit"));
        }
        let _copy = (self.observe_copy)(bytes.len());
        // Vec's normal growth may exceed the wire limit. Request only the
        // exact bounded target before accepting each serializer chunk.
        if self.bytes.capacity() - self.bytes.len() < bytes.len() {
            let required = self.bytes.len() + bytes.len();
            let target = required
                .max(self.bytes.capacity().saturating_mul(2))
                .min(self.limit);
            self.bytes.reserve_exact(target - self.bytes.len());
        }
        self.bytes.extend_from_slice(bytes);
        if COUNT {
            self.accepted_writes = self.accepted_writes.saturating_add(1);
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Refuses an oversized serializer chunk before growth. Arbitrary Serialize
/// implementations must still be cooperative, not a preemptible sandbox.
pub(crate) fn bounded_json<T: Serialize + ?Sized>(
    value: &T,
    limit: usize,
) -> Result<Vec<u8>, BoundedJsonError> {
    encode::<false, _, _>(value, limit, |_| ()).0
}

/// Local accepted-write counts, published by the caller once per frame even on
/// a serialization/limit error. The optional guard spans accepted growth/copy;
/// rejected chunks are neither counted nor observed.
pub(crate) fn bounded_json_counted<T: Serialize + ?Sized, G>(
    value: &T,
    limit: usize,
    observe_copy: impl FnMut(usize) -> G,
) -> (Result<Vec<u8>, BoundedJsonError>, CopyCounts) {
    encode::<true, _, _>(value, limit, observe_copy)
}

fn encode<const COUNT: bool, T: Serialize + ?Sized, G>(
    value: &T,
    limit: usize,
    observe_copy: impl FnMut(usize) -> G,
) -> (Result<Vec<u8>, BoundedJsonError>, CopyCounts) {
    let mut writer = BoundedJsonWriter::<_, COUNT>::new(limit, observe_copy);
    let result = serde_json::to_writer(&mut writer, value);
    let counts = CopyCounts {
        calls: writer.accepted_writes,
        bytes: writer.bytes.len(),
    };
    let result = if writer.exceeded {
        Err(BoundedJsonError::Limit)
    } else {
        result
            .map(|()| writer.bytes)
            .map_err(BoundedJsonError::Serialize)
    };
    (result, counts)
}

#[cfg(test)]
#[path = "../../tests/unit/parallel/codec.rs"]
mod tests;
