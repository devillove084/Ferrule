//! Bounded CPU consumer driven by the transaction-owned CUDA transport.
//! Cancellation is not a payload; exact ticket identity precedes consumption.
use super::plan::error;
use crate::cpu::{
    StreamingF32GqaConfig, StreamingF32GqaPartial, StreamingF32GqaRequest,
    streaming_f32_gqa_partial,
};
use ferrule_common::Result;
use ferrule_common::host_kv::HostKvAppendReadWitness;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PrefixDownload {
    Query,
    Key { start: usize, tokens: usize },
    Value { start: usize, tokens: usize },
    TailKey,
    TailValue,
}

/// Payload custody only: one outstanding download, one K/V chunk and partial.
pub(super) struct AppendStream<I> {
    config: StreamingF32GqaConfig,
    page_tokens: usize,
    end: usize,
    next: Option<PrefixDownload>,
    pending: Option<I>,
    query: Option<Box<[f32]>>,
    key: Option<Box<[f32]>>,
    total: StreamingF32GqaPartial,
    unknown: bool,
    consumed: bool,
    include_tail: bool,
}
impl<I: PartialEq> AppendStream<I> {
    pub fn validate_config(
        witness: &HostKvAppendReadWitness,
        config: StreamingF32GqaConfig,
    ) -> Result<()> {
        config.validate()?;
        if config.rows != 1
            || witness
                .source()
                .schema
                .planes()
                .iter()
                .any(|p| Some(p.elements_per_token) != config.kv_elements(1).ok())
        {
            return Err(error("append DMA query/schema geometry mismatch"));
        }
        Ok(())
    }
    pub fn new(witness: &HostKvAppendReadWitness, config: StreamingF32GqaConfig) -> Result<Self> {
        Self::validate_config(witness, config)?;
        let end = witness.source().committed_tokens;
        Ok(Self {
            config,
            page_tokens: witness.source().schema.page_size(),
            end,
            next: Some(PrefixDownload::Query),
            pending: None,
            query: None,
            key: None,
            total: StreamingF32GqaPartial::empty(config, end as u64)?,
            unknown: false,
            consumed: false,
            include_tail: false,
        })
    }
    pub fn include_tail(&mut self) -> Result<()> {
        if self.next != Some(PrefixDownload::Query) || self.pending.is_some() {
            return Err(error("tail must be declared before Q DMA"));
        }
        self.include_tail = true;
        Ok(())
    }
    fn healthy(&self) -> Result<()> {
        if self.unknown {
            return Err(error("append DMA/consumer completion unknown"));
        }
        Ok(())
    }
    pub fn request(&self) -> Result<Option<PrefixDownload>> {
        self.healthy()?;
        if self.pending.is_some() {
            return Err(error("append DMA ticket still pending"));
        }
        Ok(self.next)
    }
    pub fn submitted(&mut self, id: I) -> Result<()> {
        self.healthy()?;
        if self.pending.is_some() || self.next.is_none() || self.consumed {
            return Err(error("append DMA submission out of order"));
        }
        self.pending = Some(id);
        Ok(())
    }
    fn next_key(&mut self, start: usize) {
        if start == self.end && self.include_tail {
            self.next = Some(PrefixDownload::TailKey);
            return;
        }
        self.next = (start < self.end).then(|| PrefixDownload::Key {
            start,
            tokens: self
                .config
                .block_tokens
                .min(self.end - start)
                .min(self.page_tokens - start % self.page_tokens),
        });
    }
    pub fn complete(&mut self, id: I, values: Box<[f32]>) -> Result<()> {
        self.healthy()?;
        // Failed checks or an unwinding CPU consumer are sticky Unknown.
        self.unknown = true;
        if self.pending.as_ref() != Some(&id) {
            return Err(error("append DMA completion identity mismatch"));
        }
        match self
            .next
            .ok_or_else(|| error("append DMA already complete"))?
        {
            PrefixDownload::Query => {
                if values.len() != self.config.query_elements()? {
                    return Err(error("append query D2H length mismatch"));
                }
                self.query = Some(values);
                self.next_key(0);
            }
            PrefixDownload::TailKey => {
                if values.len() != self.config.kv_elements(1)? {
                    return Err(error("append shadow key length mismatch"));
                }
                self.key = Some(values);
                self.next = Some(PrefixDownload::TailValue);
            }
            PrefixDownload::TailValue => {
                if values.len() != self.config.kv_elements(1)? {
                    return Err(error("append shadow value length mismatch"));
                }
                let tail = streaming_f32_gqa_partial(StreamingF32GqaRequest {
                    config: self.config,
                    query: self
                        .query
                        .as_deref()
                        .ok_or_else(|| error("append Q absent"))?,
                    key: self
                        .key
                        .as_deref()
                        .ok_or_else(|| error("append shadow K absent"))?,
                    value: &values,
                    query_start: self.end as u64,
                    kv_start: self.end as u64,
                })?;
                // Both operands derive from this one downloaded Q. Prefix is
                // exactly 0..end; the only appended row is end..end+1.
                self.total.merge(&tail)?;
                self.key = None;
                self.next = None;
            }
            PrefixDownload::Key { start, tokens } => {
                if values.len() != self.config.kv_elements(tokens)? {
                    return Err(error("append key D2H length mismatch"));
                }
                self.key = Some(values);
                self.next = Some(PrefixDownload::Value { start, tokens });
            }
            PrefixDownload::Value { start, tokens } => {
                if values.len() != self.config.kv_elements(tokens)? {
                    return Err(error("append value D2H length mismatch"));
                }
                let partial = streaming_f32_gqa_partial(StreamingF32GqaRequest {
                    config: self.config,
                    query: self
                        .query
                        .as_deref()
                        .ok_or_else(|| error("append query absent"))?,
                    key: self
                        .key
                        .as_deref()
                        .ok_or_else(|| error("append key absent"))?,
                    value: &values,
                    query_start: self.end as u64,
                    kv_start: start as u64,
                })?;
                self.total.merge(&partial)?;
                self.key = None;
                self.next_key(start + tokens);
            }
        }
        self.pending = None;
        self.unknown = false;
        Ok(())
    }
    pub fn ready(&self) -> Result<bool> {
        self.healthy()?;
        Ok(self.next.is_none() && self.pending.is_none())
    }
    /// The trusted adapter must consume synchronously into separately charged
    /// tail/output storage. No asynchronous borrow or output buffer is exported.
    pub fn consume(
        &mut self,
        consumer: impl FnOnce(&StreamingF32GqaPartial) -> Result<()>,
    ) -> Result<()> {
        if !self.ready()? || self.consumed {
            return Err(error("append partial is not ready or was already consumed"));
        }
        self.unknown = true;
        consumer(&self.total)?;
        self.consumed = true;
        self.unknown = false;
        Ok(())
    }
    pub fn consumed(&self) -> Result<bool> {
        self.healthy()?;
        Ok(self.consumed)
    }
    pub fn is_unknown(&self) -> bool {
        self.unknown
    }
}
