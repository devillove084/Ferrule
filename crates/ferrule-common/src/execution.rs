//! Dependency-neutral model execution vocabulary and contracts.
//!
//! This module deliberately contains no runtime request identifiers, model-family
//! types, graph IR, or backend resources. Runtime and model crates can therefore
//! share the same packed/ragged execution boundary without depending on each
//! other.

use std::collections::HashSet;
use std::num::{NonZeroU32, TryFromIntError};
use std::ops::Range;

use crate::{Error, Result};

macro_rules! define_u32_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(u32);

        impl $name {
            /// Creates an identifier from its wire representation.
            pub const fn new(value: u32) -> Self {
                Self(value)
            }

            /// Returns the identifier's wire representation.
            pub const fn get(self) -> u32 {
                self.0
            }

            /// Converts this identifier to a host index without assuming that
            /// `usize` is at least 32 bits wide.
            pub fn try_as_usize(self) -> std::result::Result<usize, TryFromIntError> {
                usize::try_from(self.0)
            }
        }

        impl From<u32> for $name {
            fn from(value: u32) -> Self {
                Self::new(value)
            }
        }

        impl From<$name> for u32 {
            fn from(value: $name) -> Self {
                value.get()
            }
        }

        impl TryFrom<usize> for $name {
            type Error = TryFromIntError;

            fn try_from(value: usize) -> std::result::Result<Self, Self::Error> {
                u32::try_from(value).map(Self::new)
            }
        }

        impl TryFrom<$name> for usize {
            type Error = TryFromIntError;

            fn try_from(value: $name) -> std::result::Result<Self, Self::Error> {
                value.try_as_usize()
            }
        }
    };
}

define_u32_id! {
    /// Index of mutable sequence state within the state slice supplied to one
    /// execution call.
    StateSlot
}

define_u32_id! {
    /// Backend-defined physical destination for one token's KV write.
    KvWriteSlot
}

define_u32_id! {
    /// Backend-defined physical KV block identifier.
    KvBlockId
}

/// Forward phase for one sequence in a packed batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ForwardPhase {
    Prefill,
    Decode,
}

/// Aggregate phase represented by a packed batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ForwardMode {
    Prefill,
    Decode,
    Mixed,
}

/// State-publication semantics for one execution call.
///
/// Ordinary prefill/decode publishes all successful rows. Speculative target
/// verification executes against a branch and publishes only an accepted prefix
/// through an explicit transaction; it must never be inferred from row count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionIntent {
    Committed,
    ProvisionalVerification,
}

/// Request and row dimensions used for executable-plan/resource selection.
/// `sequence_count` and `total_rows` are intentionally separate: decode with
/// four requests is not the same workload as one request verifying four rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionShape {
    pub intent: ExecutionIntent,
    pub sequence_count: usize,
    pub total_rows: usize,
    pub max_query_tokens: usize,
    pub decode_rows: usize,
}

/// Logits requested for one packed input row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogitsRequest {
    None,
    TopK(NonZeroU32),
    Full,
}

/// One sequence's view into the packed query and flattened KV block table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionSequence {
    pub state_slot: StateSlot,
    pub phase: ForwardPhase,
    pub query: Range<u32>,
    pub context_len: u32,
    pub sequence_len: u32,
    pub block_table: Range<u32>,
}

impl ExecutionSequence {
    pub fn new(
        state_slot: StateSlot,
        phase: ForwardPhase,
        query: Range<u32>,
        context_len: u32,
        sequence_len: u32,
        block_table: Range<u32>,
    ) -> Self {
        Self {
            state_slot,
            phase,
            query,
            context_len,
            sequence_len,
            block_table,
        }
    }
}

/// Owned packed/ragged input for one model execution call.
///
/// Construction intentionally does not validate cross-field invariants because
/// validation requires both a state-slice length and a prepared plan's
/// capabilities. Call [`ExecutionBatch::validate`] immediately before lowering or
/// execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionBatch {
    intent: ExecutionIntent,
    mode: ForwardMode,
    token_ids: Vec<u32>,
    positions: Vec<u32>,
    kv_write_slots: Vec<Option<KvWriteSlot>>,
    logits: Vec<LogitsRequest>,
    sequences: Vec<ExecutionSequence>,
    kv_block_ids: Vec<KvBlockId>,
}

impl ExecutionBatch {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mode: ForwardMode,
        token_ids: Vec<u32>,
        positions: Vec<u32>,
        kv_write_slots: Vec<Option<KvWriteSlot>>,
        logits: Vec<LogitsRequest>,
        sequences: Vec<ExecutionSequence>,
        kv_block_ids: Vec<KvBlockId>,
    ) -> Self {
        Self {
            intent: ExecutionIntent::Committed,
            mode,
            token_ids,
            positions,
            kv_write_slots,
            logits,
            sequences,
            kv_block_ids,
        }
    }

    /// Changes publication semantics without changing the packed payload.
    /// Callers constructing speculative verification work must opt in explicitly.
    pub fn with_intent(mut self, intent: ExecutionIntent) -> Self {
        self.intent = intent;
        self
    }

    pub const fn intent(&self) -> ExecutionIntent {
        self.intent
    }

    pub const fn mode(&self) -> ForwardMode {
        self.mode
    }

    pub fn shape(&self) -> Result<ExecutionShape> {
        let mut max_query_tokens = 0usize;
        let mut decode_rows = 0usize;
        for sequence in &self.sequences {
            let query_tokens = sequence
                .query
                .end
                .checked_sub(sequence.query.start)
                .ok_or_else(|| {
                    execution_error("cannot derive shape from a reversed query range")
                })?;
            let query_tokens = usize::try_from(query_tokens)
                .map_err(|_| execution_error("query length cannot be represented as usize"))?;
            max_query_tokens = max_query_tokens.max(query_tokens);
            if sequence.phase == ForwardPhase::Decode {
                decode_rows = decode_rows.checked_add(query_tokens).ok_or_else(|| {
                    execution_error("decode row count overflow while deriving execution shape")
                })?;
            }
        }
        Ok(ExecutionShape {
            intent: self.intent,
            sequence_count: self.sequences.len(),
            total_rows: self.len(),
            max_query_tokens,
            decode_rows,
        })
    }

    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn positions(&self) -> &[u32] {
        &self.positions
    }

    pub fn kv_write_slots(&self) -> &[Option<KvWriteSlot>] {
        &self.kv_write_slots
    }

    pub fn logits(&self) -> &[LogitsRequest] {
        &self.logits
    }

    pub fn sequences(&self) -> &[ExecutionSequence] {
        &self.sequences
    }

    pub fn kv_block_ids(&self) -> &[KvBlockId] {
        &self.kv_block_ids
    }

    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }

    /// Validates all structural and capability-dependent execution invariants.
    pub fn validate(&self, state_count: usize, capabilities: &ExecutionCapabilities) -> Result<()> {
        self.validate_packed_lengths()?;
        self.validate_batch_capabilities(capabilities)?;

        let row_count_u32 = u32::try_from(self.len()).map_err(|_| {
            execution_error(format!(
                "packed row count {} cannot be represented by query ranges",
                self.len()
            ))
        })?;
        let block_count_u32 = u32::try_from(self.kv_block_ids.len()).map_err(|_| {
            execution_error(format!(
                "KV block count {} cannot be represented by block-table ranges",
                self.kv_block_ids.len()
            ))
        })?;

        let mut expected_query_start = 0_u32;
        let mut state_slots = HashSet::with_capacity(self.sequences.len());
        let mut saw_prefill = false;
        let mut saw_decode = false;

        for (sequence_index, sequence) in self.sequences.iter().enumerate() {
            if sequence.query.start > sequence.query.end {
                return Err(execution_error(format!(
                    "sequence {sequence_index} query range {}..{} is reversed",
                    sequence.query.start, sequence.query.end
                )));
            }
            if sequence.query.start == sequence.query.end {
                return Err(execution_error(format!(
                    "sequence {sequence_index} has an empty query range"
                )));
            }
            if sequence.query.start != expected_query_start {
                return Err(execution_error(format!(
                    "sequence {sequence_index} query starts at {}, expected {expected_query_start}",
                    sequence.query.start
                )));
            }
            if sequence.query.end > row_count_u32 {
                return Err(execution_error(format!(
                    "sequence {sequence_index} query ends at {}, beyond packed row count {row_count_u32}",
                    sequence.query.end
                )));
            }
            expected_query_start = sequence.query.end;

            let state_index = sequence.state_slot.try_as_usize().map_err(|_| {
                execution_error(format!(
                    "sequence {sequence_index} state slot {} cannot be represented as usize",
                    sequence.state_slot.get()
                ))
            })?;
            if state_index >= state_count {
                return Err(execution_error(format!(
                    "sequence {sequence_index} state slot {} is out of range for {state_count} states",
                    sequence.state_slot.get()
                )));
            }
            if !state_slots.insert(sequence.state_slot) {
                return Err(execution_error(format!(
                    "state slot {} is referenced by more than one sequence",
                    sequence.state_slot.get()
                )));
            }

            self.validate_sequence_phase(sequence_index, sequence, capabilities)?;
            match sequence.phase {
                ForwardPhase::Prefill => saw_prefill = true,
                ForwardPhase::Decode => saw_decode = true,
            }

            let query_len = sequence.query.end - sequence.query.start;
            let query_len_usize = usize::try_from(query_len).map_err(|_| {
                execution_error(format!(
                    "sequence {sequence_index} query length {query_len} cannot be represented as usize"
                ))
            })?;
            let query_limit = match sequence.phase {
                ForwardPhase::Prefill => capabilities.max_prefill_query_tokens_per_sequence,
                ForwardPhase::Decode => capabilities.max_decode_query_tokens_per_sequence,
            };
            if query_len_usize > query_limit {
                return Err(execution_error(format!(
                    "sequence {sequence_index} {:?} query length {query_len_usize} exceeds capability limit {query_limit}",
                    sequence.phase
                )));
            }

            let expected_sequence_len = sequence.context_len.checked_add(query_len).ok_or_else(|| {
                execution_error(format!(
                    "sequence {sequence_index} context length {} plus query length {query_len} overflows u32",
                    sequence.context_len
                ))
            })?;
            if sequence.sequence_len != expected_sequence_len {
                return Err(execution_error(format!(
                    "sequence {sequence_index} sequence length {} does not equal context {} plus query {query_len}",
                    sequence.sequence_len, sequence.context_len
                )));
            }

            let query_start = usize::try_from(sequence.query.start).map_err(|_| {
                execution_error(format!(
                    "sequence {sequence_index} query start {} cannot be represented as usize",
                    sequence.query.start
                ))
            })?;
            let query_end = usize::try_from(sequence.query.end).map_err(|_| {
                execution_error(format!(
                    "sequence {sequence_index} query end {} cannot be represented as usize",
                    sequence.query.end
                ))
            })?;
            for (offset, row) in (query_start..query_end).enumerate() {
                let offset = u32::try_from(offset).map_err(|_| {
                    execution_error(format!(
                        "sequence {sequence_index} position offset cannot be represented as u32"
                    ))
                })?;
                let expected_position =
                    sequence.context_len.checked_add(offset).ok_or_else(|| {
                        execution_error(format!(
                            "sequence {sequence_index} expected position overflows u32"
                        ))
                    })?;
                if self.positions[row] != expected_position {
                    return Err(execution_error(format!(
                        "packed row {row} position {} is not the expected contiguous position {expected_position}",
                        self.positions[row]
                    )));
                }
                self.validate_logits_request(row, row + 1 == query_end, capabilities)?;
            }

            if sequence.block_table.start > sequence.block_table.end {
                return Err(execution_error(format!(
                    "sequence {sequence_index} block-table range {}..{} is reversed",
                    sequence.block_table.start, sequence.block_table.end
                )));
            }
            if sequence.block_table.end > block_count_u32 {
                return Err(execution_error(format!(
                    "sequence {sequence_index} block table ends at {}, beyond flattened block count {block_count_u32}",
                    sequence.block_table.end
                )));
            }

            match capabilities.kv_binding_mode {
                KvBindingMode::None => {
                    if sequence.block_table.start != sequence.block_table.end {
                        return Err(execution_error(format!(
                            "sequence {sequence_index} supplies a block table while KV binding mode is None"
                        )));
                    }
                }
                KvBindingMode::Paged => {
                    if sequence.block_table.start == sequence.block_table.end {
                        return Err(execution_error(format!(
                            "sequence {sequence_index} has no block table for paged KV"
                        )));
                    }
                }
            }
        }

        if expected_query_start != row_count_u32 {
            return Err(execution_error(format!(
                "sequence queries cover {expected_query_start} packed rows, expected {row_count_u32}"
            )));
        }

        if self.intent == ExecutionIntent::ProvisionalVerification
            && self.mode != ForwardMode::Prefill
        {
            return Err(execution_error(
                "provisional verification requires prefill-phase sequences",
            ));
        }

        match self.mode {
            ForwardMode::Prefill if !saw_prefill || saw_decode => {
                return Err(execution_error(
                    "Prefill mode must contain only prefill sequences",
                ));
            }
            ForwardMode::Decode if saw_prefill || !saw_decode => {
                return Err(execution_error(
                    "Decode mode must contain only decode sequences",
                ));
            }
            ForwardMode::Mixed if !saw_prefill || !saw_decode => {
                return Err(execution_error(
                    "Mixed mode must contain both prefill and decode sequences",
                ));
            }
            _ => {}
        }

        self.validate_kv_bindings(capabilities)
    }

    fn validate_packed_lengths(&self) -> Result<()> {
        if self.token_ids.is_empty() {
            return Err(execution_error(
                "execution batch must contain at least one packed row",
            ));
        }

        let expected = self.token_ids.len();
        for (name, actual) in [
            ("positions", self.positions.len()),
            ("kv_write_slots", self.kv_write_slots.len()),
            ("logits", self.logits.len()),
        ] {
            if actual != expected {
                return Err(execution_error(format!(
                    "packed vector {name} has length {actual}, expected {expected}"
                )));
            }
        }

        if self.sequences.is_empty() {
            return Err(execution_error(
                "execution batch must contain at least one sequence",
            ));
        }

        Ok(())
    }

    fn validate_batch_capabilities(&self, capabilities: &ExecutionCapabilities) -> Result<()> {
        if self.len() > capabilities.max_batch_tokens {
            return Err(execution_error(format!(
                "batch has {} packed tokens, exceeding capability limit {}",
                self.len(),
                capabilities.max_batch_tokens
            )));
        }
        if self.sequences.len() > capabilities.max_sequences {
            return Err(execution_error(format!(
                "batch has {} sequences, exceeding capability limit {}",
                self.sequences.len(),
                capabilities.max_sequences
            )));
        }

        let supported = match self.mode {
            ForwardMode::Prefill => capabilities.supports_prefill,
            ForwardMode::Decode => capabilities.supports_decode,
            ForwardMode::Mixed => capabilities.supports_mixed,
        };
        if !supported {
            return Err(execution_error(format!(
                "batch mode {:?} is not supported by the prepared plan",
                self.mode
            )));
        }

        Ok(())
    }

    fn validate_sequence_phase(
        &self,
        sequence_index: usize,
        sequence: &ExecutionSequence,
        capabilities: &ExecutionCapabilities,
    ) -> Result<()> {
        let matches_mode = matches!(
            (self.mode, sequence.phase),
            (ForwardMode::Prefill, ForwardPhase::Prefill)
                | (ForwardMode::Decode, ForwardPhase::Decode)
                | (ForwardMode::Mixed, _)
        );
        if !matches_mode {
            return Err(execution_error(format!(
                "sequence {sequence_index} phase {:?} does not match batch mode {:?}",
                sequence.phase, self.mode
            )));
        }

        let supported = match sequence.phase {
            ForwardPhase::Prefill => capabilities.supports_prefill,
            ForwardPhase::Decode => capabilities.supports_decode,
        };
        if !supported {
            return Err(execution_error(format!(
                "sequence {sequence_index} phase {:?} is not supported by the prepared plan",
                sequence.phase
            )));
        }

        Ok(())
    }

    fn validate_logits_request(
        &self,
        row: usize,
        is_last_in_sequence: bool,
        capabilities: &ExecutionCapabilities,
    ) -> Result<()> {
        let request = self.logits[row];
        match request {
            LogitsRequest::None => return Ok(()),
            LogitsRequest::TopK(k) => match capabilities.max_top_k {
                Some(maximum) if k <= maximum => {}
                Some(maximum) => {
                    return Err(execution_error(format!(
                        "packed row {row} requests top-k {}, exceeding capability limit {}",
                        k.get(),
                        maximum.get()
                    )));
                }
                None => {
                    return Err(execution_error(format!(
                        "packed row {row} requests top-k logits, which are unsupported"
                    )));
                }
            },
            LogitsRequest::Full if capabilities.full_logits_width.is_none() => {
                return Err(execution_error(format!(
                    "packed row {row} requests full logits, which are unsupported"
                )));
            }
            LogitsRequest::Full => {}
        }

        match capabilities.logits_row_policy {
            LogitsRowPolicy::None => Err(execution_error(format!(
                "packed row {row} requests logits, but this plan returns no logits rows"
            ))),
            LogitsRowPolicy::LastPerSequence if !is_last_in_sequence => Err(execution_error(
                format!("packed row {row} requests logits before the last row of its sequence"),
            )),
            LogitsRowPolicy::LastPerSequence | LogitsRowPolicy::Any => Ok(()),
        }
    }

    fn validate_kv_bindings(&self, capabilities: &ExecutionCapabilities) -> Result<()> {
        match capabilities.kv_binding_mode {
            KvBindingMode::None => {
                if let Some(row) = self.kv_write_slots.iter().position(Option::is_some) {
                    return Err(execution_error(format!(
                        "packed row {row} supplies a KV write slot while KV binding mode is None"
                    )));
                }
                if !self.kv_block_ids.is_empty() {
                    return Err(execution_error(format!(
                        "batch supplies {} KV blocks while KV binding mode is None",
                        self.kv_block_ids.len()
                    )));
                }
            }
            KvBindingMode::Paged => {
                let mut destinations = HashSet::with_capacity(self.kv_write_slots.len());
                for (row, write_slot) in self.kv_write_slots.iter().enumerate() {
                    let Some(write_slot) = write_slot else {
                        return Err(execution_error(format!(
                            "packed row {row} has no KV write slot for paged KV"
                        )));
                    };
                    if !destinations.insert(*write_slot) {
                        return Err(execution_error(format!(
                            "packed row {row} reuses paged KV write slot {} within the same batch",
                            write_slot.get()
                        )));
                    }
                }
            }
        }

        Ok(())
    }
}

/// Shape and feature limits of one prepared model plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionCapabilities {
    pub max_batch_tokens: usize,
    pub max_sequences: usize,
    pub max_prefill_query_tokens_per_sequence: usize,
    pub max_decode_query_tokens_per_sequence: usize,
    pub max_top_k: Option<NonZeroU32>,
    pub supports_prefill: bool,
    pub supports_decode: bool,
    pub supports_mixed: bool,
    /// Exact vocabulary width of a full-logits row, or `None` when full logits
    /// are unsupported.
    pub full_logits_width: Option<NonZeroU32>,
    pub kv_binding_mode: KvBindingMode,
    pub logits_row_policy: LogitsRowPolicy,
}

/// KV metadata expected on every execution batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvBindingMode {
    /// Compatibility mode with no physical write slots or block tables.
    None,
    /// Every row has a physical write slot and every sequence has a packed block
    /// table.
    Paged,
}

/// Packed rows on which a prepared plan can produce logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogitsRowPolicy {
    None,
    LastPerSequence,
    Any,
}

/// One token/logit pair in a top-k result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenLogit {
    pub token_id: u32,
    pub logit: f32,
}

impl TokenLogit {
    pub const fn new(token_id: u32, logit: f32) -> Self {
        Self { token_id, logit }
    }
}

/// Logits payload corresponding to one requested packed input row.
#[derive(Debug, Clone, PartialEq)]
pub enum LogitsOutput {
    TopK(Vec<TokenLogit>),
    Full(Vec<f32>),
}

/// One output row, correlated only by its packed input-row index.
#[derive(Debug, Clone, PartialEq)]
pub struct LogitsRow {
    pub input_row: u32,
    pub logits: LogitsOutput,
}

impl LogitsRow {
    pub fn new(input_row: u32, logits: LogitsOutput) -> Self {
        Self { input_row, logits }
    }
}

/// Model output for one [`ExecutionBatch`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExecutionOutput {
    pub logits: Vec<LogitsRow>,
}

impl ExecutionOutput {
    pub fn new(logits: Vec<LogitsRow>) -> Self {
        Self { logits }
    }

    pub fn validate(&self, batch: &ExecutionBatch) -> Result<()> {
        let mut previous_input_row = None;
        let mut seen = vec![false; batch.len()];

        for (output_index, row) in self.logits.iter().enumerate() {
            if let Some(previous) = previous_input_row
                && row.input_row <= previous
            {
                return Err(execution_error(format!(
                    "output logits row {output_index} input_row {} is not strictly greater than {previous}",
                    row.input_row
                )));
            }
            previous_input_row = Some(row.input_row);

            let input_row = usize::try_from(row.input_row).map_err(|_| {
                execution_error(format!(
                    "output logits row {output_index} input_row {} cannot be represented as usize",
                    row.input_row
                ))
            })?;
            let Some(request) = batch.logits.get(input_row) else {
                return Err(execution_error(format!(
                    "output logits row {output_index} references input row {}, but the batch has {} rows",
                    row.input_row,
                    batch.len()
                )));
            };

            match (request, &row.logits) {
                (LogitsRequest::None, _) => {
                    return Err(execution_error(format!(
                        "output was provided for input row {input_row}, which requested no logits"
                    )));
                }
                (LogitsRequest::TopK(k), LogitsOutput::TopK(logits)) => {
                    Self::validate_top_k_payload(input_row, *k, logits)?;
                }
                (LogitsRequest::Full, LogitsOutput::Full(logits)) => {
                    Self::validate_full_payload(input_row, logits)?;
                }
                (LogitsRequest::TopK(_), LogitsOutput::Full(_)) => {
                    return Err(execution_error(format!(
                        "input row {input_row} requested top-k logits but received full logits"
                    )));
                }
                (LogitsRequest::Full, LogitsOutput::TopK(_)) => {
                    return Err(execution_error(format!(
                        "input row {input_row} requested full logits but received top-k logits"
                    )));
                }
            }

            seen[input_row] = true;
        }

        for (input_row, request) in batch.logits.iter().enumerate() {
            if !matches!(request, LogitsRequest::None) && !seen[input_row] {
                return Err(execution_error(format!(
                    "input row {input_row} requested logits but has no output row"
                )));
            }
        }

        Ok(())
    }

    /// Validates intrinsic output structure and payload invariants, then applies
    /// logits-related prepared-plan capabilities.
    pub fn validate_with_capabilities(
        &self,
        batch: &ExecutionBatch,
        capabilities: &ExecutionCapabilities,
    ) -> Result<()> {
        self.validate(batch)?;

        for row in &self.logits {
            let input_row = usize::try_from(row.input_row).map_err(|_| {
                execution_error(format!(
                    "output input row {} cannot be represented as usize",
                    row.input_row
                ))
            })?;
            let request = batch.logits.get(input_row).ok_or_else(|| {
                execution_error(format!(
                    "output references input row {}, but the batch has {} logits requests",
                    row.input_row,
                    batch.logits.len()
                ))
            })?;

            match (request, &row.logits) {
                (LogitsRequest::TopK(k), LogitsOutput::TopK(_)) => match capabilities.max_top_k {
                    Some(maximum) if *k <= maximum => {}
                    Some(maximum) => {
                        return Err(execution_error(format!(
                            "input row {input_row} requests top-k {}, exceeding capability limit {}",
                            k.get(),
                            maximum.get()
                        )));
                    }
                    None => {
                        return Err(execution_error(format!(
                            "input row {input_row} requests top-k logits, which are unsupported"
                        )));
                    }
                },
                (LogitsRequest::Full, LogitsOutput::Full(logits)) => {
                    let width = capabilities.full_logits_width.ok_or_else(|| {
                        execution_error(format!(
                            "input row {input_row} requests full logits, which are unsupported"
                        ))
                    })?;
                    let expected_width = usize::try_from(width.get()).map_err(|_| {
                        execution_error(format!(
                            "full-logits width {} cannot be represented as usize",
                            width.get()
                        ))
                    })?;
                    if logits.len() != expected_width {
                        return Err(execution_error(format!(
                            "input row {input_row} full-logits payload has width {}, expected {expected_width}",
                            logits.len()
                        )));
                    }
                }
                _ => {
                    return Err(execution_error(format!(
                        "input row {input_row} output no longer matches its logits request"
                    )));
                }
            }
        }

        Ok(())
    }

    fn validate_top_k_payload(
        input_row: usize,
        k: NonZeroU32,
        logits: &[TokenLogit],
    ) -> Result<()> {
        let maximum_len = usize::try_from(k.get()).map_err(|_| {
            execution_error(format!(
                "input row {input_row} requested top-k {} which cannot be represented as usize",
                k.get()
            ))
        })?;
        if logits.len() > maximum_len {
            return Err(execution_error(format!(
                "input row {input_row} top-k payload has {} candidates, exceeding requested k {maximum_len}",
                logits.len()
            )));
        }

        let mut token_ids = HashSet::with_capacity(logits.len());
        let mut previous: Option<&TokenLogit> = None;
        for (candidate_index, candidate) in logits.iter().enumerate() {
            if !candidate.logit.is_finite() {
                return Err(execution_error(format!(
                    "input row {input_row} top-k candidate {candidate_index} has non-finite logit {}",
                    candidate.logit
                )));
            }
            if !token_ids.insert(candidate.token_id) {
                return Err(execution_error(format!(
                    "input row {input_row} top-k payload repeats token ID {}",
                    candidate.token_id
                )));
            }

            if let Some(previous_candidate) = previous {
                let wrong_logit_order = previous_candidate.logit < candidate.logit;
                let wrong_tie_order = previous_candidate.logit == candidate.logit
                    && previous_candidate.token_id > candidate.token_id;
                if wrong_logit_order || wrong_tie_order {
                    return Err(execution_error(format!(
                        "input row {input_row} top-k candidate {candidate_index} is not in deterministic logit-descending/token-ascending order"
                    )));
                }
            }
            previous = Some(candidate);
        }

        Ok(())
    }

    fn validate_full_payload(input_row: usize, logits: &[f32]) -> Result<()> {
        if logits.is_empty() {
            return Err(execution_error(format!(
                "input row {input_row} full-logits payload must not be empty"
            )));
        }
        if let Some(index) = logits.iter().position(|logit| !logit.is_finite()) {
            return Err(execution_error(format!(
                "input row {input_row} full-logits value {index} is non-finite ({})",
                logits[index]
            )));
        }

        Ok(())
    }
}

/// Initial cursor and capacity requested for a new backend sequence state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceStateInit {
    pub initial_position: u32,
    pub max_sequence_len: u32,
}

impl SequenceStateInit {
    pub const fn new(initial_position: u32, max_sequence_len: u32) -> Self {
        Self {
            initial_position,
            max_sequence_len,
        }
    }
}

/// Immutable model-global prepared execution description.
pub trait PreparedModelPlan {
    fn capabilities(&self) -> &ExecutionCapabilities;

    fn validate_batch(&self, state_count: usize, batch: &ExecutionBatch) -> Result<()> {
        batch.validate(state_count, self.capabilities())
    }

    fn validate_output(&self, batch: &ExecutionBatch, output: &ExecutionOutput) -> Result<()> {
        output.validate_with_capabilities(batch, self.capabilities())
    }
}

/// Backend lifecycle and execution boundary for a prepared model plan.
pub trait ModelBatchExecutor {
    type Plan: PreparedModelPlan;
    type SequenceState;

    fn create_sequence_state(
        &mut self,
        plan: &Self::Plan,
        init: SequenceStateInit,
    ) -> Result<Self::SequenceState>;

    fn reset_sequence_state(
        &mut self,
        plan: &Self::Plan,
        state: &mut Self::SequenceState,
    ) -> Result<()>;

    fn release_sequence_state(
        &mut self,
        _plan: &Self::Plan,
        state: Self::SequenceState,
    ) -> Result<()> {
        drop(state);
        Ok(())
    }

    fn execute(
        &mut self,
        plan: &Self::Plan,
        states: &mut [Self::SequenceState],
        batch: &ExecutionBatch,
    ) -> Result<ExecutionOutput>;
}

fn execution_error(message: impl Into<String>) -> Error {
    Error::Execution(message.into())
}

// ── E5: Physical paged KV layout schema ───────────────────────────────────

/// Describes one logical KV plane (e.g. window, compressed, indexer).
///
/// Each plane has its own element width and page-size semantics. The runtime
/// page manager allocates pages per plane; the backend maps them to physical
/// device buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvPlaneDescriptor {
    /// Human-readable plane name for diagnostics (e.g. "window", "compressed").
    pub name: &'static str,
    /// Number of f32-equivalent elements per token in this plane.
    pub elements_per_token: usize,
    /// Number of layers this plane spans (usually equal to the model layer count).
    pub layer_count: usize,
}

/// Model-supplied schema describing how the model's KV cache is laid out
/// across physical pages.
///
/// The runtime `KvPageManager` uses this schema to compute page requirements
/// and allocate pages. The backend uses it to size physical pools.
///
/// This trait is model-agnostic. Concrete models (DSV4, Qwen3, etc.) implement
/// it to describe their specific KV planes.
pub trait KvLayoutSchema: std::fmt::Debug + Send + Sync {
    /// All KV planes this model requires.
    fn planes(&self) -> &[KvPlaneDescriptor];

    /// Page size in tokens. All planes use the same page granularity.
    fn page_size(&self) -> usize;

    /// Maximum sequence length the schema supports.
    fn max_sequence_len(&self) -> usize;

    /// Number of pages needed for a sequence of `token_count` tokens.
    fn pages_for_tokens(&self, token_count: usize) -> usize {
        let page_size = self.page_size();
        token_count.div_ceil(page_size)
    }
}

/// A logical KV page identifier within one plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvPageId(pub u32);

/// One delayed copy-on-write replacement of a shared tail page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCowReplacement {
    pub logical_page: usize,
    pub source: KvPageId,
    pub replacement: KvPageId,
}

/// Reservation of pages for one sequence before execution.
///
/// Pages are reserved before model execution and committed only after
/// successful completion. A failure rolls back all newly allocated pages.
#[derive(Debug, Clone)]
pub struct KvReservation {
    /// State slot this reservation belongs to.
    pub state_slot: StateSlot,
    /// Position range covered by this reservation.
    pub positions: std::ops::Range<usize>,
    /// Newly allocated page IDs that will be committed on success or rolled
    /// back on failure.
    pub newly_allocated: Vec<KvPageId>,
    /// Generation of the sequence state when the reservation was made.
    pub generation: u64,
    /// Delayed COW replacement for a partially filled shared tail page.
    pub cow_replacement: Option<KvCowReplacement>,
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use super::*;

    fn nz(value: u32) -> NonZeroU32 {
        NonZeroU32::new(value).unwrap()
    }

    fn capabilities() -> ExecutionCapabilities {
        ExecutionCapabilities {
            max_batch_tokens: 16,
            max_sequences: 4,
            max_prefill_query_tokens_per_sequence: 8,
            max_decode_query_tokens_per_sequence: 2,
            max_top_k: Some(nz(5)),
            supports_prefill: true,
            supports_decode: true,
            supports_mixed: true,
            full_logits_width: Some(nz(2)),
            kv_binding_mode: KvBindingMode::None,
            logits_row_policy: LogitsRowPolicy::Any,
        }
    }

    fn sequence(
        state_slot: u32,
        phase: ForwardPhase,
        query: Range<u32>,
        context_len: u32,
        sequence_len: u32,
    ) -> ExecutionSequence {
        ExecutionSequence::new(
            StateSlot::new(state_slot),
            phase,
            query,
            context_len,
            sequence_len,
            0..0,
        )
    }

    fn prefill_batch() -> ExecutionBatch {
        ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![10, 11, 20, 21],
            vec![0, 1, 5, 6],
            vec![None; 4],
            vec![
                LogitsRequest::None,
                LogitsRequest::TopK(nz(2)),
                LogitsRequest::None,
                LogitsRequest::Full,
            ],
            vec![
                sequence(0, ForwardPhase::Prefill, 0..2, 0, 2),
                sequence(1, ForwardPhase::Prefill, 2..4, 5, 7),
            ],
            vec![],
        )
    }

    fn decode_batch() -> ExecutionBatch {
        ExecutionBatch::new(
            ForwardMode::Decode,
            vec![10, 20],
            vec![4, 9],
            vec![None; 2],
            vec![LogitsRequest::TopK(nz(2)), LogitsRequest::Full],
            vec![
                sequence(0, ForwardPhase::Decode, 0..1, 4, 5),
                sequence(1, ForwardPhase::Decode, 1..2, 9, 10),
            ],
            vec![],
        )
    }

    fn one_row_prefill(logits: LogitsRequest) -> ExecutionBatch {
        ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![10],
            vec![0],
            vec![None],
            vec![logits],
            vec![sequence(0, ForwardPhase::Prefill, 0..1, 0, 1)],
            vec![],
        )
    }

    fn assert_execution_error(result: Result<()>) {
        assert!(matches!(result, Err(Error::Execution(_))));
    }

    #[test]
    fn identifiers_round_trip_and_check_host_conversions() {
        let state = StateSlot::new(7);
        assert_eq!(state.get(), 7);
        assert_eq!(state.try_as_usize().unwrap(), 7);
        assert_eq!(usize::try_from(state).unwrap(), 7);
        assert_eq!(StateSlot::try_from(7_usize).unwrap(), state);
        assert_eq!(u32::from(state), 7);

        let write = KvWriteSlot::new(8);
        let block = KvBlockId::new(9);
        assert_eq!(write.try_as_usize().unwrap(), 8);
        assert_eq!(block.try_as_usize().unwrap(), 9);

        if usize::BITS > u32::BITS {
            assert!(StateSlot::try_from(usize::MAX).is_err());
            assert!(KvWriteSlot::try_from(usize::MAX).is_err());
            assert!(KvBlockId::try_from(usize::MAX).is_err());
        }
    }

    #[test]
    fn validates_ragged_prefill_and_read_only_accessors() {
        let batch = prefill_batch();
        batch.validate(2, &capabilities()).unwrap();

        assert_eq!(batch.mode(), ForwardMode::Prefill);
        assert_eq!(batch.token_ids(), &[10, 11, 20, 21]);
        assert_eq!(batch.positions(), &[0, 1, 5, 6]);
        assert_eq!(batch.kv_write_slots(), &[None; 4]);
        assert_eq!(batch.logits().len(), 4);
        assert_eq!(batch.sequences().len(), 2);
        assert!(batch.kv_block_ids().is_empty());
        assert_eq!(batch.len(), 4);
        assert!(!batch.is_empty());
    }

    #[test]
    fn validates_multi_sequence_decode() {
        let batch = decode_batch();
        batch.validate(2, &capabilities()).unwrap();
        assert_eq!(batch.intent(), ExecutionIntent::Committed);
        assert_eq!(
            batch.shape().unwrap(),
            ExecutionShape {
                intent: ExecutionIntent::Committed,
                sequence_count: 2,
                total_rows: 2,
                max_query_tokens: 1,
                decode_rows: 2,
            }
        );
    }

    #[test]
    fn provisional_verification_is_explicit_and_shape_distinct_from_decode() {
        let batch = ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![10, 11, 12, 20, 21],
            vec![8, 9, 10, 17, 18],
            vec![None; 5],
            vec![LogitsRequest::TopK(nz(1)); 5],
            vec![
                sequence(0, ForwardPhase::Prefill, 0..3, 8, 11),
                sequence(1, ForwardPhase::Prefill, 3..5, 17, 19),
            ],
            vec![],
        )
        .with_intent(ExecutionIntent::ProvisionalVerification);
        batch.validate(2, &capabilities()).unwrap();
        assert_eq!(
            batch.shape().unwrap(),
            ExecutionShape {
                intent: ExecutionIntent::ProvisionalVerification,
                sequence_count: 2,
                total_rows: 5,
                max_query_tokens: 3,
                decode_rows: 0,
            }
        );
    }

    #[test]
    fn validates_mixed_prefill_and_decode() {
        let batch = ExecutionBatch::new(
            ForwardMode::Mixed,
            vec![10, 11, 20],
            vec![0, 1, 7],
            vec![None; 3],
            vec![
                LogitsRequest::None,
                LogitsRequest::TopK(nz(1)),
                LogitsRequest::Full,
            ],
            vec![
                sequence(0, ForwardPhase::Prefill, 0..2, 0, 2),
                sequence(1, ForwardPhase::Decode, 2..3, 7, 8),
            ],
            vec![],
        );

        batch.validate(2, &capabilities()).unwrap();
    }

    #[test]
    fn rejects_empty_and_mismatched_packed_vectors() {
        let empty = ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
        );
        assert_execution_error(empty.validate(0, &capabilities()));

        let mismatched = ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![1],
            vec![],
            vec![None],
            vec![LogitsRequest::None],
            vec![sequence(0, ForwardPhase::Prefill, 0..1, 0, 1)],
            vec![],
        );
        assert_execution_error(mismatched.validate(1, &capabilities()));
    }

    #[test]
    fn rejects_query_gaps_overlaps_empty_ranges_and_trailing_rows() {
        for sequences in [
            vec![sequence(0, ForwardPhase::Prefill, 1..2, 0, 1)],
            vec![
                sequence(0, ForwardPhase::Prefill, 0..1, 0, 1),
                sequence(1, ForwardPhase::Prefill, 0..2, 0, 2),
            ],
            vec![sequence(0, ForwardPhase::Prefill, 0..0, 0, 0)],
            vec![sequence(0, ForwardPhase::Prefill, 0..1, 0, 1)],
        ] {
            let batch = ExecutionBatch::new(
                ForwardMode::Prefill,
                vec![1, 2],
                vec![0, 1],
                vec![None; 2],
                vec![LogitsRequest::None; 2],
                sequences,
                vec![],
            );
            assert_execution_error(batch.validate(2, &capabilities()));
        }
    }

    #[test]
    fn rejects_reversed_and_out_of_bounds_query_ranges() {
        let reversed = Range { start: 1, end: 0 };
        for query in [reversed, 0..2] {
            let batch = ExecutionBatch::new(
                ForwardMode::Prefill,
                vec![1],
                vec![0],
                vec![None],
                vec![LogitsRequest::None],
                vec![sequence(0, ForwardPhase::Prefill, query, 0, 1)],
                vec![],
            );
            assert_execution_error(batch.validate(1, &capabilities()));
        }
    }

    #[test]
    fn rejects_out_of_range_and_duplicate_state_slots() {
        let out_of_range = ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![1],
            vec![0],
            vec![None],
            vec![LogitsRequest::None],
            vec![sequence(1, ForwardPhase::Prefill, 0..1, 0, 1)],
            vec![],
        );
        assert_execution_error(out_of_range.validate(1, &capabilities()));

        let duplicate = ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![1, 2],
            vec![0, 0],
            vec![None; 2],
            vec![LogitsRequest::None; 2],
            vec![
                sequence(0, ForwardPhase::Prefill, 0..1, 0, 1),
                sequence(0, ForwardPhase::Prefill, 1..2, 0, 1),
            ],
            vec![],
        );
        assert_execution_error(duplicate.validate(1, &capabilities()));
    }

    #[test]
    fn rejects_phase_mode_mismatches_and_degenerate_mixed_mode() {
        let mismatch = ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![1],
            vec![0],
            vec![None],
            vec![LogitsRequest::None],
            vec![sequence(0, ForwardPhase::Decode, 0..1, 0, 1)],
            vec![],
        );
        assert_execution_error(mismatch.validate(1, &capabilities()));

        let only_prefill = ExecutionBatch::new(
            ForwardMode::Mixed,
            vec![1],
            vec![0],
            vec![None],
            vec![LogitsRequest::None],
            vec![sequence(0, ForwardPhase::Prefill, 0..1, 0, 1)],
            vec![],
        );
        assert_execution_error(only_prefill.validate(1, &capabilities()));
    }

    #[test]
    fn rejects_position_discontinuity_sequence_length_mismatch_and_overflow() {
        let wrong_position = ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![1, 2],
            vec![3, 5],
            vec![None; 2],
            vec![LogitsRequest::None; 2],
            vec![sequence(0, ForwardPhase::Prefill, 0..2, 3, 5)],
            vec![],
        );
        assert_execution_error(wrong_position.validate(1, &capabilities()));

        let wrong_length = ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![1],
            vec![3],
            vec![None],
            vec![LogitsRequest::None],
            vec![sequence(0, ForwardPhase::Prefill, 0..1, 3, 3)],
            vec![],
        );
        assert_execution_error(wrong_length.validate(1, &capabilities()));

        let overflow = ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![1],
            vec![u32::MAX],
            vec![None],
            vec![LogitsRequest::None],
            vec![sequence(0, ForwardPhase::Prefill, 0..1, u32::MAX, u32::MAX)],
            vec![],
        );
        assert_execution_error(overflow.validate(1, &capabilities()));
    }

    #[test]
    fn validates_complete_paged_kv_bindings() {
        let mut caps = capabilities();
        caps.kv_binding_mode = KvBindingMode::Paged;
        let batch = ExecutionBatch::new(
            ForwardMode::Mixed,
            vec![1, 2, 3],
            vec![0, 1, 4],
            vec![
                Some(KvWriteSlot::new(10)),
                Some(KvWriteSlot::new(11)),
                Some(KvWriteSlot::new(20)),
            ],
            vec![LogitsRequest::None; 3],
            vec![
                ExecutionSequence::new(StateSlot::new(0), ForwardPhase::Prefill, 0..2, 0, 2, 0..2),
                ExecutionSequence::new(StateSlot::new(1), ForwardPhase::Decode, 2..3, 4, 5, 2..3),
            ],
            vec![
                KvBlockId::new(100),
                KvBlockId::new(101),
                KvBlockId::new(200),
            ],
        );

        batch.validate(2, &caps).unwrap();
    }

    #[test]
    fn validates_unique_paged_kv_write_slots() {
        let mut caps = capabilities();
        caps.kv_binding_mode = KvBindingMode::Paged;
        let batch = ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![1, 2],
            vec![0, 1],
            vec![Some(KvWriteSlot::new(7)), Some(KvWriteSlot::new(8))],
            vec![LogitsRequest::None; 2],
            vec![ExecutionSequence::new(
                StateSlot::new(0),
                ForwardPhase::Prefill,
                0..2,
                0,
                2,
                0..1,
            )],
            vec![KvBlockId::new(100)],
        );

        batch.validate(1, &caps).unwrap();
    }

    #[test]
    fn rejects_duplicate_paged_kv_write_slots() {
        let mut caps = capabilities();
        caps.kv_binding_mode = KvBindingMode::Paged;
        let batch = ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![1, 2],
            vec![0, 1],
            vec![Some(KvWriteSlot::new(7)), Some(KvWriteSlot::new(7))],
            vec![LogitsRequest::None; 2],
            vec![ExecutionSequence::new(
                StateSlot::new(0),
                ForwardPhase::Prefill,
                0..2,
                0,
                2,
                0..1,
            )],
            vec![KvBlockId::new(100)],
        );

        assert_execution_error(batch.validate(1, &caps));
    }

    #[test]
    fn rejects_kv_data_in_none_mode() {
        let with_write = ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![1],
            vec![0],
            vec![Some(KvWriteSlot::new(0))],
            vec![LogitsRequest::None],
            vec![sequence(0, ForwardPhase::Prefill, 0..1, 0, 1)],
            vec![],
        );
        assert_execution_error(with_write.validate(1, &capabilities()));

        let with_blocks = ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![1],
            vec![0],
            vec![None],
            vec![LogitsRequest::None],
            vec![ExecutionSequence::new(
                StateSlot::new(0),
                ForwardPhase::Prefill,
                0..1,
                0,
                1,
                0..1,
            )],
            vec![KvBlockId::new(0)],
        );
        assert_execution_error(with_blocks.validate(1, &capabilities()));
    }

    #[test]
    fn rejects_incomplete_and_invalid_paged_kv_bindings() {
        let mut caps = capabilities();
        caps.kv_binding_mode = KvBindingMode::Paged;

        let reversed_block_table = Range { start: 1, end: 0 };
        let cases = [
            (
                vec![None],
                ExecutionSequence::new(StateSlot::new(0), ForwardPhase::Prefill, 0..1, 0, 1, 0..1),
                vec![KvBlockId::new(0)],
            ),
            (
                vec![Some(KvWriteSlot::new(0))],
                ExecutionSequence::new(StateSlot::new(0), ForwardPhase::Prefill, 0..1, 0, 1, 0..0),
                vec![],
            ),
            (
                vec![Some(KvWriteSlot::new(0))],
                ExecutionSequence::new(
                    StateSlot::new(0),
                    ForwardPhase::Prefill,
                    0..1,
                    0,
                    1,
                    reversed_block_table,
                ),
                vec![KvBlockId::new(0)],
            ),
            (
                vec![Some(KvWriteSlot::new(0))],
                ExecutionSequence::new(StateSlot::new(0), ForwardPhase::Prefill, 0..1, 0, 1, 0..2),
                vec![KvBlockId::new(0)],
            ),
        ];

        for (write_slots, sequence, blocks) in cases {
            let batch = ExecutionBatch::new(
                ForwardMode::Prefill,
                vec![1],
                vec![0],
                write_slots,
                vec![LogitsRequest::None],
                vec![sequence],
                blocks,
            );
            assert_execution_error(batch.validate(1, &caps));
        }
    }

    #[test]
    fn enforces_batch_sequence_and_phase_query_limits() {
        let mut caps = capabilities();
        caps.max_batch_tokens = 1;
        assert_execution_error(prefill_batch().validate(2, &caps));

        let mut caps = capabilities();
        caps.max_sequences = 1;
        assert_execution_error(prefill_batch().validate(2, &caps));

        let mut caps = capabilities();
        caps.max_prefill_query_tokens_per_sequence = 1;
        assert_execution_error(prefill_batch().validate(2, &caps));

        let mut caps = capabilities();
        caps.max_decode_query_tokens_per_sequence = 0;
        assert_execution_error(decode_batch().validate(2, &caps));
    }

    #[test]
    fn rejects_unsupported_forward_capabilities() {
        let mut caps = capabilities();
        caps.supports_prefill = false;
        assert_execution_error(prefill_batch().validate(2, &caps));

        let mut caps = capabilities();
        caps.supports_decode = false;
        assert_execution_error(decode_batch().validate(2, &caps));

        let mut caps = capabilities();
        caps.supports_mixed = false;
        let mixed = ExecutionBatch::new(
            ForwardMode::Mixed,
            vec![1, 2],
            vec![0, 1],
            vec![None; 2],
            vec![LogitsRequest::None; 2],
            vec![
                sequence(0, ForwardPhase::Prefill, 0..1, 0, 1),
                sequence(1, ForwardPhase::Decode, 1..2, 1, 2),
            ],
            vec![],
        );
        assert_execution_error(mixed.validate(2, &caps));
    }

    #[test]
    fn enforces_top_k_and_full_logits_capabilities() {
        let mut caps = capabilities();
        caps.max_top_k = None;
        assert_execution_error(one_row_prefill(LogitsRequest::TopK(nz(1))).validate(1, &caps));

        let mut caps = capabilities();
        caps.max_top_k = Some(nz(2));
        assert_execution_error(one_row_prefill(LogitsRequest::TopK(nz(3))).validate(1, &caps));

        let mut caps = capabilities();
        caps.full_logits_width = None;
        assert_execution_error(one_row_prefill(LogitsRequest::Full).validate(1, &caps));
    }

    #[test]
    fn enforces_logits_row_placement_policy() {
        let mut caps = capabilities();
        caps.logits_row_policy = LogitsRowPolicy::None;
        assert_execution_error(one_row_prefill(LogitsRequest::TopK(nz(1))).validate(1, &caps));
        one_row_prefill(LogitsRequest::None)
            .validate(1, &caps)
            .unwrap();

        let mut caps = capabilities();
        caps.logits_row_policy = LogitsRowPolicy::LastPerSequence;
        let early = ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![1, 2],
            vec![0, 1],
            vec![None; 2],
            vec![LogitsRequest::TopK(nz(1)), LogitsRequest::None],
            vec![sequence(0, ForwardPhase::Prefill, 0..2, 0, 2)],
            vec![],
        );
        assert_execution_error(early.validate(1, &caps));

        let last = ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![1, 2],
            vec![0, 1],
            vec![None; 2],
            vec![LogitsRequest::None, LogitsRequest::TopK(nz(1))],
            vec![sequence(0, ForwardPhase::Prefill, 0..2, 0, 2)],
            vec![],
        );
        last.validate(1, &caps).unwrap();
    }

    #[test]
    fn validates_complete_ordered_execution_output() {
        let batch = prefill_batch();
        let output = ExecutionOutput::new(vec![
            LogitsRow::new(
                1,
                LogitsOutput::TopK(vec![TokenLogit::new(100, 1.5), TokenLogit::new(101, 1.0)]),
            ),
            LogitsRow::new(3, LogitsOutput::Full(vec![0.25, 0.5])),
        ]);

        output.validate(&batch).unwrap();
        output
            .validate_with_capabilities(&batch, &capabilities())
            .unwrap();
    }

    #[test]
    fn validates_empty_top_k_and_deterministically_ordered_ties() {
        let batch = one_row_prefill(LogitsRequest::TopK(nz(3)));
        let no_candidate =
            ExecutionOutput::new(vec![LogitsRow::new(0, LogitsOutput::TopK(vec![]))]);
        no_candidate.validate(&batch).unwrap();

        let tied = ExecutionOutput::new(vec![LogitsRow::new(
            0,
            LogitsOutput::TopK(vec![
                TokenLogit::new(2, 1.0),
                TokenLogit::new(7, 1.0),
                TokenLogit::new(1, 0.5),
            ]),
        )]);
        tied.validate(&batch).unwrap();
        tied.validate_with_capabilities(&batch, &capabilities())
            .unwrap();
    }

    #[test]
    fn rejects_oversized_top_k_payload() {
        let batch = one_row_prefill(LogitsRequest::TopK(nz(2)));
        let output = ExecutionOutput::new(vec![LogitsRow::new(
            0,
            LogitsOutput::TopK(vec![
                TokenLogit::new(1, 3.0),
                TokenLogit::new(2, 2.0),
                TokenLogit::new(3, 1.0),
            ]),
        )]);

        assert_execution_error(output.validate(&batch));
    }

    #[test]
    fn rejects_duplicate_top_k_token_ids() {
        let batch = one_row_prefill(LogitsRequest::TopK(nz(2)));
        let output = ExecutionOutput::new(vec![LogitsRow::new(
            0,
            LogitsOutput::TopK(vec![TokenLogit::new(1, 2.0), TokenLogit::new(1, 1.0)]),
        )]);

        assert_execution_error(output.validate(&batch));
    }

    #[test]
    fn rejects_unsorted_top_k_payload() {
        let batch = one_row_prefill(LogitsRequest::TopK(nz(2)));
        for logits in [
            vec![TokenLogit::new(1, 1.0), TokenLogit::new(2, 2.0)],
            vec![TokenLogit::new(2, 1.0), TokenLogit::new(1, 1.0)],
        ] {
            let output = ExecutionOutput::new(vec![LogitsRow::new(0, LogitsOutput::TopK(logits))]);
            assert_execution_error(output.validate(&batch));
        }
    }

    #[test]
    fn rejects_non_finite_top_k_logits() {
        let batch = one_row_prefill(LogitsRequest::TopK(nz(1)));
        for logit in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let output = ExecutionOutput::new(vec![LogitsRow::new(
                0,
                LogitsOutput::TopK(vec![TokenLogit::new(1, logit)]),
            )]);
            assert_execution_error(output.validate(&batch));
        }
    }

    #[test]
    fn rejects_empty_and_non_finite_full_logits() {
        let batch = one_row_prefill(LogitsRequest::Full);
        for logits in [
            vec![],
            vec![f32::NAN],
            vec![f32::INFINITY],
            vec![f32::NEG_INFINITY],
        ] {
            let output = ExecutionOutput::new(vec![LogitsRow::new(0, LogitsOutput::Full(logits))]);
            assert_execution_error(output.validate(&batch));
        }
    }

    #[test]
    fn capabilities_validation_enforces_full_width_and_request_support() {
        let full_batch = one_row_prefill(LogitsRequest::Full);
        let wrong_width =
            ExecutionOutput::new(vec![LogitsRow::new(0, LogitsOutput::Full(vec![1.0]))]);
        wrong_width.validate(&full_batch).unwrap();
        assert_execution_error(
            wrong_width.validate_with_capabilities(&full_batch, &capabilities()),
        );

        let mut caps = capabilities();
        caps.full_logits_width = None;
        assert_execution_error(wrong_width.validate_with_capabilities(&full_batch, &caps));

        let top_k_batch = one_row_prefill(LogitsRequest::TopK(nz(3)));
        let no_candidate =
            ExecutionOutput::new(vec![LogitsRow::new(0, LogitsOutput::TopK(vec![]))]);
        caps.max_top_k = Some(nz(2));
        assert_execution_error(no_candidate.validate_with_capabilities(&top_k_batch, &caps));
        caps.max_top_k = None;
        assert_execution_error(no_candidate.validate_with_capabilities(&top_k_batch, &caps));
    }

    #[test]
    fn rejects_unordered_duplicate_and_out_of_range_output_rows() {
        let batch = decode_batch();
        for output in [
            ExecutionOutput::new(vec![
                LogitsRow::new(1, LogitsOutput::Full(vec![0.0])),
                LogitsRow::new(0, LogitsOutput::TopK(vec![])),
            ]),
            ExecutionOutput::new(vec![
                LogitsRow::new(0, LogitsOutput::TopK(vec![])),
                LogitsRow::new(0, LogitsOutput::TopK(vec![])),
            ]),
            ExecutionOutput::new(vec![
                LogitsRow::new(0, LogitsOutput::TopK(vec![])),
                LogitsRow::new(2, LogitsOutput::Full(vec![0.0])),
            ]),
        ] {
            assert_execution_error(output.validate(&batch));
        }
    }

    #[test]
    fn rejects_output_for_none_request_and_payload_type_mismatches() {
        let none_batch = one_row_prefill(LogitsRequest::None);
        assert_execution_error(
            ExecutionOutput::new(vec![LogitsRow::new(0, LogitsOutput::TopK(vec![]))])
                .validate(&none_batch),
        );

        let top_k_batch = one_row_prefill(LogitsRequest::TopK(nz(1)));
        assert_execution_error(
            ExecutionOutput::new(vec![LogitsRow::new(0, LogitsOutput::Full(vec![]))])
                .validate(&top_k_batch),
        );

        let full_batch = one_row_prefill(LogitsRequest::Full);
        assert_execution_error(
            ExecutionOutput::new(vec![LogitsRow::new(0, LogitsOutput::TopK(vec![]))])
                .validate(&full_batch),
        );
    }

    #[test]
    fn rejects_missing_requested_output_rows() {
        assert_execution_error(
            ExecutionOutput::default().validate(&one_row_prefill(LogitsRequest::TopK(nz(1)))),
        );
        ExecutionOutput::default()
            .validate(&one_row_prefill(LogitsRequest::None))
            .unwrap();
    }

    struct TestPlan {
        capabilities: ExecutionCapabilities,
    }

    impl PreparedModelPlan for TestPlan {
        fn capabilities(&self) -> &ExecutionCapabilities {
            &self.capabilities
        }
    }

    #[test]
    fn prepared_plan_default_validation_uses_its_capabilities() {
        let plan = TestPlan {
            capabilities: capabilities(),
        };
        plan.validate_batch(2, &prefill_batch()).unwrap();
        assert_execution_error(plan.validate_batch(1, &prefill_batch()));
    }

    #[test]
    fn prepared_plan_default_output_validation_uses_its_capabilities() {
        let plan = TestPlan {
            capabilities: capabilities(),
        };
        let batch = one_row_prefill(LogitsRequest::Full);
        let valid =
            ExecutionOutput::new(vec![LogitsRow::new(0, LogitsOutput::Full(vec![1.0, 0.5]))]);
        plan.validate_output(&batch, &valid).unwrap();

        let wrong_width =
            ExecutionOutput::new(vec![LogitsRow::new(0, LogitsOutput::Full(vec![1.0]))]);
        assert_execution_error(plan.validate_output(&batch, &wrong_width));
    }

    struct DropState(Rc<Cell<bool>>);

    impl Drop for DropState {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }

    struct TestExecutor;

    impl ModelBatchExecutor for TestExecutor {
        type Plan = TestPlan;
        type SequenceState = DropState;

        fn create_sequence_state(
            &mut self,
            _plan: &Self::Plan,
            _init: SequenceStateInit,
        ) -> Result<Self::SequenceState> {
            Ok(DropState(Rc::new(Cell::new(false))))
        }

        fn reset_sequence_state(
            &mut self,
            _plan: &Self::Plan,
            _state: &mut Self::SequenceState,
        ) -> Result<()> {
            Ok(())
        }

        fn execute(
            &mut self,
            plan: &Self::Plan,
            states: &mut [Self::SequenceState],
            batch: &ExecutionBatch,
        ) -> Result<ExecutionOutput> {
            plan.validate_batch(states.len(), batch)?;
            Ok(ExecutionOutput::default())
        }
    }

    #[test]
    fn executor_default_release_drops_sequence_state() {
        let plan = TestPlan {
            capabilities: capabilities(),
        };
        let dropped = Rc::new(Cell::new(false));
        let state = DropState(Rc::clone(&dropped));

        TestExecutor.release_sequence_state(&plan, state).unwrap();

        assert!(dropped.get());
    }
}
