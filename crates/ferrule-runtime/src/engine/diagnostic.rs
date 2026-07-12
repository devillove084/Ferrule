//! Explicit page-managed execution for model-family diagnostics.

use ferrule_common::execution::{
    ExecutionBatch, ExecutionSequence, ForwardMode, ForwardPhase, KvBindingMode, KvLayoutSchema,
    LogitsRequest, StateSlot,
};
use ferrule_common::{Error, Result};
use ferrule_model::MultiSessionRunner;

use crate::cache::{KvPageManager, KvPageManagerStats};

use super::NativeMultiSessionExecutor;

/// A diagnostic executor with authoritative logical pages and a bounded backend
/// physical pool.
///
/// Each sequence is a fresh explicit model state with its own page-table entry.
/// Diagnostic mutations use the same reserve/bind/prepare/commit transaction as
/// resident execution, while allowing callers to capture model-specific
/// intermediate values.
pub struct PageManagedDiagnosticHarness<R: MultiSessionRunner> {
    executor: NativeMultiSessionExecutor<R>,
    page_manager: KvPageManager,
    states: Vec<R::SequenceState>,
    generations: Vec<u64>,
    max_sequences: usize,
}

impl<R: MultiSessionRunner> PageManagedDiagnosticHarness<R> {
    /// Configure a bounded physical pool sized for `max_sequences` independent
    /// sequences of at most `max_tokens` tokens each.
    pub fn new(
        runner: R,
        schema: Box<dyn KvLayoutSchema>,
        max_tokens: usize,
        max_sequences: usize,
    ) -> Result<Self> {
        if max_tokens == 0 {
            return Err(Error::Execution(
                "page-managed diagnostic max_tokens must be greater than zero".into(),
            ));
        }
        if max_sequences == 0 {
            return Err(Error::Execution(
                "page-managed diagnostic max_sequences must be greater than zero".into(),
            ));
        }
        if max_tokens > schema.max_sequence_len() {
            return Err(Error::Execution(format!(
                "page-managed diagnostic max_tokens {max_tokens} exceeds model KV limit {}",
                schema.max_sequence_len()
            )));
        }
        let max_pages = schema
            .pages_for_tokens(max_tokens)
            .checked_mul(max_sequences)
            .filter(|pages| *pages > 0)
            .ok_or_else(|| Error::Execution("diagnostic KV page capacity overflow".into()))?;
        let page_manager = KvPageManager::new(schema, max_pages);
        let mut executor = NativeMultiSessionExecutor::new(runner);
        executor.configure_kv_page_capacity(max_pages)?;
        Ok(Self {
            executor,
            page_manager,
            states: Vec::with_capacity(max_sequences),
            generations: Vec::with_capacity(max_sequences),
            max_sequences,
        })
    }

    /// Allocate one fresh explicit model sequence and authoritative page table.
    pub fn create_sequence(&mut self, generation: u64) -> Result<StateSlot> {
        if self.states.len() >= self.max_sequences {
            return Err(Error::Execution(format!(
                "page-managed diagnostic sequence capacity {} is exhausted",
                self.max_sequences
            )));
        }
        let slot_u32 = u32::try_from(self.states.len())
            .map_err(|_| Error::Execution("diagnostic state slot exceeds u32".into()))?;
        let slot = StateSlot::new(slot_u32);
        let state = self.executor.create_sequence_state()?;
        self.page_manager.alloc_sequence(slot, generation)?;
        self.states.push(state);
        self.generations.push(generation);
        Ok(slot)
    }

    /// Execute one prefill chunk or decode token against an explicit sequence.
    pub fn execute_sequence_step<T>(
        &mut self,
        slot: StateSlot,
        phase: ForwardPhase,
        token_ids: &[u32],
        execute: impl FnOnce(&mut R) -> Result<T>,
    ) -> Result<T> {
        if token_ids.is_empty() {
            return Err(Error::Execution(
                "page-managed diagnostic step requires at least one token".into(),
            ));
        }
        if phase == ForwardPhase::Decode && token_ids.len() != 1 {
            return Err(Error::Execution(
                "page-managed diagnostic decode step requires exactly one token".into(),
            ));
        }
        let state_index = slot.try_as_usize().map_err(|_| {
            Error::Execution("page-managed diagnostic state slot exceeds usize".into())
        })?;
        let generation = *self.generations.get(state_index).ok_or_else(|| {
            Error::Execution(format!(
                "page-managed diagnostic state slot {} is not allocated",
                slot.get()
            ))
        })?;
        let reservation = self
            .page_manager
            .reserve(slot, generation, token_ids.len())?;
        let bindings = match self.page_manager.reservation_bindings(&reservation) {
            Ok(bindings) => bindings,
            Err(error) => {
                let _ = self.page_manager.rollback(reservation);
                return Err(error);
            }
        };
        let context_len = u32::try_from(reservation.positions.start)
            .map_err(|_| Error::Execution("diagnostic context length exceeds u32".into()))?;
        let sequence_len = u32::try_from(reservation.positions.end)
            .map_err(|_| Error::Execution("diagnostic sequence length exceeds u32".into()))?;
        let query_len = u32::try_from(token_ids.len())
            .map_err(|_| Error::Execution("diagnostic query length exceeds u32".into()))?;
        let uses_paged_kv = self.executor.capabilities().kv_binding_mode == KvBindingMode::Paged;
        let block_count = if uses_paged_kv {
            u32::try_from(bindings.block_ids.len())
                .map_err(|_| Error::Execution("diagnostic block table exceeds u32".into()))?
        } else {
            0
        };
        let mode = match phase {
            ForwardPhase::Prefill => ForwardMode::Prefill,
            ForwardPhase::Decode => ForwardMode::Decode,
        };
        let batch = ExecutionBatch::new(
            mode,
            token_ids.to_vec(),
            reservation
                .positions
                .clone()
                .map(|position| {
                    u32::try_from(position)
                        .map_err(|_| Error::Execution("diagnostic position exceeds u32".into()))
                })
                .collect::<Result<Vec<_>>>()?,
            if uses_paged_kv {
                bindings.write_slots.iter().copied().map(Some).collect()
            } else {
                vec![None; token_ids.len()]
            },
            vec![LogitsRequest::None; token_ids.len()],
            vec![ExecutionSequence::new(
                slot,
                phase,
                0..query_len,
                context_len,
                sequence_len,
                0..block_count,
            )],
            if uses_paged_kv {
                bindings.block_ids
            } else {
                Vec::new()
            },
        );

        let execution = self.executor.execute_diagnostic_batch_with_kv(
            &mut self.states,
            &batch,
            std::slice::from_ref(&reservation),
            |runner, states| runner.with_sequence_state(&mut states[state_index], execute),
        );
        let output = match execution {
            Ok(output) => output,
            Err(error) => {
                let _ = self.page_manager.rollback(reservation);
                return Err(error);
            }
        };

        if let Err(error) = self.page_manager.commit(reservation) {
            let rollback = self.executor.rollback_prepared_batch();
            return match rollback {
                Ok(()) => Err(error),
                Err(rollback_error) => Err(Error::Internal(format!(
                    "diagnostic KV commit failed ({error}); backend rollback also failed ({rollback_error})"
                ))),
            };
        }
        self.executor.commit_prepared_batch()?;
        Ok(output)
    }

    pub fn sequence_state(&self, slot: StateSlot) -> Result<&R::SequenceState> {
        let index = slot.try_as_usize().map_err(|_| {
            Error::Execution("page-managed diagnostic state slot exceeds usize".into())
        })?;
        self.states.get(index).ok_or_else(|| {
            Error::Execution(format!(
                "page-managed diagnostic state slot {} is not allocated",
                slot.get()
            ))
        })
    }

    pub fn runner(&self) -> &R {
        self.executor.runner()
    }

    pub fn runner_mut(&mut self) -> &mut R {
        self.executor.runner_mut()
    }

    pub fn page_stats(&self) -> KvPageManagerStats {
        self.page_manager.stats()
    }

    /// Release all diagnostic sequence/page state and return the prepared runner.
    pub fn into_runner(mut self) -> Result<R> {
        while let Some(state) = self.states.pop() {
            let index = self.states.len();
            let slot = StateSlot::new(u32::try_from(index).map_err(|_| {
                Error::Execution("diagnostic state slot exceeds u32 during release".into())
            })?);
            let pages = self.page_manager.free_sequence_pages(slot)?;
            self.executor.release_kv_pages(&pages)?;
            self.executor.release_sequence_state(state)?;
            self.generations.pop();
        }
        self.executor.into_runner()
    }
}
