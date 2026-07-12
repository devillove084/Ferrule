//! Resident request/session scheduler.
//!
//! This module ties together the runtime vocabulary that was previously adjacent
//! but not connected: `GenerateRequest`, `SequenceState`, `SequenceKvCache`,
//! `SchedulerAction`, neutral execution lowering, and runtime output correlation.
//!
//! It is intentionally synchronous and single-process. It does not execute a
//! model and does not know concrete model families; it only decides which
//! resident sequence should prefill/decode next and owns KV allocation lifecycle.

use std::collections::{BTreeMap, VecDeque};

use ferrule_common::execution::{LogitsOutput, TokenLogit};
use ferrule_common::{Error, Result};

use crate::cache::SequenceKvCache;

use super::actions::{plan_prefill_chunk, DecodeAction, PrefillChunkAction, SchedulerAction};
use super::session::{GenerateRequest, SequenceFinishReason, SequenceState, SessionId};

#[derive(Debug, Clone)]
struct WaitingRequest {
    request: GenerateRequest,
    position_start: Option<usize>,
}

impl WaitingRequest {
    fn new(request: GenerateRequest) -> Self {
        Self {
            request,
            position_start: None,
        }
    }

    fn at_position(request: GenerateRequest, position_start: usize) -> Self {
        Self {
            request,
            position_start: Some(position_start),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentSchedulerConfig {
    pub prefill_chunk_size: usize,
    pub max_active_sequences: usize,
    pub max_decode_batch: usize,
}

impl Default for ResidentSchedulerConfig {
    fn default() -> Self {
        Self {
            prefill_chunk_size: super::actions::DEFAULT_CHUNK_SIZE,
            max_active_sequences: 1,
            max_decode_batch: 1,
        }
    }
}

impl ResidentSchedulerConfig {
    fn normalized(self) -> Self {
        Self {
            prefill_chunk_size: self.prefill_chunk_size.max(1),
            max_active_sequences: self.max_active_sequences.max(1),
            max_decode_batch: self.max_decode_batch.max(1),
        }
    }
}

#[derive(Debug)]
pub struct ResidentScheduler {
    config: ResidentSchedulerConfig,
    waiting: VecDeque<WaitingRequest>,
    active: BTreeMap<SessionId, SequenceState>,
    prefill_queue: VecDeque<SessionId>,
    decode_ready: VecDeque<SessionId>,
    finished: Vec<SequenceState>,
    cancelled: Vec<SequenceState>,
    failed: Vec<SequenceState>,
    next_session_id: u64,
    total_submitted: u64,
}

impl Default for ResidentScheduler {
    fn default() -> Self {
        Self::new(ResidentSchedulerConfig::default())
    }
}

impl ResidentScheduler {
    pub fn new(config: ResidentSchedulerConfig) -> Self {
        Self {
            config: config.normalized(),
            waiting: VecDeque::new(),
            active: BTreeMap::new(),
            prefill_queue: VecDeque::new(),
            decode_ready: VecDeque::new(),
            finished: Vec::new(),
            cancelled: Vec::new(),
            failed: Vec::new(),
            next_session_id: 1,
            total_submitted: 0,
        }
    }

    pub fn config(&self) -> ResidentSchedulerConfig {
        self.config
    }

    pub fn submit(&mut self, request: GenerateRequest) {
        self.total_submitted = self.total_submitted.saturating_add(1);
        self.waiting.push_back(WaitingRequest::new(request));
    }

    /// Submit a request turn whose prompt should append to an already-resident
    /// backend session at `position_start`.
    ///
    /// This is the bridge needed by single-runner resident backends such as the
    /// current DSV4 runner: the runner owns the physical session/KV state, while
    /// the scheduler owns request-turn accounting and metadata KV lifecycle.
    pub fn submit_at_position(&mut self, request: GenerateRequest, position_start: usize) {
        self.total_submitted = self.total_submitted.saturating_add(1);
        self.waiting
            .push_back(WaitingRequest::at_position(request, position_start));
    }

    pub fn total_submitted(&self) -> u64 {
        self.total_submitted
    }

    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    pub fn active_len(&self) -> usize {
        self.active.len()
    }

    pub fn prefill_queue_len(&self) -> usize {
        self.prefill_queue.len()
    }

    pub fn decode_ready_len(&self) -> usize {
        self.decode_ready.len()
    }

    pub fn finished_len(&self) -> usize {
        self.finished.len()
    }

    pub fn cancelled_len(&self) -> usize {
        self.cancelled.len()
    }

    pub fn failed_len(&self) -> usize {
        self.failed.len()
    }

    pub fn active_sequence(&self, session_id: SessionId) -> Option<&SequenceState> {
        self.active.get(&session_id)
    }

    pub fn active_sequence_mut(&mut self, session_id: SessionId) -> Option<&mut SequenceState> {
        self.active.get_mut(&session_id)
    }

    pub fn drain_finished(&mut self) -> Vec<SequenceState> {
        std::mem::take(&mut self.finished)
    }

    pub fn drain_cancelled(&mut self) -> Vec<SequenceState> {
        std::mem::take(&mut self.cancelled)
    }

    pub fn drain_failed(&mut self) -> Vec<SequenceState> {
        std::mem::take(&mut self.failed)
    }

    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty()
            && self.active.is_empty()
            && self.prefill_queue.is_empty()
            && self.decode_ready.is_empty()
    }

    pub fn admit_waiting<C>(&mut self, kv_cache: &mut C) -> Result<usize>
    where
        C: SequenceKvCache,
    {
        let mut admitted = 0;
        while self.active.len() < self.config.max_active_sequences {
            let Some(waiting) = self.waiting.pop_front() else {
                break;
            };
            let session_id = self.resolve_session_id(waiting.request.session_id);
            if self.active.contains_key(&session_id) {
                self.waiting.push_front(waiting);
                return Err(Error::Internal(format!(
                    "session {:?} is already active",
                    session_id
                )));
            }

            let kv_handle = match kv_cache.alloc_sequence() {
                Ok(handle) => handle,
                Err(_) => {
                    self.waiting.push_front(waiting);
                    break;
                }
            };

            let mut sequence = SequenceState::from_request(&waiting.request, session_id);
            if let Some(position_start) = waiting.position_start {
                sequence.position = position_start;
            }
            sequence.bind_kv(kv_handle);
            if !sequence.prompt_prefill_done() {
                self.prefill_queue.push_back(sequence.session_id);
            }
            self.active.insert(sequence.session_id, sequence);
            admitted += 1;
        }
        Ok(admitted)
    }

    pub fn next_prefill_action<C>(&mut self, kv_cache: &mut C) -> Result<Option<SchedulerAction>>
    where
        C: SequenceKvCache,
    {
        if self.prefill_queue.is_empty() {
            self.admit_waiting(kv_cache)?;
        }

        while let Some(session_id) = self.prefill_queue.front().copied() {
            let Some(sequence) = self.active.get(&session_id) else {
                self.prefill_queue.pop_front();
                continue;
            };
            if sequence.prompt_prefill_done() {
                self.prefill_queue.pop_front();
                continue;
            }
            return plan_prefill_chunk(sequence, self.config.prefill_chunk_size);
        }

        Ok(None)
    }

    /// Pick the next executable action, preferring ready decode work over new
    /// prefill chunks for token latency. Callers that need stricter prefill-first
    /// behavior can keep using `next_prefill_action` and `next_decode_action`
    /// directly.
    pub fn next_action<C>(&mut self, kv_cache: &mut C) -> Result<Option<SchedulerAction>>
    where
        C: SequenceKvCache,
    {
        if let Some(action) = self.next_decode_action()? {
            return Ok(Some(action));
        }
        self.next_prefill_action(kv_cache)
    }

    pub fn commit_prefill_action(&mut self, action: &PrefillChunkAction) -> Result<()> {
        let sequence = self.active.get_mut(&action.session_id).ok_or_else(|| {
            Error::Internal(format!(
                "cannot commit prefill for inactive session {:?}",
                action.session_id
            ))
        })?;
        action.commit(sequence)?;
        if sequence.prompt_prefill_done() {
            self.remove_from_queue(action.session_id, QueueKind::Prefill);
        }
        Ok(())
    }

    pub fn stage_decode_token(&mut self, session_id: SessionId, token_id: u32) -> Result<()> {
        self.stage_decode_candidate(session_id, token_id, None)
    }

    pub fn stage_decode_candidate(
        &mut self,
        session_id: SessionId,
        token_id: u32,
        logit: Option<f32>,
    ) -> Result<()> {
        let sequence = self.active.get_mut(&session_id).ok_or_else(|| {
            Error::Internal(format!(
                "cannot stage decode token for inactive session {:?}",
                session_id
            ))
        })?;
        sequence.stage_decode_candidate(token_id, logit)?;
        if !self.decode_ready.iter().any(|queued| *queued == session_id) {
            self.decode_ready.push_back(session_id);
        }
        Ok(())
    }

    /// Stage a greedy candidate after runtime has explicitly correlated neutral
    /// output back to a service-level session.
    pub fn stage_greedy_decode_from_logits(
        &mut self,
        session_id: SessionId,
        logits: &LogitsOutput,
    ) -> Result<bool> {
        let Some(candidate) = greedy_candidate(logits) else {
            return Ok(false);
        };
        self.stage_decode_candidate(session_id, candidate.token_id, Some(candidate.logit))?;
        Ok(true)
    }

    pub fn next_decode_action(&mut self) -> Result<Option<SchedulerAction>> {
        let mut actions = Vec::new();
        while actions.len() < self.config.max_decode_batch {
            let Some(session_id) = self.decode_ready.pop_front() else {
                break;
            };
            let Some(sequence) = self.active.get(&session_id) else {
                continue;
            };
            let Some(token_id) = sequence.next_decode_token else {
                continue;
            };
            actions.push(DecodeAction::from_sequence(sequence, token_id));
        }

        if actions.is_empty() {
            Ok(None)
        } else {
            Ok(Some(SchedulerAction::DecodeBatch(actions)))
        }
    }

    pub fn commit_decode_action(&mut self, action: &DecodeAction) -> Result<()> {
        let sequence = self.active.get_mut(&action.session_id).ok_or_else(|| {
            Error::Internal(format!(
                "cannot commit decode for inactive session {:?}",
                action.session_id
            ))
        })?;
        if sequence.position != action.position {
            return Err(Error::Internal(format!(
                "decode position mismatch for session {:?}: sequence at {}, action at {}",
                action.session_id, sequence.position, action.position
            )));
        }
        if sequence.kv_handle != action.kv_handle {
            return Err(Error::Internal(format!(
                "decode KV handle mismatch for session {:?}: sequence {:?}, action {:?}",
                action.session_id, sequence.kv_handle, action.kv_handle
            )));
        }
        sequence.commit_staged_decode_token(action.token_id)
    }

    pub fn commit_decode_batch(&mut self, actions: &[DecodeAction]) -> Result<usize> {
        for action in actions {
            self.commit_decode_action(action)?;
        }
        Ok(actions.len())
    }

    /// Commit scheduler-owned state after an action has been executed by a backend.
    ///
    /// This updates prefill cursors or committed decode tokens; logits/output
    /// staging remains explicit after runtime correlation so sampling policy stays
    /// separate from state commits.
    pub fn commit_action(&mut self, action: &SchedulerAction) -> Result<usize> {
        match action {
            SchedulerAction::PrefillChunk(prefill) => {
                self.commit_prefill_action(prefill)?;
                Ok(prefill.token_range.len())
            }
            SchedulerAction::DecodeBatch(actions) => self.commit_decode_batch(actions),
            SchedulerAction::Finish { .. } | SchedulerAction::Cancel { .. } => Ok(0),
        }
    }

    pub fn finish_sequence<C>(
        &mut self,
        session_id: SessionId,
        reason: SequenceFinishReason,
        kv_cache: &mut C,
    ) -> Result<SchedulerAction>
    where
        C: SequenceKvCache,
    {
        let mut sequence = self.remove_active_sequence(session_id)?;
        if let Some(handle) = sequence.kv_handle {
            if let Err(error) = kv_cache.free_sequence(handle) {
                sequence.mark_error();
                self.failed.push(sequence);
                return Err(error);
            }
            sequence.clear_kv();
        }
        sequence.mark_finished(reason);
        let action = SchedulerAction::Finish {
            request_id: sequence.request_id,
            session_id,
            reason,
        };
        self.finished.push(sequence);
        Ok(action)
    }

    pub fn cancel_sequence<C>(
        &mut self,
        session_id: SessionId,
        kv_cache: &mut C,
    ) -> Result<SchedulerAction>
    where
        C: SequenceKvCache,
    {
        let mut sequence = self.remove_active_sequence(session_id)?;
        if let Some(handle) = sequence.kv_handle {
            if let Err(error) = kv_cache.free_sequence(handle) {
                sequence.mark_error();
                self.failed.push(sequence);
                return Err(error);
            }
            sequence.clear_kv();
        }
        sequence.mark_cancelled();
        let action = SchedulerAction::Cancel {
            request_id: sequence.request_id,
            session_id,
        };
        self.cancelled.push(sequence);
        Ok(action)
    }

    /// Remove one sequence from scheduling after backend execution may have
    /// partially committed state. Metadata KV is released when possible; a failed
    /// release leaves the handle attached to the terminal error record.
    pub fn fail_sequence<C>(&mut self, session_id: SessionId, kv_cache: &mut C) -> Result<()>
    where
        C: SequenceKvCache,
    {
        let mut sequence = self.remove_active_sequence(session_id)?;
        sequence.mark_error();
        let release_result = if let Some(handle) = sequence.kv_handle {
            match kv_cache.free_sequence(handle) {
                Ok(()) => {
                    sequence.clear_kv();
                    Ok(())
                }
                Err(error) => Err(error),
            }
        } else {
            Ok(())
        };
        self.failed.push(sequence);
        release_result
    }

    /// Fail every sequence referenced by an action. Cleanup is attempted for all
    /// rows even if one metadata KV release fails.
    pub fn fail_action<C>(&mut self, action: &SchedulerAction, kv_cache: &mut C) -> Result<usize>
    where
        C: SequenceKvCache,
    {
        let session_ids: Vec<SessionId> = match action {
            SchedulerAction::PrefillChunk(prefill) => vec![prefill.session_id],
            SchedulerAction::DecodeBatch(actions) => {
                actions.iter().map(|action| action.session_id).collect()
            }
            SchedulerAction::Finish { .. } | SchedulerAction::Cancel { .. } => Vec::new(),
        };

        let mut failed = 0;
        let mut seen = Vec::new();
        let mut first_error = None;
        for session_id in session_ids {
            if seen.contains(&session_id) {
                continue;
            }
            seen.push(session_id);
            if !self.active.contains_key(&session_id) {
                continue;
            }
            match self.fail_sequence(session_id, kv_cache) {
                Ok(()) => failed += 1,
                Err(error) => {
                    failed += 1;
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
        }

        match first_error {
            Some(error) => Err(error),
            None => Ok(failed),
        }
    }

    fn resolve_session_id(&mut self, requested: Option<SessionId>) -> SessionId {
        if let Some(session_id) = requested {
            return session_id;
        }
        loop {
            let session_id = SessionId(self.next_session_id);
            self.next_session_id = self.next_session_id.saturating_add(1);
            if !self.active.contains_key(&session_id) {
                return session_id;
            }
        }
    }

    fn remove_active_sequence(&mut self, session_id: SessionId) -> Result<SequenceState> {
        self.remove_from_queue(session_id, QueueKind::Prefill);
        self.remove_from_queue(session_id, QueueKind::Decode);
        self.active.remove(&session_id).ok_or_else(|| {
            Error::Internal(format!(
                "cannot remove inactive resident session {:?}",
                session_id
            ))
        })
    }

    fn remove_from_queue(&mut self, session_id: SessionId, queue: QueueKind) {
        let target = match queue {
            QueueKind::Prefill => &mut self.prefill_queue,
            QueueKind::Decode => &mut self.decode_ready,
        };
        target.retain(|queued| *queued != session_id);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueueKind {
    Prefill,
    Decode,
}

pub(crate) fn greedy_candidate(logits: &LogitsOutput) -> Option<TokenLogit> {
    match logits {
        LogitsOutput::TopK(topk) => topk.first().copied(),
        LogitsOutput::Full(logits) => logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, logit)| TokenLogit {
                token_id: index as u32,
                logit,
            }),
    }
}

#[cfg(test)]
mod tests {
    use ferrule_common::execution::LogitsOutput;
    use ferrule_model::TokenLogit;

    use crate::cache::{
        KvCacheLayout, KvHandle, KvLayerView, MultiSessionKvCache, SequenceKvCache,
    };
    use crate::sampling::SamplingConfig;
    use crate::scheduling::RequestId;

    use super::*;

    fn request(id: u64, tokens: Vec<u32>) -> GenerateRequest {
        GenerateRequest {
            id: RequestId(id),
            session_id: None,
            prompt_tokens: tokens,
            sampling: SamplingConfig::greedy(),
            max_new_tokens: 16,
            stop: Vec::new(),
        }
    }

    struct FailingFreeKvCache;

    impl SequenceKvCache for FailingFreeKvCache {
        fn storage_layout(&self) -> KvCacheLayout {
            KvCacheLayout::SingleContiguous
        }

        fn layer_count(&self) -> usize {
            1
        }

        fn entry_dim(&self) -> usize {
            1
        }

        fn sequence_capacity(&self, _handle: KvHandle) -> usize {
            4
        }

        fn alloc_sequence(&mut self) -> Result<KvHandle> {
            Ok(KvHandle(0))
        }

        fn free_sequence(&mut self, _handle: KvHandle) -> Result<()> {
            Err(Error::Internal("simulated KV free failure".into()))
        }

        fn append_to_sequence(
            &mut self,
            _handle: KvHandle,
            _layer: usize,
            _pos: usize,
            _k: &[f32],
            _v: &[f32],
        ) -> Result<()> {
            Ok(())
        }

        fn layer_view(&self, _handle: KvHandle, _layer: usize) -> Result<KvLayerView<'_>> {
            Err(Error::Internal("unused failing cache layer view".into()))
        }

        fn sequence_len(&self, _handle: KvHandle) -> usize {
            0
        }
    }

    #[test]
    fn resident_scheduler_admits_prefills_decodes_and_finishes() {
        let mut scheduler = ResidentScheduler::new(ResidentSchedulerConfig {
            prefill_chunk_size: 2,
            max_active_sequences: 2,
            max_decode_batch: 2,
        });
        let mut kv = MultiSessionKvCache::new(1, 1, 8, 2);
        scheduler.submit(request(10, vec![1, 2, 3]));

        let SchedulerAction::PrefillChunk(first) = scheduler
            .next_prefill_action(&mut kv)
            .unwrap()
            .expect("first prefill action")
        else {
            panic!("expected prefill action");
        };
        assert_eq!(first.request_id, Some(RequestId(10)));
        assert_eq!(first.session_id, SessionId(1));
        assert_eq!(first.tokens, vec![1, 2]);
        assert_eq!(first.position_start, 0);
        assert_eq!(first.kv_handle, Some(crate::cache::KvHandle(0)));
        scheduler.commit_prefill_action(&first).unwrap();
        assert_eq!(scheduler.active_sequence(SessionId(1)).unwrap().position, 2);

        let SchedulerAction::PrefillChunk(last) = scheduler
            .next_prefill_action(&mut kv)
            .unwrap()
            .expect("last prefill action")
        else {
            panic!("expected prefill action");
        };
        assert_eq!(last.tokens, vec![3]);
        assert_eq!(last.position_start, 2);
        scheduler.commit_prefill_action(&last).unwrap();
        assert_eq!(scheduler.prefill_queue_len(), 0);
        assert_eq!(scheduler.active_sequence(SessionId(1)).unwrap().position, 3);

        let logits = LogitsOutput::TopK(vec![TokenLogit {
            token_id: 99,
            logit: 1.0,
        }]);
        assert!(scheduler
            .stage_greedy_decode_from_logits(SessionId(1), &logits)
            .unwrap());
        assert_eq!(scheduler.decode_ready_len(), 1);

        let SchedulerAction::DecodeBatch(actions) = scheduler
            .next_decode_action()
            .unwrap()
            .expect("decode action")
        else {
            panic!("expected decode batch");
        };
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].token_id, 99);
        assert_eq!(actions[0].position, 3);
        scheduler.commit_decode_action(&actions[0]).unwrap();
        let seq = scheduler.active_sequence(SessionId(1)).unwrap();
        assert_eq!(seq.tokens, vec![1, 2, 3, 99]);
        assert_eq!(seq.position, 4);
        assert_eq!(seq.next_decode_token, None);

        let finish = scheduler
            .finish_sequence(SessionId(1), SequenceFinishReason::MaxTokens, &mut kv)
            .unwrap();
        assert!(matches!(finish, SchedulerAction::Finish { .. }));
        assert_eq!(scheduler.active_len(), 0);
        assert_eq!(scheduler.finished_len(), 1);
        assert_eq!(kv.active_count(), 0);
    }

    #[test]
    fn resident_scheduler_batches_ready_decode_rows() {
        let mut scheduler = ResidentScheduler::new(ResidentSchedulerConfig {
            prefill_chunk_size: 4,
            max_active_sequences: 2,
            max_decode_batch: 2,
        });
        let mut kv = MultiSessionKvCache::new(1, 1, 8, 2);
        scheduler.submit(request(1, vec![1]));
        scheduler.submit(request(2, vec![2]));
        assert_eq!(scheduler.admit_waiting(&mut kv).unwrap(), 2);

        for session_id in [SessionId(1), SessionId(2)] {
            let action = scheduler.next_prefill_action(&mut kv).unwrap().unwrap();
            let SchedulerAction::PrefillChunk(prefill) = action else {
                panic!("expected prefill");
            };
            assert_eq!(prefill.session_id, session_id);
            scheduler.commit_prefill_action(&prefill).unwrap();
        }

        scheduler.stage_decode_token(SessionId(1), 10).unwrap();
        scheduler.stage_decode_token(SessionId(2), 20).unwrap();
        let Some(SchedulerAction::DecodeBatch(actions)) = scheduler.next_decode_action().unwrap()
        else {
            panic!("expected decode batch");
        };
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0].token_id, 10);
        assert_eq!(actions[1].token_id, 20);
    }

    #[test]
    fn resident_scheduler_cancel_frees_kv_and_drains_cancelled() {
        let mut scheduler = ResidentScheduler::default();
        let mut kv = MultiSessionKvCache::new(1, 1, 4, 1);
        scheduler.submit(request(7, vec![1]));
        assert!(scheduler.next_prefill_action(&mut kv).unwrap().is_some());
        assert_eq!(kv.active_count(), 1);

        let action = scheduler.cancel_sequence(SessionId(1), &mut kv).unwrap();
        assert!(matches!(action, SchedulerAction::Cancel { .. }));
        assert_eq!(kv.active_count(), 0);
        assert_eq!(scheduler.cancelled_len(), 1);
        let cancelled = scheduler.drain_cancelled();
        assert_eq!(
            cancelled[0].finish_reason,
            Some(SequenceFinishReason::Cancelled)
        );
        assert_eq!(cancelled[0].kv_handle, None);
        assert!(scheduler.is_idle());
    }

    #[test]
    fn greedy_decode_stages_full_logits_argmax() {
        let mut scheduler = ResidentScheduler::default();
        let mut kv = MultiSessionKvCache::new(1, 1, 4, 1);
        scheduler.submit(request(1, vec![1]));
        let SchedulerAction::PrefillChunk(prefill) = scheduler
            .next_prefill_action(&mut kv)
            .unwrap()
            .expect("prefill")
        else {
            panic!("expected prefill");
        };
        scheduler.commit_prefill_action(&prefill).unwrap();

        let logits = LogitsOutput::Full(vec![0.1, 2.0, 1.5]);
        assert!(scheduler
            .stage_greedy_decode_from_logits(SessionId(1), &logits)
            .unwrap());
        let Some(SchedulerAction::DecodeBatch(actions)) = scheduler.next_decode_action().unwrap()
        else {
            panic!("expected decode batch");
        };
        assert_eq!(actions[0].token_id, 1);
    }

    #[test]
    fn finish_and_cancel_preserve_sequence_ownership_when_kv_free_fails() {
        for cancel in [false, true] {
            let mut scheduler = ResidentScheduler::default();
            let mut kv = FailingFreeKvCache;
            scheduler.submit(request(if cancel { 21 } else { 20 }, vec![1]));
            assert!(scheduler.next_prefill_action(&mut kv).unwrap().is_some());

            let result = if cancel {
                scheduler.cancel_sequence(SessionId(1), &mut kv)
            } else {
                scheduler.finish_sequence(SessionId(1), SequenceFinishReason::MaxTokens, &mut kv)
            };
            assert!(format!("{}", result.unwrap_err()).contains("KV free failure"));
            assert_eq!(scheduler.active_len(), 0);
            assert_eq!(scheduler.failed_len(), 1);
            let failed = scheduler.drain_failed();
            assert_eq!(
                failed[0].status,
                super::super::session::SequenceStatus::Error
            );
            assert_eq!(failed[0].kv_handle, Some(KvHandle(0)));
        }
    }

    #[test]
    fn resident_scheduler_waits_when_kv_is_full() {
        let mut scheduler = ResidentScheduler::new(ResidentSchedulerConfig {
            prefill_chunk_size: 4,
            max_active_sequences: 2,
            max_decode_batch: 2,
        });
        let mut kv = MultiSessionKvCache::new(1, 1, 4, 1);
        scheduler.submit(request(1, vec![1]));
        scheduler.submit(request(2, vec![2]));
        assert_eq!(scheduler.admit_waiting(&mut kv).unwrap(), 1);
        assert_eq!(scheduler.waiting_len(), 1);
        assert_eq!(scheduler.active_len(), 1);
    }
}
