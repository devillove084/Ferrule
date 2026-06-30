//! Request scheduler — simple FIFO with priority levels.
//!
//! Queues: waiting prefill, running decode, finished.
//! Designed to be replaced later with continuous batching.

#[allow(unused_imports)]
use crate::session::{GenerateRequest, RequestId, SequenceState, SequenceStatus, SessionId};
use std::collections::VecDeque;

/// Maximum tokens to process in one prefill chunk.
/// Larger = higher throughput but worse latency for concurrent decodes.
pub const DEFAULT_CHUNK_SIZE: usize = 256;

// ── Continuous batching types ──────────────────────────────────────────

/// A batch of sequences being decoded concurrently.
#[derive(Debug, Clone)]
pub struct DecodeBatch {
    pub sequences: Vec<SequenceState>,
}

impl DecodeBatch {
    pub fn new(sequences: Vec<SequenceState>) -> Self {
        Self { sequences }
    }

    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }
}

/// A request being incrementally prefilled.
#[derive(Debug, Clone)]
pub struct ChunkedPrefill {
    pub request: GenerateRequest,
    /// How many prompt tokens have been processed so far.
    pub consumed: usize,
    /// Total prompt tokens.
    pub total: usize,
    /// Chunk size for this request.
    pub chunk_size: usize,
}

impl ChunkedPrefill {
    pub fn new(request: GenerateRequest, chunk_size: usize) -> Self {
        let total = request.prompt_tokens.len();
        Self {
            request,
            consumed: 0,
            total,
            chunk_size,
        }
    }

    /// Return the next chunk of token IDs, or None if done.
    pub fn next_chunk(&mut self) -> Option<&[u32]> {
        if self.consumed >= self.total {
            return None;
        }
        let end = (self.consumed + self.chunk_size).min(self.total);
        let chunk = &self.request.prompt_tokens[self.consumed..end];
        self.consumed = end;
        Some(chunk)
    }

    pub fn is_done(&self) -> bool {
        self.consumed >= self.total
    }
    pub fn remaining(&self) -> usize {
        self.total.saturating_sub(self.consumed)
    }
}

/// Scheduler state for a single inference backend.
#[derive(Debug, Default)]
pub struct Scheduler {
    /// Requests waiting to be prefilled.
    waiting: VecDeque<GenerateRequest>,
    /// Requests currently being decoded (supports multiple concurrent).
    running: VecDeque<GenerateRequest>,
    /// Finished requests pending result collection.
    finished: Vec<GenerateRequest>,
    /// Maximum concurrent decode requests.
    max_concurrent: usize,
    /// Total requests ever submitted.
    total_submitted: u64,
    /// Running sequences for batch decode stepping.
    running_sequences: Vec<SequenceState>,
}

impl Scheduler {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            max_concurrent,
            ..Default::default()
        }
    }

    /// Submit a request to the waiting queue.
    pub fn submit(&mut self, req: GenerateRequest) {
        self.total_submitted += 1;
        self.waiting.push_back(req);
    }

    /// Try to admit requests from waiting → running.
    /// Returns the number admitted.
    pub fn schedule(&mut self) -> usize {
        let slots = self.max_concurrent.saturating_sub(self.running.len());
        let mut admitted = 0;
        while admitted < slots {
            if let Some(req) = self.waiting.pop_front() {
                self.running.push_back(req);
                admitted += 1;
            } else {
                break;
            }
        }
        admitted
    }

    /// Mark a running request as finished.
    pub fn finish(&mut self, id: RequestId) {
        if let Some(pos) = self.running.iter().position(|r| r.id == id) {
            let req = self.running.remove(pos).unwrap();
            self.finished.push(req);
        }
    }

    /// Cancel a request (remove from any queue).
    pub fn cancel(&mut self, id: RequestId) -> bool {
        // Check running
        if let Some(pos) = self.running.iter().position(|r| r.id == id) {
            self.running.remove(pos);
            return true;
        }
        // Check waiting
        let len_before = self.waiting.len();
        self.waiting.retain(|r| r.id != id);
        if self.waiting.len() < len_before {
            return true;
        }
        false
    }

    /// Take all finished results.
    pub fn drain_finished(&mut self) -> Vec<GenerateRequest> {
        std::mem::take(&mut self.finished)
    }

    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    pub fn running_id(&self) -> Option<RequestId> {
        self.running.front().map(|r| r.id)
    }

    pub fn running_ids(&self) -> Vec<RequestId> {
        self.running.iter().map(|r| r.id).collect()
    }

    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    pub fn total_submitted(&self) -> u64 {
        self.total_submitted
    }

    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    /// Set the running sequences for batch decode stepping.
    pub fn set_running_sequences(&mut self, sequences: Vec<SequenceState>) {
        self.running_sequences = sequences;
    }

    pub fn running_sequences(&self) -> &[SequenceState] {
        &self.running_sequences
    }

    /// Run one decode step for all running sequences.
    /// Calls `decode_one` on each sequence; if it returns true, the
    /// sequence is considered finished and collected.
    /// Returns sequences that finished this step.
    pub fn step_decode<F>(&mut self, mut decode_one: F) -> Vec<SequenceState>
    where
        F: FnMut(&mut SequenceState) -> bool,
    {
        let mut finished = Vec::new();
        let mut still_running = Vec::new();

        for mut seq in std::mem::take(&mut self.running_sequences) {
            if decode_one(&mut seq) {
                seq.status = SequenceStatus::Finished;
                finished.push(seq);
            } else {
                still_running.push(seq);
            }
        }
        self.running_sequences = still_running;

        // Also mark corresponding requests as finished
        for seq in &finished {
            // Find matching running request by session_id if available
            if let Some(pos) = self
                .running
                .iter()
                .position(|r| r.session_id == Some(seq.session_id))
            {
                let req = self.running.remove(pos).unwrap();
                self.finished.push(req);
            }
        }

        finished
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::SamplingConfig;

    fn req(id: u64) -> GenerateRequest {
        GenerateRequest {
            id: RequestId(id),
            session_id: None,
            prompt_tokens: vec![1, 2, 3],
            sampling: SamplingConfig::greedy(),
            max_new_tokens: 16,
            stop: vec![],
        }
    }

    fn seq(id: u64) -> SequenceState {
        let mut s = SequenceState::new(SessionId(id));
        s.set_prompt(&[1, 2, 3]);
        s
    }

    #[test]
    fn submit_and_schedule_single() {
        let mut s = Scheduler::new(1);
        s.submit(req(1));
        assert_eq!(s.waiting_len(), 1);
        assert!(s.running_id().is_none());

        let admitted = s.schedule();
        assert_eq!(admitted, 1);
        assert_eq!(s.waiting_len(), 0);
        assert_eq!(s.running_id(), Some(RequestId(1)));
    }

    #[test]
    fn schedule_respects_concurrency() {
        let mut s = Scheduler::new(1);
        s.submit(req(1));
        s.submit(req(2));
        let admitted = s.schedule();
        assert_eq!(admitted, 1);
        assert_eq!(s.running_id(), Some(RequestId(1)));
        assert_eq!(s.waiting_len(), 1);
    }

    #[test]
    fn schedule_multiple_concurrent() {
        let mut s = Scheduler::new(3);
        s.submit(req(1));
        s.submit(req(2));
        s.submit(req(3));
        s.submit(req(4));
        let admitted = s.schedule();
        assert_eq!(admitted, 3);
        assert_eq!(s.running_len(), 3);
        assert_eq!(s.waiting_len(), 1);
        let ids = s.running_ids();
        assert!(ids.contains(&RequestId(1)));
        assert!(ids.contains(&RequestId(2)));
        assert!(ids.contains(&RequestId(3)));
    }

    #[test]
    fn finish_frees_slot() {
        let mut s = Scheduler::new(1);
        s.submit(req(1));
        s.schedule();
        s.finish(RequestId(1));
        assert!(s.running_id().is_none());
        assert_eq!(s.drain_finished().len(), 1);

        // Next can be scheduled
        s.submit(req(2));
        s.schedule();
        assert_eq!(s.running_id(), Some(RequestId(2)));
    }

    #[test]
    fn cancel_from_waiting() {
        let mut s = Scheduler::new(1);
        s.submit(req(1));
        s.submit(req(2));
        assert!(s.cancel(RequestId(2)));
        assert_eq!(s.waiting_len(), 1);
    }

    #[test]
    fn cancel_from_running() {
        let mut s = Scheduler::new(1);
        s.submit(req(1));
        s.schedule();
        assert!(s.cancel(RequestId(1)));
        assert!(s.running_id().is_none());
    }

    #[test]
    fn drain_only_returns_finished() {
        let mut s = Scheduler::new(1);
        s.submit(req(1));
        s.submit(req(2));
        s.schedule(); // req(1) running
        s.finish(RequestId(1));
        s.schedule(); // req(2) running
        s.finish(RequestId(2));

        let done = s.drain_finished();
        assert_eq!(done.len(), 2);
        assert_eq!(s.drain_finished().len(), 0); // drained
    }

    #[test]
    fn idle_detection() {
        let mut s = Scheduler::new(1);
        assert!(s.is_idle());
        s.submit(req(1));
        assert!(!s.is_idle());
        s.schedule();
        assert!(!s.is_idle());
        s.finish(RequestId(1));
        assert!(s.is_idle());
    }

    #[test]
    fn total_submitted_counter() {
        let mut s = Scheduler::new(2);
        s.submit(req(1));
        s.submit(req(2));
        assert_eq!(s.total_submitted(), 2);
    }

    #[test]
    fn step_decode_finishes_sequences() {
        let mut s = Scheduler::new(2);
        let seq1 = seq(1);
        let seq2 = seq(2);
        assert_eq!(seq1.status, SequenceStatus::Running);
        assert_eq!(seq2.status, SequenceStatus::Running);
        s.set_running_sequences(vec![seq1, seq2]);

        // Decode function: mark sequence 2 as finished
        let finished = s.step_decode(|seq| seq.session_id == SessionId(2));

        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].session_id, SessionId(2));
        assert_eq!(finished[0].status, SequenceStatus::Finished);
        // Sequence 1 should still be running
        assert_eq!(s.running_sequences().len(), 1);
        assert_eq!(s.running_sequences()[0].session_id, SessionId(1));
    }

    #[test]
    fn step_decode_all_finish() {
        let mut s = Scheduler::new(2);
        s.set_running_sequences(vec![seq(1), seq(2), seq(3)]);

        let finished = s.step_decode(|_seq| true);

        assert_eq!(finished.len(), 3);
        assert!(s.running_sequences().is_empty());
    }

    #[test]
    fn step_decode_none_finish() {
        let mut s = Scheduler::new(2);
        s.set_running_sequences(vec![seq(1), seq(2)]);

        let finished = s.step_decode(|_seq| false);

        assert_eq!(finished.len(), 0);
        assert_eq!(s.running_sequences().len(), 2);
    }

    #[test]
    fn step_decode_empty_batch() {
        let mut s = Scheduler::new(2);
        let finished = s.step_decode(|_seq| true);
        assert!(finished.is_empty());
    }

    #[test]
    fn decode_batch_construction() {
        let batch = DecodeBatch::new(vec![seq(1), seq(2)]);
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());

        let empty = DecodeBatch::new(vec![]);
        assert!(empty.is_empty());
    }
}

// ── Chunked prefill tests ──────────────────────────────────────────────

#[cfg(test)]
mod chunked_tests {
    use super::*;
    use crate::sampler::SamplingConfig;

    fn long_req(id: u64, prompt_len: usize) -> GenerateRequest {
        GenerateRequest {
            id: RequestId(id),
            session_id: None,
            prompt_tokens: (0..prompt_len as u32).collect(),
            sampling: SamplingConfig::greedy(),
            max_new_tokens: 16,
            stop: vec![],
        }
    }

    #[test]
    fn chunked_prefill_splits_prompt() {
        let req = long_req(1, 500);
        let mut cp = ChunkedPrefill::new(req, 200);
        assert!(!cp.is_done());
        assert_eq!(cp.remaining(), 500);

        let c1 = cp.next_chunk().unwrap();
        assert_eq!(c1.len(), 200);
        assert_eq!(c1[0], 0);
        assert_eq!(c1[199], 199);
        assert!(!cp.is_done());
        assert_eq!(cp.remaining(), 300);

        let c2 = cp.next_chunk().unwrap();
        assert_eq!(c2.len(), 200);
        assert!(!cp.is_done());

        let c3 = cp.next_chunk().unwrap();
        assert_eq!(c3.len(), 100);
        assert!(cp.is_done());
        assert_eq!(cp.remaining(), 0);

        assert!(cp.next_chunk().is_none());
    }

    #[test]
    fn chunked_prefill_small_prompt() {
        let req = long_req(2, 5);
        let mut cp = ChunkedPrefill::new(req, 256);
        let c = cp.next_chunk().unwrap();
        assert_eq!(c.len(), 5);
        assert!(cp.is_done());
    }

    #[test]
    fn chunked_prefill_empty_prompt() {
        let req = long_req(3, 0);
        let mut cp = ChunkedPrefill::new(req, 256);
        assert!(cp.is_done());
        assert!(cp.next_chunk().is_none());
    }
}

// ── Preemption ───────────────────────────────────────────────────────────

/// Policy for how to free KV cache blocks when under memory pressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreemptionPolicy {
    /// Never preempt.
    Never,
    /// Preempt the sequence with fewest generated tokens (swap to CPU/disk).
    SwapLeastGenerated,
    /// Recompute: drop the sequence entirely, restart later.
    Recompute,
}

// ── BatchedScheduler ────────────────────────────────────────────────────

/// Scheduler that manages a batch of sequences with preemption support.
#[derive(Debug)]
pub struct BatchedScheduler {
    /// Maximum sequences in one decode batch.
    pub max_batch_size: usize,
    /// Currently active batch.
    active: Vec<SequenceState>,
    /// Waiting queue.
    waiting: VecDeque<SequenceState>,
    /// Finished sequences pending drain.
    finished: Vec<SequenceState>,
    /// Preempted sequences that can be resumed later.
    preempted: Vec<SequenceState>,
    /// Current preemption policy.
    policy: PreemptionPolicy,
    /// Total KV blocks in use (simulated).
    kv_blocks_used: usize,
    /// Maximum KV blocks available.
    max_kv_blocks: usize,
    /// Blocks per sequence (simplified: 1 block = 1 token).
    blocks_per_token: usize,
}

impl BatchedScheduler {
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            max_batch_size,
            active: Vec::new(),
            waiting: VecDeque::new(),
            finished: Vec::new(),
            preempted: Vec::new(),
            policy: PreemptionPolicy::Never,
            kv_blocks_used: 0,
            max_kv_blocks: usize::MAX,
            blocks_per_token: 1,
        }
    }

    /// Set the preemption policy.
    pub fn set_preemption_policy(&mut self, policy: PreemptionPolicy) {
        self.policy = policy;
    }

    /// Set KV cache limits for preemption.
    pub fn set_kv_limits(&mut self, max_blocks: usize, blocks_per_token: usize) {
        self.max_kv_blocks = max_blocks;
        self.blocks_per_token = blocks_per_token;
    }

    /// Submit a sequence. If there's room in the active batch and no
    /// preemption is needed, it is scheduled immediately; otherwise it
    /// goes to the waiting queue.
    pub fn submit(&mut self, seq: SequenceState) {
        let needed = seq.tokens.len() * self.blocks_per_token;
        if self.active.len() < self.max_batch_size
            && self.kv_blocks_used + needed <= self.max_kv_blocks
        {
            self.kv_blocks_used += needed;
            self.active.push(seq);
        } else {
            self.waiting.push_back(seq);
        }
    }

    /// Admit sequences from the waiting queue into the active batch.
    /// Returns the number admitted.
    pub fn schedule(&mut self) -> usize {
        let mut admitted = 0;
        while self.active.len() < self.max_batch_size {
            let Some(seq) = self.waiting.pop_front() else {
                break;
            };
            let needed = seq.tokens.len() * self.blocks_per_token;
            if self.kv_blocks_used + needed > self.max_kv_blocks {
                // Try preemption
                let freed = self.preempt_for_blocks(needed, self.policy);
                if freed < needed {
                    // Not enough room even after preemption; put it back
                    self.waiting.push_front(seq);
                    break;
                }
            }
            self.kv_blocks_used += needed;
            self.active.push(seq);
            admitted += 1;
        }
        admitted
    }

    /// Run one decode step. Calls `decode` on each active sequence; if it
    /// returns true the sequence is finished and moved to the finished list.
    /// The closure receives a mutable reference to the sequence and returns
    /// true when the sequence has completed.
    pub fn step<F>(&mut self, mut decode: F) -> Vec<SequenceState>
    where
        F: FnMut(&mut SequenceState) -> bool,
    {
        let mut newly_finished = Vec::new();
        let mut still_active = Vec::new();

        for mut seq in std::mem::take(&mut self.active) {
            let blocks = seq.tokens.len() * self.blocks_per_token;
            if decode(&mut seq) {
                seq.status = SequenceStatus::Finished;
                self.kv_blocks_used = self.kv_blocks_used.saturating_sub(blocks);
                newly_finished.push(seq);
            } else {
                still_active.push(seq);
            }
        }
        self.active = still_active;
        self.finished.extend(newly_finished.clone());
        newly_finished
    }

    /// Drain and return all finished sequences.
    pub fn drain_finished(&mut self) -> Vec<SequenceState> {
        std::mem::take(&mut self.finished)
    }

    /// Try to make room for `needed_blocks` by preempting sequences.
    /// Returns the number of freed blocks.
    pub fn preempt_for_blocks(&mut self, needed_blocks: usize, policy: PreemptionPolicy) -> usize {
        match policy {
            PreemptionPolicy::Never => 0,
            PreemptionPolicy::SwapLeastGenerated => self.preempt_least_generated(needed_blocks),
            PreemptionPolicy::Recompute => self.preempt_recompute(needed_blocks),
        }
    }

    /// Preempt the active sequence with the fewest generated tokens.
    fn preempt_least_generated(&mut self, _needed_blocks: usize) -> usize {
        if self.active.is_empty() {
            return 0;
        }

        // Find the sequence with the fewest generated tokens
        let idx = self
            .active
            .iter()
            .enumerate()
            .min_by_key(|(_, seq)| seq.generated)
            .map(|(i, _)| i);

        if let Some(i) = idx {
            let mut seq = self.active.remove(i);
            let blocks = seq.tokens.len() * self.blocks_per_token;
            seq.status = SequenceStatus::Cancelled;
            self.kv_blocks_used = self.kv_blocks_used.saturating_sub(blocks);
            self.preempted.push(seq);
            blocks
        } else {
            0
        }
    }

    /// Preempt by recomputing: drop the sequence entirely, it can restart.
    fn preempt_recompute(&mut self, needed_blocks: usize) -> usize {
        if self.active.is_empty() {
            return 0;
        }

        let mut freed = 0;
        let mut still_active = Vec::new();

        for seq in std::mem::take(&mut self.active) {
            if freed >= needed_blocks {
                still_active.push(seq);
            } else {
                let blocks = seq.tokens.len() * self.blocks_per_token;
                freed += blocks;
                self.kv_blocks_used = self.kv_blocks_used.saturating_sub(blocks);
                // Put back at front of waiting queue to restart later
                self.waiting.push_front(seq);
            }
        }
        self.active = still_active;
        freed
    }

    // ── Accessors ──────────────────────────────────────────────────────

    pub fn active_len(&self) -> usize {
        self.active.len()
    }

    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    pub fn finished_len(&self) -> usize {
        self.finished.len()
    }

    pub fn preempted_len(&self) -> usize {
        self.preempted.len()
    }

    pub fn kv_blocks_used(&self) -> usize {
        self.kv_blocks_used
    }

    pub fn is_idle(&self) -> bool {
        self.active.is_empty() && self.waiting.is_empty()
    }
}

// ── BatchedScheduler tests ───────────────────────────────────────────────

#[cfg(test)]
mod batched_tests {
    use super::*;

    fn seq(id: u64, prompt_len: usize, generated: usize) -> SequenceState {
        let mut s = SequenceState::new(SessionId(id));
        let prompt: Vec<u32> = (0..prompt_len as u32).collect();
        s.set_prompt(&prompt);
        if generated > 0 {
            let gen: Vec<u32> = (100..100 + generated as u32).collect();
            s.extend_generated(&gen);
        }
        s
    }

    #[test]
    fn submit_admits_when_room() {
        let mut bs = BatchedScheduler::new(3);
        bs.set_kv_limits(100, 1);
        bs.submit(seq(1, 5, 0));
        bs.submit(seq(2, 5, 0));
        assert_eq!(bs.active_len(), 2);
        assert_eq!(bs.waiting_len(), 0);
        assert_eq!(bs.kv_blocks_used(), 10);
    }

    #[test]
    fn submit_queues_when_full() {
        let mut bs = BatchedScheduler::new(2);
        bs.set_kv_limits(100, 1);
        bs.submit(seq(1, 5, 0));
        bs.submit(seq(2, 5, 0));
        bs.submit(seq(3, 5, 0)); // should queue
        assert_eq!(bs.active_len(), 2);
        assert_eq!(bs.waiting_len(), 1);
    }

    #[test]
    fn schedule_admits_from_waiting() {
        let mut bs = BatchedScheduler::new(3);
        bs.set_kv_limits(100, 1);
        bs.submit(seq(1, 5, 0));
        bs.submit(seq(2, 5, 0));
        bs.submit(seq(3, 5, 0));
        bs.submit(seq(4, 5, 0)); // queued
        assert_eq!(bs.waiting_len(), 1);

        // Finish one active sequence to make room
        bs.step(|seq| seq.session_id == SessionId(1));
        assert_eq!(bs.active_len(), 2);

        let admitted = bs.schedule();
        assert_eq!(admitted, 1);
        assert_eq!(bs.active_len(), 3);
        assert_eq!(bs.waiting_len(), 0);
    }

    #[test]
    fn step_returns_finished() {
        let mut bs = BatchedScheduler::new(4);
        bs.set_kv_limits(100, 1);
        bs.submit(seq(1, 5, 0));
        bs.submit(seq(2, 5, 0));
        bs.submit(seq(3, 5, 0));

        // Finish sequences 1 and 3
        let finished =
            bs.step(|seq| seq.session_id == SessionId(1) || seq.session_id == SessionId(3));

        assert_eq!(finished.len(), 2);
        assert_eq!(bs.active_len(), 1);
        assert_eq!(bs.active()[0].session_id, SessionId(2));

        let drained = bs.drain_finished();
        assert_eq!(drained.len(), 2);
        assert_eq!(bs.drain_finished().len(), 0);
    }

    #[test]
    fn step_updates_kv_blocks_on_finish() {
        let mut bs = BatchedScheduler::new(4);
        bs.set_kv_limits(100, 1);
        bs.submit(seq(1, 5, 0)); // 5 blocks
        bs.submit(seq(2, 10, 0)); // 10 blocks
        assert_eq!(bs.kv_blocks_used(), 15);

        // Finish sequence 1 (frees 5 blocks)
        bs.step(|seq| seq.session_id == SessionId(1));
        assert_eq!(bs.kv_blocks_used(), 10);
    }

    #[test]
    fn preempt_never_does_nothing() {
        let mut bs = BatchedScheduler::new(2);
        bs.set_kv_limits(10, 1);
        bs.submit(seq(1, 5, 0));
        bs.submit(seq(2, 5, 0));
        bs.set_preemption_policy(PreemptionPolicy::Never);

        let freed = bs.preempt_for_blocks(5, PreemptionPolicy::Never);
        assert_eq!(freed, 0);
        assert_eq!(bs.active_len(), 2);
    }

    #[test]
    fn preempt_least_generated_frees_blocks() {
        let mut bs = BatchedScheduler::new(4);
        bs.set_kv_limits(100, 1);
        // Sequence 1: 5 generated tokens (most)
        let mut s1 = SequenceState::new(SessionId(1));
        s1.set_prompt(&(0..10u32).collect::<Vec<_>>());
        s1.extend_generated(&(100..105u32).collect::<Vec<_>>());
        // Sequence 2: 1 generated token (fewest — should be preempted)
        let mut s2 = SequenceState::new(SessionId(2));
        s2.set_prompt(&(0..10u32).collect::<Vec<_>>());
        s2.extend_generated(&[100u32]);
        bs.submit(s1);
        bs.submit(s2);

        let freed = bs.preempt_for_blocks(5, PreemptionPolicy::SwapLeastGenerated);
        assert_eq!(freed, 11); // 10 prompt + 1 generated = 11 tokens
        assert_eq!(bs.active_len(), 1);
        assert_eq!(bs.preempted_len(), 1);
        // The preempted one should be the one with fewer generated tokens
        assert_eq!(bs.preempted()[0].session_id, SessionId(2));
    }

    #[test]
    fn preempt_recompute_drops_and_requeues() {
        let mut bs = BatchedScheduler::new(4);
        bs.set_kv_limits(15, 1);
        bs.submit(seq(1, 5, 0));
        bs.submit(seq(2, 5, 0));
        bs.submit(seq(3, 5, 0));
        assert_eq!(bs.kv_blocks_used(), 15);

        // Need 5 more blocks but only have 15 max
        let freed = bs.preempt_for_blocks(5, PreemptionPolicy::Recompute);
        assert!(freed >= 5);
        // Sequences were requeued to waiting
        assert!(bs.waiting_len() > 0);
        // KV blocks were freed
        assert!(bs.kv_blocks_used() < 15);
    }

    #[test]
    fn batched_scheduler_idle_detection() {
        let mut bs = BatchedScheduler::new(4);
        assert!(bs.is_idle());
        bs.submit(seq(1, 5, 0));
        assert!(!bs.is_idle());
        bs.step(|_seq| true);
        assert!(bs.is_idle());
    }

    #[test]
    fn batched_scheduler_drain_empty() {
        let mut bs = BatchedScheduler::new(4);
        let drained = bs.drain_finished();
        assert!(drained.is_empty());
    }

    #[test]
    fn schedule_with_preemption_for_kv_pressure() {
        let mut bs = BatchedScheduler::new(3);
        bs.set_kv_limits(10, 1);
        bs.set_preemption_policy(PreemptionPolicy::SwapLeastGenerated);
        // Fill KV cache with two sequences (10 blocks total)
        bs.submit(seq(1, 5, 0));
        bs.submit(seq(2, 5, 0));
        // Third sequence is queued because no room
        bs.submit(seq(3, 5, 0));
        assert_eq!(bs.active_len(), 2);
        assert_eq!(bs.waiting_len(), 1);

        // Schedule should preempt and admit
        let admitted = bs.schedule();
        // Should have admitted seq(3) by preempting least-generated
        assert_eq!(admitted, 1);
        assert_eq!(bs.active_len(), 2); // one was preempted, one admitted
    }

    // Accessor helper — not part of BatchedScheduler API, used in tests
    impl BatchedScheduler {
        fn active(&self) -> &[SequenceState] {
            &self.active
        }

        fn preempted(&self) -> &[SequenceState] {
            &self.preempted
        }
    }
}
