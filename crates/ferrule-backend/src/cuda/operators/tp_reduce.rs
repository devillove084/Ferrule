//! INTERNAL ONLY: two sealed members, eager BF16 TP Sum, no model dispatch.
//!
//! NCCL 2.27.5/CUDA12 is dynamically loaded by native core. Private NCCL host/
//! device allocation bounds are unknown: no public hard-cap/resource admission.
//! Nonblocking init/enqueue is advanced once per caller invocation; no loop,
//! watchdog, env mutation, site synchronization or payload host copy. A Pending
//! enqueue MUST be advanced before scheduling downstream compute kernels.
//! Numeric status is sticky and checked at existing quiescence before publishing.
use crate::cuda::{
    operator_owner::{CudaBf16Buffer, CudaOperators, cu},
    operators::OperatorOwner,
    providers::tp_reduce::Rank,
};
use ferrule_common::Result;
use std::sync::Arc;

pub use super::tp_reduce_contract::TpInputCustodyEstimate;
pub use super::tp_reduce_contract::{
    HIDDEN, INPUT_NONFINITE, MAX_COUNT, MAX_ROWS, NCCL_HEADER_SHA256, NCCL_LIBRARY_SHA256,
    NCCL_VERSION, OUTPUT_NONFINITE, PrivateAllocationBound, STORAGE_BYTES, STORAGE_GRANT_BYTES,
    TpProgress, TpReducePlan, TpResourceEstimate,
};
use super::tp_reduce_contract::{Phase, checked_input_custody, invalid};
pub use crate::cuda::providers::tp_reduce::NcclLibrary;
use crate::cuda::runtime::DeviceBuffer;

pub struct InternalTpReduceTwo {
    plan: TpReducePlan,
    ranks: Option<[Rank; 2]>,
    phase: Phase,
    warmed: bool,
    submitted: [bool; 2],
    count: usize,
    validated_output: bool,
    input_custody: Vec<[DeviceBuffer<u16>; 2]>,
    queued_site_capacity: usize,
}
impl InternalTpReduceTwo {
    /// Both owners/backings are preflighted before either native init call.
    /// CommInitRankConfig uses one shared unique ID, and two existing CUDA owners
    /// bound on two scoped threads. Calls are nonblocking; no peer wait/barrier.
    pub fn start(
        owners: [&CudaOperators; 2],
        plan: TpReducePlan,
        library: Arc<NcclLibrary>,
    ) -> Result<Self> {
        let budget = Self::input_custody_estimate(2)?.input_view_bytes;
        Self::start_with_custody(owners, plan, library, 2, budget)
    }
    pub fn input_custody_estimate(queued_sites: usize) -> Result<TpInputCustodyEstimate> {
        Ok(TpInputCustodyEstimate {
            queued_sites,
            input_view_bytes: checked_input_custody(
                queued_sites,
                std::mem::size_of::<[DeviceBuffer<u16>; 2]>(),
                usize::MAX,
            )?,
        })
    }
    /// Preallocate enough joint input views for every site before the caller's
    /// existing quiescence. No hot-path container growth or site status read.
    pub fn start_with_custody(
        owners: [&CudaOperators; 2],
        plan: TpReducePlan,
        library: Arc<NcclLibrary>,
        queued_sites: usize,
        input_custody_host_budget: usize,
    ) -> Result<Self> {
        let slot_bytes = std::mem::size_of::<[DeviceBuffer<u16>; 2]>();
        checked_input_custody(queued_sites, slot_bytes, input_custody_host_budget)?;
        let mut input_custody = Vec::new();
        input_custody
            .try_reserve_exact(queued_sites)
            .map_err(|e| invalid(e.to_string()))?;
        checked_input_custody(
            input_custody.capacity(),
            slot_bytes,
            input_custody_host_budget,
        )?;
        let streams = owners.map(CudaOperators::stream_clone);
        if streams[0].context().ordinal() == streams[1].context().ordinal() {
            return Err(invalid("two distinct CUDA devices required"));
        }
        for stream in &streams {
            cu(stream.context().check_graph_ready())?;
        }
        let id = library.unique_id()?;
        let ranks = [
            Rank::allocate(Arc::clone(&streams[0]), Arc::clone(&library))?,
            Rank::allocate(Arc::clone(&streams[1]), library)?,
        ];
        let mut pair = Self {
            plan,
            ranks: Some(ranks),
            phase: Phase::Initializing,
            warmed: false,
            submitted: [false; 2],
            count: 0,
            validated_output: false,
            input_custody,
            queued_site_capacity: queued_sites,
        };
        #[cfg(test)]
        gpu_tests::initialization_snapshot(&pair, "beforeNccl_afterOwnedGrant");
        let [first, second] = pair.ranks.as_mut().unwrap();
        let initialized = std::thread::scope(|scope| {
            let a = scope.spawn(|| first.init(&id, 0));
            let b = scope.spawn(|| second.init(&id, 1));
            let a = a
                .join()
                .map_err(|_| invalid("rank zero initialization panicked"));
            let b = b
                .join()
                .map_err(|_| invalid("rank one initialization panicked"));
            a.and_then(|v| v)?;
            b.and_then(|v| v)
        });
        #[cfg(test)]
        println!(
            "TP_INIT_THREADS {{\"rank0_joined\":true,\"rank1_joined\":true,\"native_calls_succeeded\":{}}}",
            initialized.is_ok()
        );
        if let Err(error) = initialized {
            pair.abort_unknown();
            return Err(error);
        }
        Ok(pair)
    }
    pub fn plan(&self) -> TpReducePlan {
        self.plan
    }
    pub fn is_unknown(&self) -> bool {
        self.phase == Phase::Unknown
            || self.ranks.as_ref().is_some_and(|ranks| {
                ranks
                    .iter()
                    .any(|r| r.stream.context().graph_completion_unknown())
            })
    }
    fn ranks(&self) -> &[Rank; 2] {
        self.ranks.as_ref().unwrap()
    }
    fn validate_owners(&self, owners: [&CudaOperators; 2]) -> Result<()> {
        if self.is_unknown() {
            for rank in self.ranks() {
                rank.stream.context().mark_graph_unknown();
            }
            return Err(invalid("terminal Unknown"));
        }
        for (rank, op) in self.ranks().iter().zip(owners) {
            if !Arc::ptr_eq(&rank.stream, &op.stream_clone()) {
                return Err(invalid("exact compute-stream owner mismatch"));
            }
            cu(rank.stream.context().check_graph_ready())?;
        }
        Ok(())
    }
    fn quarantine(&mut self) {
        self.phase = Phase::Unknown;
        if let Some(ranks) = &self.ranks {
            for rank in ranks {
                rank.stream.context().mark_graph_unknown();
            }
        }
    }
    /// Cancellation/deadline belongs to the existing caller. Abort neither clears
    /// Unknown nor authorizes destruction of handles/backing/primary contexts.
    pub fn abort_unknown(&mut self) {
        self.quarantine();
        for rank in self.ranks() {
            rank.request_abort();
        }
    }
    fn post_enqueue<T>(&mut self, result: Result<T>) -> Result<T> {
        if result.is_err() {
            self.quarantine();
        }
        result
    }
    fn record_enqueued(&mut self) -> Result<TpProgress> {
        let result = (|| {
            for rank in self.ranks() {
                cu(rank.event.record(&rank.stream))?;
            }
            Ok(TpProgress::Enqueued)
        })();
        self.post_enqueue(result)?;
        self.phase = Phase::InFlight;
        Ok(TpProgress::Enqueued)
    }
    /// One-shot init/submission progress, no busy polling. Warmup always uses
    /// rows=8/count=16384 before any real invocation is admitted.
    pub fn advance(&mut self, owners: [&CudaOperators; 2]) -> Result<TpProgress> {
        self.validate_owners(owners)?;
        match self.phase {
            Phase::Initializing => {
                let result = (|| {
                    let a = self.ranks()[0].async_status()?;
                    let b = self.ranks()[1].async_status()?;
                    Ok(a && b)
                })();
                if !self.post_enqueue(result)? {
                    return Ok(TpProgress::Pending);
                }
                #[cfg(test)]
                gpu_tests::initialization_snapshot(self, "afterinit_async_ready_beforeWarm");
                self.phase = Phase::Submitting;
                self.count = MAX_COUNT;
                let result = (|| {
                    let a = owners[0].submit_operator(2, |_| self.ranks()[0].warm())?;
                    let b = owners[1].submit_operator(2, |_| self.ranks()[1].warm())?;
                    Ok([a, b])
                })();
                self.submitted = self.post_enqueue(result)?;
            }
            Phase::Submitting => {
                let result = (|| {
                    let mut completed = self.submitted;
                    for (i, ready) in completed.iter_mut().enumerate() {
                        if !*ready {
                            *ready = self.ranks()[i].progress()?;
                        }
                    }
                    Ok(completed)
                })();
                self.submitted = self.post_enqueue(result)?;
            }
            Phase::InFlight => return Ok(TpProgress::Enqueued),
            Phase::Ready => return Ok(TpProgress::Complete),
            Phase::Rejected | Phase::Unknown => return Err(invalid("terminal reduction failure")),
        }
        if self.submitted == [true; 2] {
            self.record_enqueued()
        } else {
            Ok(TpProgress::Pending)
        }
    }
    /// No site fence or payload D2H. Both inputs must be contiguous [rows,2048],
    /// on the exact original compute owners. Custody is installed before enqueue.
    pub fn enqueue(
        &mut self,
        owners: [&CudaOperators; 2],
        inputs: [&CudaBf16Buffer; 2],
        rows: usize,
    ) -> Result<TpProgress> {
        self.validate_owners(owners)?;
        if !matches!(self.phase, Phase::Ready | Phase::InFlight) || !self.warmed {
            return Err(invalid("warmup/previous host enqueue not proven"));
        }
        if self.input_custody.len() >= self.queued_site_capacity {
            return Err(invalid(
                "preallocated joint input custody exhausted; checkpoint at existing quiescence",
            ));
        }
        let count = self.plan.count(rows)?;
        for (rank, input) in self.ranks().iter().zip(inputs) {
            let input = input.as_device_buffer();
            cu(input.check_context(rank.stream.context(), "TP reduction input"))?;
            if input.len() != count {
                return Err(invalid("input byte count/contiguous shape mismatch"));
            }
            let start = input.cu_deviceptr();
            let backing = rank.storage.cu_deviceptr();
            if start < backing + STORAGE_BYTES as u64 && backing < start + input.num_bytes() as u64
            {
                return Err(invalid("input aliases communicator workspace"));
            }
        }
        let custody = [
            cu(inputs[0].as_device_buffer().slice(0, count))?,
            cu(inputs[1].as_device_buffer().slice(0, count))?,
        ];
        for (rank, input) in self.ranks.as_mut().unwrap().iter_mut().zip(&custody) {
            rank.input = Some(cu(input.slice(0, count))?);
        }
        self.input_custody.push(custody);
        self.phase = Phase::Submitting;
        self.count = count;
        self.validated_output = false;
        let result = (|| {
            let a = owners[0].submit_operator(2, |_| self.ranks()[0].gather(count))?;
            let b = owners[1].submit_operator(2, |_| self.ranks()[1].gather(count))?;
            Ok([a, b])
        })();
        self.submitted = self.post_enqueue(result)?;
        if self.submitted == [true; 2] {
            self.record_enqueued()
        } else {
            Ok(TpProgress::Pending)
        }
    }
    /// Polls existing reduction-completion events, then reads just 4 bytes/rank.
    /// No synchronization or loop. Rejected numeric output can NEVER be published.
    pub fn check_at_quiescence(&mut self, owners: [&CudaOperators; 2]) -> Result<TpProgress> {
        self.validate_owners(owners)?;
        if self.phase != Phase::InFlight {
            return Err(invalid("nothing enqueued for completion"));
        }
        let result = (|| {
            let mut complete = true;
            for rank in self.ranks() {
                complete &= rank.async_status()?;
                complete &= cu(rank.event.query())?;
            }
            Ok(complete)
        })();
        if !self.post_enqueue(result)? {
            return Ok(TpProgress::Pending);
        }
        let result = (|| {
            let a = self.ranks()[0].device_status()?;
            let b = self.ranks()[1].device_status()?;
            Ok(a | b)
        })();
        let flags = self.post_enqueue(result)?;
        for rank in self.ranks.as_mut().unwrap() {
            rank.input = None;
        }
        self.input_custody.clear();
        if flags != 0 {
            self.phase = Phase::Rejected;
            return Err(invalid(format!(
                "nonfinite input/output rejected (sticky flags {flags:#x})"
            )));
        }
        self.validated_output = self.warmed;
        self.warmed = true;
        self.phase = Phase::Ready;
        Ok(TpProgress::Complete)
    }
    pub fn output(&self, member_index: usize) -> Result<CudaBf16Buffer> {
        if self.phase != Phase::Ready || !self.validated_output || self.is_unknown() {
            return Err(invalid("output is not validated for publication"));
        }
        self.output_view(member_index)
    }
    /// Borrow provisional data only for subsequent kernels on its original stream.
    /// # Safety
    /// No externally visible publication or reuse until joint status validation;
    /// retain this pair through all consumers. Unknown invalidates all consumers.
    pub unsafe fn tentative_output(&self, member_index: usize) -> Result<CudaBf16Buffer> {
        if self.phase != Phase::InFlight || self.is_unknown() {
            return Err(invalid("enqueue not complete"));
        }
        self.output_view(member_index)
    }
    fn output_view(&self, member_index: usize) -> Result<CudaBf16Buffer> {
        let rank = self
            .ranks()
            .get(member_index)
            .ok_or_else(|| invalid("member index must be 0 or 1"))?;
        Ok(CudaBf16Buffer::from_device_buffer(cu(rank
            .storage
            .slice(3 * MAX_COUNT, self.count))?))
    }
}
impl Drop for InternalTpReduceTwo {
    fn drop(&mut self) {
        // No waiting on a failed/missing peer. The native destructor also queries
        // the full stream tail (including downstream consumers) before destruction.
        let proof = !self.is_unknown()
            && self.phase.can_release(true)
            && self.ranks().iter().all(|rank| {
                matches!(rank.event.query(), Ok(true)) && matches!(rank.async_status(), Ok(true))
            });
        if !proof {
            self.quarantine();
            std::mem::forget(self.ranks.take());
            std::mem::forget(std::mem::take(&mut self.input_custody));
            return;
        }
        for rank in self.ranks.as_mut().unwrap() {
            if rank.destroy().is_err() {
                self.quarantine();
                std::mem::forget(self.ranks.take());
                std::mem::forget(std::mem::take(&mut self.input_custody));
                return;
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/tp_reduce/gpu.rs"]
pub(crate) mod gpu_tests;
