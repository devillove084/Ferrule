//! Synchronous three-linear upload using the baseline pageable copy primitive.
//! Six copies share one exact compute-stream completion fence. The driver may
//! block/stage each pageable copy internally; this is not an async/overlap claim.
use crate::cuda::operator_owner::{CudaOperators, cu};
use crate::cuda::operators::linear::CudaNumericFp8Artifact;
use ferrule_common::numeric_fp8::ImmutableValidatedNumericFp8Payload;
use ferrule_common::{Error, Result};
use std::sync::Arc;

// Only owned resources enter this bundle. In production `owner` is the exact
// compute stream, which itself keeps its context alive even after operator Drop.
struct Group<R, D> {
    owner: R,
    payloads: [ImmutableValidatedNumericFp8Payload; 3],
    destinations: [D; 6],
}

#[derive(Clone, Copy)]
enum Phase {
    Copying,
    Proved,
    Unknown,
}

struct Guard<R, D, F: FnMut(&R) -> Result<()>> {
    group: Option<Group<R, D>>,
    fence: F,
    phase: Phase,
}

impl<R, D, F: FnMut(&R) -> Result<()>> Drop for Guard<R, D, F> {
    fn drop(&mut self) {
        if let Some(group) = self.group.take() {
            // The closure is NOT part of retained custody. It may borrow facade
            // counters, whereas the bundle retains only independently owned data.
            crate::cuda::transport::drop_after_proof(group, |group| match self.phase {
                Phase::Proved => true,
                Phase::Copying => match (self.fence)(&group.owner) {
                    Ok(()) => true,
                    Err(error) => {
                        tracing::error!(%error, "grouped numeric FP8 upload unwind cleanup failed; entire group retained");
                        false
                    }
                },
                Phase::Unknown => {
                    tracing::error!(
                        "grouped numeric FP8 upload completion Unknown; entire group retained; restart worker to recover"
                    );
                    false
                }
            });
        }
    }
}

// Resource operations are injectable ONLY here, so CPU contracts exercise the
// production allocation/copy/fence ordering without introducing a public ticket.
fn upload_group<R, D>(
    owner: R,
    payloads: [ImmutableValidatedNumericFp8Payload; 3],
    mut allocate: impl FnMut(usize) -> Result<D>,
    mut copy: impl FnMut(&R, &D, &[u8]) -> Result<()>,
    fence: impl FnMut(&R) -> Result<()>,
) -> Result<Group<R, D>> {
    for payload in &payloads {
        payload
            .layout()
            .validate_lengths(payload.weight_bytes().len(), payload.scale_bytes().len())?;
    }
    let lengths: [usize; 6] = std::array::from_fn(|i| {
        let payload = &payloads[i / 2];
        if i % 2 == 0 {
            payload.weight_bytes().len()
        } else {
            payload.scale_bytes().len()
        }
    });
    // Allocation-only: all six destinations must exist before any DMA. Failure
    // here drops only unsubmitted resources, with no completion fence needed.
    let destinations = [
        allocate(lengths[0])?,
        allocate(lengths[1])?,
        allocate(lengths[2])?,
        allocate(lengths[3])?,
        allocate(lengths[4])?,
        allocate(lengths[5])?,
    ];
    let mut guard = Guard {
        group: Some(Group {
            owner,
            payloads,
            destinations,
        }),
        fence,
        // Installed and marked BEFORE the first possibly submitting driver call.
        phase: Phase::Copying,
    };
    let copied = (0..6).try_for_each(|i| {
        let group = guard.group.as_ref().expect("installed group custody");
        let payload = &group.payloads[i / 2];
        let source = if i % 2 == 0 {
            payload.weight_bytes()
        } else {
            payload.scale_bytes()
        };
        copy(&group.owner, &group.destinations[i], source)
    });
    // Exactly one uncancellable proof attempt, including a partial-copy error.
    // Failed or unwinding proof is terminal: Drop must NEVER retry and heal it.
    guard.phase = Phase::Unknown;
    let completion = (guard.fence)(&guard.group.as_ref().expect("submitted group").owner);
    if completion.is_ok() {
        guard.phase = Phase::Proved;
    }
    match (copied, completion) {
        (Ok(()), Ok(())) => Ok(guard.group.take().expect("completed group")),
        (Err(source), cleanup) => Err(Error::with_cleanup(
            "grouped numeric FP8 upload",
            source,
            cleanup,
        )),
        (Ok(()), Err(error)) => Err(error),
    }
}

impl CudaOperators {
    /// Upload three validated linears in input order, synchronously, using the
    /// SAME pageable H2D primitive and compute stream as the single-linear API.
    /// All six destinations and immutable sources enter custody before any copy;
    /// one actual stream fence covers the whole group (also on copy failure).
    ///
    /// No pinned slab, concatenation buffer, upload event, poll or sleep is used.
    /// Pageable driver copies may themselves block/stage/synchronize internally;
    /// one explicit fence is not a claim of one total native synchronization.
    /// Original E4M3FN bytes and BF16/F32 numeric scale bytes are not converted.
    ///
    /// Failed completion retains the entire group and its payload aliases, with
    /// no retry or publication. Upper owners must quarantine on error, not refund
    /// source credit merely because this method returned. Normal completion
    /// releases source custody before return. Existing upload defaults are unchanged.
    pub fn upload_validated_numeric_fp8_group_sync(
        &self,
        payloads: [ImmutableValidatedNumericFp8Payload; 3],
    ) -> Result<[CudaNumericFp8Artifact; 3]> {
        self.check_capture_safe("synchronous grouped numeric FP8 upload")?;
        cu(self._ctx.check_graph_ready())?;
        let group = upload_group(
            Arc::clone(&self.stream),
            payloads,
            |len| self.uninitialized_device_buffer::<u8>(len),
            |stream, destination, source| {
                cu(destination.copy_from_host(stream, source))?;
                self.counters.add_host_to_device(source.len() as u64);
                Ok(())
            },
            |stream| {
                // Like submitted_copy_cleanup: assertion/failpoint prechecks
                // cannot suppress a fence once a copy may have been submitted.
                self.counters
                    .checked_sync(|| Ok(()), || false, || cu(stream.synchronize()))
            },
        )?;
        let layouts = group.payloads.each_ref().map(|payload| payload.layout());
        let [w0, s0, w1, s1, w2, s2] = group.destinations;
        let artifacts = [
            (layouts[0], w0, s0),
            (layouts[1], w1, s1),
            (layouts[2], w2, s2),
        ]
        .map(|(layout, weight, scales)| {
            CudaNumericFp8Artifact::from_device_buffers(layout, weight, scales)
        });
        for artifact in &artifacts {
            self.counters
                .add_artifact_upload(artifact.storage_bytes() as u64);
        }
        Ok(artifacts)
    }
}

#[cfg(test)]
#[path = "../../../tests/numeric_fp8_group_sync/contract.rs"]
mod contract_tests;
