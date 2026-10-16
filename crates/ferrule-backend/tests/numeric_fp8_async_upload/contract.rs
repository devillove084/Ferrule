//! CPU-only contracts for the exact production custody state machine. No CUDA
//! context, stream, event or allocation is created by any test in this file.
//! The separate `gpu` module is real-driver coverage and explicitly ignored.
use super::*;
use ferrule_common::numeric_fp8::{NumericFp8Layout, NumericFp8ScaleType};
use std::cell::{Cell, RefCell};
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::cuda::operator_owner::numeric_fp8_fixture::payload;

fn payloads() -> [ImmutableValidatedNumericFp8Payload; 3] {
    [
        payload(NumericFp8ScaleType::Bf16, 0x38),
        payload(NumericFp8ScaleType::F32, 0x40),
        payload(NumericFp8ScaleType::Bf16, 0x7e),
    ]
}

#[test]
fn six_ranges_are_exact_contiguous_nonoverlapping_and_include_the_tail() {
    let layout = SlabLayout::new([1, 2, 3, 4, 5, 6]).unwrap();
    assert_eq!(layout.bytes, 21);
    assert_eq!(
        layout.ranges.map(|r| (r.offset, r.len)),
        [(0, 1), (1, 2), (3, 3), (6, 4), (10, 5), (15, 6)]
    );
    for range in layout.ranges {
        assert!(ByteRange::checked(range.offset, range.len, layout.bytes).is_ok());
    }
    assert!(ByteRange::checked(20, 2, 21).is_err());
    assert!(ByteRange::checked(21, 1, 21).is_err());
    assert!(ByteRange::checked(0, 0, 21).is_err());
    assert!(ByteRange::checked(usize::MAX, 1, usize::MAX).is_err());
    assert!(ByteRange::checked(0, 1, usize::MAX).is_err());
    assert!(SlabLayout::new([1, 2, 0, 4, 5, 6]).is_err());
    assert!(SlabLayout::new([usize::MAX, 1, 1, 1, 1, 1]).is_err());
    assert!(SlabLayout::new([isize::MAX as usize, 1, 1, 1, 1, 1]).is_err());
}

#[test]
fn slab_preserves_every_packed_byte_original_scale_dtype_and_128_grid() {
    let payloads = payloads();
    let layout = SlabLayout::for_payloads(&payloads).unwrap();
    assert_eq!(layout.ranges.map(|r| r.len), [4, 8, 4, 16, 4, 8]);
    let mut guarded = vec![0xa5; layout.bytes + 2];
    layout
        .pack(&payloads, &mut guarded[1..layout.bytes + 1])
        .unwrap();
    assert_eq!(guarded[0], 0xa5);
    assert_eq!(guarded[layout.bytes + 1], 0xa5);
    for (i, range) in layout.ranges.iter().enumerate() {
        let payload = &payloads[i / 2];
        assert_eq!(payload.layout().scale_shape().unwrap(), [2, 2]);
        let expected = if i % 2 == 0 {
            payload.weight_bytes()
        } else {
            payload.scale_bytes()
        };
        assert_eq!(
            &guarded[1 + range.offset..1 + range.offset + range.len],
            expected
        );
    }
    assert!(layout.pack(&payloads, &mut guarded).is_err());
    let mut wrong_layout = layout;
    wrong_layout.ranges[5].offset += 1;
    assert!(
        wrong_layout
            .pack(&payloads, &mut guarded[1..layout.bytes + 1])
            .is_err()
    );
}

#[test]
fn admission_charges_pageable_plus_one_whole_pinned_allocation_before_dma() {
    let payloads = payloads();
    let p = payloads.iter().map(|p| p.storage_bytes()).sum::<usize>();
    let charge = CudaNumericFp8UploadHostCharge::for_payloads(&payloads).unwrap();
    assert_eq!(charge.pageable_payload_bytes, p);
    assert_eq!(charge.pinned_allocation_bytes, p);
    assert_eq!(
        charge.ticket_metadata_bytes,
        std::mem::size_of::<CudaNumericFp8UploadTicket>()
    );
    assert_eq!(charge.total_bytes, 2 * p + charge.ticket_metadata_bytes);
    assert!(charge.admit(p).is_err());
    assert!(charge.admit(charge.total_bytes - 1).is_err());
    assert!(charge.admit(charge.total_bytes).is_ok());
    // Full extents remain charged even if the caller retains an Arc alias.
    let aliases = [
        payloads[0].clone(),
        payloads[0].clone(),
        payloads[0].clone(),
    ];
    assert_eq!(
        CudaNumericFp8UploadHostCharge::for_payloads(&aliases)
            .unwrap()
            .pageable_payload_bytes,
        3 * payloads[0].storage_bytes()
    );
    assert!(CudaNumericFp8UploadHostCharge::for_bytes(usize::MAX).is_err());
    assert!(CudaNumericFp8UploadHostCharge::for_bytes(isize::MAX as usize / 2).is_err());
    let expert_charge = CudaNumericFp8UploadHostCharge::for_bytes(3_146_112).unwrap();
    assert_eq!(
        expert_charge.total_bytes,
        6_292_224 + expert_charge.ticket_metadata_bytes
    );
}

#[test]
fn identity_requires_exact_context_and_both_stream_owners_not_equal_ordinals() {
    let owner = Arc::new(0usize);
    let upload = Arc::new(1usize);
    let compute = Arc::new(2usize);
    assert!(exact_owner(
        &owner,
        &upload,
        &compute,
        &owner.clone(),
        &upload.clone(),
        &compute.clone()
    ));
    assert!(!exact_owner(
        &owner,
        &upload,
        &compute,
        &Arc::new(0),
        &upload,
        &compute
    ));
    assert!(!exact_owner(
        &owner,
        &upload,
        &compute,
        &owner,
        &Arc::new(1),
        &compute
    ));
    assert!(!exact_owner(
        &owner,
        &upload,
        &compute,
        &owner,
        &upload,
        &Arc::new(2)
    ));
    assert!(!exact_owner(
        &owner, &upload, &compute, &owner, &compute, &upload
    ));
    assert_eq!(next_generation(0).unwrap(), 1);
    assert_eq!(next_generation(u64::MAX - 1).unwrap(), u64::MAX);
    assert!(next_generation(u64::MAX).is_err());
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Call {
    Copy(usize),
    Record,
    Wait,
    Query,
    FenceEvent,
    FenceStream,
}

#[derive(Default)]
struct State {
    calls: RefCell<Vec<Call>>,
    fail: Cell<Option<Call>>,
    panic: Cell<Option<Call>>,
    ready: Cell<bool>,
    drops: Cell<usize>,
}
impl State {
    fn call(&self, call: Call) -> Result<()> {
        self.calls.borrow_mut().push(call);
        assert_ne!(self.panic.get(), Some(call), "injected driver unwind");
        if self.fail.get() == Some(call) {
            Err(invalid(format!("injected {call:?}")))
        } else {
            Ok(())
        }
    }
}

struct Probe(Rc<State>);
impl Drop for Probe {
    fn drop(&mut self) {
        self.0.drops.set(self.0.drops.get() + 1);
    }
}
struct MockBundle {
    state: Rc<State>,
    _payloads: [ImmutableValidatedNumericFp8Payload; 3],
    // Independently observable source/device/event/stream/context custody.
    _resources: [Probe; 7],
}
impl UploadResources for MockBundle {
    fn copy(&self, i: usize) -> Result<()> {
        self.state.call(Call::Copy(i))
    }
    fn record(&self) -> Result<()> {
        self.state.call(Call::Record)
    }
    fn wait_compute(&self) -> Result<()> {
        self.state.call(Call::Wait)
    }
    fn query(&self) -> Result<bool> {
        self.state.call(Call::Query)?;
        Ok(self.state.ready.get())
    }
    fn fence_event(&self) -> Result<()> {
        self.state.call(Call::FenceEvent)
    }
    fn fence_stream(&self) -> Result<()> {
        self.state.call(Call::FenceStream)
    }
}
fn staged() -> (
    Custody<MockBundle>,
    Rc<State>,
    [ferrule_common::numeric_fp8::NumericFp8PayloadLiveness; 3],
) {
    let state = Rc::new(State::default());
    let payloads = payloads();
    let liveness = payloads.each_ref().map(|p| p.allocation_liveness());
    let bundle = MockBundle {
        state: state.clone(),
        _payloads: payloads,
        _resources: std::array::from_fn(|_| Probe(state.clone())),
    };
    (Custody::new(bundle), state, liveness)
}
fn assert_retained(
    state: &State,
    liveness: &[ferrule_common::numeric_fp8::NumericFp8PayloadLiveness; 3],
) {
    assert_eq!(state.drops.get(), 0);
    assert!(liveness.iter().all(|p| p.is_live()));
}
fn assert_released(
    state: &State,
    liveness: &[ferrule_common::numeric_fp8::NumericFp8PayloadLiveness; 3],
) {
    assert_eq!(state.drops.get(), 7);
    assert!(liveness.iter().all(|p| !p.is_live()));
}

#[test]
fn staging_drop_submits_nothing_and_can_release_without_a_fence() {
    let (custody, state, liveness) = staged();
    assert_retained(&state, &liveness);
    drop(custody);
    assert!(state.calls.borrow().is_empty());
    assert_released(&state, &liveness);
}

#[test]
fn wait_is_only_a_dependency_pending_keeps_everything_and_ready_publishes_once() {
    let (mut custody, state, liveness) = staged();
    assert!(custody.poll().is_err()); // unrecorded events are never queried
    custody.submit().unwrap();
    assert_eq!(
        *state.calls.borrow(),
        [
            Call::Copy(0),
            Call::Copy(1),
            Call::Copy(2),
            Call::Copy(3),
            Call::Copy(4),
            Call::Copy(5),
            Call::Record,
            Call::Wait
        ]
    );
    assert_retained(&state, &liveness);
    assert!(custody.submit().is_err());
    for _ in 0..3 {
        assert!(custody.poll().unwrap().is_none());
        assert_retained(&state, &liveness);
    }
    state.ready.set(true);
    let published = custody.poll().unwrap().unwrap();
    assert_retained(&state, &liveness); // ownership moved, not freed during publication
    assert!(custody.poll().is_err());
    drop(published);
    drop(custody);
    assert_released(&state, &liveness);
    assert_eq!(
        state
            .calls
            .borrow()
            .iter()
            .filter(|&&c| c == Call::Record)
            .count(),
        1
    );
    assert!(
        !state
            .calls
            .borrow()
            .iter()
            .any(|c| matches!(c, Call::FenceEvent | Call::FenceStream))
    );
}

#[test]
fn every_partial_copy_and_record_failure_retains_before_any_cleanup_or_publication() {
    for fault in (0..6).map(Call::Copy).chain([Call::Record]) {
        let (mut custody, state, liveness) = staged();
        state.fail.set(Some(fault));
        assert!(custody.submit().is_err());
        assert_retained(&state, &liveness);
        assert!(custody.poll().is_err());
        assert!(custody.submit().is_err());
        assert_eq!(state.calls.borrow().last(), Some(&fault));
        assert!(!state.calls.borrow().contains(&Call::Query));
        assert!(!state.calls.borrow().contains(&Call::Wait));
        custody.drain().unwrap();
        assert_eq!(state.calls.borrow().last(), Some(&Call::FenceStream));
        assert!(!state.calls.borrow().contains(&Call::FenceEvent));
        assert_released(&state, &liveness);
    }
}

#[test]
fn wait_failure_cannot_publish_even_when_event_is_ready_and_cleanup_uses_exact_event() {
    let (mut custody, state, liveness) = staged();
    state.fail.set(Some(Call::Wait));
    state.ready.set(true);
    assert!(custody.submit().is_err());
    assert_retained(&state, &liveness);
    assert!(custody.poll().is_err());
    custody.drain().unwrap();
    assert_eq!(state.calls.borrow().last(), Some(&Call::FenceEvent));
    assert!(!state.calls.borrow().contains(&Call::Query));
    assert!(!state.calls.borrow().contains(&Call::FenceStream));
    assert_released(&state, &liveness);
}

#[test]
fn query_error_is_sticky_unknown_and_later_readiness_or_fences_cannot_recover() {
    let (mut custody, state, liveness) = staged();
    custody.submit().unwrap();
    state.fail.set(Some(Call::Query));
    assert!(custody.poll().is_err());
    state.fail.set(None);
    state.ready.set(true);
    assert!(custody.poll().is_err());
    assert!(custody.drain().is_err());
    drop(custody);
    assert_eq!(state.calls.borrow().last(), Some(&Call::Query));
    assert_retained(&state, &liveness);
}

#[test]
fn failed_explicit_event_or_stream_proof_is_sticky_and_never_uses_an_unrelated_fence() {
    for fence in [Call::FenceEvent, Call::FenceStream] {
        let (mut custody, state, liveness) = staged();
        if fence == Call::FenceStream {
            state.fail.set(Some(Call::Record));
        }
        let _ = custody.submit();
        state.fail.set(Some(fence));
        assert!(custody.drain().is_err());
        state.fail.set(None);
        assert!(custody.drain().is_err());
        drop(custody);
        assert_eq!(
            state
                .calls
                .borrow()
                .iter()
                .filter(|&&c| matches!(c, Call::FenceEvent | Call::FenceStream))
                .copied()
                .collect::<Vec<_>>(),
            [fence]
        );
        assert_retained(&state, &liveness);
    }
}

#[test]
fn successful_drop_cleanup_uses_event_only_if_recorded_and_stream_only_if_unrecorded() {
    for recorded in [false, true] {
        let (mut custody, state, liveness) = staged();
        if !recorded {
            state.fail.set(Some(Call::Record));
        }
        let _ = custody.submit();
        drop(custody);
        assert_eq!(
            state.calls.borrow().last(),
            Some(&if recorded {
                Call::FenceEvent
            } else {
                Call::FenceStream
            })
        );
        assert_released(&state, &liveness);
    }
}

#[test]
fn failed_or_unwinding_drop_proof_retains_the_entire_bundle() {
    for fence in [Call::FenceEvent, Call::FenceStream] {
        for panic in [false, true] {
            let (mut custody, state, liveness) = staged();
            if fence == Call::FenceStream {
                state.fail.set(Some(Call::Record));
            }
            let _ = custody.submit();
            if panic {
                state.panic.set(Some(fence));
            } else {
                state.fail.set(Some(fence));
            }
            let result = catch_unwind(AssertUnwindSafe(|| drop(custody)));
            assert_eq!(result.is_err(), panic);
            assert_retained(&state, &liveness);
        }
    }
}

#[test]
fn submission_unwind_keeps_custody_and_never_queries_an_unrecorded_event() {
    for fault in [Call::Copy(0), Call::Copy(3), Call::Record, Call::Wait] {
        let (mut custody, state, liveness) = staged();
        state.panic.set(Some(fault));
        assert!(catch_unwind(AssertUnwindSafe(|| custody.submit())).is_err());
        assert_retained(&state, &liveness);
        assert!(custody.poll().is_err());
        state.panic.set(None);
        custody.drain().unwrap();
        assert_eq!(
            state.calls.borrow().last(),
            Some(&if fault == Call::Wait {
                Call::FenceEvent
            } else {
                Call::FenceStream
            })
        );
        assert_released(&state, &liveness);
    }
}

#[test]
fn query_unwind_is_sticky_unknown_and_retains_all_resources() {
    let (mut custody, state, liveness) = staged();
    custody.submit().unwrap();
    state.panic.set(Some(Call::Query));
    assert!(catch_unwind(AssertUnwindSafe(|| custody.poll())).is_err());
    state.panic.set(None);
    state.ready.set(true);
    assert!(custody.poll().is_err());
    drop(custody);
    assert_retained(&state, &liveness);
}

#[test]
fn full_three_linear_geometry_charges_the_actual_bf16_or_f32_packed_extent() {
    for (scale_type, expected) in [
        (NumericFp8ScaleType::Bf16, 3_146_112),
        (NumericFp8ScaleType::F32, 3_146_496),
    ] {
        let payloads = [(512, 2048), (512, 2048), (2048, 512)].map(|(n, k)| {
            let layout = NumericFp8Layout {
                n,
                k,
                row_origin: 0,
                column_origin: 0,
                scale_type,
            };
            let scale = match scale_type {
                NumericFp8ScaleType::Bf16 => 0x3f80u16.to_le_bytes().to_vec(),
                NumericFp8ScaleType::F32 => 1.0f32.to_le_bytes().to_vec(),
            };
            ImmutableValidatedNumericFp8Payload::new(layout, vec![0x38; n * k], scale.repeat(64))
                .unwrap()
        });
        let layout = SlabLayout::for_payloads(&payloads).unwrap();
        let charge = CudaNumericFp8UploadHostCharge::for_payloads(&payloads).unwrap();
        assert_eq!(layout.bytes, expected);
        assert_eq!(charge.pageable_payload_bytes, expected);
        assert_eq!(charge.pinned_allocation_bytes, expected);
        assert_eq!(
            charge.total_bytes,
            expected * 2 + charge.ticket_metadata_bytes
        );
        assert!(charge.admit(expected).is_err());
        let mut slab = vec![0; layout.bytes];
        layout.pack(&payloads, &mut slab).unwrap();
        let last = layout.ranges[5];
        assert_eq!(last.offset + last.len, expected);
        assert_eq!(&slab[last.offset..], payloads[2].scale_bytes());
    }
}

#[test]
fn one_tickets_ready_event_does_not_release_another_tickets_pending_custody() {
    let (mut first, first_state, first_liveness) = staged();
    let (mut second, second_state, second_liveness) = staged();
    first.submit().unwrap();
    second.submit().unwrap();
    first_state.ready.set(true);
    drop(first.poll().unwrap().unwrap());
    assert_released(&first_state, &first_liveness);
    assert!(second.poll().unwrap().is_none());
    assert_retained(&second_state, &second_liveness);
    assert!(first.poll().is_err());
    second_state.ready.set(true);
    drop(second.poll().unwrap().unwrap());
    assert_released(&second_state, &second_liveness);
}

#[path = "gpu.rs"]
mod gpu;
