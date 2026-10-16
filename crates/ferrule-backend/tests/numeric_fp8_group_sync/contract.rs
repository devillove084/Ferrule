//! CPU contracts exercise the actual production allocation/copy/fence flow.
//! The separate real-driver test is ignored; these tests initialize no CUDA.
use super::*;
use ferrule_common::numeric_fp8::{NumericFp8PayloadLiveness, NumericFp8ScaleType};
use std::cell::{Cell, RefCell};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use crate::cuda::operator_owner::numeric_fp8_fixture::payload;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Allocate(usize),
    Copy(usize),
    Fence,
}

struct State {
    calls: RefCell<Vec<Op>>,
    faults: RefCell<Vec<Op>>,
    panic: Cell<Option<Op>>,
    expected: [Vec<u8>; 6],
    liveness: [NumericFp8PayloadLiveness; 3],
    allocated: Cell<usize>,
    device_drops: Cell<usize>,
    owner_drops: Cell<usize>,
    in_flight: Cell<bool>,
    premature_drop: Cell<bool>,
    identity: Rc<()>,
}
impl State {
    fn hit(&self, op: Op) -> Result<()> {
        self.calls.borrow_mut().push(op);
        assert_ne!(
            self.panic.get(),
            Some(op),
            "injected group operation unwind"
        );
        if self.faults.borrow().contains(&op) {
            Err(Error::Internal {
                message: format!("injected {op:?}"),
            })
        } else {
            Ok(())
        }
    }
    fn fence_count(&self) -> usize {
        self.calls
            .borrow()
            .iter()
            .filter(|&&op| op == Op::Fence)
            .count()
    }
    fn assert_live(&self) {
        assert!(self.liveness.iter().all(|payload| payload.is_live()));
        assert_eq!(self.device_drops.get(), 0);
        assert_eq!(self.owner_drops.get(), 0);
        assert!(!self.premature_drop.get());
    }
    fn assert_released(&self) {
        assert!(self.liveness.iter().all(|payload| !payload.is_live()));
        assert_eq!(self.device_drops.get(), self.allocated.get());
        assert_eq!(self.owner_drops.get(), 1);
        assert!(!self.premature_drop.get());
    }
}
struct Owner {
    state: Rc<State>,
    identity: Rc<()>,
}
impl Drop for Owner {
    fn drop(&mut self) {
        self.state.owner_drops.set(self.state.owner_drops.get() + 1);
        self.state
            .premature_drop
            .set(self.state.premature_drop.get() || self.state.in_flight.get());
    }
}
struct Device {
    index: usize,
    len: usize,
    state: Rc<State>,
}
impl Drop for Device {
    fn drop(&mut self) {
        self.state
            .device_drops
            .set(self.state.device_drops.get() + 1);
        self.state
            .premature_drop
            .set(self.state.premature_drop.get() || self.state.in_flight.get());
    }
}
fn setup() -> (Rc<State>, [ImmutableValidatedNumericFp8Payload; 3]) {
    let payloads = [
        payload(NumericFp8ScaleType::Bf16, 0x38),
        payload(NumericFp8ScaleType::F32, 0x40),
        payload(NumericFp8ScaleType::Bf16, 0x7e),
    ];
    let state = Rc::new(State {
        calls: RefCell::new(Vec::new()),
        faults: RefCell::new(Vec::new()),
        panic: Cell::new(None),
        expected: std::array::from_fn(|i| {
            if i % 2 == 0 {
                payloads[i / 2].weight_bytes().to_vec()
            } else {
                payloads[i / 2].scale_bytes().to_vec()
            }
        }),
        liveness: payloads
            .each_ref()
            .map(|payload| payload.allocation_liveness()),
        allocated: Cell::new(0),
        device_drops: Cell::new(0),
        owner_drops: Cell::new(0),
        in_flight: Cell::new(false),
        premature_drop: Cell::new(false),
        identity: Rc::new(()),
    });
    (state, payloads)
}
fn run(
    state: &Rc<State>,
    payloads: [ImmutableValidatedNumericFp8Payload; 3],
) -> Result<Group<Owner, Device>> {
    upload_group(
        Owner {
            state: Rc::clone(state),
            identity: Rc::clone(&state.identity),
        },
        payloads,
        |len| {
            let index = state.allocated.get();
            state.hit(Op::Allocate(index))?;
            assert_eq!(len, state.expected[index].len());
            state.allocated.set(index + 1);
            Ok(Device {
                index,
                len,
                state: Rc::clone(state),
            })
        },
        |owner, destination, bytes| {
            assert!(Rc::ptr_eq(&owner.identity, &state.identity));
            assert_eq!(state.allocated.get(), 6, "all allocations precede DMA");
            state.assert_live();
            assert_eq!(destination.len, bytes.len());
            assert_eq!(bytes, state.expected[destination.index]);
            // A failing/panicking copy may already have submitted work.
            state.in_flight.set(true);
            state.hit(Op::Copy(destination.index))
        },
        |owner| {
            assert!(Rc::ptr_eq(&owner.identity, &state.identity));
            state.assert_live();
            state.hit(Op::Fence)?;
            state.in_flight.set(false);
            Ok(())
        },
    )
}

#[test]
fn success_allocates_six_then_copies_six_exact_sources_and_fences_once() {
    let (state, payloads) = setup();
    let group = run(&state, payloads).expect("group success");
    let expected = (0..6)
        .map(Op::Allocate)
        .chain((0..6).map(Op::Copy))
        .chain([Op::Fence])
        .collect::<Vec<_>>();
    assert_eq!(*state.calls.borrow(), expected);
    state.assert_live();
    assert!(!state.in_flight.get());
    drop(group);
    state.assert_released();
    assert_eq!(state.fence_count(), 1, "normal completion disarms cleanup");
}

#[test]
fn every_allocation_failure_submits_nothing_and_needs_no_fence() {
    for index in 0..6 {
        let (state, payloads) = setup();
        state.faults.borrow_mut().push(Op::Allocate(index));
        assert!(run(&state, payloads).is_err());
        assert_eq!(
            *state.calls.borrow(),
            (0..=index).map(Op::Allocate).collect::<Vec<_>>()
        );
        state.assert_released();
        assert_eq!(state.fence_count(), 0);
    }
}

#[test]
fn every_allocation_unwind_submits_nothing_and_releases_only_unsubmitted_resources() {
    for index in 0..6 {
        let (state, payloads) = setup();
        state.panic.set(Some(Op::Allocate(index)));
        assert!(catch_unwind(AssertUnwindSafe(|| run(&state, payloads))).is_err());
        state.assert_released();
        assert_eq!(state.fence_count(), 0);
    }
}

#[test]
fn every_copy_failure_stops_further_copies_and_drains_exactly_once() {
    for index in 0..6 {
        let (state, payloads) = setup();
        state.faults.borrow_mut().push(Op::Copy(index));
        let error = run(&state, payloads).err().expect("copy failure");
        assert!(error.to_string().contains(&format!("Copy({index})")));
        let expected = (0..6)
            .map(Op::Allocate)
            .chain((0..=index).map(Op::Copy))
            .chain([Op::Fence])
            .collect::<Vec<_>>();
        assert_eq!(*state.calls.borrow(), expected);
        state.assert_released();
        assert_eq!(state.fence_count(), 1);
    }
}

#[test]
fn fence_failure_retains_payload_aliases_all_destinations_and_exact_owner_without_retry() {
    let (state, payloads) = setup();
    state.faults.borrow_mut().push(Op::Fence);
    assert!(run(&state, payloads).is_err());
    state.faults.borrow_mut().clear(); // a later successful fence cannot heal this group
    state.assert_live();
    assert!(state.in_flight.get());
    assert_eq!(state.fence_count(), 1);
}

#[test]
fn every_copy_plus_fence_failure_reports_both_errors_and_retains_the_entire_group() {
    for index in 0..6 {
        let (state, payloads) = setup();
        state
            .faults
            .borrow_mut()
            .extend([Op::Copy(index), Op::Fence]);
        let error = run(&state, payloads)
            .err()
            .expect("copy and cleanup failure");
        assert!(matches!(&error, Error::Cleanup { .. }));
        assert!(error.to_string().contains(&format!("Copy({index})")));
        assert!(error.to_string().contains("Fence"));
        assert_eq!(
            state
                .calls
                .borrow()
                .iter()
                .filter(|op| matches!(op, Op::Copy(_)))
                .count(),
            index + 1
        );
        state.assert_live();
        assert_eq!(state.fence_count(), 1);
    }
}

#[test]
fn every_copy_unwind_gets_one_exact_fence_before_releasing_resources() {
    for index in 0..6 {
        let (state, payloads) = setup();
        state.panic.set(Some(Op::Copy(index)));
        assert!(catch_unwind(AssertUnwindSafe(|| run(&state, payloads))).is_err());
        assert_eq!(state.calls.borrow().last(), Some(&Op::Fence));
        state.assert_released();
        assert_eq!(state.fence_count(), 1);
    }
}

#[test]
fn copy_unwind_and_failed_cleanup_keep_every_resource_live() {
    let (state, payloads) = setup();
    state.panic.set(Some(Op::Copy(3)));
    state.faults.borrow_mut().push(Op::Fence);
    assert!(catch_unwind(AssertUnwindSafe(|| run(&state, payloads))).is_err());
    state.assert_live();
    assert_eq!(state.fence_count(), 1);
}

#[test]
fn fence_unwind_is_terminal_and_never_attempts_a_second_fence() {
    let (state, payloads) = setup();
    state.panic.set(Some(Op::Fence));
    assert!(catch_unwind(AssertUnwindSafe(|| run(&state, payloads))).is_err());
    state.panic.set(None);
    state.assert_live();
    assert_eq!(state.fence_count(), 1);
}

#[test]
fn borrowed_fence_callback_is_not_leaked_with_owned_unknown_custody() {
    let (state, payloads) = setup();
    let callback_hold = Rc::new(());
    let callback_weak = Rc::downgrade(&callback_hold);
    let result = upload_group(
        Rc::clone(&state),
        payloads,
        |_| Ok(()),
        |_, _, _| Ok(()),
        move |_| {
            let _hold = &callback_hold;
            Err(Error::Internal {
                message: "no proof".into(),
            })
        },
    );
    assert!(result.is_err());
    assert!(callback_weak.upgrade().is_none());
    assert!(state.liveness.iter().all(|payload| payload.is_live()));
}

#[path = "gpu.rs"]
mod gpu;
