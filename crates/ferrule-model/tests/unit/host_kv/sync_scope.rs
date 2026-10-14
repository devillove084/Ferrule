//! Driver-free checks of the production synchronous continuation gate.
use super::{DecoderForwardProgress, DecoderWait, NoTerminalGuard};
use ferrule_common::{DependencySet, Error, LogicalDependency, OperationId};
use std::cell::Cell;
use std::rc::Rc;

struct Held(Rc<Cell<usize>>);
impl Drop for Held {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}
#[test]
fn waiting_keeps_exact_continuation_and_rejects_resume_edge() {
    let drops = Rc::new(Cell::new(0));
    let held = Held(drops.clone());
    let progress: DecoderForwardProgress<Held, NoTerminalGuard> = DecoderForwardProgress::Waiting {
        continuation: held,
        wait: DecoderWait::Dependencies(
            DependencySet::new(
                [LogicalDependency::operation_retired(OperationId::new(1)).unwrap()],
            )
            .unwrap(),
        ),
    };
    let DecoderForwardProgress::FailedActive {
        continuation,
        error,
    } = progress.require_synchronous()
    else {
        panic!("borrowed synchronous scope escaped into waiting");
    };
    assert!(error.to_string().contains("abort required"));
    assert!(Rc::ptr_eq(&continuation.0, &drops));
    assert_eq!(drops.get(), 0);
    drop(continuation);
    assert_eq!(drops.get(), 1);
}
#[test]
fn unknown_failure_is_not_downgraded_to_quiescence() {
    let drops = Rc::new(Cell::new(0));
    let progress: DecoderForwardProgress<Held, NoTerminalGuard> =
        DecoderForwardProgress::FailedActive {
            continuation: Held(drops.clone()),
            error: Error::Execution {
                message: "physical completion Unknown".into(),
            },
        };
    let DecoderForwardProgress::FailedActive {
        continuation,
        error,
    } = progress.require_synchronous()
    else {
        panic!("lost active failure custody");
    };
    assert!(error.to_string().contains("Unknown"));
    assert_eq!(drops.get(), 0);
    drop(continuation);
    assert_eq!(drops.get(), 1);
}
