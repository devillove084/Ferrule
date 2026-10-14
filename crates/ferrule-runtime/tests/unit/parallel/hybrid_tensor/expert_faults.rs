//! Owner-thread-local fault injection, installed only by the CPU hybrid tests.
use super::ExpertDispatchContext;
use ferrule_common::Result;
use std::{cell::RefCell, sync::Arc};
#[derive(Clone, Copy)]
pub(crate) enum Event {
    Compute(ExpertDispatchContext),
    Shutdown,
}
pub(crate) type Hook = Arc<dyn Fn(Event) -> Result<()> + Send + Sync>;
std::thread_local! {pub(crate) static HOOK:RefCell<Option<Hook>>=RefCell::new(None);}
pub(crate) fn call(event: Event) -> Result<()> {
    let hook = HOOK.with_borrow(Clone::clone);
    hook.map_or(Ok(()), |hook| hook(event))
}
