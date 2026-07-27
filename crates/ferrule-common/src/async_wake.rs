//! Lock-free completion wakeups shared by storage, CUDA, model, and runtime.
//!
//! Ferrule has one inference owner per device. Producers may run on io_uring,
//! staging, or CUDA callback threads, but exactly one owner consumes completion
//! notifications. A monotonic epoch plus [`atomic_waker::AtomicWaker`] gives that
//! topology a race-free wake path without a business-state lock or allocation on
//! producer threads.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::task::{Context, Poll};

use atomic_waker::AtomicWaker;

/// Result of waiting for the next completion epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionWake {
    /// At least one producer published progress. Multiple publications may be
    /// coalesced into this latest epoch; the owner must drain all ready work.
    Progress(u64),
    /// The completion source is permanently closed and will publish no more work.
    Closed,
}

#[derive(Debug)]
struct CompletionHubInner {
    epoch: AtomicU64,
    closed: AtomicBool,
    owner_waker: AtomicWaker,
}

/// Cloneable producer/consumer handle for one inference owner's completion bus.
///
/// `notify` is allocation-free and does not acquire a mutex. The hub intentionally
/// stores one waker: Ferrule has one owner task per device, and that task drains
/// every model/I/O completion after each wake.
#[derive(Debug, Clone)]
pub struct CompletionHub {
    inner: Arc<CompletionHubInner>,
}

impl Default for CompletionHub {
    fn default() -> Self {
        Self::new()
    }
}

impl CompletionHub {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(CompletionHubInner {
                epoch: AtomicU64::new(0),
                closed: AtomicBool::new(false),
                owner_waker: AtomicWaker::new(),
            }),
        }
    }

    /// Return the latest published epoch.
    pub fn epoch(&self) -> u64 {
        self.inner.epoch.load(Ordering::Acquire)
    }

    /// Arm an owned listener before checking producer-owned state.
    ///
    /// The owner should create one listener, inspect/drain all completion state,
    /// and await the listener only when no immediately runnable work remains.
    pub fn listen(&self) -> CompletionListener {
        CompletionListener {
            inner: Arc::clone(&self.inner),
            observed_epoch: self.epoch(),
        }
    }

    /// Publish progress from an io_uring pump, staging worker, or CUDA host
    /// callback. Returns the newly visible epoch.
    pub fn notify(&self) -> u64 {
        let epoch = self
            .inner
            .epoch
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1);
        self.inner.owner_waker.wake();
        epoch
    }

    /// Permanently close the source and wake the owner. Closing is idempotent.
    pub fn close(&self) {
        if !self.inner.closed.swap(true, Ordering::AcqRel) {
            self.inner.owner_waker.wake();
        }
    }

    pub fn is_closed(&self) -> bool {
        self.inner.closed.load(Ordering::Acquire)
    }
}

/// Owned future for one completion epoch change.
///
/// This future does not borrow a runner, CUDA context, or io_uring state, so the
/// inference owner may continue scheduling independent transactions while it is
/// pending. Only one listener may be actively polled for a hub at a time.
#[derive(Debug)]
pub struct CompletionListener {
    inner: Arc<CompletionHubInner>,
    observed_epoch: u64,
}

impl Future for CompletionListener {
    type Output = CompletionWake;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let current = this.inner.epoch.load(Ordering::Acquire);
        if current != this.observed_epoch {
            return Poll::Ready(CompletionWake::Progress(current));
        }
        if this.inner.closed.load(Ordering::Acquire) {
            return Poll::Ready(CompletionWake::Closed);
        }

        this.inner.owner_waker.register(context.waker());

        // Check again after registration. A producer racing between the first
        // check and register either changed the epoch or will wake this waker.
        let current = this.inner.epoch.load(Ordering::Acquire);
        if current != this.observed_epoch {
            return Poll::Ready(CompletionWake::Progress(current));
        }
        if this.inner.closed.load(Ordering::Acquire) {
            return Poll::Ready(CompletionWake::Closed);
        }
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Wake, Waker};

    #[derive(Default)]
    struct TestWake(AtomicUsize);

    impl Wake for TestWake {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::AcqRel);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn poll_listener(listener: &mut CompletionListener, waker: &Waker) -> Poll<CompletionWake> {
        let mut context = Context::from_waker(waker);
        Pin::new(listener).poll(&mut context)
    }

    #[test]
    fn producer_wake_is_coalesced_without_losing_epoch_progress() {
        let hub = CompletionHub::new();
        let mut listener = hub.listen();
        let wake = Arc::new(TestWake::default());
        let waker = Waker::from(Arc::clone(&wake));

        assert_eq!(poll_listener(&mut listener, &waker), Poll::Pending);
        assert_eq!(hub.notify(), 1);
        assert_eq!(hub.notify(), 2);
        assert!(wake.0.load(Ordering::Acquire) >= 1);
        assert_eq!(
            poll_listener(&mut listener, &waker),
            Poll::Ready(CompletionWake::Progress(2))
        );
    }

    #[test]
    fn close_wakes_listener_and_is_idempotent() {
        let hub = CompletionHub::new();
        let mut listener = hub.listen();
        let wake = Arc::new(TestWake::default());
        let waker = Waker::from(Arc::clone(&wake));

        assert_eq!(poll_listener(&mut listener, &waker), Poll::Pending);
        hub.close();
        hub.close();
        assert_eq!(
            poll_listener(&mut listener, &waker),
            Poll::Ready(CompletionWake::Closed)
        );
        assert_eq!(wake.0.load(Ordering::Acquire), 1);
    }

    #[test]
    fn listener_created_after_progress_waits_for_the_next_epoch() {
        let hub = CompletionHub::new();
        hub.notify();
        let mut listener = hub.listen();
        let waker = Waker::from(Arc::new(TestWake::default()));

        assert_eq!(poll_listener(&mut listener, &waker), Poll::Pending);
        hub.notify();
        assert_eq!(
            poll_listener(&mut listener, &waker),
            Poll::Ready(CompletionWake::Progress(2))
        );
    }
}
