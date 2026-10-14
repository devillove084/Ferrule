use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Barrier, Mutex, MutexGuard, Weak};

use crate::allocator::CudaDeviceAllocator;

pub type DevicePtr = u64;
#[derive(Debug)]
pub struct CudaError;
pub type CudaResult<T> = Result<T, CudaError>;
impl CudaError {
    pub fn internal(_: &str) -> Self {
        Self
    }
}
#[derive(Default)]
pub struct CaptureState {
    pub active: bool,
    pub unknown: bool,
}
impl CaptureState {
    pub fn blocks_allocation(&self) -> bool {
        self.active || self.unknown
    }
    pub fn completion_unknown(&self) -> bool {
        self.unknown
    }
}

#[derive(Default)]
pub struct Driver {
    pub streams: AtomicUsize,
    pub recordings: AtomicUsize,
    pub creates: AtomicUsize,
    pub records: AtomicUsize,
    pub destroys: AtomicUsize,
    pub queries: AtomicUsize,
    pub event_syncs: AtomicUsize,
    pub context_syncs: AtomicUsize,
    pub frees: AtomicUsize,
    pub ready_through: AtomicUsize,
    pub pending_streams: AtomicUsize,
    pub fail_record_at: AtomicUsize,
    pub fail_query: AtomicBool,
    pub fail_sync: AtomicBool,
    pub panic_record: AtomicBool,
    pub panic_query: AtomicBool,
    pub panic_sync: AtomicBool,
    pub record_hook: Mutex<Option<(Arc<Barrier>, Arc<Barrier>)>>,
    pub sync_hook: Mutex<Option<(Arc<Barrier>, Arc<Barrier>)>>,
    pub reenter: Mutex<Option<Weak<CudaDeviceAllocator>>>,
}
thread_local! {
    static CURRENT: RefCell<Option<Arc<Driver>>> = const { RefCell::new(None) };
}
static NEXT_PTR: AtomicU64 = AtomicU64::new(0x10000000);

pub fn driver_alloc(bytes: usize) -> CudaResult<DevicePtr> {
    Ok(NEXT_PTR.fetch_add(bytes as u64 + (2 << 20), SeqCst))
}
pub fn driver_free(_: DevicePtr) -> CudaResult<()> {
    CURRENT.with(|current| {
        current
            .borrow()
            .as_ref()
            .unwrap()
            .frees
            .fetch_add(1, SeqCst)
    });
    Ok(())
}
pub struct CudaContext {
    pub allocator: Arc<CudaDeviceAllocator>,
    pub driver: Arc<Driver>,
    pub capture: Mutex<CaptureState>,
}
impl CudaContext {
    pub fn new(streams: usize) -> Arc<Self> {
        Arc::new_cyclic(|weak| Self {
            allocator: CudaDeviceAllocator::new(weak.clone()),
            driver: Arc::new(Driver {
                streams: AtomicUsize::new(streams),
                ..Driver::default()
            }),
            capture: Mutex::new(CaptureState::default()),
        })
    }
    pub fn capture_state(&self) -> MutexGuard<'_, CaptureState> {
        self.capture.lock().unwrap()
    }
    pub fn is_capturing(&self) -> bool {
        self.capture_state().blocks_allocation()
    }
    pub fn graph_completion_unknown(&self) -> bool {
        self.capture_state().unknown
    }
    pub fn mark_graph_unknown(&self) {
        self.capture_state().unknown = true;
    }
    pub fn bind_to_thread(&self) -> CudaResult<()> {
        CURRENT.with(|current| *current.borrow_mut() = Some(self.driver.clone()));
        Ok(())
    }
    pub fn record_retirement_events(&self) -> Result<Option<Vec<CudaRetirementEvent>>, ()> {
        let d = &self.driver;
        let generation = d.recordings.fetch_add(1, SeqCst) + 1;
        if let Some(allocator) = d
            .reenter
            .lock()
            .unwrap()
            .take()
            .and_then(|weak| weak.upgrade())
        {
            allocator.metrics();
            allocator.poll_retirements();
        }
        let mut events = Vec::new();
        for index in 0..d.streams.load(SeqCst) {
            d.creates.fetch_add(1, SeqCst);
            let event = CudaRetirementEvent {
                driver: d.clone(),
                generation,
                stream: index,
            };
            d.records.fetch_add(1, SeqCst);
            if index == 0 {
                let hook = d.record_hook.lock().unwrap().take();
                if let Some((reached, resume)) = hook {
                    reached.wait();
                    resume.wait();
                }
                assert!(!d.panic_record.load(SeqCst), "mock record panic");
            }
            if d.fail_record_at.load(SeqCst) == index + 1 {
                return Err(());
            }
            events.push(event);
        }
        Ok(Some(events))
    }
    pub fn synchronize_registered_streams(&self) -> CudaResult<()> {
        if self.is_capturing() {
            return Err(CudaError);
        }
        self.driver.context_syncs.fetch_add(1, SeqCst);
        let hook = self.driver.sync_hook.lock().unwrap().take();
        if let Some((reached, resume)) = hook {
            reached.wait();
            resume.wait();
        }
        assert!(!self.driver.panic_sync.load(SeqCst), "mock sync panic");
        if self.driver.fail_sync.load(SeqCst) {
            self.mark_graph_unknown();
            return Err(CudaError);
        }
        self.driver
            .ready_through
            .store(self.driver.recordings.load(SeqCst), SeqCst);
        Ok(())
    }
}
impl Drop for CudaContext {
    fn drop(&mut self) {
        if self.graph_completion_unknown()
            || self.allocator.shutdown_with_context(self, true).is_err()
        {
            self.allocator.retain_after_context_failure();
        }
    }
}
#[derive(Debug)]
pub struct CudaRetirementEvent {
    driver: Arc<Driver>,
    generation: usize,
    stream: usize,
}
impl std::fmt::Debug for Driver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MockDriver")
    }
}
impl CudaRetirementEvent {
    pub fn query(&self) -> CudaResult<bool> {
        self.driver.queries.fetch_add(1, SeqCst);
        assert!(!self.driver.panic_query.load(SeqCst), "mock query panic");
        if self.driver.fail_query.load(SeqCst) {
            return Err(CudaError);
        }
        Ok(self.generation <= self.driver.ready_through.load(SeqCst)
            && self.driver.pending_streams.load(SeqCst) & (1 << self.stream) == 0)
    }
    pub fn synchronize(&self) -> CudaResult<()> {
        self.driver.event_syncs.fetch_add(1, SeqCst);
        assert!(
            !self.driver.panic_sync.load(SeqCst),
            "mock event sync panic"
        );
        if self.driver.fail_sync.load(SeqCst) {
            Err(CudaError)
        } else {
            Ok(())
        }
    }
}
impl Drop for CudaRetirementEvent {
    fn drop(&mut self) {
        self.driver.destroys.fetch_add(1, SeqCst);
    }
}

// Models the unchanged runtime stream teardown contract, not a new allocator proof.
pub struct MockStream {
    context: Arc<CudaContext>,
}
impl MockStream {
    pub fn new(context: &Arc<CudaContext>) -> Self {
        context.driver.streams.fetch_add(1, SeqCst);
        Self {
            context: context.clone(),
        }
    }
}
impl Drop for MockStream {
    fn drop(&mut self) {
        if self.context.driver.fail_sync.load(SeqCst) {
            self.context.capture.lock().unwrap().unknown = true;
        }
        self.context.driver.streams.fetch_sub(1, SeqCst);
    }
}
