use std::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
}
fn symbol(name: &std::ffi::CStr) -> *mut c_void {
    static LIB: OnceLock<usize> = OnceLock::new();
    let handle = *LIB.get_or_init(|| {
        let handle = unsafe { dlopen(c"libcuda.so.1".as_ptr(), 2) };
        assert!(!handle.is_null(), "real libcuda is required");
        handle as usize
    });
    let ptr = unsafe { dlsym(handle as *mut c_void, name.as_ptr()) };
    assert!(!ptr.is_null(), "missing real CUDA symbol {name:?}");
    ptr
}

pub static CREATES: AtomicUsize = AtomicUsize::new(0);
pub static RECORDS: AtomicUsize = AtomicUsize::new(0);
pub static QUERIES: AtomicUsize = AtomicUsize::new(0);
pub static EVENT_SYNCS: AtomicUsize = AtomicUsize::new(0);
pub static CONTEXT_SYNCS: AtomicUsize = AtomicUsize::new(0);
pub static STREAM_SYNCS: AtomicUsize = AtomicUsize::new(0);
pub static FREES: AtomicUsize = AtomicUsize::new(0);
pub static RELEASES: AtomicUsize = AtomicUsize::new(0);
pub static FAIL_QUERY: AtomicBool = AtomicBool::new(false);
pub static FAIL_EVENT_SYNC: AtomicBool = AtomicBool::new(false);
pub static QUERY_GATE_AT: AtomicUsize = AtomicUsize::new(0);
pub static QUERY_GATE: Mutex<Option<Arc<Gate>>> = Mutex::new(None);
pub static STREAM_GATE: Mutex<Option<Arc<Gate>>> = Mutex::new(None);

#[derive(Default)]
pub struct Gate {
    state: Mutex<(bool, bool)>,
    changed: Condvar,
    pub timed_out: AtomicBool,
}
impl Gate {
    pub fn block(&self) {
        let mut state = self.state.lock().unwrap();
        state.0 = true;
        self.changed.notify_all();
        let (state, timeout) = self
            .changed
            .wait_timeout_while(state, Duration::from_secs(20), |s| !s.1)
            .unwrap();
        if timeout.timed_out() && !state.1 {
            self.timed_out.store(true, SeqCst);
        }
    }
    pub fn entered(&self) {
        let state = self.state.lock().unwrap();
        let (state, _) = self
            .changed
            .wait_timeout_while(state, Duration::from_secs(10), |s| !s.0)
            .unwrap();
        assert!(state.0, "real CUDA callback / driver gate did not arrive");
    }
    pub fn release(&self) {
        self.state.lock().unwrap().1 = true;
        self.changed.notify_all();
    }
}
unsafe extern "C" fn callback(data: *mut c_void) {
    let gate = unsafe { Arc::from_raw(data.cast::<Gate>()) };
    gate.block();
}
pub fn block_stream(stream: &crate::runtime::CudaStream) -> Arc<Gate> {
    let gate = Arc::new(Gate::default());
    let launch: unsafe extern "C" fn(
        *mut c_void,
        unsafe extern "C" fn(*mut c_void),
        *mut c_void,
    ) -> c_int = unsafe { std::mem::transmute(symbol(c"cuLaunchHostFunc")) };
    let data = Arc::into_raw(gate.clone()) as *mut c_void;
    let code = unsafe { launch(stream.cu_stream(), callback, data) };
    assert_eq!(code, 0, "cuLaunchHostFunc failed");
    gate.entered();
    gate
}
macro_rules! forward {
    ($name:ident, $counter:ident, ($($arg:ident: $ty:ty),*)) => {
        #[unsafe(no_mangle)]
        unsafe extern "C" fn $name($($arg: $ty),*) -> c_int {
            $counter.fetch_add(1, SeqCst);
            let name = std::ffi::CString::new(stringify!($name)).unwrap();
            let real: unsafe extern "C" fn($($ty),*) -> c_int = unsafe { std::mem::transmute(symbol(&name)) };
            unsafe { real($($arg),*) }
        }
    };
}
forward!(cuEventCreate, CREATES, (event: *mut *mut c_void, flags: c_uint));
forward!(cuEventRecord, RECORDS, (event: *mut c_void, stream: *mut c_void));
forward!(cuCtxSynchronize, CONTEXT_SYNCS, ());
forward!(cuMemFree_v2, FREES, (ptr: u64));
forward!(cuDevicePrimaryCtxRelease_v2, RELEASES, (device: c_int));
#[unsafe(no_mangle)]
unsafe extern "C" fn cuStreamSynchronize(stream: *mut c_void) -> c_int {
    STREAM_SYNCS.fetch_add(1, SeqCst);
    let gate = STREAM_GATE.lock().unwrap().take();
    if let Some(gate) = gate {
        gate.block();
    }
    let real: unsafe extern "C" fn(*mut c_void) -> c_int =
        unsafe { std::mem::transmute(symbol(c"cuStreamSynchronize")) };
    unsafe { real(stream) }
}
#[unsafe(no_mangle)]
unsafe extern "C" fn cuEventQuery(event: *mut c_void) -> c_int {
    let n = QUERIES.fetch_add(1, SeqCst) + 1;
    let real: unsafe extern "C" fn(*mut c_void) -> c_int =
        unsafe { std::mem::transmute(symbol(c"cuEventQuery")) };
    let code = unsafe { real(event) };
    if QUERY_GATE_AT.load(SeqCst) == n {
        let gate = QUERY_GATE.lock().unwrap().take().unwrap();
        gate.block();
    }
    if FAIL_QUERY.swap(false, SeqCst) {
        assert_eq!(code, 0);
        999
    } else {
        code
    }
}
#[unsafe(no_mangle)]
unsafe extern "C" fn cuEventSynchronize(event: *mut c_void) -> c_int {
    EVENT_SYNCS.fetch_add(1, SeqCst);
    let real: unsafe extern "C" fn(*mut c_void) -> c_int =
        unsafe { std::mem::transmute(symbol(c"cuEventSynchronize")) };
    let code = unsafe { real(event) };
    if FAIL_EVENT_SYNC.swap(false, SeqCst) {
        assert_eq!(code, 0);
        999
    } else {
        code
    }
}
pub fn counts(phase: &str) {
    println!(
        "DRIVER {phase} creates={} records={} queries={} event_syncs={} context_syncs={} stream_syncs={} frees={} primary_releases={}",
        CREATES.load(SeqCst),
        RECORDS.load(SeqCst),
        QUERIES.load(SeqCst),
        EVENT_SYNCS.load(SeqCst),
        CONTEXT_SYNCS.load(SeqCst),
        STREAM_SYNCS.load(SeqCst),
        FREES.load(SeqCst),
        RELEASES.load(SeqCst)
    );
}
