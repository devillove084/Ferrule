#![allow(non_snake_case)]
//! These symbols exist only in this dedicated test executable. No CUDA call in
//! the tested lifecycle is forwarded to the GPU, even with real libcuda linked.
use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Barrier, Mutex};

#[derive(Default)]
pub struct Driver {
    pub sync_code: AtomicI32,
    pub stream_sync_code: AtomicI32,
    pub query_code: AtomicI32,
    pub record_code: AtomicI32,
    pub event_sync_code: AtomicI32,
    pub free_code: AtomicI32,
    pub release_code: AtomicI32,
    pub fail_bind: AtomicBool,
    pub ready: AtomicBool,
    pub retains: AtomicUsize,
    pub releases: AtomicUsize,
    pub frees: AtomicUsize,
    pub syncs: AtomicUsize,
    pub creates: AtomicUsize,
    pub destroys: AtomicUsize,
    pub sync_hook: Mutex<Option<(Arc<Barrier>, Arc<Barrier>)>>,
    pub stream_sync_hook: Mutex<Option<(Arc<Barrier>, Arc<Barrier>)>>,
    pub query_hook: Mutex<Option<(Arc<Barrier>, Arc<Barrier>)>>,
    pub calls: Mutex<Vec<&'static str>>,
}
thread_local! {
    static CURRENT: RefCell<Option<Arc<Driver>>> = const { RefCell::new(None) };
}
pub struct Guard(Option<Arc<Driver>>);
impl Drop for Guard {
    fn drop(&mut self) {
        CURRENT.with(|current| *current.borrow_mut() = self.0.take());
    }
}
impl Driver {
    pub fn enter(self: &Arc<Self>) -> Guard {
        Guard(CURRENT.with(|current| current.borrow_mut().replace(self.clone())))
    }
}
fn run(name: &'static str, f: impl FnOnce(&Driver) -> c_int) -> c_int {
    let driver = CURRENT.with(|current| current.borrow().clone());
    let Some(driver) = driver else {
        return 999;
    };
    driver.calls.lock().unwrap().push(name);
    f(&driver)
}
static NEXT: AtomicUsize = AtomicUsize::new(0x10000000);

#[unsafe(no_mangle)]
extern "C" fn cuInit(_: c_uint) -> c_int {
    run("init", |_| 0)
}
#[unsafe(no_mangle)]
unsafe extern "C" fn cuDeviceGet(out: *mut c_int, ordinal: c_int) -> c_int {
    run("device", |_| {
        unsafe { *out = ordinal };
        0
    })
}
#[unsafe(no_mangle)]
unsafe extern "C" fn cuDevicePrimaryCtxRetain(out: *mut *mut c_void, _: c_int) -> c_int {
    run("retain", |driver| {
        driver.retains.fetch_add(1, SeqCst);
        unsafe { *out = NEXT.fetch_add(256, SeqCst) as *mut c_void };
        0
    })
}
#[unsafe(no_mangle)]
extern "C" fn cuDevicePrimaryCtxRelease_v2(_: c_int) -> c_int {
    run("release", |driver| {
        driver.releases.fetch_add(1, SeqCst);
        driver.release_code.load(SeqCst)
    })
}
#[unsafe(no_mangle)]
unsafe extern "C" fn cuCtxGetCurrent(out: *mut *mut c_void) -> c_int {
    run("get_current", |_| {
        unsafe { *out = std::ptr::null_mut() };
        0
    })
}
#[unsafe(no_mangle)]
extern "C" fn cuCtxSetCurrent(_: *mut c_void) -> c_int {
    run("bind", |driver| {
        if driver.fail_bind.load(SeqCst) {
            999
        } else {
            0
        }
    })
}
#[unsafe(no_mangle)]
extern "C" fn cuCtxSynchronize() -> c_int {
    run("context_sync", |driver| {
        driver.syncs.fetch_add(1, SeqCst);
        let hook = driver.sync_hook.lock().unwrap().take();
        if let Some((reached, resume)) = hook {
            reached.wait();
            resume.wait();
        }
        driver.sync_code.load(SeqCst)
    })
}
#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemAlloc_v2(out: *mut u64, bytes: usize) -> c_int {
    run("alloc", |_| {
        unsafe { *out = NEXT.fetch_add(bytes + (2 << 20), SeqCst) as u64 };
        0
    })
}
#[unsafe(no_mangle)]
extern "C" fn cuMemFree_v2(_: u64) -> c_int {
    run("free", |driver| {
        driver.frees.fetch_add(1, SeqCst);
        driver.free_code.load(SeqCst)
    })
}
#[unsafe(no_mangle)]
unsafe extern "C" fn cuEventCreate(out: *mut *mut c_void, _: c_uint) -> c_int {
    run("event_create", |driver| {
        driver.creates.fetch_add(1, SeqCst);
        unsafe { *out = NEXT.fetch_add(256, SeqCst) as *mut c_void };
        0
    })
}
#[unsafe(no_mangle)]
extern "C" fn cuEventRecord(_: *mut c_void, _: *mut c_void) -> c_int {
    run("event_record", |driver| driver.record_code.load(SeqCst))
}
#[unsafe(no_mangle)]
extern "C" fn cuEventQuery(_: *mut c_void) -> c_int {
    run("event_query", |driver| {
        let hook = driver.query_hook.lock().unwrap().take();
        if let Some((reached, resume)) = hook {
            reached.wait();
            resume.wait();
        }
        let code = driver.query_code.load(SeqCst);
        if code != 0 {
            code
        } else if driver.ready.load(SeqCst) {
            0
        } else {
            600
        }
    })
}
#[unsafe(no_mangle)]
extern "C" fn cuEventSynchronize(_: *mut c_void) -> c_int {
    run("event_sync", |driver| driver.event_sync_code.load(SeqCst))
}
#[unsafe(no_mangle)]
extern "C" fn cuEventDestroy_v2(_: *mut c_void) -> c_int {
    run("event_destroy", |driver| {
        driver.destroys.fetch_add(1, SeqCst);
        0
    })
}
#[unsafe(no_mangle)]
extern "C" fn cuStreamSynchronize(_: *mut c_void) -> c_int {
    run("stream_sync", |driver| {
        let hook = driver.stream_sync_hook.lock().unwrap().take();
        if let Some((reached, resume)) = hook {
            reached.wait();
            resume.wait();
        }
        driver.stream_sync_code.load(SeqCst)
    })
}
#[unsafe(no_mangle)]
extern "C" fn cuStreamBeginCapture(_: *mut c_void, _: c_int) -> c_int {
    run("begin_capture", |_| 0)
}
#[unsafe(no_mangle)]
unsafe extern "C" fn cuGetErrorName(_: c_int, out: *mut *const c_char) -> c_int {
    unsafe { *out = c"MOCK_CUDA_ERROR".as_ptr() };
    0
}
#[unsafe(no_mangle)]
unsafe extern "C" fn cuGetErrorString(_: c_int, out: *mut *const c_char) -> c_int {
    unsafe { *out = c"injected CPU driver failure; no GPU access".as_ptr() };
    0
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuStreamEndCapture(_: *mut c_void, out: *mut *mut c_void) -> c_int {
    run("end_capture", |_| {
        unsafe { *out = NEXT.fetch_add(256, SeqCst) as *mut c_void };
        0
    })
}
#[unsafe(no_mangle)]
extern "C" fn cuGraphDestroy(_: *mut c_void) -> c_int {
    run("graph_destroy", |_| 0)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuStreamCreate(out: *mut *mut c_void, _: c_uint) -> c_int {
    run("stream_create", |_| {
        unsafe { *out = NEXT.fetch_add(256, SeqCst) as *mut c_void };
        0
    })
}
#[unsafe(no_mangle)]
extern "C" fn cuStreamDestroy_v2(_: *mut c_void) -> c_int {
    run("stream_destroy", |_| 0)
}
