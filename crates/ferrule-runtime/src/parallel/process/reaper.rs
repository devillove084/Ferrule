use std::io;
use std::process::Child;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use super::{PROCESS_REAPER_CAPACITY, ProcessError};

struct Registry {
    reserved: usize,
    pending: Vec<Child>,
}
type Shared = Arc<Mutex<Registry>>;
static REAPER: OnceLock<Result<Shared, String>> = OnceLock::new();

/// A slot is reserved before spawn, so Drop cannot overflow the quarantine
/// registry. A permanently unreapable child keeps its slot forever, stopping
/// future admission instead of accumulating detached waiter threads.
pub(super) struct Slot {
    shared: Shared,
    held: bool,
}
impl Slot {
    pub(super) fn reserve() -> Result<Self, ProcessError> {
        let registry = REAPER.get_or_init(|| {
            let shared = Arc::new(Mutex::new(Registry {
                reserved: 0,
                pending: Vec::with_capacity(PROCESS_REAPER_CAPACITY),
            }));
            let worker = Arc::clone(&shared);
            thread::Builder::new()
                .name("ferrule-process-reaper".into())
                .spawn(move || {
                    loop {
                        {
                            let mut registry = worker.lock().unwrap_or_else(|e| e.into_inner());
                            let mut index = 0;
                            while index < registry.pending.len() {
                                // Only try_wait: one D-state child cannot stop others.
                                if matches!(registry.pending[index].try_wait(), Ok(Some(_))) {
                                    registry.pending.swap_remove(index);
                                    registry.reserved -= 1;
                                } else {
                                    index += 1;
                                }
                            }
                        }
                        thread::sleep(Duration::from_millis(20));
                    }
                })
                .map_err(|e| e.to_string())?;
            Ok(shared)
        });
        let shared = registry.as_ref().map_err(|message| ProcessError::Io {
            operation: "start bounded process reaper",
            source: io::Error::other(message.clone()),
        })?;
        let mut registry = shared.lock().unwrap_or_else(|e| e.into_inner());
        if registry.reserved == PROCESS_REAPER_CAPACITY {
            return Err(ProcessError::Capacity {
                resource: "process reaper",
                limit: PROCESS_REAPER_CAPACITY,
            });
        }
        registry.reserved += 1;
        Ok(Self {
            shared: Arc::clone(shared),
            held: true,
        })
    }

    pub(super) fn adopt(mut self, child: Child) {
        let mut registry = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        registry.pending.push(child);
        self.held = false;
    }
}
impl Drop for Slot {
    fn drop(&mut self) {
        if self.held {
            self.shared
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .reserved -= 1;
        }
    }
}

/// Only signal while the direct child is still owned/unreaped. Never retain a
/// bare PID for later signalling after releasing the Child (PID reuse).
#[expect(
    unsafe_code,
    reason = "kill targets the still-owned Child or its isolated process group"
)]
pub(super) fn signal_owned(child: &Child, signal: libc::c_int, group: bool) -> io::Result<()> {
    let pid = libc::pid_t::try_from(child.id()).map_err(io::Error::other)?;
    if pid <= 0 {
        return Err(io::Error::other("invalid child pid"));
    }
    if unsafe { libc::kill(if group { -pid } else { pid }, signal) } < 0 {
        let error = io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::ESRCH) {
            return Err(error);
        }
    }
    // ESRCH is not reap evidence. The caller still owns and polls the Child.
    Ok(())
}
