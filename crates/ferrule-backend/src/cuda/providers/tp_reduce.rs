//! Internal NCCL custody. All CUDA/NCCL calls go through the native core ABI.
use crate::cuda::{
    ffi::tp_reduce as ffi,
    operator_owner::cu,
    operators::tp_reduce_contract::{NCCL_LIBRARY_SHA256, STORAGE_BYTES, STORAGE_WORDS, invalid},
    runtime::{CudaEvent, CudaStream, DeviceBuffer},
};
use ferrule_common::Result;
use std::{
    ffi::CString,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    sync::Arc,
};

pub(crate) fn status(code: i32) -> Result<()> {
    if code == ffi::OK {
        Ok(())
    } else {
        Err(invalid(format!("native status {code}")))
    }
}

/// Opaque native metadata only; excludes dlopen/NCCL-private/Rust allocations.
/// This lower bound must not be misrepresented as a complete host hard cap.
pub(crate) fn native_handle_bytes() -> Result<usize> {
    let library = unsafe { ffi::ferrule_tp_library_host_bytes() };
    let comm = unsafe { ffi::ferrule_tp_comm_host_bytes() };
    if library == 0 || comm == 0 {
        return Err(invalid("pinned NCCL provider not compiled"));
    }
    library
        .checked_add(
            comm.checked_mul(2)
                .ok_or_else(|| invalid("host size overflow"))?,
        )
        .ok_or_else(|| invalid("host size overflow"))
}

pub struct NcclLibrary {
    raw: *mut ffi::Library,
    source: PathBuf,
}
// Only immutable library/symbol data is shared; each communicator is single-owner.
unsafe impl Send for NcclLibrary {}
unsafe impl Sync for NcclLibrary {}
impl NcclLibrary {
    /// CPU-only loader: exact absolute canonical source, SHA256 and version 22705.
    /// Trusted pinned files must remain immutable during the load (not a sandbox).
    pub fn load(path: &Path, native_handle_budget: usize) -> Result<Arc<Self>> {
        if native_handle_budget < native_handle_bytes()? {
            return Err(invalid("native handle budget insufficient"));
        }
        if !path.is_absolute() {
            return Err(invalid("library path must be absolute"));
        }
        let source = path.canonicalize().map_err(|e| invalid(e.to_string()))?;
        let hash = super::nccl_sha256::digest(BufReader::new(
            File::open(&source).map_err(|e| invalid(e.to_string()))?,
        ))
        .map_err(|e| invalid(e.to_string()))?;
        if hash != NCCL_LIBRARY_SHA256 {
            return Err(invalid("library SHA256 does not match CUDA12 NCCL 2.27.5"));
        }
        let path = CString::new(source.as_os_str().as_encoded_bytes())
            .map_err(|e| invalid(e.to_string()))?;
        let mut raw = std::ptr::null_mut();
        status(unsafe { ffi::ferrule_tp_library_open(path.as_ptr(), &mut raw) })?;
        if raw.is_null() {
            return Err(invalid("native library returned null handle"));
        }
        Ok(Arc::new(Self { raw, source }))
    }
    pub fn source(&self) -> &Path {
        &self.source
    }
    pub fn sha256(&self) -> &'static str {
        NCCL_LIBRARY_SHA256
    }
    pub fn required_native_handle_bytes() -> Result<usize> {
        native_handle_bytes()
    }
    pub(crate) fn unique_id(&self) -> Result<ffi::UniqueId> {
        let mut id = ffi::UniqueId { bytes: [0; 128] };
        status(unsafe { ffi::ferrule_tp_unique_id(self.raw, &mut id) })?;
        Ok(id)
    }
}
impl Drop for NcclLibrary {
    fn drop(&mut self) {
        unsafe { ffi::ferrule_tp_library_close(self.raw) };
    }
}

pub(crate) struct Rank {
    pub(crate) raw: *mut ffi::Comm,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) storage: DeviceBuffer<u16>,
    pub(crate) event: CudaEvent,
    pub(crate) input: Option<DeviceBuffer<u16>>,
    // Kept in joint custody even if native init/enqueue/abort loses completion.
    pub(crate) library: Arc<NcclLibrary>,
}
unsafe impl Send for Rank {}
impl Rank {
    pub(crate) fn allocate(stream: Arc<CudaStream>, library: Arc<NcclLibrary>) -> Result<Self> {
        cu(stream.context().check_graph_ready())?;
        cu(stream.context().bind_to_thread())?;
        let storage = cu(DeviceBuffer::uninitialized_dedicated(
            &stream,
            STORAGE_WORDS,
        ))?;
        let event = cu(stream.context().new_event(false))?;
        Ok(Self {
            raw: std::ptr::null_mut(),
            stream,
            storage,
            event,
            input: None,
            library,
        })
    }
    pub(crate) fn init(&mut self, id: &ffi::UniqueId, rank: u32) -> Result<()> {
        cu(self.stream.context().bind_to_thread())?;
        let code = unsafe {
            ffi::ferrule_tp_comm_create(
                self.library.raw,
                id,
                rank,
                self.stream.cu_stream() as usize as u64,
                self.storage.cu_deviceptr(),
                STORAGE_BYTES as u64,
                &mut self.raw,
            )
        };
        if code == ffi::PENDING {
            Ok(())
        } else {
            status(code)
        }
    }
    pub(crate) fn bind(&self) -> Result<()> {
        cu(self.stream.context().check_graph_ready())?;
        cu(self.stream.context().bind_to_thread())
    }
    pub(crate) fn async_status(&self) -> Result<bool> {
        self.bind()?;
        let code = unsafe { ffi::ferrule_tp_comm_async_status(self.raw) };
        if code == ffi::PENDING {
            Ok(false)
        } else {
            status(code)?;
            Ok(true)
        }
    }
    pub(crate) fn progress(&self) -> Result<bool> {
        self.bind()?;
        let code = unsafe { ffi::ferrule_tp_comm_progress(self.raw) };
        if code == ffi::PENDING {
            Ok(false)
        } else {
            status(code)?;
            Ok(true)
        }
    }
    pub(crate) fn warm(&self) -> Result<bool> {
        self.bind()?;
        let code = unsafe { ffi::ferrule_tp_comm_warmup(self.raw) };
        if code == ffi::PENDING {
            Ok(false)
        } else {
            status(code)?;
            Ok(true)
        }
    }
    pub(crate) fn gather(&self, count: usize) -> Result<bool> {
        self.bind()?;
        let input = self
            .input
            .as_ref()
            .ok_or_else(|| invalid("input custody missing"))?;
        let code = unsafe {
            ffi::ferrule_tp_allgather_bf16_two(
                self.raw,
                input.cu_deviceptr(),
                input.num_bytes() as u64,
                count as u32,
            )
        };
        if code == ffi::PENDING {
            Ok(false)
        } else {
            status(code)?;
            Ok(true)
        }
    }
    pub(crate) fn device_status(&self) -> Result<u32> {
        self.bind()?;
        let mut value = 0;
        status(unsafe { ffi::ferrule_tp_comm_read_status(self.raw, &mut value) })?;
        #[cfg(test)]
        crate::cuda::operators::tp_reduce::gpu_tests::status_copied();
        Ok(value)
    }
    pub(crate) fn destroy(&mut self) -> Result<()> {
        self.bind()?;
        if !self.raw.is_null() {
            status(unsafe { ffi::ferrule_tp_comm_destroy(self.raw) })?;
            self.raw = std::ptr::null_mut();
        }
        Ok(())
    }
    pub(crate) fn request_abort(&self) {
        // Never interpret abort success as a storage/context release proof.
        if !self.raw.is_null() && self.stream.context().bind_to_thread().is_ok() {
            unsafe { ffi::ferrule_tp_comm_abort(self.raw) };
        }
    }
}
