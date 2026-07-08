use std::cell::Cell;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaOpCounters {
    pub kernel_launches: u64,
    pub host_to_device_copies: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_copies: u64,
    pub device_to_host_bytes: u64,
    pub artifact_uploads: u64,
    pub artifact_upload_bytes: u64,
    pub moe_calls: u64,
    pub moe_tc_calls: u64,
    pub moe_scalar_calls: u64,
    pub moe_reduce_calls: u64,
    pub moe_total_us: u64,
    pub moe_pointer_upload_us: u64,
    pub moe_input_prepare_us: u64,
    pub moe_gate_up_us: u64,
    pub moe_swiglu_us: u64,
    pub moe_hidden_pack_us: u64,
    pub moe_down_us: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaMoeExecutionPath {
    TensorCore,
    Scalar,
    Reduce,
}

#[derive(Default)]
pub(crate) struct CudaOpCounterCells {
    kernel_launches: Cell<u64>,
    host_to_device_copies: Cell<u64>,
    host_to_device_bytes: Cell<u64>,
    device_to_host_copies: Cell<u64>,
    device_to_host_bytes: Cell<u64>,
    artifact_uploads: Cell<u64>,
    artifact_upload_bytes: Cell<u64>,
    moe_calls: Cell<u64>,
    moe_tc_calls: Cell<u64>,
    moe_scalar_calls: Cell<u64>,
    moe_reduce_calls: Cell<u64>,
    moe_total_us: Cell<u64>,
    moe_pointer_upload_us: Cell<u64>,
    moe_input_prepare_us: Cell<u64>,
    moe_gate_up_us: Cell<u64>,
    moe_swiglu_us: Cell<u64>,
    moe_hidden_pack_us: Cell<u64>,
    moe_down_us: Cell<u64>,
}

impl CudaOpCounterCells {
    pub(crate) fn snapshot(&self) -> CudaOpCounters {
        CudaOpCounters {
            kernel_launches: self.kernel_launches.get(),
            host_to_device_copies: self.host_to_device_copies.get(),
            host_to_device_bytes: self.host_to_device_bytes.get(),
            device_to_host_copies: self.device_to_host_copies.get(),
            device_to_host_bytes: self.device_to_host_bytes.get(),
            artifact_uploads: self.artifact_uploads.get(),
            artifact_upload_bytes: self.artifact_upload_bytes.get(),
            moe_calls: self.moe_calls.get(),
            moe_tc_calls: self.moe_tc_calls.get(),
            moe_scalar_calls: self.moe_scalar_calls.get(),
            moe_reduce_calls: self.moe_reduce_calls.get(),
            moe_total_us: self.moe_total_us.get(),
            moe_pointer_upload_us: self.moe_pointer_upload_us.get(),
            moe_input_prepare_us: self.moe_input_prepare_us.get(),
            moe_gate_up_us: self.moe_gate_up_us.get(),
            moe_swiglu_us: self.moe_swiglu_us.get(),
            moe_hidden_pack_us: self.moe_hidden_pack_us.get(),
            moe_down_us: self.moe_down_us.get(),
        }
    }

    pub(crate) fn reset(&self) {
        self.kernel_launches.set(0);
        self.host_to_device_copies.set(0);
        self.host_to_device_bytes.set(0);
        self.device_to_host_copies.set(0);
        self.device_to_host_bytes.set(0);
        self.artifact_uploads.set(0);
        self.artifact_upload_bytes.set(0);
        self.moe_calls.set(0);
        self.moe_tc_calls.set(0);
        self.moe_scalar_calls.set(0);
        self.moe_reduce_calls.set(0);
        self.moe_total_us.set(0);
        self.moe_pointer_upload_us.set(0);
        self.moe_input_prepare_us.set(0);
        self.moe_gate_up_us.set(0);
        self.moe_swiglu_us.set(0);
        self.moe_hidden_pack_us.set(0);
        self.moe_down_us.set(0);
    }

    pub(crate) fn add_kernel_launch(&self) {
        self.kernel_launches
            .set(self.kernel_launches.get().saturating_add(1));
    }

    pub(crate) fn add_host_to_device(&self, bytes: u64) {
        self.host_to_device_copies
            .set(self.host_to_device_copies.get().saturating_add(1));
        self.host_to_device_bytes
            .set(self.host_to_device_bytes.get().saturating_add(bytes));
    }

    pub(crate) fn add_device_to_host(&self, bytes: u64) {
        self.device_to_host_copies
            .set(self.device_to_host_copies.get().saturating_add(1));
        self.device_to_host_bytes
            .set(self.device_to_host_bytes.get().saturating_add(bytes));
    }

    pub(crate) fn add_artifact_upload(&self, bytes: u64) {
        self.artifact_uploads
            .set(self.artifact_uploads.get().saturating_add(1));
        self.artifact_upload_bytes
            .set(self.artifact_upload_bytes.get().saturating_add(bytes));
    }

    pub(crate) fn add_moe_call(&self, path: CudaMoeExecutionPath) {
        self.moe_calls.set(self.moe_calls.get().saturating_add(1));
        match path {
            CudaMoeExecutionPath::TensorCore => self
                .moe_tc_calls
                .set(self.moe_tc_calls.get().saturating_add(1)),
            CudaMoeExecutionPath::Scalar => self
                .moe_scalar_calls
                .set(self.moe_scalar_calls.get().saturating_add(1)),
            CudaMoeExecutionPath::Reduce => self
                .moe_reduce_calls
                .set(self.moe_reduce_calls.get().saturating_add(1)),
        }
    }

    pub(crate) fn add_moe_total_us(&self, us: u64) {
        self.moe_total_us
            .set(self.moe_total_us.get().saturating_add(us));
    }

    pub(crate) fn add_moe_pointer_upload_us(&self, us: u64) {
        self.moe_pointer_upload_us
            .set(self.moe_pointer_upload_us.get().saturating_add(us));
    }

    pub(crate) fn add_moe_input_prepare_us(&self, us: u64) {
        self.moe_input_prepare_us
            .set(self.moe_input_prepare_us.get().saturating_add(us));
    }

    pub(crate) fn add_moe_gate_up_us(&self, us: u64) {
        self.moe_gate_up_us
            .set(self.moe_gate_up_us.get().saturating_add(us));
    }

    pub(crate) fn add_moe_swiglu_us(&self, us: u64) {
        self.moe_swiglu_us
            .set(self.moe_swiglu_us.get().saturating_add(us));
    }

    pub(crate) fn add_moe_down_us(&self, us: u64) {
        self.moe_down_us
            .set(self.moe_down_us.get().saturating_add(us));
    }
}
