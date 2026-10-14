//! Private handle bound once to caller-admitted storage and one owner stream.
use crate::cuda::operators::linear::CudaNumericFp8Artifact;
use crate::cuda::{
    ffi::cutlass::*,
    operator_owner::cu,
    runtime::{CudaStream, DeviceBuffer, DeviceCopy},
};
use crate::cuda_plan::bf16_gemmex::Bf16GemmExPlan;
use ferrule_common::{Error, Result};
use std::{cell::Cell, sync::Arc};

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Internal {
        message: format!("BF16 GemmEx: {}", message.into()),
    }
}
fn status(code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(invalid(format!(
            "native status {code}; requires CUDA12.4/sm86/cuBLAS120405, no fallback"
        )))
    }
}
pub(crate) fn available() -> Result<()> {
    if unsafe { ferrule_bf16_gemmex_available() } == 1 {
        Ok(())
    } else {
        status(2)
    }
}

use super::admission::{Admission as SharedAdmission, Charge, ProviderFamily};
#[cfg(test)]
use super::admission::{Ledger, ferrule_bf16_gemmex_process_config_valid, ledger};
pub(crate) struct Admission {
    _lease: SharedAdmission,
}
impl Admission {
    pub(crate) fn prepare<T>(
        stream: &CudaStream,
        plan: Bf16GemmExPlan,
        prepare: impl FnOnce(Self) -> Result<T>,
    ) -> Result<T> {
        SharedAdmission::prepare(
            stream,
            ProviderFamily::GemmEx,
            Charge {
                storage_grant: plan.storage_grant_bytes(),
                external: plan.provider_bytes(),
            },
            |a| prepare(Self { _lease: a }),
        )
    }
}

#[cfg(test)]
thread_local! { static WARMUP_FAULT: Cell<u8> = const { Cell::new(0) }; }
#[cfg(test)]
fn injected_warmup_code(stream: &CudaStream, code: i32, phase: u8) -> i32 {
    if code == 0 && WARMUP_FAULT.get() == phase {
        WARMUP_FAULT.set(0);
        // The completed native call supplies a real live handle/backing. Model
        // a failed earlier native fence with status 9, then explicitly prove a
        // later raw driver fence succeeds before handing that status to Rust.
        cu(stream.synchronize()).expect("later fence succeeds in injected warmup case");
        return COMPLETION_UNKNOWN;
    }
    code
}
const COMPLETION_UNKNOWN: i32 = 9;
/// Native completion failure is sticky even if a later fence would succeed.
fn warmup_completion(
    code: i32,
    poison: impl FnOnce(),
    fence: impl FnOnce() -> Result<()>,
) -> Result<()> {
    if code == COMPLETION_UNKNOWN {
        poison();
        return Err(invalid(
            "native warmup completion unknown (status 9); terminal quarantine, no later-fence recovery",
        ));
    }
    if let Err(error) = fence() {
        poison();
        return Err(error);
    }
    Ok(())
}

pub(crate) struct NativeBf16GemmEx {
    raw: u64,
    admission: Option<Admission>,
    pub(crate) stream: Arc<CudaStream>,
    storage: Option<DeviceBuffer<u8>>,
    pub(crate) plan: Bf16GemmExPlan,
    /// Device-global free-memory delta during coordinated initialization. This
    /// is diagnostic, NOT attributable allocation accounting or a hard cap.
    pub(crate) growth: u64,
    views: Vec<Bf16GemmExPlan>,
    pub(crate) version: i32,
    pub(crate) poisoned: Cell<bool>,
}
impl NativeBf16GemmEx {
    pub(crate) fn new(
        stream: &CudaStream,
        storage: DeviceBuffer<u8>,
        plan: Bf16GemmExPlan,
        admission: Admission,
    ) -> Result<Arc<Self>> {
        available()?;
        cu(stream.context().check_graph_ready())?;
        cu(stream.context().bind_to_thread())?;
        cu(storage.check_context(stream.context(), "GemmEx preparation"))?;
        if storage.num_bytes() < plan.storage_bytes() {
            return Err(invalid("storage below admitted plan"));
        }
        let mut p = Self {
            raw: 0,
            admission: Some(admission),
            stream: cu(stream.owner())?,
            storage: Some(storage),
            plan,
            growth: 0,
            views: vec![plan],
            version: 0,
            poisoned: Cell::new(false),
        };
        #[cfg(test)]
        if WARMUP_FAULT.get() == 3 {
            WARMUP_FAULT.set(0);
            return Err(invalid("injected prehandle initialization failure"));
        }
        let b = p.storage.as_ref().unwrap();
        let l = plan.layout();
        let code = unsafe {
            ferrule_bf16_gemmex_create(
                plan.rows() as u32,
                l.n as u32,
                l.k as u32,
                stream.cu_stream() as usize as u64,
                b.cu_deviceptr(),
                b.num_bytes() as u64,
                plan.provider_bytes() as u64,
                &mut p.raw,
                &mut p.growth,
                &mut p.version,
            )
        };
        // Failed initialization may have submitted work. Drop retains the entire
        // bound allocation if this completion proof fails, never just the stream.
        #[cfg(test)]
        let code = injected_warmup_code(stream, code, 1);
        warmup_completion(
            code,
            || stream.context().mark_graph_unknown(),
            || cu(stream.synchronize()),
        )?;
        if code == 5 || code == 6 {
            return Err(invalid(format!(
                "provider admission status={code}, observed_growth={}, reserve={}; caller workspace is separately {} bytes; CUBLAS_WORKSPACE_CONFIG=:0:0 avoids an additional library-default pool",
                p.growth,
                plan.provider_bytes(),
                plan.cublas_workspace_bytes()
            )));
        }
        if code == 7 {
            return Err(invalid(format!(
                "requires cuBLAS120405, loaded {}; no fallback",
                p.version
            )));
        }
        status(code)?;
        // Graph leases use Arc; Cell deliberately keeps this eager owner !Sync.
        #[allow(clippy::arc_with_non_send_sync)]
        let owner = Arc::new(p);
        Ok(owner)
    }

    pub(crate) fn prepare_views(
        stream: &CudaStream,
        mut owner: Arc<Self>,
        views: &[Bf16GemmExPlan],
    ) -> Result<Arc<Self>> {
        let largest = *views
            .iter()
            .max_by_key(|p| p.storage_bytes())
            .ok_or_else(|| invalid("empty shape set"))?;
        if owner.plan != largest {
            return Err(invalid("maximum shape owner mismatch"));
        }
        let p = Arc::get_mut(&mut owner).ok_or_else(|| invalid("shared preparation"))?;
        for &view in views {
            if p.views.contains(&view) {
                continue;
            }
            let l = view.layout();
            let code = unsafe {
                ferrule_bf16_gemmex_warm_view(
                    p.raw,
                    view.rows() as u32,
                    l.n as u32,
                    l.k as u32,
                    p.storage.as_ref().unwrap().cu_deviceptr(),
                    largest.staging_bytes() as u64,
                    p.growth,
                    &mut p.growth,
                )
            };
            #[cfg(test)]
            let code = injected_warmup_code(stream, code, 2);
            warmup_completion(
                code,
                || stream.context().mark_graph_unknown(),
                || cu(stream.synchronize()),
            )?;
            status(code)?;
            p.views.push(view);
        }
        // Revisit the complete set after all shapes have initialized. Changing
        // shape must not require an additional provider allocation on reuse.
        for &view in views {
            let l = view.layout();
            let before = p.growth;
            let code = unsafe {
                ferrule_bf16_gemmex_warm_view(
                    p.raw,
                    view.rows() as u32,
                    l.n as u32,
                    l.k as u32,
                    p.storage.as_ref().unwrap().cu_deviceptr(),
                    largest.staging_bytes() as u64,
                    before,
                    &mut p.growth,
                )
            };
            #[cfg(test)]
            let code = injected_warmup_code(stream, code, 2);
            warmup_completion(
                code,
                || stream.context().mark_graph_unknown(),
                || cu(stream.synchronize()),
            )?;
            status(code)?;
            if before != p.growth {
                return Err(invalid("shape revisit grew provider allocation"));
            }
        }
        Ok(owner)
    }
    pub(crate) fn launch_view(
        &self,
        stream: &CudaStream,
        plan: Bf16GemmExPlan,
        w: &CudaNumericFp8Artifact,
        x: &DeviceBuffer<u16>,
        y: &DeviceBuffer<u16>,
    ) -> Result<()> {
        if !self.views.contains(&plan) {
            return Err(invalid("unprepared shape view"));
        }
        let a = self.args_for_plan(stream, w, x, y, plan)?;
        self.submit_view(stream, || unsafe {
            ferrule_bf16_gemmex_launch(&a, self.raw)
        })
    }
    pub(crate) fn launch_resident(
        &self,
        stream: &CudaStream,
        plan: Bf16GemmExPlan,
        weight: &DeviceBuffer<u16>,
        input: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
    ) -> Result<()> {
        self.owner(stream)?;
        if !self.views.contains(&plan) {
            return Err(invalid("unprepared resident BF16 shape"));
        }
        let l = plan.layout();
        let counts = [plan.rows() * l.k, l.n * l.k, plan.rows() * l.n];
        let storage = self.storage.as_ref().unwrap();
        let workspace = storage.cu_deviceptr()..storage.cu_deviceptr() + storage.num_bytes() as u64;
        for (buffer, count) in [input, weight, output].into_iter().zip(counts) {
            if buffer.len() != count {
                return Err(invalid("resident BF16 packed extent mismatch"));
            }
            let ptr = self.buffer(buffer, count * 2)?;
            let end = ptr
                .checked_add((count * 2) as u64)
                .ok_or_else(|| invalid("resident span overflow"))?;
            if ptr < workspace.end && workspace.start < end {
                return Err(invalid("resident BF16 operand aliases provider storage"));
            }
        }
        let dim = |v| u32::try_from(v).map_err(|_| invalid("resident dimension overflow"));
        let args = ResidentArgs {
            m: dim(plan.rows())?,
            n: dim(l.n)?,
            k: dim(l.k)?,
            reserved: 0,
            input: input.cu_deviceptr(),
            weight: weight.cu_deviceptr(),
            output: output.cu_deviceptr(),
            stream: stream.cu_stream() as usize as u64,
        };
        status(unsafe { ferrule_bf16_gemmex_resident_can(&args) })?;
        self.submit_view(stream, || unsafe {
            ferrule_bf16_gemmex_resident_launch(&args, self.raw)
        })
    }

    pub(crate) fn launch_pair<const SWIGLU: bool>(
        &self,
        stream: &CudaStream,
        plan: Bf16GemmExPlan,
        weights: [&CudaNumericFp8Artifact; 2],
        x: &DeviceBuffer<u16>,
        outputs: [&DeviceBuffer<u16>; 2],
    ) -> Result<()> {
        let l = weights[0].layout();
        if !self.views.contains(&plan)
            || weights[1].layout() != l
            || l.n.checked_mul(2) != Some(plan.layout().n)
            || plan.layout()
                != (crate::cuda_plan::NumericFp8Layout {
                    n: plan.layout().n,
                    ..l
                })
        {
            return Err(invalid("unprepared or incompatible concatenated shape"));
        }
        let half = Bf16GemmExPlan::new(
            l,
            plan.rows(),
            self.plan.storage_grant_bytes(),
            self.plan.provider_bytes(),
        )?;
        if SWIGLU {
            self.owner(stream)?;
            self.resident_operand(x, plan.rows() * l.k)?;
            self.resident_operand(outputs[0], plan.rows() * l.n)?;
        }
        let a = self.args_for_plan(stream, weights[0], x, outputs[0], half)?;
        let b = self.args_for_plan(stream, weights[1], x, outputs[1], half)?;
        let (can, launch) = if SWIGLU {
            (
                ferrule_bf16_gemmex_can_pair_swiglu
                    as unsafe extern "C" fn(*const NumericArgs, *const NumericArgs) -> i32,
                ferrule_bf16_gemmex_launch_pair_swiglu
                    as unsafe extern "C" fn(*const NumericArgs, *const NumericArgs, u64) -> i32,
            )
        } else {
            (
                ferrule_bf16_gemmex_can_pair
                    as unsafe extern "C" fn(*const NumericArgs, *const NumericArgs) -> i32,
                ferrule_bf16_gemmex_launch_pair
                    as unsafe extern "C" fn(*const NumericArgs, *const NumericArgs, u64) -> i32,
            )
        };
        status(unsafe { can(&a, &b) })?;
        self.submit_view(stream, || unsafe { launch(&a, &b, self.raw) })
    }
    fn submit_view(&self, stream: &CudaStream, f: impl FnOnce() -> i32) -> Result<()> {
        if stream.context().is_capturing() {
            return Err(invalid("shape views require eager retained owner"));
        }
        cu(stream.context().bind_to_thread())?;
        let result = status(f());
        if result.is_err() {
            self.poisoned.set(true);
            stream.context().mark_graph_unknown();
        }
        result
    }
    pub(crate) fn owner(&self, stream: &CudaStream) -> Result<()> {
        if !std::ptr::eq(self.stream.as_ref(), stream)
            || self.poisoned.get()
            || stream.context().graph_completion_unknown()
        {
            return Err(invalid(
                "stream owner mismatch or terminal unknown/poisoned state",
            ));
        }
        Ok(())
    }
    fn buffer<T: DeviceCopy>(&self, b: &DeviceBuffer<T>, bytes: usize) -> Result<u64> {
        cu(b.check_context(self.stream.context(), "GemmEx operand"))?;
        if b.num_bytes() < bytes {
            return Err(invalid("operand extent"));
        }
        Ok(b.cu_deviceptr())
    }
    pub(crate) fn args(
        &self,
        stream: &CudaStream,
        w: &CudaNumericFp8Artifact,
        x: &DeviceBuffer<u16>,
        y: &DeviceBuffer<u16>,
    ) -> Result<NumericArgs> {
        self.args_for_plan(stream, w, x, y, self.plan)
    }
    fn args_for_plan(
        &self,
        stream: &CudaStream,
        w: &CudaNumericFp8Artifact,
        x: &DeviceBuffer<u16>,
        y: &DeviceBuffer<u16>,
        p: Bf16GemmExPlan,
    ) -> Result<NumericArgs> {
        self.owner(stream)?;
        if p.is_resident() || w.layout() != p.layout() {
            return Err(invalid("artifact/plan mismatch"));
        }
        let l = p.layout();
        let (weight, scales) = w.buffers();
        let (wb, sb) = l.storage_lengths()?;
        let storage = self.storage.as_ref().unwrap();
        let a = NumericArgs {
            m: p.rows() as u32,
            n: l.n as u32,
            k: l.k as u32,
            padded_k: l.k as u32,
            tile_rows: l.n as u32,
            scale_cols: (l.k / 128) as u32,
            row_origin: 0,
            column_origin: 0,
            scale_bytes: 2,
            lda: l.k as u32,
            ldd: l.n as u32,
            reserved: 0,
            activation: self.buffer(x, p.rows() * l.k * 2)?,
            weight: self.buffer(weight, wb)?,
            scales: self.buffer(scales, sb)?,
            output: self.buffer(y, p.rows() * l.n * 2)?,
            workspace: storage.cu_deviceptr(),
            workspace_bytes: storage.num_bytes() as u64,
            stream: stream.cu_stream() as usize as u64,
        };
        status(unsafe { ferrule_bf16_gemmex_can_implement(&a) })?;
        Ok(a)
    }
    pub(crate) fn launch(
        &self,
        stream: &CudaStream,
        w: &CudaNumericFp8Artifact,
        x: &DeviceBuffer<u16>,
        y: &DeviceBuffer<u16>,
        retained_capture: bool,
    ) -> Result<()> {
        let a = self.args(stream, w, x, y)?;
        if stream.context().is_capturing() && !retained_capture {
            return Err(invalid(
                "use the retained GemmEx capture API, not unowned graph capture",
            ));
        }
        cu(stream.context().bind_to_thread())?;
        let result = status(unsafe { ferrule_bf16_gemmex_launch(&a, self.raw) });
        if result.is_err() {
            self.poisoned.set(true);
            stream.context().mark_graph_unknown();
        }
        result
    }
}
impl Drop for NativeBf16GemmEx {
    fn drop(&mut self) {
        if self.poisoned.get()
            || self.stream.context().graph_completion_unknown()
            || self.stream.context().is_capturing()
            || self.stream.context().bind_to_thread().is_err()
            || self.stream.synchronize().is_err()
        {
            self.stream.context().mark_graph_unknown();
            std::mem::forget(self.admission.take());
            std::mem::forget(self.storage.take());
            std::mem::forget(Arc::clone(&self.stream));
            return;
        }
        if self.raw != 0 && unsafe { ferrule_bf16_gemmex_destroy(self.raw) } != 0 {
            self.stream.context().mark_graph_unknown();
            std::mem::forget(self.admission.take());
            std::mem::forget(self.storage.take());
            std::mem::forget(Arc::clone(&self.stream));
        }
        if self.admission.is_none() {
            return;
        }
        // Keep this owner's physical/external reservation until its allocation
        // retires. Other owners have their own independently admitted charges.
        drop(self.storage.take());
        if self
            .stream
            .context()
            .synchronize_registered_streams()
            .is_err()
            || self.stream.context().trim_allocator().is_err()
        {
            self.stream.context().mark_graph_unknown();
            std::mem::forget(self.admission.take());
            std::mem::forget(Arc::clone(&self.stream));
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/numeric_bf16_gemv/backend_gpu.rs"]
mod tests;

unsafe extern "C" {
    fn ferrule_bf16_gemmex_warm_view(
        handle: u64,
        m: u32,
        n: u32,
        k: u32,
        memory: u64,
        staging_capacity: u64,
        initial_growth: u64,
        growth: *mut u64,
    ) -> i32;
    fn ferrule_bf16_gemmex_can_pair(a: *const NumericArgs, b: *const NumericArgs) -> i32;
    fn ferrule_bf16_gemmex_launch_pair(
        a: *const NumericArgs,
        b: *const NumericArgs,
        handle: u64,
    ) -> i32;
}

#[cfg(test)]
#[path = "../../../tests/numeric_bf16_gemv/bf16_gemmex_views.rs"]
mod view_tests;

#[cfg(test)]
#[path = "../../../tests/numeric_bf16_gemv/bf16_gemmex_admission.rs"]
mod admission_tests;

#[repr(C)]
struct ResidentArgs {
    m: u32,
    n: u32,
    k: u32,
    reserved: u32,
    input: u64,
    weight: u64,
    output: u64,
    stream: u64,
}
unsafe extern "C" {
    fn ferrule_bf16_gemmex_resident_can(args: *const ResidentArgs) -> i32;
    fn ferrule_bf16_gemmex_resident_launch(args: *const ResidentArgs, handle: u64) -> i32;
}
#[cfg(test)]
#[path = "../../../tests/numeric_bf16_gemv/resident_router/gpu.rs"]
mod resident_router_tests;

#[cfg(test)]
#[path = "../../../tests/numeric_bf16_gemv/resident_control/gpu.rs"]
mod resident_control_tests;

#[cfg(test)]
#[path = "../../../tests/numeric_bf16_gemv/resident_scalar/gpu.rs"]
mod resident_scalar_tests;

impl NativeBf16GemmEx {
    fn resident_operand<T: DeviceCopy>(
        &self,
        buffer: &DeviceBuffer<T>,
        count: usize,
    ) -> Result<u64> {
        if buffer.len() != count {
            return Err(invalid("resident operand exact extent mismatch"));
        }
        let ptr = self.buffer(buffer, buffer.num_bytes())?;
        let end = ptr
            .checked_add(buffer.num_bytes() as u64)
            .ok_or_else(|| invalid("resident operand span overflow"))?;
        let storage = self.storage.as_ref().unwrap();
        let start = storage.cu_deviceptr();
        if ptr < start + storage.num_bytes() as u64 && start < end {
            return Err(invalid("resident operand aliases provider storage"));
        }
        Ok(ptr)
    }

    fn decode_args(
        &self,
        stream: &CudaStream,
        weight: &CudaNumericFp8Artifact,
        target: &DeviceBuffer<u16>,
    ) -> Result<NumericArgs> {
        self.owner(stream)?;
        let l = weight.layout();
        // Decode has no M or warmed-shape dependency. Reuse numeric geometry
        // admission, without charging or allocating another provider workspace.
        Bf16GemmExPlan::new(l, 1, usize::MAX, usize::MAX)?;
        let (wb, sb) = l.storage_lengths()?;
        let (w, s) = weight.buffers();
        let args = NumericArgs {
            m: 1,
            n: l.n as u32,
            k: l.k as u32,
            padded_k: l.k as u32,
            tile_rows: l.n as u32,
            scale_cols: (l.k / 128) as u32,
            row_origin: 0,
            column_origin: 0,
            scale_bytes: 2,
            lda: l.k as u32,
            ldd: l.n as u32,
            reserved: 0,
            activation: 0,
            weight: self.resident_operand(w, wb)?,
            scales: self.resident_operand(s, sb)?,
            output: self.resident_operand(target, wb)?,
            workspace: 0,
            workspace_bytes: 0,
            stream: stream.cu_stream() as usize as u64,
        };
        status(unsafe { ferrule_bf16_gemmex_decode_can(&args) })?;
        Ok(args)
    }

    pub(crate) fn decode_into(
        &self,
        stream: &CudaStream,
        weight: &CudaNumericFp8Artifact,
        target: &DeviceBuffer<u16>,
    ) -> Result<()> {
        let args = self.decode_args(stream, weight, target)?;
        self.submit_view(stream, || unsafe {
            ferrule_bf16_gemmex_decode_launch(&args)
        })
    }

    pub(crate) fn decode_pair_into(
        &self,
        stream: &CudaStream,
        weights: [&CudaNumericFp8Artifact; 2],
        target: &DeviceBuffer<u16>,
    ) -> Result<()> {
        self.owner(stream)?;
        if weights[0].layout() != weights[1].layout() {
            return Err(invalid("pair decode requires identical numeric layouts"));
        }
        let (count, _) = weights[0].layout().storage_lengths()?;
        let total = count
            .checked_mul(2)
            .ok_or_else(|| invalid("pair decode extent overflow"))?;
        self.resident_operand(target, total)?;
        // DeviceBuffer slices retain the same allocation, without staging/copy.
        // Validate both sources and cross-source aliases before submitting either.
        let gate = cu(target.slice(0, count))?;
        let up = cu(target.slice(count, count))?;
        let a = self.decode_args(stream, weights[0], &gate)?;
        let b = self.decode_args(stream, weights[1], &up)?;
        status(unsafe { ferrule_bf16_gemmex_decode_pair_can(&a, &b) })?;
        self.submit_view(stream, || unsafe {
            ferrule_bf16_gemmex_decode_pair_launch(&a, &b)
        })
    }

    pub(crate) fn launch_resident_pair<const SWIGLU: bool>(
        &self,
        stream: &CudaStream,
        plan: Bf16GemmExPlan,
        weight: &DeviceBuffer<u16>,
        input: &DeviceBuffer<u16>,
        outputs: [&DeviceBuffer<u16>; 2],
    ) -> Result<()> {
        self.owner(stream)?;
        if !self.views.contains(&plan) {
            return Err(invalid("unprepared resident pair shape"));
        }
        let l = plan.layout();
        if !l.n.is_multiple_of(256) || !l.k.is_multiple_of(128) {
            return Err(invalid(
                "resident pair requires original numeric 2N geometry",
            ));
        }
        let storage = self.storage.as_ref().unwrap();
        let workspace = storage.cu_deviceptr();
        let args = ResidentPairArgs {
            gemm: ResidentArgs {
                m: plan.rows() as u32,
                n: l.n as u32,
                k: l.k as u32,
                reserved: 0,
                input: self.resident_operand(input, plan.rows() * l.k)?,
                weight: self.resident_operand(weight, l.n * l.k)?,
                output: workspace + ((plan.rows() + l.n) * l.k * 2) as u64,
                stream: stream.cu_stream() as usize as u64,
            },
            gate: self.resident_operand(outputs[0], plan.rows() * (l.n / 2))?,
            up: self.resident_operand(outputs[1], plan.rows() * (l.n / 2))?,
            workspace,
            workspace_bytes: storage.num_bytes() as u64,
        };
        let (can, launch) = if SWIGLU {
            (
                ferrule_bf16_gemmex_resident_pair_swiglu_can
                    as unsafe extern "C" fn(*const ResidentPairArgs) -> i32,
                ferrule_bf16_gemmex_resident_pair_swiglu_launch
                    as unsafe extern "C" fn(*const ResidentPairArgs, u64) -> i32,
            )
        } else {
            (
                ferrule_bf16_gemmex_resident_pair_can
                    as unsafe extern "C" fn(*const ResidentPairArgs) -> i32,
                ferrule_bf16_gemmex_resident_pair_launch
                    as unsafe extern "C" fn(*const ResidentPairArgs, u64) -> i32,
            )
        };
        status(unsafe { can(&args) })?;
        self.submit_view(stream, || unsafe { launch(&args, self.raw) })
    }
}

#[repr(C)]
struct ResidentPairArgs {
    gemm: ResidentArgs,
    gate: u64,
    up: u64,
    workspace: u64,
    workspace_bytes: u64,
}
unsafe extern "C" {
    fn ferrule_bf16_gemmex_can_pair_swiglu(a: *const NumericArgs, b: *const NumericArgs) -> i32;
    fn ferrule_bf16_gemmex_launch_pair_swiglu(
        a: *const NumericArgs,
        b: *const NumericArgs,
        handle: u64,
    ) -> i32;
    fn ferrule_bf16_gemmex_resident_pair_swiglu_can(args: *const ResidentPairArgs) -> i32;
    fn ferrule_bf16_gemmex_resident_pair_swiglu_launch(
        args: *const ResidentPairArgs,
        handle: u64,
    ) -> i32;
    fn ferrule_bf16_gemmex_decode_can(args: *const NumericArgs) -> i32;
    fn ferrule_bf16_gemmex_decode_launch(args: *const NumericArgs) -> i32;
    fn ferrule_bf16_gemmex_decode_pair_can(a: *const NumericArgs, b: *const NumericArgs) -> i32;
    fn ferrule_bf16_gemmex_decode_pair_launch(a: *const NumericArgs, b: *const NumericArgs) -> i32;
    fn ferrule_bf16_gemmex_resident_pair_can(args: *const ResidentPairArgs) -> i32;
    fn ferrule_bf16_gemmex_resident_pair_launch(args: *const ResidentPairArgs, handle: u64) -> i32;
}

#[cfg(test)]
#[path = "../../../tests/numeric_bf16_gemv/mixed_decode_reuse.rs"]
mod mixed_decode_reuse_tests;

#[cfg(test)]
#[path = "../../../tests/numeric_bf16_gemv/pair_swiglu.rs"]
mod pair_swiglu_tests;
