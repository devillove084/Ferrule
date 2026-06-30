//! CUDA context helpers — probe, GEMV benchmarks, kernel dispatch.

use cuda_core::stream::CudaStream;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use ferrule_core::{Error, Result};
use ferrule_quant::QuantType;

use crate::kernels::kernels::LoadedModule;

/// Convert a CUDA result into a ferrule Result.
pub(crate) fn cu<T, E: std::fmt::Debug>(r: std::result::Result<T, E>) -> Result<T> {
    r.map_err(|e| Error::Internal(format!("CUDA {e:?}")))
}

// ── Kernel dispatch (selects Q4_0 vs Q2S at runtime) ─────────────────

pub(crate) fn gemv_quant(
    m: &LoadedModule,
    s: &CudaStream,
    qt: QuantType,
    cfg: LaunchConfig,
    x: &DeviceBuffer<f32>,
    packed: &DeviceBuffer<u8>,
    scales: &DeviceBuffer<f32>,
    y: &mut DeviceBuffer<f32>,
    k: u32,
) -> Result<()> {
    match qt {
        QuantType::Q4_0 => cu(m.gemv_q4(s, cfg, x, packed, scales, y, k)),
        QuantType::Q8_0 => cu(m.gemv_q8(s, cfg, x, packed, scales, y, k)),
        QuantType::Q2S => cu(m.gemv_q2(s, cfg, x, packed, scales, y, k)),
        QuantType::T1S => cu(m.gemv_t1(s, cfg, x, packed, scales, y, k)),
    }
}

pub(crate) fn gemv_quant_off(
    m: &LoadedModule,
    s: &CudaStream,
    qt: QuantType,
    cfg: LaunchConfig,
    x: &DeviceBuffer<f32>,
    packed: &DeviceBuffer<u8>,
    scales: &DeviceBuffer<f32>,
    y: &mut DeviceBuffer<f32>,
    k: u32,
    packed_off: u32,
    scales_off: u32,
) -> Result<()> {
    match qt {
        QuantType::Q4_0 => {
            cu(m.gemv_q4_off(s, cfg, x, packed, scales, y, k, packed_off, scales_off))
        }
        QuantType::Q8_0 => {
            cu(m.gemv_q8_off(s, cfg, x, packed, scales, y, k, packed_off, scales_off))
        }
        QuantType::Q2S => {
            cu(m.gemv_q2_off(s, cfg, x, packed, scales, y, k, packed_off, scales_off))
        }
        QuantType::T1S => {
            cu(m.gemv_t1_off(s, cfg, x, packed, scales, y, k, packed_off, scales_off))
        }
    }
}

// ── Device probe ──────────────────────────────────────────────────────

/// Probe the CUDA device and print basic info.
/// No-op when no GPU is available (returns an error).
pub fn cuda_probe() -> Result<()> {
    let ctx = cu(CudaContext::new(0))?;
    let name = cu(ctx.device_name())?;
    cu(ctx.bind_to_thread())?;
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        cuda_bindings::cuMemGetInfo_v2(&mut free, &mut total);
    }
    println!(
        "  Device: {name}\n  Memory: {:.1} GB free / {:.1} GB total",
        free as f64 / 1e9,
        total as f64 / 1e9
    );
    Ok(())
}

// ── Standalone GEMV (benchmark) ───────────────────────────────────────

/// Run a single GEMV on GPU — used for microbenchmarking.
pub fn cuda_gemv(x: &[f32], w: &[f32], out_f: usize) -> Result<Vec<f32>> {
    let ctx = cu(CudaContext::new(0))?;
    cu(ctx.bind_to_thread())?;
    let module = cu(crate::kernels::kernels::load(&ctx))?;
    let s = ctx.default_stream();
    let xd = cu(DeviceBuffer::from_host(&s, x))?;
    let wd = cu(DeviceBuffer::from_host(&s, w))?;
    let mut yd = cu(DeviceBuffer::<f32>::zeroed(&s, out_f))?;
    cu(module.gemv_f32(
        &s,
        LaunchConfig::for_num_elems(out_f as u32),
        &xd,
        &wd,
        &mut yd,
        x.len() as u32,
    ))?;
    cu(yd.to_host_vec(&s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_probe_compiles() {
        // This test just verifies the function signature compiles.
        // cuda_probe requires a real GPU to succeed, so we only
        // check that it doesn't panic or cause a link error.
        let _ = cuda_probe(); // may fail without GPU — that's fine
    }
}
