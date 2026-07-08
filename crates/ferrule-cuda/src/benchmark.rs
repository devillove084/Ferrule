use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use ferrule_common::Result;

use crate::context::cu;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CudaSmokeBenchmark {
    pub dim: usize,
    pub cpu_ms: f64,
    pub gpu_gemv_ms: f64,
    pub kernel_launch_overhead_us: f64,
    pub rms_us: f64,
}

impl CudaSmokeBenchmark {
    pub fn speedup(&self) -> f64 {
        self.cpu_ms / self.gpu_gemv_ms
    }
}

pub fn run_smoke_benchmark() -> Result<CudaSmokeBenchmark> {
    run_gemv_rms_smoke_benchmark(2048, 2_000, 5_000, 1_000)
}

pub fn run_gemv_rms_smoke_benchmark(
    dim: usize,
    gemv_iters: usize,
    empty_iters: usize,
    rms_iters: usize,
) -> Result<CudaSmokeBenchmark> {
    let x: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
    let w: Vec<f32> = (0..dim * dim).map(|i| (i as f32).cos()).collect();

    let ctx = cu(CudaContext::new(0))?;
    cu(ctx.bind_to_thread())?;
    let module = cu(crate::kernels::kernels::load(&ctx))?;
    let stream = ctx.default_stream();
    let xd = cu(DeviceBuffer::from_host(&stream, &x))?;
    let wd = cu(DeviceBuffer::from_host(&stream, &w))?;
    let mut yd = cu(DeviceBuffer::<f32>::zeroed(&stream, dim))?;

    for _ in 0..10 {
        unsafe {
            module.gemv_f32(
                &stream,
                LaunchConfig::for_num_elems(dim as u32),
                &xd,
                &wd,
                &mut yd,
                dim as u32,
                dim as u32,
            )
        }
        .map_err(|e| ferrule_common::Error::Internal(format!("CUDA {e:?}")))?;
    }

    let t0 = std::time::Instant::now();
    for _ in 0..gemv_iters {
        unsafe {
            module.gemv_f32(
                &stream,
                LaunchConfig::for_num_elems(dim as u32),
                &xd,
                &wd,
                &mut yd,
                dim as u32,
                dim as u32,
            )
        }
        .map_err(|e| ferrule_common::Error::Internal(format!("CUDA {e:?}")))?;
    }
    let gpu_gemv_ms = t0.elapsed().as_secs_f64() * 1000.0 / gemv_iters as f64;

    let mut rms_buf = cu(DeviceBuffer::<f32>::zeroed(&stream, 1))?;
    let dummy = cu(DeviceBuffer::<f32>::zeroed(&stream, 1))?;
    let t0 = std::time::Instant::now();
    for _ in 0..empty_iters {
        unsafe {
            module.compute_rms(
                &stream,
                LaunchConfig::for_num_elems(1u32),
                &dummy,
                &mut rms_buf,
                1u32,
                1e-5f32,
            )
        }
        .map_err(|e| ferrule_common::Error::Internal(format!("CUDA {e:?}")))?;
    }
    let kernel_launch_overhead_us = t0.elapsed().as_secs_f64() * 1e6 / empty_iters as f64;

    let hidden_buf = cu(DeviceBuffer::<f32>::zeroed(&stream, dim))?;
    let t0 = std::time::Instant::now();
    for _ in 0..rms_iters {
        unsafe {
            module.compute_rms(
                &stream,
                LaunchConfig::for_num_elems(1u32),
                &hidden_buf,
                &mut rms_buf,
                dim as u32,
                1e-5f32,
            )
        }
        .map_err(|e| ferrule_common::Error::Internal(format!("CUDA {e:?}")))?;
    }
    let rms_us = t0.elapsed().as_secs_f64() * 1e6 / rms_iters as f64;

    let t0 = std::time::Instant::now();
    for _ in 0..gemv_iters {
        let mut out = vec![0f32; dim];
        for j in 0..dim {
            let row = &w[j * dim..(j + 1) * dim];
            out[j] = row.iter().zip(x.iter()).map(|(r, xi)| r * xi).sum();
        }
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / gemv_iters as f64;

    Ok(CudaSmokeBenchmark {
        dim,
        cpu_ms,
        gpu_gemv_ms,
        kernel_launch_overhead_us,
        rms_us,
    })
}
