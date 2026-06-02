pub mod mha;
// use cuda_device::{kernel, thread, DisjointSlice};
// use cuda_host::cuda_module;
// use cuda_async::device_context::init_device_contexts;
// use cuda_async::device_operation::DeviceOperation;
// use cuda_core::LaunchConfig;

// #[cuda_module]
// mod kernels {
//     use super::*;

//     #[kernel]
//     pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
//         let idx = thread::index_1d();
//         let idx_raw = idx.get();
//         if let Some(c_elem) = c.get_mut(idx) {
//             *c_elem = a[idx_raw] + b[idx_raw];
//         }
//     }
// }

// async fn run() -> Result<(), Box<dyn std::error::Error>> {
//     use cuda_async::device_box::DeviceBox;
//     use cuda_core::memory::{malloc_async, memcpy_dtoh_async, memcpy_htod_async};
//     use std::mem;

//     init_device_contexts(0, 1)?;
//     let module = kernels::load_async(0)?;

//     const N: usize = 1024;
//     let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
//     let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

//     let (a_dev, b_dev, mut c_dev) = cuda_async::device_context::with_cuda_context(0, |ctx| {
//         let stream = ctx.default_stream();
//         let num_bytes = N * mem::size_of::<f32>();
//         unsafe {
//             let a = malloc_async(stream.cu_stream(), num_bytes).unwrap();
//             let b = malloc_async(stream.cu_stream(), num_bytes).unwrap();
//             let c = malloc_async(stream.cu_stream(), num_bytes).unwrap();
//             memcpy_htod_async(a, a_host.as_ptr(), num_bytes, stream.cu_stream()).unwrap();
//             memcpy_htod_async(b, b_host.as_ptr(), num_bytes, stream.cu_stream()).unwrap();
//             stream.synchronize().unwrap();
//             (
//                 DeviceBox::<[f32]>::from_raw_parts(a, N, 0),
//                 DeviceBox::<[f32]>::from_raw_parts(b, N, 0),
//                 DeviceBox::<[f32]>::from_raw_parts(c, N, 0),
//             )
//         }
//     })?;

//     module
//         .vecadd_async(
//             LaunchConfig::for_num_elems(N as u32),
//             &a_dev,
//             &b_dev,
//             &mut c_dev,
//         )?
//         .sync()?;

//     let mut c_host = vec![0.0f32; N];
//     cuda_async::device_context::with_cuda_context(0, |ctx| {
//         let stream = ctx.default_stream();
//         unsafe {
//             memcpy_dtoh_async(
//                 c_host.as_mut_ptr(),
//                 c_dev.cu_deviceptr(),
//                 N * mem::size_of::<f32>(),
//                 stream.cu_stream(),
//             )
//             .unwrap();
//             stream.synchronize().unwrap();
//         }
//     })?;

//     let errors = (0..N)
//         .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
//         .count();

//     if errors == 0 {
//         println!("PASSED: all {} elements correct", N);
//     } else {
//         eprintln!("FAILED: {} errors", errors);
//         std::process::exit(1);
//     }

//     Ok(())
// }

pub mod mha;
