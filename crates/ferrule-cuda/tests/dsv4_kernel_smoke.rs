//! DSV4 CUDA kernel smoke — runs all kernels on GPU (RTX 3090).

use cuda_core::stream::CudaStream;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use ferrule_core::Result;
use ferrule_cuda::context::{cuda_gemv_fp4_e2m1_e8m0, cuda_gemv_fp8_e4m3fn_e8m0_2d};
use ferrule_cuda::kernels::kernels;
use std::sync::Arc;

fn rc<T, E: std::fmt::Debug>(r: std::result::Result<T, E>) -> Result<T> {
    r.map_err(|e| ferrule_core::Error::Internal(format!("{e:?}")))
}
fn has_cuda() -> bool {
    CudaContext::new(0).is_ok()
}
fn load() -> Result<(Arc<CudaContext>, kernels::LoadedModule, Arc<CudaStream>)> {
    let ctx = rc(CudaContext::new(0))?;
    rc(ctx.bind_to_thread())?;
    let m = rc(kernels::load(&ctx))?;
    let s = ctx.default_stream();
    Ok((ctx, m, s))
}
fn pass<T>(r: std::result::Result<T, impl std::fmt::Debug>, l: &str) {
    match r {
        Ok(_) => eprintln!("  [PASS] {l}"),
        Err(e) => eprintln!("  [FAIL] {l}: {e:?}"),
    }
}

#[test]
fn smoke_all_kernels() {
    if !has_cuda() {
        eprintln!("SKIP: no CUDA");
        return;
    }
    pass(
        cuda_gemv_fp4_e2m1_e8m0(&vec![1.0; 32], &vec![0x42u8; 16], &vec![127u8; 1], 1, 32),
        "FP4 GEMV",
    );
    pass(
        cuda_gemv_fp8_e4m3fn_e8m0_2d(
            &vec![1.0, 2.0],
            &vec![0x38u8; 4],
            &vec![127u8; 4],
            2,
            2,
            1,
            1,
        ),
        "FP8 GEMV 2D",
    );
    let (_c, m, s) = match load() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("load fail: {e}");
            return;
        }
    };

    let x = vec![1.0f32; 64];
    let p = vec![0x42u8; 16];
    let sc = vec![127u8; 1];
    let xd = rc(DeviceBuffer::from_host(&s, &x)).unwrap();
    let pd = rc(DeviceBuffer::from_host(&s, &p)).unwrap();
    let sd = rc(DeviceBuffer::from_host(&s, &sc)).unwrap();
    let mut yd = rc(DeviceBuffer::<f32>::zeroed(&s, 2)).unwrap();
    pass(
        rc(m.gemm_fp4_e2m1_e8m0(
            &s,
            LaunchConfig::for_num_elems(2),
            &xd,
            &pd,
            &sd,
            &mut yd,
            2,
            1,
            32,
        )),
        "Batched FP4 GEMM",
    );

    let (hs, qr, hd) = (64, 16, 32);
    let xd = rc(DeviceBuffer::from_host(&s, &vec![1.0; hs])).unwrap();
    let ad = rc(DeviceBuffer::from_host(&s, &vec![0.0; qr * hs])).unwrap();
    let bd = rc(DeviceBuffer::from_host(&s, &vec![0.0; hd * qr])).unwrap();
    let nd = rc(DeviceBuffer::from_host(&s, &vec![1.0; qr])).unwrap();
    let mut qd = rc(DeviceBuffer::<f32>::zeroed(&s, hd)).unwrap();
    pass(
        rc(m.mla_q_projection_f32(
            &s,
            LaunchConfig::for_num_elems(hd as u32),
            &xd,
            &ad,
            &bd,
            &nd,
            &mut qd,
            hs as u32,
            qr as u32,
            hd as u32,
            1e-6,
        )),
        "MLA Q-proj",
    );

    let (rd, nh) = (64, 4);
    let mut rk = vec![0.0f32; nh * rd];
    for i in 0..rk.len() {
        rk[i] = (i as f32) * 0.01;
    }
    let mut rkd = rc(DeviceBuffer::from_host(&s, &rk)).unwrap();
    let co = vec![0.5f32; rd / 2];
    let si = vec![0.866f32; rd / 2];
    let cd = rc(DeviceBuffer::from_host(&s, &co)).unwrap();
    let sid_r = rc(DeviceBuffer::from_host(&s, &si)).unwrap();
    pass(
        rc(m.rope_yarn(
            &s,
            LaunchConfig::for_num_elems((nh * rd) as u32),
            &mut rkd,
            &cd,
            &sid_r,
            (nh * rd) as u32,
            rd as u32,
            rd as u32,
        )),
        "YAARN RoPE",
    );

    let (hd2, nh2, kvl, tk) = (8, 2, 4, 2);
    let qq = vec![0.1; nh2 * hd2];
    let kvv = vec![0.5; kvl * hd2];
    let tkk: Vec<i32> = (0..tk).map(|i| i as i32).collect();
    let skk = vec![0.0; nh2];
    let qqd = rc(DeviceBuffer::from_host(&s, &qq)).unwrap();
    let kvvd = rc(DeviceBuffer::from_host(&s, &kvv)).unwrap();
    let tkd = rc(DeviceBuffer::from_host(&s, &tkk)).unwrap();
    let sid2 = rc(DeviceBuffer::from_host(&s, &skk)).unwrap();
    let mut od2 = rc(DeviceBuffer::<f32>::zeroed(&s, nh2 * hd2)).unwrap();
    pass(
        rc(m.sparse_attn_tiled_sink_f32(
            &s,
            LaunchConfig::for_num_elems(nh2 as u32),
            &qqd,
            &kvvd,
            &tkd,
            &sid2,
            &mut od2,
            nh2 as u32,
            1,
            kvl as u32,
            nh2 as u32,
            hd2 as u32,
            tk as u32,
            0.5,
        )),
        "Tiled sparse attn",
    );

    let (inter, hid) = (32, 8);
    let gt = vec![0.5; inter];
    let up = vec![0.3; inter];
    let dp = vec![0x42u8; hid * inter / 2];
    let ds = vec![127u8; hid * inter / 32];
    let gd = rc(DeviceBuffer::from_host(&s, &gt)).unwrap();
    let ud = rc(DeviceBuffer::from_host(&s, &up)).unwrap();
    let dpd = rc(DeviceBuffer::from_host(&s, &dp)).unwrap();
    let dsd = rc(DeviceBuffer::from_host(&s, &ds)).unwrap();
    let mut odd = rc(DeviceBuffer::<f32>::zeroed(&s, hid)).unwrap();
    pass(
        rc(m.swiglu_down_accumulate(
            &s,
            LaunchConfig::for_num_elems(hid as u32),
            &gd,
            &ud,
            &dpd,
            &dsd,
            &mut odd,
            inter as u32,
            hid as u32,
            1.0,
            10.0,
        )),
        "SwiGLU accumulate",
    );
}
