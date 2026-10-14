use super::*;
use crate::cuda::operators::linear::{CudaBf16Buffer, CudaOperators};
use crate::cuda_plan::{NumericFp8Layout, NumericFp8ScaleType};
fn plan(m: usize, n: usize, k: usize) -> Bf16GemmExPlan {
    Bf16GemmExPlan::new(
        NumericFp8Layout {
            n,
            k,
            row_origin: 0,
            column_origin: 0,
            scale_type: NumericFp8ScaleType::Bf16,
        },
        m,
        128 << 20,
        16 << 20,
    )
    .unwrap()
}
#[test]
#[ignore = "sole GPU owner; bounded multi-shape provider warmup/reuse and rejection gate"]
fn bf16_gemmex_shared_shapes_resource_gate() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let plans = [
        plan(1, 8192, 2048),
        plan(5, 8192, 2048),
        plan(2, 1024, 2048),
        plan(1, 2048, 512),
        plan(5, 256, 256),
        plan(32, 128, 128),
    ];
    let max = *plans.iter().max_by_key(|p| p.storage_bytes()).unwrap();
    let baseline = op.allocator_metrics().live_requested_bytes;
    assert!(
        op.prepare_bf16_gemmex_views(&plans, max.storage_grant_bytes() - 1, max.provider_bytes())
            .is_err()
    );
    assert!(
        op.prepare_bf16_gemmex_views(&plans, max.storage_grant_bytes(), max.provider_bytes() - 1)
            .is_err()
    );
    let mut provider = op
        .prepare_bf16_gemmex_views(&plans, max.storage_grant_bytes(), max.provider_bytes())
        .unwrap();
    assert_eq!(
        op.allocator_metrics().live_requested_bytes - baseline,
        max.storage_bytes()
    );
    assert!(provider.provider_observed_growth_bytes() <= 16 << 20);
    eprintln!(
        "GEMMEX_SHARED_POOL shapes={} storage={} provider_observed={} workspace_bound_once=true",
        plans.len(),
        max.storage_bytes(),
        provider.provider_observed_growth_bytes()
    );
    for p in plans.into_iter().chain(plans.into_iter().rev()) {
        let l = p.layout();
        let w = op
            .upload_numeric_fp8_linear(
                l,
                &vec![0x38; l.n * l.k],
                &[0x80, 0x3f].repeat(l.n * l.k / 16384),
            )
            .unwrap();
        let x = op.upload_bf16_words(&vec![0x3f80; p.rows() * l.k]).unwrap();
        let mut y = op.zero_bf16_buffer(p.rows() * l.n).unwrap();
        let memory = op.memory_info().unwrap().0;
        op.reset_counters();
        provider.execute_view(&op, p, &w, &x, &mut y).unwrap();
        assert_eq!(op.counters().device_allocation_attempts, 0);
        assert_eq!(op.counters().stream_wide_syncs, 0);
        assert_eq!(op.counters().kernel_launches, 3);
        assert_eq!(
            op.download_bf16_buffer(&y).unwrap(),
            vec![l.k as f32; p.rows() * l.n]
        );
        assert_eq!(op.memory_info().unwrap().0, memory);
        let wrong = plan(3, l.n, l.k);
        assert!(provider.execute_view(&op, wrong, &w, &x, &mut y).is_err());
        assert!(!provider.is_poisoned());
    }
    let p = plan(2, 1024, 2048);
    let half = plan(2, 512, 2048);
    let w = op
        .upload_numeric_fp8_linear(
            half.layout(),
            &vec![0x38; 512 * 2048],
            &[0x80, 0x3f].repeat(64),
        )
        .unwrap();
    let x = op.upload_bf16_words(&vec![0x3f80; 4096]).unwrap();
    let mut a = op.zero_bf16_buffer(1024).unwrap();
    let mut b = op.zero_bf16_buffer(1024).unwrap();
    op.reset_counters();
    provider
        .execute_pair(&op, p, [&w, &w], &x, [&mut a, &mut b])
        .unwrap();
    assert_eq!(op.counters().kernel_launches, 5);
    assert_eq!(op.counters().device_allocation_attempts, 0);
    assert_eq!(op.counters().stream_wide_syncs, 0);
    for y in [&a, &b] {
        assert_eq!(op.download_bf16_buffer(y).unwrap(), vec![2048.; 1024]);
    }
    let mut alias =
        CudaBf16Buffer::from_device_buffer(a.as_device_buffer().slice(0, 1024).unwrap());
    assert!(
        provider
            .execute_pair(&op, p, [&w, &w], &x, [&mut a, &mut alias])
            .is_err()
    );
    assert!(!provider.is_poisoned());
    let foreign = CudaOperators::new_on_device(0).unwrap();
    assert!(
        provider
            .execute_pair(&foreign, p, [&w, &w], &x, [&mut a, &mut b])
            .is_err()
    );
    assert!(!provider.is_poisoned());
    drop(foreign);
    let stream = op.stream_clone();
    stream.begin_capture().unwrap();
    assert!(
        provider
            .execute_pair(&op, p, [&w, &w], &x, [&mut a, &mut b])
            .is_err()
    );
    drop(stream.end_capture().unwrap());
    assert!(!provider.is_poisoned());
    drop((provider, w, x, a, b, alias));
    op.trim_device_allocator().unwrap();
    assert_eq!(op.allocator_metrics().live_requested_bytes, baseline);
}

#[test]
fn bf16_gemmex_pair_preflight_without_device_calls() {
    let p = plan(2, 256, 128);
    let args = || NumericArgs {
        m: 2,
        n: 128,
        k: 128,
        padded_k: 128,
        tile_rows: 128,
        scale_cols: 1,
        row_origin: 0,
        column_origin: 0,
        scale_bytes: 2,
        lda: 128,
        ldd: 128,
        reserved: 0,
        activation: 0x1000,
        weight: 0x10000,
        scales: 0x20000,
        output: 0x30000,
        workspace: 0x100000,
        workspace_bytes: p.storage_bytes() as u64,
        stream: 0,
    };
    let a = args();
    let mut b = args();
    b.weight = 0x40000;
    b.scales = 0x50000;
    b.output = 0x60000;
    assert_eq!(unsafe { ferrule_bf16_gemmex_can_pair(&a, &b) }, 0);
    for which in 0..8 {
        let mut bad = NumericArgs {
            weight: b.weight,
            scales: b.scales,
            output: b.output,
            ..args()
        };
        match which {
            0 => bad.output = a.output,
            1 => bad.weight = a.output,
            2 => bad.output = a.scales,
            3 => bad.activation += 256,
            4 => bad.n = 256,
            5 => bad.workspace += 256,
            6 => bad.scale_bytes = 4,
            _ => bad.output = a.workspace + p.staging_bytes() as u64,
        };
        assert_eq!(unsafe { ferrule_bf16_gemmex_can_pair(&a, &bad) }, 1);
    }
    let mut short = a;
    short.workspace_bytes = p.storage_bytes() as u64 - 1;
    assert_eq!(unsafe { ferrule_bf16_gemmex_can_pair(&short, &b) }, 1);
}

#[test]
#[ignore = "sole GPU owner; prepared/graph-retained physical reservation slot"]
fn bf16_gemmex_graph_retained_pool_blocks_reprepare() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let p = plan(1, 128, 128);
    let baseline = op.allocator_metrics().live_requested_bytes;
    let w = op
        .upload_numeric_fp8_linear(p.layout(), &vec![0x38; 128 * 128], &[0x80, 0x3f])
        .unwrap();
    let x = op.upload_bf16_words(&vec![0x3f80; 128]).unwrap();
    let mut y = op.zero_bf16_buffer(128).unwrap();
    let mut provider = op.prepare_bf16_gemmex(p).unwrap();
    let graph = provider.capture(&op, &w, &x, &mut y).unwrap();
    let key = (
        Arc::as_ptr(&op.stream_clone()) as usize,
        ProviderFamily::GemmEx,
    );
    let expected = Charge {
        storage_grant: p.storage_grant_bytes(),
        external: p.provider_bytes(),
    };
    assert_eq!(ledger().owners[&key].1, expected);
    assert!(op.prepare_bf16_gemmex(p).is_err());
    drop(provider);
    assert_eq!(ledger().owners[&key].1, expected);
    assert!(op.prepare_bf16_gemmex(p).is_err());
    graph.launch(&op).unwrap();
    assert_eq!(graph.download(&op).unwrap(), vec![128.; 128]);
    drop(graph);
    assert!(!ledger().owners.contains_key(&key));
    let provider = op.prepare_bf16_gemmex(p).unwrap();
    drop((provider, w, x, y));
    op.trim_device_allocator().unwrap();
    assert_eq!(op.allocator_metrics().live_requested_bytes, baseline);
}
fn graph_failure(fault: crate::cuda::runtime::GraphFault, destroy: bool) {
    let op = CudaOperators::new_on_device(0).unwrap();
    let p = plan(1, 128, 128);
    let w = op
        .upload_numeric_fp8_linear(p.layout(), &vec![0x38; 128 * 128], &[0x80, 0x3f])
        .unwrap();
    let x = op.upload_bf16_words(&vec![0x3f80; 128]).unwrap();
    let mut y = op.zero_bf16_buffer(128).unwrap();
    let mut provider = op.prepare_bf16_gemmex(p).unwrap();
    let graph = provider.capture(&op, &w, &x, &mut y).unwrap();
    graph.launch(&op).unwrap();
    assert_eq!(graph.download(&op).unwrap(), vec![128.; 128]);
    let foreign = CudaOperators::new_on_device(0).unwrap();
    assert!(graph.download(&foreign).is_err());
    assert!(!provider.is_poisoned());
    drop(foreign);
    let before = op.allocator_metrics().live_requested_bytes;
    graph.launch(&op).unwrap();
    op.stream_clone().context().arm_graph_fault(fault);
    if destroy {
        drop(graph);
    } else {
        assert!(graph.download(&op).is_err());
        assert!(graph.launch(&op).is_err());
        assert!(graph.upload(&op).is_err());
        drop(graph);
    }
    assert!(provider.is_poisoned());
    assert!(provider.execute(&op, &w, &x, &mut y).is_err());
    assert!(op.prepare_bf16_gemmex(p).is_err());
    drop((provider, w, x, y));
    assert!(op.allocator_metrics().live_requested_bytes >= before);
    assert!(ledger().owners.contains_key(&(
        Arc::as_ptr(&op.stream_clone()) as usize,
        ProviderFamily::GemmEx
    )));
    eprintln!(
        "GEMMEX_GRAPH_UNKNOWN workspace_and_operands_retained=true execute_replay_prepare_blocked=true live_before={before} live_after={}",
        op.allocator_metrics().live_requested_bytes
    );
}
#[test]
#[ignore = "isolated sole-GPU child; submitted result copy failure"]
fn bf16_gemmex_graph_download_copy_unknown() {
    graph_failure(crate::cuda::runtime::GraphFault::DownloadCopy, false)
}
#[test]
#[ignore = "isolated sole-GPU child; submitted result completion failure"]
fn bf16_gemmex_graph_download_sync_unknown() {
    graph_failure(crate::cuda::runtime::GraphFault::DownloadSynchronize, false)
}
#[test]
#[ignore = "isolated sole-GPU child; graph destruction failure"]
fn bf16_gemmex_graph_destroy_unknown() {
    graph_failure(crate::cuda::runtime::GraphFault::Destroy, true)
}
#[test]
#[ignore = "isolated sole-GPU child; executable destruction failure"]
fn bf16_gemmex_graph_exec_destroy_unknown() {
    graph_failure(crate::cuda::runtime::GraphFault::ExecDestroy, true)
}
#[test]
fn bf16_gemmex_rejects_bad_process_config_before_cuda() {
    if std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref() == Ok(":0:0") {
        return;
    }
    assert_eq!(unsafe { ferrule_bf16_gemmex_process_config_valid() }, 0);
}

#[test]
fn bf16_gemmex_startup_config_is_not_late_mutable_cpu() {
    if std::env::var_os("FERRULE_GEMMEX_STARTUP_CHILD").is_some() {
        // This isolated child runs just this test, with no CUDA initialization
        // and no concurrent environment readers. Production never does this.
        unsafe { std::env::set_var("CUBLAS_WORKSPACE_CONFIG", ":0:0") };
        assert_eq!(unsafe { ferrule_bf16_gemmex_process_config_valid() }, 0);
        return;
    }
    let status=std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact","cuda::providers::bf16_gemmex::view_tests::bf16_gemmex_startup_config_is_not_late_mutable_cpu","--test-threads=1"])
        .env("CUDA_VISIBLE_DEVICES","").env("NVIDIA_VISIBLE_DEVICES","void")
        .env("CUBLAS_WORKSPACE_CONFIG",":4096:8").env("FERRULE_GEMMEX_STARTUP_CHILD","1")
        .status().unwrap();
    assert!(status.success());
}
#[test]
#[ignore = "sole GPU owner; launch this child with literal bad config to prove early rejection"]
fn bf16_gemmex_bad_startup_rejects_before_provider_allocation() {
    assert_ne!(
        std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref(),
        Ok(":0:0")
    );
    let op = CudaOperators::new_on_device(0).unwrap();
    let p = plan(1, 128, 128);
    op.reset_counters();
    let error = op
        .prepare_bf16_gemmex(p)
        .err()
        .expect("bad startup rejected");
    assert!(error.to_string().contains("BEFORE any cuBLAS"), "{error}");
    assert_eq!(op.counters().device_allocation_attempts, 0);
    assert_eq!(op.counters().kernel_launches, 0);
    assert!(!op.stream_clone().context().graph_completion_unknown());
    assert!(!ledger().owners.contains_key(&(
        Arc::as_ptr(&op.stream_clone()) as usize,
        ProviderFamily::GemmEx
    )));
}

#[path = "bf16_gemmex_owner_completion.rs"]
mod owner_completion;
