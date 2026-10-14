//! Ignored, authenticated, supervised two-card backend gates. Never model execution.
use super::super::tp_reduce_contract::ordered_host_word;
use super::*;
use crate::cuda::runtime::{CudaContext, GraphFault, copy_device_to_device};
use std::{cell::Cell, path::Path, process::Command, thread, time::Duration};

const UUIDS: &str =
    "GPU-b980f22b-58a3-fc88-21c1-ae86053d8535,GPU-63f1bd01-ddf9-a742-ee48-9389892493d9";
thread_local! { static STATUS_BYTES: Cell<usize> = const { Cell::new(0) }; }
pub(crate) fn status_copied() {
    STATUS_BYTES.set(STATUS_BYTES.get() + 4);
}

fn authorize_only() {
    assert_eq!(
        std::env::var("FERRULE_TP_REDUCE_GPU_AUTH").as_deref(),
        Ok("main-authorized-two-card-proof")
    );
    assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").as_deref(), Ok(UUIDS));
    assert_eq!(
        std::env::var("NVIDIA_VISIBLE_DEVICES").as_deref(),
        Ok(UUIDS)
    );
    assert!(
        !std::env::var("FERRULE_PAPER_WORKER_TOKEN")
            .unwrap()
            .is_empty()
    );
    let receipt = std::env::var("FERRULE_TP_REDUCE_AUTH_RECEIPT").unwrap();
    assert!(
        Path::new(&receipt).is_file(),
        "maintained supervisor authentication required"
    );
}
fn authorized_library() -> Arc<NcclLibrary> {
    authorize_only();
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/vendor/vllm-baseline/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2")
        .canonicalize().unwrap();
    let library =
        NcclLibrary::load(&path, NcclLibrary::required_native_handle_bytes().unwrap()).unwrap();
    println!(
        "TP_LIBRARY {{\"source\":{:?},\"sha256\":{:?},\"version\":{},\"native_metadata_minimum_bytes\":{}}}",
        path.display().to_string(),
        library.sha256(),
        NCCL_VERSION,
        NcclLibrary::required_native_handle_bytes().unwrap()
    );
    library
}
fn nvml(field: &str) -> String {
    // Every query is a supervised descendant with its own 5s timeout.
    let out = Command::new("/usr/bin/timeout")
        .args([
            "--signal=TERM",
            "--kill-after=2",
            "5",
            "/usr/bin/nvidia-smi",
            field,
            "--format=csv,noheader,nounits",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "bounded NVML query failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}
fn task_roster() -> Vec<String> {
    let mut tasks = std::fs::read_dir("/proc/self/task")
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            format!(
                "{}:{}",
                entry.file_name().to_string_lossy(),
                std::fs::read_to_string(entry.path().join("comm"))
                    .unwrap_or_default()
                    .trim()
            )
        })
        .collect::<Vec<_>>();
    tasks.sort();
    tasks
}
fn libraries() -> Vec<String> {
    let mut libs = std::fs::read_to_string("/proc/self/maps")
        .unwrap()
        .lines()
        .filter_map(|line| line.split_whitespace().last())
        .filter(|path| {
            path.starts_with('/')
                && [
                    "libcuda",
                    "libnccl",
                    "libcublas",
                    "libnvidia",
                    "libnvrtc",
                    "libcupti",
                ]
                .iter()
                .any(|s| path.contains(s))
        })
        .map(str::to_owned)
        .collect::<Vec<_>>();
    libs.sort();
    libs.dedup();
    libs
}
fn snapshot(stage: &str, contexts: &[Arc<CudaContext>]) {
    let csv = nvml("--query-gpu=index,uuid,memory.total,memory.free,memory.used,utilization.gpu");
    assert_eq!(csv.lines().count(), 8);
    let ranks = contexts.iter().map(|ctx| {
        let (free, total) = ctx.memory_info().unwrap();
        let m = ctx.allocator_metrics();
        format!("{{\"ordinal\":{},\"cu_free_bytes\":{},\"cu_total_bytes\":{},\"live_grant_bytes\":{},\"reserved_bytes\":{},\"driver_allocations\":{},\"driver_allocation_bytes\":{},\"driver_frees\":{},\"pending_batches\":{},\"unknown\":{}}}",
            ctx.ordinal(), free, total, m.live_granted_bytes, m.reserved_bytes, m.driver_allocations,
            m.driver_allocation_bytes, m.driver_frees, m.pending_retirement_batches, ctx.graph_completion_unknown())
    }).collect::<Vec<_>>().join(",");
    println!(
        "TP_MEMORY {{\"stage\":{:?},\"pid\":{},\"all_eight_nvml_csv\":{:?},\"nvml_unit\":\"MiB; not CU byte precision\",\"selected_cu\":[{}],\"known_storage_grant_per_rank\":{},\"private_bound\":\"Unknown\",\"tasks\":{:?},\"libraries\":{:?}}}",
        stage,
        std::process::id(),
        csv,
        ranks,
        STORAGE_GRANT_BYTES,
        task_roster(),
        libraries()
    );
}
pub(crate) fn initialization_snapshot(pair: &InternalTpReduceTwo, stage: &str) {
    if std::env::var("FERRULE_TP_REDUCE_MEMORY_PROBE").as_deref() == Ok("1") {
        snapshot(
            stage,
            &pair
                .ranks()
                .each_ref()
                .map(|r| Arc::clone(r.stream.context())),
        );
    }
}
fn assert_process_gpu_idle() {
    let apps = nvml("--query-compute-apps=gpu_uuid,pid,used_memory");
    assert!(
        apps.trim().is_empty(),
        "owners dropped but CUDA primary/device custody remains: {apps}"
    );
    let csv = nvml("--query-gpu=index,uuid,memory.total,memory.free,memory.used,utilization.gpu");
    for row in csv.lines() {
        let fields: Vec<_> = row.split(',').map(str::trim).collect();
        assert_eq!(fields[4], "0", "owner-drop GPU memory did not drain: {row}");
        assert_eq!(
            fields[5], "0",
            "owner-drop utilization did not drain: {row}"
        );
    }
}
fn prepared(owners: [&CudaOperators; 2], library: Arc<NcclLibrary>) -> InternalTpReduceTwo {
    let mut pair = InternalTpReduceTwo::start(
        owners,
        TpReducePlan::new([37, 9], STORAGE_GRANT_BYTES).unwrap(),
        library,
    )
    .unwrap();
    complete(&mut pair, owners);
    assert!(!pair.validated_output);
    assert_eq!(pair.plan().members(), [37, 9]);
    initialization_snapshot(&pair, "afterwarm_rows8_count16384");
    pair
}
fn enqueued(pair: &mut InternalTpReduceTwo, owners: [&CudaOperators; 2]) {
    for _ in 0..10000 {
        if pair.advance(owners).unwrap() == TpProgress::Enqueued {
            return;
        }
        thread::sleep(Duration::from_millis(1));
    }
    pair.abort_unknown();
    panic!("bounded host enqueue timed out");
}
fn complete(pair: &mut InternalTpReduceTwo, owners: [&CudaOperators; 2]) {
    for _ in 0..10000 {
        match pair.advance(owners).unwrap() {
            TpProgress::Enqueued => {
                if pair.check_at_quiescence(owners).unwrap() == TpProgress::Complete {
                    return;
                }
            }
            TpProgress::Complete => return,
            TpProgress::Pending => {}
        }
        thread::sleep(Duration::from_millis(1));
    }
    pair.abort_unknown();
    panic!("bounded two-card completion timed out; no release proof");
}
fn words(buffer: &CudaBf16Buffer, op: &CudaOperators) -> Vec<u16> {
    buffer
        .as_device_buffer()
        .to_host_vec(&op.stream_clone())
        .unwrap()
}
struct ProducedInput {
    input: CudaBf16Buffer,
    _source: CudaBf16Buffer,
    _producer: Arc<crate::cuda::runtime::CudaStream>,
    _completion: crate::cuda::runtime::CudaEvent,
}
impl std::ops::Deref for ProducedInput {
    type Target = CudaBf16Buffer;
    fn deref(&self) -> &Self::Target {
        &self.input
    }
}
fn producer_input(op: &CudaOperators, values: &[u16]) -> ProducedInput {
    let source = op.upload_bf16_words(values).unwrap();
    let compute = op.stream_clone();
    let producer = compute.fork().unwrap();
    let destination = DeviceBuffer::<u16>::zeroed(&producer, values.len()).unwrap();
    copy_device_to_device(
        &producer,
        destination.cu_deviceptr(),
        source.as_device_buffer().cu_deviceptr(),
        destination.num_bytes(),
    )
    .unwrap();
    let produced = producer.record_event(None).unwrap();
    compute.wait(&produced).unwrap(); // Actual producer -> compute dependency, NO host completion fence.
    ProducedInput {
        input: CudaBf16Buffer::from_device_buffer(destination),
        _source: source,
        _producer: producer,
        _completion: produced,
    }
}
fn assert_no_site_host_work(ops: [&CudaOperators; 2]) {
    for op in ops {
        let c = op.counters();
        assert_eq!(c.device_to_host_bytes, 0);
        assert_eq!(c.host_to_device_bytes, 0);
        assert_eq!(c.stream_wide_syncs, 0);
    }
}

#[test]
#[ignore = "authenticated main sole GPU authorization; fresh supervised process only"]
fn two_card_mapping_m1_m5_rows8_edges_and_normal_owner_drop() {
    let library = authorized_library();
    snapshot("beforecontext", &[]);
    let a = CudaOperators::new_on_device(0).unwrap();
    let b = CudaOperators::new_on_device(1).unwrap();
    let contexts = [
        a.stream_clone().context().clone(),
        b.stream_clone().context().clone(),
    ];
    snapshot("aftercontext", &contexts);
    let baseline_tasks: std::collections::BTreeSet<_> = task_roster()
        .into_iter()
        .map(|t| t.split(':').next().unwrap().to_owned())
        .collect();
    let weak_contexts = contexts.each_ref().map(Arc::downgrade);
    let weak_library = Arc::downgrade(&library);
    let ops = [&a, &b];
    let mut pair = prepared(ops, library);
    let post_init_tasks: std::collections::BTreeSet<_> = task_roster()
        .into_iter()
        .map(|t| t.split(':').next().unwrap().to_owned())
        .filter(|t| !baseline_tasks.contains(t))
        .collect();
    let edges = [
        (0x8000, 0x8000),
        (0x0000, 0x8000),
        (0x0001, 0x0001),
        (0x0001, 0x8001),
        (0x3f80, 0x3b80),
        (0x3f81, 0x3b80),
        (0x7f7f, 0xff7f),
        (0x8001, 0x8001),
        (0x3f80, 0x4000),
    ];
    for rows in [1, 5, 8] {
        let count = rows * HIDDEN;
        let first: Vec<u16> = (0..count).map(|i| edges[i % edges.len()].0).collect();
        let second: Vec<u16> = (0..count).map(|i| edges[i % edges.len()].1).collect();
        let x = producer_input(&a, &first);
        let y = producer_input(&b, &second);
        assert!(pair.enqueue([&b, &a], [&x, &y], rows).is_err());
        assert!(pair.enqueue(ops, [&y, &x], rows).is_err());
        let short = y.slice(0, count - 1).unwrap();
        assert!(pair.enqueue(ops, [&x, &short], rows).is_err());
        assert_eq!(pair.phase, Phase::Ready);
        for op in ops {
            op.reset_counters();
        }
        STATUS_BYTES.set(0);
        pair.enqueue(ops, [&x, &y], rows).unwrap();
        assert_eq!(STATUS_BYTES.get(), 0);
        assert!(pair.output(0).is_err());
        complete(&mut pair, ops);
        assert_no_site_host_work(ops);
        assert_eq!(STATUS_BYTES.get(), 8); // Exactly one actual 4-byte status read per rank.
        let expected: Vec<u16> = first
            .iter()
            .zip(&second)
            .map(|(&x, &y)| ordered_host_word(x, y).unwrap())
            .collect();
        for rank in 0..2 {
            assert_eq!(words(&pair.output(rank).unwrap(), ops[rank]), expected);
            let gather = pair.ranks()[rank]
                .storage
                .slice(MAX_COUNT, 2 * count)
                .unwrap()
                .to_host_vec(&pair.ranks()[rank].stream)
                .unwrap();
            assert_eq!(&gather[..count], &first);
            assert_eq!(&gather[count..], &second);
        }
        println!(
            "TP_EXACT {{\"rows\":{},\"count\":{},\"members\":[37,9],\"word_exact\":true,\"producer_event_dependencies\":true,\"site_payload_d2h_bytes\":0,\"completion_status_d2h_bytes\":8}}",
            rows, count
        );
    }
    snapshot("afterfinite_checks", &contexts);
    drop(pair);
    assert!(
        weak_library.upgrade().is_none(),
        "normal comm/library teardown quarantined unexpectedly"
    );
    snapshot("afterNcclDrop_beforeOwnerDrop", &contexts);
    let after_drop_tasks: std::collections::BTreeSet<_> = task_roster()
        .into_iter()
        .map(|t| t.split(':').next().unwrap().to_owned())
        .collect();
    assert!(
        post_init_tasks.is_disjoint(&after_drop_tasks),
        "post-init threads not drained"
    );
    drop(contexts);
    drop(a);
    drop(b);
    assert!(weak_contexts.iter().all(|ctx| ctx.upgrade().is_none()));
    snapshot("aftertwoOwnerDrop_normal", &[]);
    assert_process_gpu_idle();
    println!(
        "TP_NORMAL_DRAIN {{\"rust_owners_released\":true,\"nccl_threads_drained\":true,\"all_eight_idle_in_process\":true}}"
    );
}

#[test]
#[ignore = "authenticated main sole GPU authorization; fresh supervised process only"]
fn post_enqueue_unknown_abort_is_not_completion_joint_retained_until_process_exit() {
    let library = authorized_library();
    let weak_library = Arc::downgrade(&library);
    let a = CudaOperators::new_on_device(0).unwrap();
    let b = CudaOperators::new_on_device(1).unwrap();
    let ops = [&a, &b];
    let mut pair = prepared(ops, library);
    let x = producer_input(&a, &vec![0x3f80; HIDDEN]);
    let y = producer_input(&b, &vec![0x4000; HIDDEN]);
    pair.enqueue(ops, [&x, &y], 1).unwrap();
    enqueued(&mut pair, ops);
    let output = unsafe { pair.tentative_output(0) }.unwrap();
    let handles = pair.ranks().each_ref().map(|r| r.raw as usize);
    let contexts = pair
        .ranks()
        .each_ref()
        .map(|r| Arc::clone(r.stream.context()));
    contexts[0].arm_graph_fault(GraphFault::EventQuery);
    assert!(pair.check_at_quiescence(ops).is_err());
    assert!(pair.is_unknown());
    assert_eq!(pair.input_custody.len(), 1);
    assert!(pair.output(0).is_err());
    assert!(pair.enqueue(ops, [&x, &y], 1).is_err());
    assert!(pair.advance(ops).is_err());
    pair.abort_unknown(); // No completion claim; every opaque wrapper/backing stays retained.
    assert!(pair.check_at_quiescence(ops).is_err());
    assert_eq!(handles, pair.ranks().each_ref().map(|r| r.raw as usize));
    drop(output);
    drop(x);
    drop(y);
    let counts = contexts.each_ref().map(Arc::strong_count);
    let grants = contexts
        .each_ref()
        .map(|ctx| ctx.allocator_metrics().live_granted_bytes);
    snapshot("unknown_beforePairDrop", &contexts);
    drop(pair);
    drop(a);
    drop(b);
    for i in 0..2 {
        assert!(contexts[i].graph_completion_unknown());
        // Dropping owners removes their own context/stream leases, but not the retained bundle.
        assert!(Arc::strong_count(&contexts[i]) > 1);
        assert!(counts[i] > 1);
        assert_eq!(
            contexts[i].allocator_metrics().live_granted_bytes,
            grants[i]
        );
        assert_eq!(contexts[i].graph_calls(GraphFault::PrimaryRelease), 0);
    }
    assert!(weak_library.upgrade().is_some());
    snapshot("unknown_afterPairAndOwnerDrop_retained_notFreed", &contexts);
    println!(
        "TP_UNKNOWN {{\"injected_event_query_failure\":true,\"post_enqueue\":true,\"no_replay\":true,\"abort_is_completion\":false,\"comm_input_output_library_context_joint_retained\":true,\"recovery\":\"independent process exit only\"}}"
    );
}

#[path = "gpu_numeric.rs"]
mod numeric;

#[path = "ownership.rs"]
mod ownership;
