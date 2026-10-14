//! Fresh-process A/B/C ownership controls; traced siblings are not mixed with main NVML evidence.
use super::*;
use crate::cuda::ffi::tp_reduce as probe;
use std::{ffi::CString, time::Instant};

fn point(role: &str, traced: bool, stage: &str, owners: Option<[&CudaOperators; 2]>) {
    let contexts = owners.map(|ops| ops.map(|op| op.stream_clone().context().clone()));
    snapshot(stage, contexts.as_ref().map_or(&[], |c| &c[..]));
    let primary = (0..2)
        .map(|ordinal| {
            let (mut flags, mut active) = (0, 0);
            assert_eq!(
                unsafe { probe::ferrule_tp_probe_primary(ordinal, &mut flags, &mut active) },
                0
            );
            format!(
                "{{\"ordinal\":{},\"flags\":{},\"active\":{},\"refcount\":null}}",
                ordinal, flags, active
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let counts = if traced {
        let mut data = [0u64; 18];
        assert_eq!(
            unsafe { probe::ferrule_tp_probe_counts(data.as_mut_ptr(), 18) },
            0
        );
        format!("{data:?}")
    } else {
        "null".into()
    };
    println!(
        "TP_PROBE {{\"role\":{:?},\"instrumented\":{},\"stage\":{:?},\"primary\":[{}],\"callback_counts\":{},\"counts_scope\":\"successful driver API callbacks; aggregate; not refcount/private bound\"}}",
        role, traced, stage, primary, counts
    );
}
fn settle_retirement(ops: [&CudaOperators; 2]) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let mut pending = 0;
        for op in ops {
            pending += op.allocator_metrics().pending_retirement_batches;
            op.trim_device_allocator().unwrap(); // Existing completed-retirement proof; no forced device reset/sync.
        }
        if pending == 0 {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "retirement did not settle; no forced cleanup"
        );
        thread::sleep(Duration::from_millis(1));
    }
}
fn core_materialize(ops: [&CudaOperators; 2]) {
    for op in ops {
        let stream = op.stream_clone();
        stream.context().bind_to_thread().unwrap();
        let storage =
            DeviceBuffer::<u16>::uninitialized_dedicated(&stream, 4 * MAX_COUNT + 2).unwrap();
        let code = unsafe {
            probe::ferrule_tp_probe_core(
                stream.cu_stream() as usize as u64,
                storage.cu_deviceptr(),
                storage.num_bytes() as u64,
            )
        };
        if code != 0 {
            stream.context().mark_graph_unknown();
            panic!("core materialization status {code}");
        }
        stream.record_event(None).unwrap().synchronize().unwrap();
        let mut flag = 0;
        assert_eq!(
            unsafe { probe::ferrule_tp_probe_core_status(storage.cu_deviceptr(), &mut flag) },
            0
        );
        assert_eq!(flag, 0);
    }
    settle_retirement(ops);
}
fn resident_input(op: &CudaOperators, value: u16) -> CudaBf16Buffer {
    let stream = op.stream_clone();
    let values = vec![value; MAX_COUNT];
    let buffer = DeviceBuffer::<u16>::uninitialized_dedicated(&stream, MAX_COUNT).unwrap();
    buffer.copy_from_host(&stream, &values).unwrap();
    stream.record_event(None).unwrap().synchronize().unwrap();
    CudaBf16Buffer::from_device_buffer(buffer)
}
fn run(role: &str, traced: bool) {
    authorize_only();
    if traced {
        let path = std::env::var("FERRULE_TP_PROBE_CUPTI_PATH").unwrap();
        let path = CString::new(path).unwrap();
        let result = unsafe { probe::ferrule_tp_probe_trace_start(path.as_ptr()) };
        assert_eq!(
            result, 0,
            "pinned callback collector unavailable; do not invent counts"
        );
    }
    point(role, traced, "baseline_no_owner", None);
    let a = CudaOperators::new_on_device(0).unwrap();
    let b = CudaOperators::new_on_device(1).unwrap();
    let ops = [&a, &b];
    let weak_contexts = ops.map(|op| Arc::downgrade(op.stream_clone().context()));
    point(role, traced, "after_owners", Some(ops));
    core_materialize(ops); // All roles materialize identical CUDA core modules/functions before NCCL.
    point(role, traced, "same_modules_before_nccl", Some(ops));
    let library = if role == "A" {
        None
    } else {
        Some(authorized_library())
    };
    point(role, traced, "after_dlopen_or_control", Some(ops));
    if role == "A" {
        assert!(!libraries().iter().any(|p| p.contains("libnccl")));
    }
    let x = resident_input(&a, 0x3f80);
    let y = resident_input(&b, 0x4000);
    point(role, traced, "same_inputs_before_cycles", Some(ops));
    for cycle in 1..=5 {
        if role == "C" {
            let mut pair = prepared(ops, Arc::clone(library.as_ref().unwrap()));
            pair.enqueue(ops, [&x, &y], 8).unwrap();
            complete(&mut pair, ops);
            for (rank, op) in ops.iter().enumerate() {
                assert_eq!(
                    words(&pair.output(rank).unwrap(), op),
                    vec![0x4040; MAX_COUNT]
                );
            }
            drop(pair); // Native comm destroy, no abort and no release compensation.
        } else {
            core_materialize(ops); // A/B same module/workspace lifecycle; never a communicator.
        }
        settle_retirement(ops);
        point(
            role,
            traced,
            &format!("cycle_{cycle}_after_drop"),
            Some(ops),
        );
        for op in ops {
            assert_eq!(op.allocator_metrics().live_granted_bytes, 32768);
            assert_eq!(op.allocator_metrics().reserved_bytes, 32768);
        }
    }
    drop(x);
    drop(y);
    settle_retirement(ops);
    point(role, traced, "after_inputs_drop", Some(ops));
    for op in ops {
        assert_eq!(op.allocator_metrics().reserved_bytes, 0);
    }
    if let Some(lib) = library {
        let weak = Arc::downgrade(&lib);
        drop(lib);
        assert!(
            weak.upgrade().is_none(),
            "comm/library custody leaked; no reset workaround"
        );
    }
    point(role, traced, "after_library_drop", Some(ops));
    drop(a);
    drop(b);
    assert!(weak_contexts.iter().all(|w| w.upgrade().is_none()));
    point(role, traced, "after_owners_drop", None);
    // Observe process-global runtime/primary state. In-process zero is NOT this probe's success criterion.
    // The outer authenticated custodian still requires all-eight idle after process exit.
    println!(
        "TP_PROBE_POLICY {{\"role\":{:?},\"instrumented\":{},\"cycles\":5,\"rust_weak_gone\":true,\"refcount_api_available\":false,\"recovery_scope\":\"independent process exit\",\"private_bound\":\"Unknown\",\"hard_cap_claim\":false,\"public_enabled\":false}}",
        role, traced
    );
}

#[test]
#[ignore = "authorized supervised ownership probe only"]
fn control_a() {
    run("A", false);
}
#[test]
#[ignore = "authorized supervised ownership probe only"]
fn control_b() {
    run("B", false);
}
#[test]
#[ignore = "authorized supervised ownership probe only"]
fn control_c() {
    run("C", false);
}
#[test]
#[ignore = "authorized supervised callback-count sibling only"]
fn control_a_counts() {
    run("A", true);
}
#[test]
#[ignore = "authorized supervised callback-count sibling only"]
fn control_b_counts() {
    run("B", true);
}
#[test]
#[ignore = "authorized supervised callback-count sibling only"]
fn control_c_counts() {
    run("C", true);
}
