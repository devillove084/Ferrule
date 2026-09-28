#![cfg(feature = "cuda")]
use ferrule_backend::cuda::context::CudaOperators;
use ferrule_common::Error;

#[test]
#[ignore = "actual CUDA GPU; PR24 exact owner validation before launch/upload"]
fn pr24_graph_rejects_wrong_owner_before_native_submission() {
    let op = CudaOperators::new_on_device(0).expect("required CUDA device");
    let other = CudaOperators::new_on_device(0).expect("required second owner");
    let graph = op.capture_decode_graph(|| Ok(())).unwrap();
    assert!(other.launch_graph(&graph).is_err());
    assert!(other.upload_graph(&graph).is_err());
    op.launch_graph(&graph).unwrap();
    op.record_compute_event().unwrap().synchronize().unwrap();
}

#[test]
#[ignore = "actual CUDA GPU; PR24 assertion is scoped and restores prior state on Err"]
fn pr24_capture_assertion_restores_prior_state_on_error() {
    let op = CudaOperators::new_on_device(0).expect("required CUDA device");
    for prior in [false, true] {
        if prior {
            op.enable_capture_safe();
        } else {
            op.disable_capture_safe();
        }
        let observed = std::cell::Cell::new(false);
        let error = op
            .capture_decode_graph(|| {
                observed.set(op.is_capture_safe());
                Err(Error::Internal {
                    message: "closure primary".into(),
                })
            })
            .err()
            .unwrap();
        assert!(error.to_string().contains("closure primary"));
        assert!(observed.get(), "capture must scope assertion automatically");
        assert_eq!(op.is_capture_safe(), prior);
    }
    op.disable_capture_safe();
    op.zero_i32_buffer(1).unwrap();
}

#[test]
#[ignore = "actual CUDA GPU; capture panic restores assertion and permits a fresh capture"]
fn pr24_capture_panic_restores_scope_with_borrowed_non_send_resources() {
    use std::cell::Cell;
    use std::rc::Rc;
    let op = CudaOperators::new_on_device(0).expect("required CUDA device");
    let visits = Rc::new(Cell::new(0));
    for prior in [false, true] {
        if prior {
            op.enable_capture_safe();
        } else {
            op.disable_capture_safe();
        }
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = op.capture_decode_graph(|| {
                visits.set(visits.get() + 1);
                assert!(op.is_capture_safe());
                panic!("injected capture closure panic");
            });
        }));
        assert!(panic.is_err());
        assert_eq!(op.is_capture_safe(), prior);
        let graph = op.capture_decode_graph(|| Ok(())).unwrap();
        op.launch_graph(&graph).unwrap();
        op.record_compute_event().unwrap().synchronize().unwrap();
    }
    assert_eq!(visits.get(), 2);
}
