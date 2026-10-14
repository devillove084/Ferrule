//! Numeric/repeated-site cases each run in its own authenticated supervised process.
use super::*;

#[test]
#[ignore = "authenticated main sole GPU authorization; fresh supervised process only"]
fn nonfinite_and_finite_overflow_never_publish_or_clear_sticky_status() {
    let library = authorized_library();
    let a = CudaOperators::new_on_device(0).unwrap();
    let b = CudaOperators::new_on_device(1).unwrap();
    let ops = [&a, &b];
    for (first, second, flag) in [
        (0x7fc1, 0x3f80, INPUT_NONFINITE),
        (0x7f80, 0x3f80, INPUT_NONFINITE),
        (0xff80, 0x3f80, INPUT_NONFINITE),
        (0x7f7f, 0x7f7f, OUTPUT_NONFINITE),
        (0x7f7f, 0x7b00, OUTPUT_NONFINITE),
    ] {
        let mut pair = prepared(ops, Arc::clone(&library));
        let x = producer_input(&a, &vec![first; HIDDEN]);
        let y = producer_input(&b, &vec![second; HIDDEN]);
        STATUS_BYTES.set(0);
        for op in ops {
            op.reset_counters();
        }
        pair.enqueue(ops, [&x, &y], 1).unwrap();
        enqueued(&mut pair, ops);
        let mut rejected = false;
        for _ in 0..10000 {
            match pair.check_at_quiescence(ops) {
                Err(_) => {
                    rejected = true;
                    break;
                }
                Ok(TpProgress::Pending) => thread::sleep(Duration::from_millis(1)),
                other => panic!("invalid output unexpectedly accepted: {other:?}"),
            }
        }
        assert!(rejected);
        assert_eq!(pair.phase, Phase::Rejected);
        assert!(!pair.is_unknown());
        assert_eq!(STATUS_BYTES.get(), 8);
        assert_no_site_host_work(ops);
        for rank in pair.ranks() {
            assert_eq!(rank.device_status().unwrap(), flag);
        }
        assert!(pair.output(0).is_err());
        assert!(pair.enqueue(ops, [&x, &y], 1).is_err());
        println!(
            "TP_REJECT {{\"first_word\":{},\"second_word\":{},\"sticky_flags\":{},\"published\":false,\"completion_proven\":true}}",
            first, second, flag
        );
    }
    drop(library);
    drop(a);
    drop(b);
    assert_process_gpu_idle();
}

#[test]
#[ignore = "authenticated main sole GPU authorization; fresh supervised process only"]
fn two_sites_enqueue_without_checkpoint_and_rejection_is_sticky_across_sites() {
    let library = authorized_library();
    let a = CudaOperators::new_on_device(0).unwrap();
    let b = CudaOperators::new_on_device(1).unwrap();
    let ops = [&a, &b];
    for reject_first in [false, true] {
        let mut pair = prepared(ops, Arc::clone(&library));
        let x = producer_input(
            &a,
            &vec![if reject_first { 0x7fc1 } else { 0x3f80 }; HIDDEN],
        );
        let y = producer_input(&b, &vec![0x4000; HIDDEN]);
        let next_x = producer_input(&a, &vec![0x4080; HIDDEN]);
        STATUS_BYTES.set(0);
        for op in ops {
            op.reset_counters();
        }
        pair.enqueue(ops, [&x, &y], 1).unwrap();
        enqueued(&mut pair, ops);
        assert_eq!(STATUS_BYTES.get(), 0);
        pair.enqueue(ops, [&next_x, &y], 1).unwrap();
        enqueued(&mut pair, ops);
        assert_eq!(pair.input_custody.len(), 2);
        assert_eq!(STATUS_BYTES.get(), 0); // No intermediate checkpoint or payload D2H.
        assert!(pair.enqueue(ops, [&next_x, &y], 1).is_err());
        assert_eq!(pair.input_custody.len(), 2);
        let mut completed = false;
        for _ in 0..10000 {
            match pair.check_at_quiescence(ops) {
                Ok(TpProgress::Complete) => {
                    assert!(!reject_first);
                    completed = true;
                    break;
                }
                Err(_) => {
                    assert!(reject_first);
                    completed = true;
                    break;
                }
                Ok(TpProgress::Pending) => thread::sleep(Duration::from_millis(1)),
                other => panic!("unexpected completion: {other:?}"),
            }
        }
        assert!(completed);
        assert!(pair.input_custody.is_empty());
        assert_eq!(STATUS_BYTES.get(), 8);
        assert_no_site_host_work(ops);
        if reject_first {
            assert_eq!(pair.phase, Phase::Rejected);
            assert!(pair.output(0).is_err());
            for rank in pair.ranks() {
                assert_eq!(rank.device_status().unwrap(), INPUT_NONFINITE);
            }
        } else {
            for rank in 0..2 {
                assert_eq!(
                    words(&pair.output(rank).unwrap(), ops[rank]),
                    vec![0x40c0; HIDDEN]
                );
            }
        }
        println!(
            "TP_TWO_SITES {{\"intermediate_status_bytes\":0,\"joint_status_bytes\":8,\"reject_first\":{},\"sticky_preserved\":true}}",
            reject_first
        );
    }
    drop(library);
    drop(a);
    drop(b);
    assert_process_gpu_idle();
}
