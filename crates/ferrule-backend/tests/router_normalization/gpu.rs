use ferrule_backend::cuda::operator_owner::CudaOperators;
use ferrule_backend::cuda::operators::moe::{RouterNormalization, SelectedSoftmaxTopKLayout};
use std::path::Path;

fn floats(path: impl AsRef<Path>) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    assert!(bytes.len().is_multiple_of(4));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f32::from_le_bytes)
        .collect()
}
fn exact(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let bad = actual
        .iter()
        .zip(expected)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let max = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max);
    eprintln!(
        "ROUTER_NORMALIZATION {name} words={} unequal={bad} max_abs={max}",
        actual.len()
    );
    assert_eq!(
        bad,
        0,
        "{name}: first {:?}",
        actual
            .iter()
            .zip(expected)
            .position(|(a, b)| a.to_bits() != b.to_bits())
    );
}

#[test]
#[ignore = "sole GPU owner; original M1/M5 refs, then independently generated synthetic fixtures"]
fn full_softmax_frozen_weights_and_no_hidden_work() {
    let dir =
        std::path::PathBuf::from(std::env::var_os("FERRULE_ROUTER_NORMALIZATION_FIXTURE").unwrap());
    let case_path = if dir.join("oracle-cases.txt").exists() {
        dir.join("oracle-cases.txt")
    } else {
        dir.join("cases.txt")
    };
    let cases = std::fs::read_to_string(case_path).unwrap();
    let op = CudaOperators::new_on_device(0).unwrap();
    let baseline = op.allocator_metrics().live_requested_bytes;
    let mut totals = (0, 0);
    for line in cases.lines() {
        let parts: Vec<_> = line.split_whitespace().collect();
        let name = parts[0];
        let (rows, experts, top_k) = (
            parts[1].parse().unwrap(),
            parts[2].parse().unwrap(),
            parts[3].parse().unwrap(),
        );
        let layout = SelectedSoftmaxTopKLayout {
            rows,
            experts,
            top_k,
            output_scale: 1.0,
        };
        let raw = floats(dir.join(format!("{name}.raw.f32")));
        assert_eq!(raw.len(), rows * experts);
        let logits = op.upload_f32_buffer(&raw).unwrap();
        let mut ids = op.zero_i32_buffer(rows * top_k).unwrap();
        let mut weights = op.zero_f32_buffer(rows * top_k).unwrap();
        let before = op.counters();
        op.router_softmax_topk_with_normalization_into(
            &logits,
            &mut ids,
            &mut weights,
            layout,
            RouterNormalization::FullSoftmaxThenSelectedRenormalization,
        )
        .unwrap();
        let after = op.counters();
        assert_eq!(after.kernel_launches - before.kernel_launches, 1);
        assert_eq!(
            after.device_allocation_attempts,
            before.device_allocation_attempts
        );
        assert_eq!(after.host_to_device_copies, before.host_to_device_copies);
        assert_eq!(after.stream_wide_syncs, before.stream_wide_syncs);
        let actual_ids = op.download_i32_buffer(&ids).unwrap();
        let expected_ids = std::fs::read(dir.join(format!("{name}.ids.i32"))).unwrap();
        assert_eq!(
            actual_ids
                .iter()
                .flat_map(|id| id.to_le_bytes())
                .collect::<Vec<_>>(),
            expected_ids,
            "{name} raw-logit IDs"
        );
        let actual = op.download_f32_buffer(&weights).unwrap();
        exact(
            name,
            &actual,
            &floats(dir.join(format!("{name}.weights.f32"))),
        );
        std::fs::write(
            dir.join(format!("{name}.production.f32")),
            actual
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        totals.0 += 1;
        totals.1 += actual.len();
    }
    op.trim_device_allocator().unwrap();
    assert_eq!(op.allocator_metrics().live_requested_bytes, baseline);
    eprintln!(
        "ROUTER_TOTAL cases={} weights={} unequal=0 reclaimed=true",
        totals.0, totals.1
    );
}

#[test]
#[ignore = "sole GPU owner; safe preflight rejection and legacy-policy invariance"]
fn normalization_rejections_and_default_unchanged() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let foreign = CudaOperators::new_on_device(0).unwrap();
    let layout = SelectedSoftmaxTopKLayout {
        rows: 1,
        experts: 256,
        top_k: 8,
        output_scale: 1.0,
    };
    let raw: Vec<_> = (0..256).map(|i| (i % 19) as f32 * 0.125).collect();
    let logits = op.upload_f32_buffer(&raw).unwrap();
    let mut ids = op.zero_i32_buffer(8).unwrap();
    let mut weights = op.zero_f32_buffer(8).unwrap();
    let other = foreign.upload_f32_buffer(&raw).unwrap();
    let mut short = op.zero_f32_buffer(7).unwrap();
    let mut alias = logits.prefix(8).unwrap();
    let before = op.counters();
    let policy = RouterNormalization::FullSoftmaxThenSelectedRenormalization;
    assert!(
        op.router_softmax_topk_with_normalization_into(
            &other,
            &mut ids,
            &mut weights,
            layout,
            policy
        )
        .is_err()
    );
    assert!(
        op.router_softmax_topk_with_normalization_into(
            &logits, &mut ids, &mut short, layout, policy
        )
        .is_err()
    );
    assert!(
        op.router_softmax_topk_with_normalization_into(
            &logits, &mut ids, &mut alias, layout, policy
        )
        .is_err()
    );
    assert!(
        op.router_softmax_topk_with_normalization_into(
            &logits,
            &mut ids,
            &mut weights,
            SelectedSoftmaxTopKLayout {
                top_k: 33,
                ..layout
            },
            policy
        )
        .is_err()
    );
    assert_eq!(op.counters().kernel_launches, before.kernel_launches);
    op.router_softmax_topk_f32_into(&logits, &mut ids, &mut weights, layout)
        .unwrap();
    let old_ids = op.download_i32_buffer(&ids).unwrap();
    let old = op.download_f32_buffer(&weights).unwrap();
    op.router_softmax_topk_with_normalization_into(
        &logits,
        &mut ids,
        &mut weights,
        layout,
        RouterNormalization::default(),
    )
    .unwrap();
    assert_eq!(op.download_i32_buffer(&ids).unwrap(), old_ids);
    exact("legacy", &op.download_f32_buffer(&weights).unwrap(), &old);
    op.router_softmax_topk_with_normalization_into(&logits, &mut ids, &mut weights, layout, policy)
        .unwrap();
    op.sync_stream().unwrap();
}

#[test]
#[ignore = "sole GPU owner; explicit IEEE nonfinite behavior, not legacy uniform fixups"]
fn full_softmax_nonfinite_contract() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let layout = SelectedSoftmaxTopKLayout {
        rows: 1,
        experts: 4,
        top_k: 2,
        output_scale: 1.0,
    };
    for (raw, expected_ids) in [
        ([f32::INFINITY, 1., f32::INFINITY, 0.], [0, 2]),
        ([f32::NEG_INFINITY; 4], [0, 1]),
        ([f32::NAN, 2., 1., 0.], [1, 2]),
        ([f32::NAN; 4], [-1, -1]),
    ] {
        let logits = op.upload_f32_buffer(&raw).unwrap();
        let mut ids = op.zero_i32_buffer(2).unwrap();
        let mut weights = op.zero_f32_buffer(2).unwrap();
        op.router_softmax_topk_with_normalization_into(
            &logits,
            &mut ids,
            &mut weights,
            layout,
            RouterNormalization::FullSoftmaxThenSelectedRenormalization,
        )
        .unwrap();
        assert_eq!(op.download_i32_buffer(&ids).unwrap(), expected_ids);
        let actual = op.download_f32_buffer(&weights).unwrap();
        for (value, id) in actual.iter().zip(expected_ids) {
            if id < 0 {
                assert_eq!(value.to_bits(), 0);
            } else {
                assert!(value.is_nan());
            }
        }
    }
}

#[test]
#[ignore = "explicit GPU grant; exact analytic uniform source probabilities and unchanged route bits"]
fn full_softmax_actual_probability_sink_uniform_and_rejections() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let layout = SelectedSoftmaxTopKLayout { rows: 2, experts: 256, top_k: 8, output_scale: 1.0 };
    let logits = op.upload_f32_buffer(&vec![0.0; 512]).unwrap();
    let mut ids = op.zero_i32_buffer(16).unwrap();
    let mut weights = op.zero_f32_buffer(16).unwrap();
    let mut probabilities = op.zero_f32_buffer(512).unwrap();
    let policy = RouterNormalization::FullSoftmaxThenSelectedRenormalization;
    op.router_softmax_topk_with_normalization_into(&logits, &mut ids, &mut weights, layout, policy).unwrap();
    let unobserved_ids = op.download_i32_buffer(&ids).unwrap();
    let unobserved_weights = op.download_f32_buffer(&weights).unwrap();
    let before = op.counters();
    op.router_softmax_topk_with_probabilities_into(&logits, &mut ids, &mut weights, layout, policy, Some(&mut probabilities)).unwrap();
    let after = op.counters();
    assert_eq!(after.kernel_launches - before.kernel_launches, 1);
    assert_eq!(after.device_allocation_attempts, before.device_allocation_attempts);
    assert_eq!(after.stream_wide_syncs, before.stream_wide_syncs);
    // Analytic oracle: exp(0)=1; exact binary sums/division at N=256,K=8.
    assert_eq!(op.download_f32_buffer(&probabilities).unwrap(), vec![1.0 / 256.0; 512]);
    assert_eq!(op.download_i32_buffer(&ids).unwrap(), [0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7]);
    exact("source-observed weights", &op.download_f32_buffer(&weights).unwrap(), &[0.125; 16]);
    assert_eq!(op.download_i32_buffer(&ids).unwrap(), unobserved_ids);
    exact("source sink does not change route bits", &op.download_f32_buffer(&weights).unwrap(), &unobserved_weights);
    let mut short = op.zero_f32_buffer(511).unwrap();
    let mut alias = logits.prefix(512).unwrap();
    assert!(op.router_softmax_topk_with_probabilities_into(&logits, &mut ids, &mut weights, layout, policy, Some(&mut short)).is_err());
    assert!(op.router_softmax_topk_with_probabilities_into(&logits, &mut ids, &mut weights, layout, policy, Some(&mut alias)).is_err());
    assert!(op.router_softmax_topk_with_probabilities_into(&logits, &mut ids, &mut weights, layout, RouterNormalization::SelectedSoftmax, Some(&mut probabilities)).is_err());
}
