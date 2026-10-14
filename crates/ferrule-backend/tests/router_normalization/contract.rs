use ferrule_backend::cuda::operators::moe::{RouterNormalization, SelectedSoftmaxTopKLayout};

#[test]
fn normalization_capabilities_are_explicit_and_checked() {
    use RouterNormalization::*;
    assert_eq!(RouterNormalization::default(), SelectedSoftmax);
    for rows in [1, 5, 32] {
        for experts in [1, 7, 32, 128, 256, 257, 1024] {
            for top_k in [1, 3, 8, 13, 32].into_iter().filter(|k| *k <= experts) {
                FullSoftmaxThenSelectedRenormalization
                    .validate(SelectedSoftmaxTopKLayout {
                        rows,
                        experts,
                        top_k,
                        output_scale: 1.0,
                    })
                    .unwrap();
            }
        }
    }
    let valid = SelectedSoftmaxTopKLayout {
        rows: 5,
        experts: 256,
        top_k: 8,
        output_scale: 1.0,
    };
    for bad in [
        SelectedSoftmaxTopKLayout { rows: 0, ..valid },
        SelectedSoftmaxTopKLayout {
            rows: usize::MAX,
            ..valid
        },
        SelectedSoftmaxTopKLayout {
            experts: 0,
            ..valid
        },
        SelectedSoftmaxTopKLayout {
            experts: 1025,
            ..valid
        },
        SelectedSoftmaxTopKLayout { top_k: 0, ..valid },
        SelectedSoftmaxTopKLayout { top_k: 33, ..valid },
        SelectedSoftmaxTopKLayout {
            top_k: 257,
            ..valid
        },
        SelectedSoftmaxTopKLayout {
            output_scale: f32::NAN,
            ..valid
        },
        SelectedSoftmaxTopKLayout {
            output_scale: f32::INFINITY,
            ..valid
        },
    ] {
        assert!(
            FullSoftmaxThenSelectedRenormalization
                .validate(bad)
                .is_err(),
            "{bad:?}"
        );
    }
    SelectedSoftmax
        .validate(SelectedSoftmaxTopKLayout {
            experts: 2048,
            top_k: 64,
            ..valid
        })
        .unwrap();
}
