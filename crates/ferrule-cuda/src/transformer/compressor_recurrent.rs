//! Model-independent recurrent compressor state layout helpers.

use ferrule_common::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressorRecurrentShape {
    pub ratio: usize,
    pub head_dim: usize,
    pub out_dim: usize,
    pub overlap: bool,
}

impl CompressorRecurrentShape {
    pub fn validate(&self) -> Result<()> {
        let minimum_out_dim = if self.overlap {
            self.head_dim.checked_mul(2).ok_or_else(|| {
                Error::Internal("compressor recurrent head dimension overflow".into())
            })?
        } else {
            self.head_dim
        };
        if self.ratio == 0 || self.head_dim == 0 || self.out_dim != minimum_out_dim {
            return Err(Error::Internal(format!(
                "invalid compressor recurrent shape: ratio={} head_dim={} out_dim={} overlap={}",
                self.ratio, self.head_dim, self.out_dim, self.overlap
            )));
        }
        for (field, value) in [
            ("ratio", self.ratio),
            ("head_dim", self.head_dim),
            ("out_dim", self.out_dim),
        ] {
            u32::try_from(value).map_err(|_| {
                Error::Internal(format!(
                    "compressor recurrent {field} exceeds u32 kernel ABI: {value}"
                ))
            })?;
        }
        self.state_elements()?;
        Ok(())
    }

    pub fn state_rows(&self) -> usize {
        if self.overlap {
            2 * self.ratio
        } else {
            self.ratio
        }
    }

    pub fn state_elements(&self) -> Result<usize> {
        self.state_rows()
            .checked_mul(self.out_dim)
            .ok_or_else(|| Error::Internal("compressor recurrent state size overflow".into()))
    }

    pub fn ape_elements(&self) -> Result<usize> {
        self.ratio
            .checked_mul(self.out_dim)
            .ok_or_else(|| Error::Internal("compressor recurrent APE size overflow".into()))
    }

    pub fn append_row(&self, position: usize) -> usize {
        let current = position % self.ratio;
        if self.overlap {
            self.ratio + current
        } else {
            current
        }
    }

    pub fn prefill_groups(&self, tokens: usize) -> usize {
        if self.ratio == 0 {
            return 0;
        }
        (tokens - tokens % self.ratio) / self.ratio
    }

    /// Returns `(source_token, ape_row)` for one final seeded state row.
    pub fn prefill_seed_source(&self, state_row: usize, tokens: usize) -> Option<(usize, usize)> {
        if self.ratio == 0 || state_row >= self.state_rows() {
            return None;
        }
        let remainder = tokens % self.ratio;
        let cutoff = tokens - remainder;
        if self.overlap && cutoff >= self.ratio && state_row < self.ratio {
            return Some((cutoff - self.ratio + state_row, state_row));
        }
        let state_offset = if self.overlap { self.ratio } else { 0 };
        if state_row >= state_offset && state_row < state_offset + remainder {
            let local = state_row - state_offset;
            return Some((cutoff + local, local));
        }
        None
    }

    pub fn is_boundary(&self, position: usize) -> bool {
        position
            .checked_add(1)
            .is_some_and(|next| next.is_multiple_of(self.ratio))
    }

    pub fn current_window_source(&self, row: usize, dim: usize) -> Option<usize> {
        if dim >= self.head_dim || row >= self.state_rows() {
            return None;
        }
        if self.overlap {
            if row < self.ratio {
                row.checked_mul(self.out_dim)?.checked_add(dim)
            } else {
                row.checked_mul(self.out_dim)?
                    .checked_add(self.head_dim)?
                    .checked_add(dim)
            }
        } else {
            row.checked_mul(self.out_dim)?.checked_add(dim)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ReferenceState {
        shape: CompressorRecurrentShape,
        kv: Vec<f32>,
        score: Vec<f32>,
    }

    impl ReferenceState {
        fn new(shape: CompressorRecurrentShape) -> Self {
            Self {
                shape,
                kv: vec![0.0; shape.state_elements().unwrap()],
                score: vec![f32::NEG_INFINITY; shape.state_elements().unwrap()],
            }
        }

        fn seed(
            &mut self,
            kv_rows: &[f32],
            score_rows: &[f32],
            ape: &[f32],
            tokens: usize,
        ) -> usize {
            self.kv.fill(0.0);
            self.score.fill(f32::NEG_INFINITY);
            for state_row in 0..self.shape.state_rows() {
                let Some((source_token, ape_row)) =
                    self.shape.prefill_seed_source(state_row, tokens)
                else {
                    continue;
                };
                for dim in 0..self.shape.out_dim {
                    let dst = state_row * self.shape.out_dim + dim;
                    let src = source_token * self.shape.out_dim + dim;
                    self.kv[dst] = kv_rows[src];
                    self.score[dst] = score_rows[src] + ape[ape_row * self.shape.out_dim + dim];
                }
            }
            self.shape.prefill_groups(tokens)
        }

        fn append(&mut self, kv: &[f32], score: &[f32], ape: &[f32], position: usize) -> bool {
            let row = self.shape.append_row(position);
            let ape_row = position % self.shape.ratio;
            let dst = row * self.shape.out_dim;
            let ape_start = ape_row * self.shape.out_dim;
            self.kv[dst..dst + self.shape.out_dim].copy_from_slice(kv);
            for dim in 0..self.shape.out_dim {
                self.score[dst + dim] = score[dim] + ape[ape_start + dim];
            }
            self.shape.is_boundary(position)
        }

        fn compress(&self) -> Vec<f32> {
            let mut output = vec![0.0; self.shape.head_dim];
            for dim in 0..self.shape.head_dim {
                let mut max_score = f32::NEG_INFINITY;
                for row in 0..self.shape.state_rows() {
                    let src = self.shape.current_window_source(row, dim).unwrap();
                    max_score = max_score.max(self.score[src]);
                }
                let mut denominator = 0.0;
                for row in 0..self.shape.state_rows() {
                    let src = self.shape.current_window_source(row, dim).unwrap();
                    denominator += libm::expf(self.score[src] - max_score);
                }
                if denominator > 0.0 && denominator.is_finite() {
                    for row in 0..self.shape.state_rows() {
                        let src = self.shape.current_window_source(row, dim).unwrap();
                        output[dim] +=
                            libm::expf(self.score[src] - max_score) / denominator * self.kv[src];
                    }
                }
            }
            output
        }

        fn advance(&mut self) {
            if self.shape.overlap {
                let half = self.shape.ratio * self.shape.out_dim;
                self.kv.copy_within(half..2 * half, 0);
                self.score.copy_within(half..2 * half, 0);
            }
        }
    }

    #[test]
    fn prefill_seed_mapping_covers_remainder_and_last_complete_group() {
        let plain = CompressorRecurrentShape {
            ratio: 3,
            head_dim: 2,
            out_dim: 2,
            overlap: false,
        };
        assert_eq!(plain.prefill_groups(8), 2);
        assert_eq!(plain.prefill_seed_source(0, 8), Some((6, 0)));
        assert_eq!(plain.prefill_seed_source(1, 8), Some((7, 1)));
        assert_eq!(plain.prefill_seed_source(2, 8), None);
        assert_eq!(plain.prefill_seed_source(0, 6), None);

        let overlap = CompressorRecurrentShape {
            ratio: 3,
            head_dim: 2,
            out_dim: 4,
            overlap: true,
        };
        assert_eq!(overlap.prefill_groups(8), 2);
        assert_eq!(overlap.prefill_seed_source(0, 8), Some((3, 0)));
        assert_eq!(overlap.prefill_seed_source(2, 8), Some((5, 2)));
        assert_eq!(overlap.prefill_seed_source(3, 8), Some((6, 0)));
        assert_eq!(overlap.prefill_seed_source(4, 8), Some((7, 1)));
        assert_eq!(overlap.prefill_seed_source(5, 8), None);
        assert_eq!(overlap.prefill_seed_source(0, 2), None);
        assert_eq!(overlap.prefill_seed_source(3, 2), Some((0, 0)));
    }

    #[test]
    fn prefill_seed_reference_resets_and_copies_exact_final_rows() {
        for shape in [
            CompressorRecurrentShape {
                ratio: 3,
                head_dim: 2,
                out_dim: 2,
                overlap: false,
            },
            CompressorRecurrentShape {
                ratio: 3,
                head_dim: 2,
                out_dim: 4,
                overlap: true,
            },
        ] {
            for tokens in [1usize, 3, 8] {
                let mut state = ReferenceState::new(shape);
                state.kv.fill(99.0);
                state.score.fill(99.0);
                let kv: Vec<f32> = (0..tokens * shape.out_dim)
                    .map(|index| index as f32)
                    .collect();
                let score: Vec<f32> = (0..tokens * shape.out_dim)
                    .map(|index| index as f32 * 0.5)
                    .collect();
                let ape: Vec<f32> = (0..shape.ape_elements().unwrap())
                    .map(|index| index as f32 * 0.25)
                    .collect();
                assert_eq!(
                    state.seed(&kv, &score, &ape, tokens),
                    shape.prefill_groups(tokens)
                );
                for row in 0..shape.state_rows() {
                    match shape.prefill_seed_source(row, tokens) {
                        Some((source, ape_row)) => {
                            for dim in 0..shape.out_dim {
                                let dst = row * shape.out_dim + dim;
                                assert_eq!(state.kv[dst], kv[source * shape.out_dim + dim]);
                                assert_eq!(
                                    state.score[dst],
                                    score[source * shape.out_dim + dim]
                                        + ape[ape_row * shape.out_dim + dim]
                                );
                            }
                        }
                        None => {
                            assert!(
                                state.kv[row * shape.out_dim..(row + 1) * shape.out_dim]
                                    .iter()
                                    .all(|value| *value == 0.0)
                            );
                            assert!(
                                state.score[row * shape.out_dim..(row + 1) * shape.out_dim]
                                    .iter()
                                    .all(|value| *value == f32::NEG_INFINITY)
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn non_overlap_appends_positions_and_compresses_at_boundaries() {
        let shape = CompressorRecurrentShape {
            ratio: 2,
            head_dim: 2,
            out_dim: 2,
            overlap: false,
        };
        let mut state = ReferenceState::new(shape);
        let ape = [0.1, 0.2, 0.3, 0.4];
        assert!(!state.append(&[1.0, 2.0], &[0.0, 0.0], &ape, 0));
        assert!(state.append(&[3.0, 4.0], &[0.0, 0.0], &ape, 1));
        let compressed = state.compress();
        assert!(compressed[0] > 2.0 && compressed[0] < 3.0);
        assert!(compressed[1] > 3.0 && compressed[1] < 4.0);
        assert_eq!(shape.append_row(2), 0);
    }

    #[test]
    fn overlap_uses_previous_left_and_current_right_then_advances() {
        let shape = CompressorRecurrentShape {
            ratio: 2,
            head_dim: 2,
            out_dim: 4,
            overlap: true,
        };
        let mut state = ReferenceState::new(shape);
        let ape = vec![0.0; shape.ape_elements().unwrap()];
        assert!(!state.append(&[10.0, 11.0, 1.0, 2.0], &[0.0; 4], &ape, 0));
        assert!(state.append(&[20.0, 21.0, 3.0, 4.0], &[0.0; 4], &ape, 1));
        assert_eq!(state.compress(), vec![2.0, 3.0]);
        state.advance();
        assert_eq!(
            &state.kv[..8],
            &[10.0, 11.0, 1.0, 2.0, 20.0, 21.0, 3.0, 4.0]
        );

        assert!(!state.append(&[30.0, 31.0, 5.0, 6.0], &[0.0; 4], &ape, 2));
        assert!(state.append(&[40.0, 41.0, 7.0, 8.0], &[0.0; 4], &ape, 3));
        assert_eq!(state.compress(), vec![10.5, 11.5]);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_prefill_seed_matches_final_reference_with_one_launch() {
        use crate::context::CudaArtifactOperatorContext;

        fn run(shape: CompressorRecurrentShape, tokens: usize) {
            let context = CudaArtifactOperatorContext::new().unwrap();
            let mut device_state = context
                .create_compressor_recurrent_state(
                    shape.ratio,
                    shape.head_dim,
                    shape.out_dim,
                    shape.overlap,
                )
                .unwrap();
            let kv: Vec<f32> = (0..tokens * shape.out_dim)
                .map(|index| index as f32 * 0.125 - 1.0)
                .collect();
            let score: Vec<f32> = (0..tokens * shape.out_dim)
                .map(|index| index as f32 * 0.0625 - 0.5)
                .collect();
            let ape: Vec<f32> = (0..shape.ape_elements().unwrap())
                .map(|index| index as f32 * 0.03125)
                .collect();
            let kv_device = context.upload_f32_buffer(&kv).unwrap();
            let score_device = context.upload_f32_buffer(&score).unwrap();
            let ape_device = context.upload_f32_buffer(&ape).unwrap();
            let mut reference = ReferenceState::new(shape);
            let expected_groups = reference.seed(&kv, &score, &ape, tokens);

            let before = context.counters();
            let groups = context
                .compressor_recurrent_seed_prefill(
                    &mut device_state,
                    &kv_device,
                    &score_device,
                    &ape_device,
                    tokens,
                )
                .unwrap();
            let after = context.counters();
            assert_eq!(groups, expected_groups);
            assert_eq!(after.kernel_launches, before.kernel_launches + 1);
            assert_eq!(after.device_allocations, before.device_allocations);
            assert_eq!(after.device_to_host_copies, before.device_to_host_copies);

            let actual_kv = context
                .download_f32_buffer(device_state.kv_state())
                .unwrap();
            let actual_score = context
                .download_f32_buffer(device_state.score_state())
                .unwrap();
            assert_eq!(
                actual_kv
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                reference
                    .kv
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                actual_score
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                reference
                    .score
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }

        for tokens in [1usize, 3, 8] {
            run(
                CompressorRecurrentShape {
                    ratio: 3,
                    head_dim: 2,
                    out_dim: 2,
                    overlap: false,
                },
                tokens,
            );
            run(
                CompressorRecurrentShape {
                    ratio: 3,
                    head_dim: 2,
                    out_dim: 4,
                    overlap: true,
                },
                tokens,
            );
        }
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_recurrent_state_matches_reference_without_allocations_or_d2h() {
        use crate::context::CudaArtifactOperatorContext;

        fn run(shape: CompressorRecurrentShape) {
            let context = CudaArtifactOperatorContext::new().unwrap();
            let mut device_state = context
                .create_compressor_recurrent_state(
                    shape.ratio,
                    shape.head_dim,
                    shape.out_dim,
                    shape.overlap,
                )
                .unwrap();
            assert!(
                context
                    .download_f32_buffer(device_state.kv_state())
                    .unwrap()
                    .iter()
                    .all(|value| *value == 0.0)
            );
            assert!(
                context
                    .download_f32_buffer(device_state.score_state())
                    .unwrap()
                    .iter()
                    .all(|value| *value == f32::NEG_INFINITY)
            );
            let ape: Vec<f32> = (0..shape.ape_elements().unwrap())
                .map(|index| index as f32 * 0.03125 - 0.125)
                .collect();
            let ape_device = context.upload_f32_buffer(&ape).unwrap();
            let mut kv_device = context.zero_f32_buffer(shape.out_dim).unwrap();
            let mut score_device = context.zero_f32_buffer(shape.out_dim).unwrap();
            let mut output = context.zero_f32_buffer(shape.head_dim).unwrap();
            let mut reference = ReferenceState::new(shape);

            for position in 0..6 {
                let kv: Vec<f32> = (0..shape.out_dim)
                    .map(|dim| position as f32 + dim as f32 * 0.25)
                    .collect();
                let score: Vec<f32> = (0..shape.out_dim)
                    .map(|dim| dim as f32 * 0.125 - position as f32 * 0.0625)
                    .collect();
                context.overwrite_f32_buffer(&kv, &mut kv_device).unwrap();
                context
                    .overwrite_f32_buffer(&score, &mut score_device)
                    .unwrap();
                let expected_boundary = reference.append(&kv, &score, &ape, position);

                let before_append = context.counters();
                let boundary = context
                    .compressor_recurrent_append_projected(
                        &mut device_state,
                        &kv_device,
                        &score_device,
                        &ape_device,
                        position,
                    )
                    .unwrap();
                let after_append = context.counters();
                assert_eq!(boundary, expected_boundary);
                assert_eq!(
                    after_append.device_allocations,
                    before_append.device_allocations
                );
                assert_eq!(
                    after_append.device_to_host_copies,
                    before_append.device_to_host_copies
                );

                if boundary {
                    let expected = reference.compress();
                    let before_boundary = context.counters();
                    context
                        .compressor_recurrent_boundary_into(&mut device_state, &mut output)
                        .unwrap();
                    let after_boundary = context.counters();
                    assert_eq!(
                        after_boundary.device_allocations,
                        before_boundary.device_allocations
                    );
                    assert_eq!(
                        after_boundary.device_to_host_copies,
                        before_boundary.device_to_host_copies
                    );
                    let actual = context.download_f32_buffer(&output).unwrap();
                    for (actual, expected) in actual.iter().zip(&expected) {
                        assert!(
                            (actual - expected).abs() <= 1e-6,
                            "actual={actual} expected={expected}"
                        );
                    }
                    reference.advance();
                }
            }
        }

        run(CompressorRecurrentShape {
            ratio: 2,
            head_dim: 2,
            out_dim: 2,
            overlap: false,
        });
        run(CompressorRecurrentShape {
            ratio: 2,
            head_dim: 2,
            out_dim: 4,
            overlap: true,
        });
    }
}
