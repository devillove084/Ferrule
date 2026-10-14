//! Frozen pre-batch routing reference. Never calls the new planner/metadata API.
use super::*;

pub(super) fn legacy_routed(
    owner: &mut CudaStandardDecoderOperators,
    input: &Rows,
    routes: &RouterRoutes,
) -> Result<Rows> {
    owner.synchronous(|this| {
        let width = input.shape().width();
        let mut buckets = BTreeMap::<usize, Vec<(usize, f32)>>::new();
        for row in 0..routes.rows() {
            let (ids, weights) = routes.row(row)?;
            for (&expert, &weight) in ids.iter().zip(weights) {
                buckets.entry(expert).or_default().push((row, weight));
            }
        }
        let packed = this
            .ops
            .zero_bf16_buffer(routes.expert_ids().len() * width)?;
        this.hold_bf16(&packed)?;
        this.hold_bf16(this.bf16_buffer(input)?)?;
        let mut row_ids = Vec::new();
        let mut route_weights = Vec::new();
        for (expert, bucket) in buckets {
            let image = this.mixed_rank_block.as_ref().unwrap();
            let tokens = [
                TensorRole::RoutedExpertGate,
                TensorRole::RoutedExpertUp,
                TensorRole::RoutedExpertDown,
            ]
            .map(|role| {
                let slot = image
                    .projections
                    .iter()
                    .position(|p| {
                        p.binding.residency() == &ParameterResidency::Expert { layer: 0, expert }
                            && p.binding.role() == &role
                    })
                    .unwrap();
                CudaMixedRankLinear {
                    generation: this.image_generation(),
                    slot,
                }
            });
            let ids = bucket
                .iter()
                .map(|(row, _)| *row as u32)
                .collect::<Vec<_>>();
            let gathered = this
                .ops
                .embedding_bf16(this.bf16_buffer(input)?, &ids, width)?;
            this.hold_bf16(&gathered)?;
            let gathered = rows_bf16(RowsShape::new(ids.len(), width)?, None, gathered)?;
            let output = packed.slice(row_ids.len() * width, ids.len() * width)?;
            this.mixed_swiglu_inner(
                [&tokens[0], &tokens[1], &tokens[2]],
                &gathered,
                input.shape().rows(),
                Some(output),
            )?;
            for (row, weight) in bucket {
                row_ids.push(row as i32);
                route_weights.push(weight);
            }
        }
        let ids = this.ops.upload_i32_buffer(&row_ids)?;
        this.route_ids
            .push(ids.as_device_buffer().slice(0, ids.len())?);
        let weights = this.ops.upload_f32_buffer(&route_weights)?;
        this.hold_route_buffer(&weights)?;
        let mut out = this.ops.zero_bf16_buffer(input.shape().elements())?;
        this.hold_bf16(&out)?;
        this.ops.weighted_combine_bf16_into(
            &packed,
            &ids,
            &weights,
            &mut out,
            input.shape().rows(),
            width,
        )?;
        rows_bf16(input.shape(), input.arena(), out)
    })
}
