//! Pure byte accounting for standard F32 SwiGLU activations and routed buckets.
//!
//! Storage encoding and numeric operand precision do not change F32 activation
//! sizes. The provider's numeric workspace reservation is charged separately,
//! once; this module neither sizes provider tiles nor allocates/reuses buffers.

use ferrule_common::Result;

use super::model_error;
use crate::transformer::{ExpertMetadata, PreparedSwiGlu, SwiGlu};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SwiGluScratchPlan {
    input_bytes: usize,
    intermediate_bytes: usize,
    output_bytes: usize,
    bias_bytes: usize,
    route_bytes: usize,
    table_bytes: usize,
    numeric_reservation_bytes: usize,
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| model_error("SwiGLU scratch overflow"))
}

fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| model_error("SwiGLU scratch overflow"))
}

impl SwiGluScratchPlan {
    /// `rows == 0` is the existing weight-only preparation path. Bias widths
    /// describe broadcast vectors; each also needs one i32 gather ID per row.
    pub(crate) fn new(
        rows: usize,
        input: usize,
        intermediate: usize,
        output: usize,
        biases: [Option<usize>; 3],
    ) -> Result<Self> {
        // Check the complete per-row geometry even for zero-row preparation.
        let intermediate = mul(intermediate, 3)?;
        let bias = biases
            .into_iter()
            .flatten()
            .try_fold(0, |n, width| add(n, add(width, 1)?))?;
        add(add(add(input, intermediate)?, output)?, bias)?;
        let bytes = |width| mul(mul(width, rows)?, 4);
        let plan = Self {
            input_bytes: bytes(input)?,
            intermediate_bytes: bytes(intermediate)?,
            output_bytes: bytes(output)?,
            bias_bytes: bytes(bias)?,
            route_bytes: 0,
            table_bytes: 0,
            numeric_reservation_bytes: 0,
        };
        plan.total_bytes()?;
        Ok(plan)
    }

    pub(crate) fn for_prepared(expert: &PreparedSwiGlu, rows: usize) -> Result<Self> {
        Self::new(
            rows,
            expert.input_width(),
            expert.gate().out_features(),
            expert.output_width(),
            [expert.gate(), expert.up(), expert.down()].map(|p| p.bias().map(|b| b.len())),
        )
    }

    pub(crate) fn for_metadata(expert: &ExpertMetadata, rows: usize) -> Result<Self> {
        Self::new(
            rows,
            expert.input_width(),
            expert.parameters()[0].spec().shape()[0],
            expert.output_width(),
            [None; 3],
        )
    }

    pub(crate) fn for_descriptor(expert: &SwiGlu, rows: usize) -> Result<Self> {
        Self::new(
            rows,
            expert.gate().in_features(),
            expert.gate().out_features(),
            expert.down().out_features(),
            [expert.gate(), expert.up(), expert.down()]
                .map(|p| p.has_bias().then_some(p.out_features())),
        )
    }

    pub(crate) fn route(
        rows: usize,
        top_k: usize,
        width: usize,
        largest_bucket: usize,
    ) -> Result<Self> {
        if top_k == 0 || width == 0 || largest_bucket > rows {
            return Err(model_error("invalid routed bucket geometry"));
        }
        let count = mul(rows, top_k)?;
        let table = mul(count, width)?;
        let input = mul(rows, width)?;
        if table > i32::MAX as usize || input > i32::MAX as usize {
            return Err(model_error("routed table exceeds i32 ABI"));
        }
        let plan = Self {
            input_bytes: mul(input, 4)?,
            intermediate_bytes: 0,
            output_bytes: mul(input, 4)?,
            bias_bytes: 0,
            table_bytes: mul(table, 4)?,
            // Route rows/weights and the largest bucket's gather/scatter IDs.
            route_bytes: mul(add(mul(count, 2)?, mul(largest_bucket, 2)?)?, 4)?,
            numeric_reservation_bytes: 0,
        };
        plan.total_bytes()?;
        Ok(plan)
    }

    pub(crate) fn with_route_bytes(mut self, route_bytes: usize) -> Result<Self> {
        self.route_bytes = add(self.route_bytes, route_bytes)?;
        self.total_bytes()?;
        Ok(self)
    }

    pub(crate) fn with_numeric_reservation(mut self, bytes: usize) -> Result<Self> {
        self.numeric_reservation_bytes = add(self.numeric_reservation_bytes, bytes)?;
        self.total_bytes()?;
        Ok(self)
    }

    pub(crate) fn add(self, other: Self) -> Result<Self> {
        let plan = Self {
            input_bytes: add(self.input_bytes, other.input_bytes)?,
            intermediate_bytes: add(self.intermediate_bytes, other.intermediate_bytes)?,
            output_bytes: add(self.output_bytes, other.output_bytes)?,
            bias_bytes: add(self.bias_bytes, other.bias_bytes)?,
            route_bytes: add(self.route_bytes, other.route_bytes)?,
            table_bytes: add(self.table_bytes, other.table_bytes)?,
            numeric_reservation_bytes: add(
                self.numeric_reservation_bytes,
                other.numeric_reservation_bytes,
            )?,
        };
        plan.total_bytes()?;
        Ok(plan)
    }

    pub(crate) fn total_bytes(&self) -> Result<usize> {
        add(
            add(
                add(self.input_bytes, self.intermediate_bytes)?,
                add(self.output_bytes, self.bias_bytes)?,
            )?,
            add(
                add(self.route_bytes, self.table_bytes)?,
                self.numeric_reservation_bytes,
            )?,
        )
    }

    /// Used at cache boundaries where route/table are already accounted for
    /// by a separate owner-local plan.
    pub(crate) fn total_with(&self, route: usize, numeric_reservation: usize) -> Result<usize> {
        self.with_route_bytes(route)?
            .with_numeric_reservation(numeric_reservation)?
            .total_bytes()
    }

    pub(crate) fn reserved_bytes(
        operation: usize,
        route: usize,
        numeric_reservation: usize,
    ) -> Result<usize> {
        Self {
            input_bytes: 0,
            intermediate_bytes: operation,
            output_bytes: 0,
            bias_bytes: 0,
            route_bytes: route,
            table_bytes: 0,
            numeric_reservation_bytes: numeric_reservation,
        }
        .total_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::ExpertCacheLimits;
    use crate::transformer::standard::expert_cache::Cache;

    #[test]
    fn prepared_metadata_and_descriptor_agree_across_storage_and_bias() {
        use crate::checkpoint::{CheckpointDType, CheckpointTensorSlice};
        use crate::nn::{
            ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec,
        };
        use crate::support::TensorRole;
        use crate::transformer::{
            ExactNameMapper, NameMapping, PreparedLinear, StateDictBinder, StateDictMaterializer,
            StateDictSchema,
        };
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "swiglu-scratch-{}-{}.bin",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        // Planning itself never opens this immutable payload. Only materializing
        // the prepared test operands reads the zero weights and positive scales.
        let mut payload = vec![0u8; 4096];
        payload[1024..1026].copy_from_slice(&0x3f80u16.to_le_bytes());
        payload[1032..1036].copy_from_slice(&1.0f32.to_le_bytes());
        std::fs::write(&path, payload).unwrap();
        for (dtype, scale) in [
            (CheckpointDType::F32, None),
            (CheckpointDType::Bf16, None),
            (CheckpointDType::F8E4M3, Some(CheckpointDType::Bf16)),
            (CheckpointDType::F8E4M3, Some(CheckpointDType::F32)),
        ] {
            let mut schema = StateDictSchema::builder();
            let mut mapper = ExactNameMapper::new();
            let mut slices = Vec::new();
            for (index, role) in [
                TensorRole::RoutedExpertGate,
                TensorRole::RoutedExpertUp,
                TensorRole::RoutedExpertDown,
                TensorRole::Auxiliary,
                TensorRole::Auxiliary,
                TensorRole::Auxiliary,
            ]
            .into_iter()
            .enumerate()
            {
                let name = format!("p{index}");
                let module = ModulePath::new(&name).unwrap();
                let bias = index >= 3;
                let shape = match index {
                    2 => vec![8, 12],
                    3 | 4 => vec![12],
                    5 => vec![8],
                    _ => vec![12, 8],
                };
                let dtype = if bias {
                    CheckpointDType::F32
                } else {
                    dtype.clone()
                };
                let parameter_dtype = match dtype {
                    CheckpointDType::F32 => ParameterDType::F32,
                    CheckpointDType::Bf16 => ParameterDType::Bf16,
                    _ => ParameterDType::F8E4M3,
                };
                let mut spec = ParameterSpec::new(
                    ParameterId::new(index as u64 + 1),
                    module.clone(),
                    parameter_dtype,
                    shape.clone(),
                    ParameterResidency::expert(0, 0),
                )
                .unwrap();
                let bytes = shape.iter().product::<usize>()
                    * match dtype {
                        CheckpointDType::F32 => 4,
                        CheckpointDType::Bf16 => 2,
                        _ => 1,
                    };
                mapper
                    .insert(&name, NameMapping::weight(module.clone()))
                    .unwrap();
                slices.push(CheckpointTensorSlice {
                    name: name.clone(),
                    role: role.clone(),
                    path: path.clone(),
                    offset: 0,
                    bytes: bytes as u64,
                    dtype,
                    shape,
                });
                if !bias && let Some(scale) = &scale {
                    let bf16 = *scale == CheckpointDType::Bf16;
                    spec = spec
                        .with_required_scale(
                            if bf16 {
                                ParameterDType::Bf16
                            } else {
                                ParameterDType::F32
                            },
                            [1, 1],
                        )
                        .unwrap();
                    mapper
                        .insert(format!("{name}_scale"), NameMapping::scale(module))
                        .unwrap();
                    slices.push(CheckpointTensorSlice {
                        name: format!("{name}_scale"),
                        role: role.clone(),
                        path: path.clone(),
                        offset: if bf16 { 1024 } else { 1032 },
                        bytes: if bf16 { 2 } else { 4 },
                        dtype: scale.clone(),
                        shape: vec![1, 1],
                    });
                }
                schema.register_with_role(spec, role).unwrap();
            }
            let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
                .bind_slices(slices)
                .unwrap();
            let parameters =
                [1, 2, 3].map(|id| bound.get_by_id(ParameterId::new(id)).unwrap().clone());
            let metadata = ExpertMetadata::new(0, 0, parameters.clone(), None).unwrap();
            let materializer = StateDictMaterializer::new(4096).unwrap();
            let linear = |i: usize| {
                PreparedLinear::from_parameter(
                    materializer.parameter(&parameters[i]).unwrap(),
                    parameters[i].role().clone(),
                )
                .unwrap()
            };
            let prepared = PreparedSwiGlu::new(linear(0), linear(1), linear(2), None).unwrap();
            let descriptor = SwiGlu::new(8, 12, false).unwrap();
            for rows in [0, 1, 3, 32] {
                let expected = SwiGluScratchPlan::for_descriptor(&descriptor, rows).unwrap();
                assert_eq!(
                    SwiGluScratchPlan::for_prepared(&prepared, rows).unwrap(),
                    expected
                );
                assert_eq!(
                    SwiGluScratchPlan::for_metadata(&metadata, rows).unwrap(),
                    expected
                );
                assert_eq!(expected.total_bytes().unwrap(), rows * (8 + 36 + 8) * 4);
            }
            let biased = |i| {
                linear(i)
                    .with_bias(
                        materializer
                            .parameter(bound.get_by_id(ParameterId::new(i as u64 + 4)).unwrap())
                            .unwrap(),
                    )
                    .unwrap()
            };
            let prepared = PreparedSwiGlu::new(biased(0), biased(1), biased(2), None).unwrap();
            let descriptor = SwiGlu::new(8, 12, true).unwrap();
            for rows in [0, 1, 3, 32] {
                let actual = SwiGluScratchPlan::for_prepared(&prepared, rows).unwrap();
                assert_eq!(
                    actual,
                    SwiGluScratchPlan::for_descriptor(&descriptor, rows).unwrap()
                );
                assert_eq!(
                    actual.total_bytes().unwrap(),
                    rows * (8 + 36 + 8 + 13 + 13 + 9) * 4
                );
            }
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rows_biases_and_asymmetric_output_have_an_exact_byte_ledger() {
        for rows in [0, 1, 3, 32] {
            for mask in 0..8 {
                let widths = [12, 12, 7];
                let biases = std::array::from_fn(|i| (mask & (1 << i) != 0).then_some(widths[i]));
                let plan = SwiGluScratchPlan::new(rows, 8, 12, 7, biases).unwrap();
                let bias = (0..3)
                    .filter(|i| mask & (1 << i) != 0)
                    .map(|i| widths[i] + 1)
                    .sum::<usize>();
                assert_eq!(plan.input_bytes, rows * 8 * 4);
                assert_eq!(plan.intermediate_bytes, rows * 12 * 3 * 4);
                assert_eq!(plan.output_bytes, rows * 7 * 4);
                assert_eq!(plan.bias_bytes, rows * bias * 4);
                assert_eq!(
                    plan.total_with(19, 4096).unwrap(),
                    rows * (8 + 36 + 7 + bias) * 4 + 19 + 4096
                );
            }
            for bias in [false, true] {
                let descriptor = SwiGlu::new(8, 12, bias).unwrap();
                assert_eq!(
                    SwiGluScratchPlan::for_descriptor(&descriptor, rows).unwrap(),
                    SwiGluScratchPlan::new(rows, 8, 12, 8, [12, 12, 8].map(|n| bias.then_some(n)))
                        .unwrap()
                );
            }
        }
    }

    #[test]
    fn route_table_and_numeric_reservation_are_charged_once_at_exact_cap() {
        for rows in [0, 1, 3, 32] {
            for top_k in [1, 2, 8] {
                for bucket in [0, rows / 2, rows] {
                    let route = SwiGluScratchPlan::route(rows, top_k, 8, bucket).unwrap();
                    assert_eq!(route.table_bytes, rows * top_k * 8 * 4);
                    assert_eq!(route.input_bytes, rows * 8 * 4);
                    assert_eq!(route.output_bytes, rows * 8 * 4);
                    assert_eq!(route.route_bytes, (2 * rows * top_k + 2 * bucket) * 4);
                    let plan = SwiGluScratchPlan::new(bucket, 8, 12, 8, [None; 3]).unwrap();
                    let expected = (bucket * (8 + 36 + 8)
                        + 2 * rows * 8
                        + rows * top_k * 8
                        + 2 * rows * top_k
                        + 2 * bucket)
                        * 4
                        + 4096;
                    let combined = plan
                        .add(route)
                        .unwrap()
                        .with_numeric_reservation(4096)
                        .unwrap();
                    assert_eq!(combined.table_bytes, route.table_bytes);
                    assert_eq!(combined.numeric_reservation_bytes, 4096);
                    let scratch = combined.total_bytes().unwrap();
                    assert_eq!(
                        scratch,
                        plan.total_with(route.total_bytes().unwrap(), 4096).unwrap()
                    );
                    assert_eq!(scratch, expected);
                    let weights = 294;
                    for (cap, admitted) in
                        [(expected + weights, true), (expected + weights - 1, false)]
                    {
                        let cache = Cache::<usize>::new(ExpertCacheLimits {
                            max_experts: 1,
                            max_bytes: cap,
                        })
                        .unwrap();
                        assert_eq!(cache.preflight(weights, scratch).is_ok(), admitted);
                    }
                }
            }
        }
    }

    #[test]
    fn every_arithmetic_boundary_is_checked_without_allocating() {
        for (rows, input, intermediate, output, biases) in [
            (0, 1, usize::MAX, 1, [None; 3]),
            (1, usize::MAX, 1, 1, [None; 3]),
            (usize::MAX, 2, 1, 1, [None; 3]),
            (1, usize::MAX / 4 + 1, 0, 0, [None; 3]),
            (1, usize::MAX / 8, usize::MAX / 8, 1, [None; 3]),
            (0, 1, 1, 1, [Some(usize::MAX), None, None]),
            (1, 1, 1, 1, [Some(usize::MAX / 2); 3]),
        ] {
            assert!(SwiGluScratchPlan::new(rows, input, intermediate, output, biases).is_err());
        }
        for (operation, route, numeric) in
            [(usize::MAX, 1, 0), (1, 0, usize::MAX), (0, usize::MAX, 1)]
        {
            assert!(SwiGluScratchPlan::reserved_bytes(operation, route, numeric).is_err());
        }
        for (rows, top_k, width, bucket) in [
            (usize::MAX, 2, 1, 0),
            (2, 1, usize::MAX, 0),
            (i32::MAX as usize, 2, 1, 0),
            (1, 1, i32::MAX as usize + 1, 0),
            (1, 0, 1, 0),
            (1, 1, 0, 0),
            (1, 1, 8, 2),
        ] {
            assert!(SwiGluScratchPlan::route(rows, top_k, width, bucket).is_err());
        }
        assert!(SwiGluScratchPlan::route(0, 8, 2048, 0).is_ok());
        if usize::BITS > 32 {
            assert!(SwiGluScratchPlan::route(1, 1, i32::MAX as usize, 1).is_ok());
        }
    }
}
