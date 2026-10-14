//! Rank geometry dispatch into existing CUDA operators. No second layer loop.
use super::*;

impl CudaStandardDecoderOperators {
    pub(in crate::transformer::standard::cuda) fn prepare_rank_or_standard_linear(
        &mut self,
        linear: &PreparedLinear,
    ) -> Result<()> {
        if self.hybrid_rank.is_none() {
            self.bindings.linear(&self.ops, linear)?;
            return Ok(());
        }
        let local = linear
            .hybrid_rank()
            .ok_or_else(|| cuda_error("rank owner requires prepared local projection"))?;
        let rank = self.hybrid_rank.as_ref().unwrap();
        if local.layout != rank.layout
            || local.partition
                != rank
                    .layout
                    .parameter_partition(linear.parameter().binding())?
            || linear.bias().is_some()
        {
            return Err(cuda_error(
                "prepared rank projection geometry/role mismatch",
            ));
        }
        let key = (linear.parameter().binding().id(), linear.role().clone());
        if rank.linears.contains_key(&key) {
            return Err(cuda_error("duplicate rank projection binding"));
        }
        let max_rows = rank.max_rows;
        let storage = match &local.storage {
            HybridRankStorage::NumericBinding(_) => {
                return Err(cuda_error(
                    "retired Mixed host source cannot rebind a rank operator",
                ));
            }
            HybridRankStorage::Numeric(prepared) => RankLinearStorage::Numeric(
                self.prepare_numeric_fp8_rank_linear(prepared, max_rows)?,
            ),
            HybridRankStorage::Bf16(_) => {
                return Err(cuda_error(
                    "native BF16 rank operator not implemented; no F32 fallback",
                ));
            }
            HybridRankStorage::Dense(values) => {
                self.bindings.validate(linear.parameter())?;
                let bytes = values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>();
                RankLinearStorage::Dense(self.ops.upload_f32_linear(
                    &bytes,
                    local.local_shape[0],
                    local.local_shape[1],
                )?)
            }
        };
        self.hybrid_rank.as_mut().unwrap().linears.insert(
            key,
            Rc::new(RankLinear {
                binding: linear.parameter().binding().clone(),
                shape: local.local_shape,
                partition: local.partition.clone(),
                storage,
            }),
        );
        Ok(())
    }

    pub(in crate::transformer::standard::cuda) fn rank_linear(
        &mut self,
        linear: &PreparedLinear,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<Rows> {
        self.root_barrier(BarrierReason::TensorCollective)?;
        self.synchronous(|this| {
            let rank = this
                .hybrid_rank
                .as_ref()
                .ok_or_else(|| cuda_error("missing CUDA rank"))?;
            let resident = Rc::clone(
                rank.linears
                    .get(&(linear.parameter().binding().id(), linear.role().clone()))
                    .ok_or_else(|| {
                        cuda_error("unprepared rank projection; runtime checkpoint I/O forbidden")
                    })?,
            );
            if !resident
                .binding
                .shares_storage_with(linear.parameter().binding())
                || resident.binding.role() != linear.role()
                || input.shape().width() != resident.shape[1]
                || input.shape().rows() > rank.max_rows
            {
                return Err(cuda_error("rank projection input/image mismatch"));
            }
            let lane = rank.layout.lane();
            let channel = Rc::clone(&rank.channel);
            let rows = input.shape().rows();
            this.f32(input)?;
            let input_hold = this.f32(input)?.prefix(input.shape().elements())?;
            this.hold_expert_buffer(&input_hold)?;
            let mut output = this.ops.zero_f32_buffer(
                rows.checked_mul(resident.shape[0])
                    .ok_or_else(|| cuda_error("rank output overflow"))?,
            )?;
            this.hold_expert_buffer(&output)?;
            match &resident.storage {
                RankLinearStorage::Dense(handle) => {
                    this.ops
                        .linear_f32_into(handle, this.f32(input)?, rows, &mut output)?
                }
                RankLinearStorage::Numeric(token) => {
                    let mut segments = token
                        .plan()
                        .reads()
                        .iter()
                        .map(|r| this.ops.zero_f32_buffer(rows * r.local_shape()[0]))
                        .collect::<Result<Vec<_>>>()?;
                    for buffer in &segments {
                        this.hold_expert_buffer(buffer)?;
                    }
                    this.numeric_fp8_rank_linear_in_root(token, input, &mut segments)?;
                    let mut offset = 0;
                    for (segment, read) in segments.iter().zip(token.plan().reads()) {
                        let width = read.local_shape()[0];
                        for row in 0..rows {
                            this.ops.copy_f32_range(
                                segment,
                                row * width,
                                &mut output,
                                row * resident.shape[0] + offset,
                                width,
                            )?;
                        }
                        offset += width;
                    }
                }
            }
            this.consumer_proof()?;
            if let Some(plan) = &resident.partition.collective {
                this.root_barrier(BarrierReason::TensorCollective)?;
                let values = this.boundary.download(&this.ops, &output)?;
                let stage = TensorParallelStagePlan::from(plan.clone());
                let packed =
                    stage.pack_local_output(ParallelRankId::new(lane as u32), &values, rows)?;
                let semantic = match linear.role() {
                    TensorRole::OutputHead => HybridCollectiveSemantic::OutputHead,
                    TensorRole::AttentionOutput => HybridCollectiveSemantic::AttentionOutput,
                    TensorRole::DenseMlpDown => HybridCollectiveSemantic::DenseMlpDown,
                    TensorRole::SharedExpertDown => HybridCollectiveSemantic::SharedExpertDown,
                    _ => return Err(cuda_error("unsupported rank reduction role")),
                };
                let layer = match linear.parameter().residency() {
                    ParameterResidency::Layer { layer } => Some(*layer),
                    _ => None,
                };
                let values = channel.borrow_mut().exchange(semantic, layer, packed)?;
                let values = stage.unpack_collective_output(&values, rows)?;
                let buffer = this.boundary.upload(&this.ops, &values)?;
                return device_rows(RowsShape::new(rows, plan.out_features())?, arena, buffer);
            }
            device_rows(RowsShape::new(rows, resident.shape[0])?, arena, output)
        })
    }
}
