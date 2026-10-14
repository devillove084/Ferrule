use super::*;

fn owner() -> CudaStandardDecoderOperators {
    CudaStandardDecoderOperators::new(
        Rc::new(CudaOperators::new_on_device(0).unwrap()),
        ExecutionPrecisionPolicy::f32(),
        &[],
    )
    .unwrap()
}
fn plan(rows: usize) -> RecurrentScratchPlan {
    RecurrentScratchPlan::new(rows, 10, 2, 4).unwrap()
}
fn pointers(scratch: &RecurrentScratch) -> [u64; 8] {
    std::array::from_fn(|i| {
        scratch.buffers[i]
            .as_ref()
            .unwrap()
            .as_device_buffer()
            .cu_deviceptr()
    })
}

#[test]
#[ignore = "requires CUDA GPU; grow then decode/mixed-length reuse"]
fn owner_scratch_grows_once_then_reuses_exact_prefixes_without_allocation() {
    let mut owner = owner();
    owner
        .synchronous(|this| this.recurrent_scratch.reserve(&this.ops, plan(4)))
        .unwrap();
    let original = pointers(&owner.recurrent_scratch);
    owner.ops.reset_counters();
    owner
        .synchronous(|this| {
            this.recurrent_scratch.reserve(&this.ops, plan(3))?;
            for rows in [3, 1, 2] {
                let views = this.recurrent_scratch.prefixes(plan(rows))?;
                assert_eq!(views.map(|b| b.len()), plan(rows).elements);
                assert_eq!(pointers(&this.recurrent_scratch), original);
            }
            assert_eq!(this.ops.counters(), Default::default());
            Ok(())
        })
        .unwrap();
    owner
        .synchronous(|this| {
            this.recurrent_scratch.reserve(&this.ops, plan(8))?;
            assert_eq!(this.recurrent_scratch.retired.len(), 8);
            assert_ne!(pointers(&this.recurrent_scratch), original);
            Ok(())
        })
        .unwrap();
    assert!(owner.recurrent_scratch.retired.is_empty());
}

#[test]
#[ignore = "requires CUDA GPU; failure retains roots and unknown freezes custody"]
fn allocation_and_operation_errors_keep_custody_and_unknown_blocks_reuse() {
    let mut owner = owner();
    owner
        .synchronous(|this| this.recurrent_scratch.reserve(&this.ops, plan(2)))
        .unwrap();
    let old = pointers(&owner.recurrent_scratch);
    owner.ops.failpoints().arm_allocation();
    assert!(
        owner
            .synchronous(|this| this.recurrent_scratch.reserve(&this.ops, plan(4)))
            .is_err()
    );
    assert_eq!(pointers(&owner.recurrent_scratch), old);
    assert!(!owner.needs_quarantine());
    let result: Result<()> = owner.synchronous(|this| {
        this.recurrent_scratch.reserve(&this.ops, plan(4))?;
        Err(cuda_error("injected operation failure"))
    });
    assert!(result.is_err());
    assert!(!owner.needs_quarantine());
    assert!(owner.recurrent_scratch.retired.is_empty());
    assert_eq!(
        owner
            .recurrent_scratch
            .prefixes(plan(4))
            .unwrap()
            .map(|b| b.len()),
        plan(4).elements
    );
    let result: Result<()> = owner.synchronous(|this| {
        this.recurrent_scratch.reserve(&this.ops, plan(8))?;
        this.ops.failpoints().arm_stream_sync();
        Err(cuda_error("injected failure with unknown completion"))
    });
    assert!(result.is_err());
    assert!(owner.needs_quarantine());
    assert_eq!(owner.recurrent_scratch.retired.len(), 8);
    let current = pointers(&owner.recurrent_scratch);
    assert!(
        owner
            .synchronous(|_| -> Result<()> { panic!("poisoned owner must not execute") })
            .is_err()
    );
    assert_eq!(pointers(&owner.recurrent_scratch), current);
    assert_eq!(owner.recurrent_scratch.retired.len(), 8);
    // Drop intentionally retains both generations when completion is unknown.
}

#[test]
#[ignore = "requires CUDA GPU; unwinding cannot discard scratch custody"]
fn panic_poison_keeps_both_scratch_generations() {
    let mut owner = owner();
    owner
        .synchronous(|this| this.recurrent_scratch.reserve(&this.ops, plan(1)))
        .unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _: Result<()> = owner.synchronous(|this| {
            this.recurrent_scratch.reserve(&this.ops, plan(3))?;
            panic!("injected recurrent panic");
        });
    }));
    assert!(result.is_err());
    assert!(owner.needs_quarantine());
    assert_eq!(owner.recurrent_scratch.retired.len(), 8);
    assert!(owner.recurrent_scratch.buffers.iter().all(Option::is_some));
}

#[test]
fn direct_selection_requires_the_entire_single_sequence_in_order() {
    assert!(direct_batch(&[vec![0]], 1));
    assert!(direct_batch(&[vec![0, 1, 2]], 3));
    for (indices, rows) in [
        (vec![], 0),
        (vec![vec![]], 0),
        (vec![vec![0, 1], vec![]], 2),
        (vec![vec![0, 2], vec![1]], 3),
        (vec![vec![0, 1]], 3),
        (vec![vec![1, 2]], 2),
        (vec![vec![1, 0]], 2),
        (vec![vec![0, 0]], 2),
    ] {
        assert!(!direct_batch(&indices, rows), "{indices:?}, rows={rows}");
    }
}

use crate::decoder::{CudaGatedDeltaState, HybridCudaDevice, HybridCudaMemoryBudget};
use crate::transformer::{GqaMetadata, PreparedGatedDeltaNetBlock};
use ferrule_backend::cpu::gated_delta::GatedDeltaShape;

// Only the recurrent operator is exercised: the unused projections are never
// launched. All math/state updates below call the production delta_rows path.
struct DeltaFixture {
    owner: CudaStandardDecoderOperators,
    device: HybridCudaDevice,
    source: PreparedGatedDeltaNetBlock,
    file: std::path::PathBuf,
}
impl DeltaFixture {
    fn new() -> Self {
        use crate::checkpoint::{CheckpointDType, CheckpointTensorSlice};
        use crate::nn::{
            ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec,
        };
        use crate::support::TensorRole;
        use crate::transformer::{
            ExactNameMapper, GatedDeltaNetAttention, NameMapping, StateDictBinder,
            StateDictMaterializer, StateDictSchema,
        };
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let file = std::env::temp_dir().join(format!(
            "ferrule-delta-direct-{}-{}.bin",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(
            &file,
            [0.8f32, 1.2, 0.1, 0.2]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let mut schema = StateDictSchema::builder();
        let mut mapper = ExactNameMapper::new();
        let mut slices = Vec::new();
        for (i, (name, shape, role)) in [
            ("norm", vec![2], TensorRole::LinearAttentionNorm),
            ("unused", vec![2, 2], TensorRole::LinearAttentionQkv),
        ]
        .into_iter()
        .enumerate()
        {
            let path = ModulePath::new(name).unwrap();
            schema
                .register_with_role(
                    ParameterSpec::new(
                        ParameterId::new(i as u64 + 1),
                        path.clone(),
                        ParameterDType::F32,
                        shape.clone(),
                        ParameterResidency::layer(0),
                    )
                    .unwrap(),
                    role.clone(),
                )
                .unwrap();
            mapper.insert(name, NameMapping::weight(path)).unwrap();
            slices.push(CheckpointTensorSlice {
                name: name.into(),
                role,
                path: file.clone(),
                offset: 0,
                bytes: (shape.iter().product::<usize>() * 4) as u64,
                dtype: CheckpointDType::F32,
                shape,
            });
        }
        let schema = schema.build().unwrap();
        let bound = StateDictBinder::new(&schema, &mapper)
            .bind_slices(slices)
            .unwrap();
        let materializer = StateDictMaterializer::new(4096).unwrap();
        let norm = materializer.parameter(&bound.parameters()[0]).unwrap();
        let linear = materializer
            .prepared_linear(&bound.parameters()[1], TensorRole::LinearAttentionQkv)
            .unwrap();
        let source = PreparedGatedDeltaNetBlock {
            descriptor: GatedDeltaNetAttention::new(8, 1, 2, 3, 2, 3, 1e-6, false).unwrap(),
            vector_parameters: [norm.clone(), norm.clone(), norm.clone()],
            input_norm: PreparedNorm::new(norm.clone(), 1e-6).unwrap(),
            qkv: linear.clone(),
            z: linear.clone(),
            beta: linear.clone(),
            a: linear.clone(),
            output: linear,
            conv: values(30, 0.04).into(),
            a_log: vec![-0.3, 0.2].into(),
            dt_bias: vec![0.1, -0.2].into(),
            norm: PreparedNorm::new(norm, 1e-6).unwrap(),
        };
        let device = HybridCudaDevice::new_on_device(
            0,
            HybridCudaMemoryBudget {
                kv_pages: 1,
                state_bytes: 4096,
                weight_bytes: 4096,
                workspace_bytes: 65536,
            },
        )
        .unwrap();
        let ops = device.operators();
        let mut owner =
            CudaStandardDecoderOperators::new(Rc::clone(ops), ExecutionPrecisionPolicy::f32(), &[])
                .unwrap();
        owner.deltas.insert(
            0,
            ResidentDelta {
                conv: Rc::new(ops.upload_f32_buffer(&source.conv).unwrap()),
                a_log: Rc::new(ops.upload_f32_buffer(&source.a_log).unwrap()),
                dt_bias: Rc::new(ops.upload_f32_buffer(&source.dt_bias).unwrap()),
                norm: Rc::new(ops.upload_f32_buffer(source.norm.weight()).unwrap()),
                source: source.clone(),
            },
        );
        Self {
            owner,
            device,
            source,
            file,
        }
    }
    fn state(&self, position: usize) -> CudaGatedDeltaState {
        let shape = GatedDeltaShape {
            key_heads: 1,
            value_heads: 2,
            key_dim: 3,
            value_dim: 2,
            kernel: 3,
        };
        let (_, conv, recurrent) = shape.sizes().unwrap();
        CudaGatedDeltaState {
            shape,
            position,
            conv: self
                .owner
                .ops
                .upload_f32_buffer(&values(conv, 0.07))
                .unwrap(),
            recurrent: self
                .owner
                .ops
                .upload_f32_buffer(&values(recurrent, 0.09))
                .unwrap(),
            device: self.device.clone(),
        }
    }
    fn inputs(&self, rows: usize) -> [Rows; 4] {
        [10, 2, 2, 4].map(|width| {
            let mut data = values(rows * width, 0.11);
            data.extend([99.; 7]);
            let root = self.owner.ops.upload_f32_buffer(&data).unwrap();
            let view = root.prefix(rows * width).unwrap();
            drop(root); // Exact view, not the parent allocation's spare capacity.
            device_rows(RowsShape::new(rows, width).unwrap(), None, view).unwrap()
        })
    }
    fn run(
        &mut self,
        inputs: &[Rows; 4],
        states: &mut [CudaGatedDeltaState],
        ids: Vec<usize>,
    ) -> Result<Rows> {
        let mut next = states
            .iter()
            .map(|state| state.position)
            .collect::<Vec<_>>();
        let positions = ids
            .iter()
            .map(|&id| {
                let p = next[id];
                next[id] = next[id].saturating_add(1);
                p
            })
            .collect::<Vec<_>>();
        let metadata = GqaMetadata::new(
            states.len(),
            ids,
            positions.clone(),
            positions.iter().map(|p| p.saturating_add(1)).collect(),
        )
        .unwrap();
        let shape = states[0].shape;
        let mut states = states
            .iter_mut()
            .map(GatedDeltaStateRef::Cuda)
            .collect::<Vec<_>>();
        match self.owner.delta_rows(GatedDeltaNetRequest {
            layer: 0,
            shape,
            qkv: &inputs[0],
            a: &inputs[1],
            b: &inputs[2],
            z: &inputs[3],
            conv: &self.source.conv,
            a_log: &self.source.a_log,
            dt_bias: &self.source.dt_bias,
            norm: &self.source.norm,
            metadata: &metadata,
            states: &mut states,
        })? {
            OperatorProgress::Ready(rows) => Ok(rows),
            _ => panic!("delta must execute"),
        }
    }
}
impl Drop for DeltaFixture {
    fn drop(&mut self) {
        std::fs::remove_file(&self.file).unwrap();
    }
}
fn values(len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|i| ((i % 13) as f32 - 6.) * scale).collect()
}
fn snapshot(ops: &CudaOperators, state: &CudaGatedDeltaState) -> (usize, Vec<f32>, Vec<f32>) {
    (
        state.position,
        ops.download_f32_buffer(&state.conv).unwrap(),
        ops.download_f32_buffer(&state.recurrent).unwrap(),
    )
}
fn output_values(owner: &CudaStandardDecoderOperators, output: &Rows) -> Vec<f32> {
    owner
        .ops
        .download_f32_buffer(owner.f32(output).unwrap())
        .unwrap()
}

#[test]
#[ignore = "requires CUDA GPU; same nonzero state and production direct/general paths"]
fn direct_and_general_match_nonzero_frontiers_and_keep_output_independent() {
    let mut fixture = DeltaFixture::new();
    let mut direct = [fixture.state(7)];
    let mut general = [fixture.state(7)];
    let mut old_outputs = Vec::new();
    for rows in [4, 1, 3, 1] {
        let mut input = fixture.inputs(rows);
        // Read/read aliases are valid; all writable ranges remain independent.
        input[2] = device_rows(
            input[1].shape(),
            None,
            fixture
                .owner
                .f32(&input[1])
                .unwrap()
                .prefix(rows * 2)
                .unwrap(),
        )
        .unwrap();
        let input_before = input
            .each_ref()
            .map(|rows| output_values(&fixture.owner, rows));
        let scratch_before = fixture
            .owner
            .recurrent_scratch
            .buffers
            .each_ref()
            .map(|buffer| {
                buffer
                    .as_ref()
                    .map(|buffer| fixture.owner.ops.download_f32_buffer(buffer).unwrap())
            });
        fixture.owner.recurrent_scratch.force_general = false;
        let copies = fixture.owner.recurrent_scratch.copy_calls.get();
        let calls = fixture.owner.recurrent_scratch.direct_batches.get();
        let output = fixture.run(&input, &mut direct, vec![0; rows]).unwrap();
        assert_eq!(
            fixture.owner.recurrent_scratch.direct_batches.get() - calls,
            1
        );
        assert_eq!(fixture.owner.recurrent_scratch.copy_calls.get() - copies, 0);
        assert!(fixture.owner.recurrent_scratch.direct_hold.is_empty());
        assert_eq!(
            input
                .each_ref()
                .map(|rows| output_values(&fixture.owner, rows)),
            input_before
        );
        for (i, before) in scratch_before.iter().enumerate() {
            if let Some(before) = before {
                let after = fixture
                    .owner
                    .ops
                    .download_f32_buffer(
                        fixture.owner.recurrent_scratch.buffers[i].as_ref().unwrap(),
                    )
                    .unwrap();
                let len = plan(rows).elements[i];
                assert_eq!(&after[len..], &before[len..], "scratch {i} spare capacity");
                if matches!(i, 0 | 1 | 2 | 3 | 7) {
                    assert_eq!(
                        &after, before,
                        "direct path must not write packed scratch {i}"
                    );
                }
            }
        }
        assert_eq!(
            fixture
                .owner
                .recurrent_scratch
                .prefixes(plan(rows))
                .unwrap()
                .map(|b| b.len()),
            plan(rows).elements
        );
        let ptr = fixture
            .owner
            .f32(&output)
            .unwrap()
            .as_device_buffer()
            .cu_deviceptr();
        assert!(!pointers(&fixture.owner.recurrent_scratch).contains(&ptr));
        for source in &input {
            assert_ne!(
                ptr,
                fixture
                    .owner
                    .f32(source)
                    .unwrap()
                    .as_device_buffer()
                    .cu_deviceptr()
            );
        }
        let expected = output_values(&fixture.owner, &output);
        assert!(expected.iter().all(|x| x.is_finite()));
        fixture.owner.recurrent_scratch.force_general = true;
        let general_output = fixture.run(&input, &mut general, vec![0; rows]).unwrap();
        assert_eq!(
            fixture.owner.recurrent_scratch.copy_calls.get() - copies,
            5 * rows
        );
        assert_eq!(
            fixture.owner.recurrent_scratch.direct_batches.get() - calls,
            1
        );
        assert_eq!(output_values(&fixture.owner, &general_output), expected);
        assert_eq!(
            snapshot(&fixture.owner.ops, &direct[0]),
            snapshot(&fixture.owner.ops, &general[0])
        );
        old_outputs.push((output, expected));
        // Subsequent scratch writes cannot change a previously returned result.
        for (output, expected) in &old_outputs {
            assert_eq!(&output_values(&fixture.owner, output), expected);
        }
    }
    assert_eq!(direct[0].position, 16);
}

#[test]
#[ignore = "requires CUDA GPU; ragged/interleaved/empty sequence fallback uses real copies"]
fn ragged_interleaved_and_extra_states_use_general_path() {
    let mut fixture = DeltaFixture::new();
    for ids in [vec![0, 1, 0], vec![0, 0, 1], vec![0, 0, 0]] {
        let input = fixture.inputs(ids.len());
        let mut states = [fixture.state(5), fixture.state(11)];
        let mut references = [fixture.state(5), fixture.state(11)];
        let copies = fixture.owner.recurrent_scratch.copy_calls.get();
        let result = fixture.run(&input, &mut states, ids.clone()).unwrap();
        assert_eq!(fixture.owner.recurrent_scratch.direct_batches.get(), 0);
        assert_eq!(
            fixture.owner.recurrent_scratch.copy_calls.get() - copies,
            ids.len() * 5
        );
        let actual = output_values(&fixture.owner, &result);
        // Each sequence alone is direct; scatter the independently computed oracle.
        let mut expected = vec![0.; ids.len() * 4];
        for sequence in 0..2 {
            let indices = ids
                .iter()
                .enumerate()
                .filter_map(|(row, &id)| (id == sequence).then_some(row))
                .collect::<Vec<_>>();
            if indices.is_empty() {
                continue;
            }
            let packed = std::array::from_fn(|i| {
                let source = output_values(&fixture.owner, &input[i]);
                let width = input[i].shape().width();
                let data = indices
                    .iter()
                    .flat_map(|&row| source[row * width..(row + 1) * width].iter().copied())
                    .collect::<Vec<_>>();
                device_rows(
                    RowsShape::new(indices.len(), width).unwrap(),
                    None,
                    fixture.owner.ops.upload_f32_buffer(&data).unwrap(),
                )
                .unwrap()
            });
            let output = fixture
                .run(
                    &packed,
                    &mut references[sequence..sequence + 1],
                    vec![0; indices.len()],
                )
                .unwrap();
            let data = output_values(&fixture.owner, &output);
            for (local, &row) in indices.iter().enumerate() {
                expected[row * 4..row * 4 + 4].copy_from_slice(&data[local * 4..local * 4 + 4]);
            }
            assert_eq!(
                snapshot(&fixture.owner.ops, &states[sequence]),
                snapshot(&fixture.owner.ops, &references[sequence])
            );
        }
        assert_eq!(actual, expected);
        fixture.owner.recurrent_scratch.direct_batches.set(0);
    }
}

#[test]
#[ignore = "requires CUDA GPU; existing owner, extent, frontier and kernel overlap guards"]
fn direct_rejects_foreign_owner_bad_window_frontier_and_write_overlap() {
    let mut fixture = DeltaFixture::new();
    let mut states = [fixture.state(7)];
    let initial = snapshot(&fixture.owner.ops, &states[0]);
    let foreign = CudaOperators::new_on_device(0).unwrap();
    let mut input = fixture.inputs(1);
    input[0] = device_rows(
        RowsShape::new(1, 10).unwrap(),
        None,
        foreign.zero_f32_buffer(10).unwrap(),
    )
    .unwrap();
    assert!(fixture.run(&input, &mut states, vec![0]).is_err());
    let mut input = fixture.inputs(1);
    input[0] = device_rows(
        RowsShape::new(1, 9).unwrap(),
        None,
        fixture.owner.f32(&input[0]).unwrap().prefix(9).unwrap(),
    )
    .unwrap();
    assert!(fixture.run(&input, &mut states, vec![0]).is_err());
    let input = fixture.inputs(1);
    states[0].position = usize::MAX;
    assert!(fixture.run(&input, &mut states, vec![0]).is_err());
    states[0].position = 7;
    let metadata = GqaMetadata::new(1, vec![0], vec![8], vec![9]).unwrap();
    let shape = states[0].shape;
    let mut refs = [GatedDeltaStateRef::Cuda(&mut states[0])];
    let error = fixture
        .owner
        .delta_rows(GatedDeltaNetRequest {
            layer: 0,
            shape,
            qkv: &input[0],
            a: &input[1],
            b: &input[2],
            z: &input[3],
            conv: &fixture.source.conv,
            a_log: &fixture.source.a_log,
            dt_bias: &fixture.source.dt_bias,
            norm: &fixture.source.norm,
            metadata: &metadata,
            states: &mut refs,
        })
        .unwrap_err();
    assert!(
        error.to_string().contains("recurrent frontier mismatch"),
        "{error}"
    );
    let mut input = fixture.inputs(1);
    input[0] = device_rows(
        RowsShape::new(1, 10).unwrap(),
        None,
        states[0].conv.prefix(10).unwrap(),
    )
    .unwrap();
    // Exercise the existing backend overlap guard on the exact borrowed view.
    let mut convolved = fixture.owner.ops.zero_f32_buffer(10).unwrap();
    let error = fixture
        .owner
        .ops
        .causal_depthwise_conv1d_silu_into(
            CausalConv1dBuffers {
                input: fixture.owner.f32(&input[0]).unwrap(),
                weight: &fixture.owner.deltas[&0].conv,
                bias: None,
                history: &mut states[0].conv,
                output: &mut convolved,
            },
            CausalConv1dLayout {
                rows: 1,
                channels: 10,
                kernel_size: 3,
            },
        )
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("overlapping writable buffer ranges"),
        "{error}"
    );
    assert!(!fixture.owner.needs_quarantine());
    assert_eq!(snapshot(&fixture.owner.ops, &states[0]), initial);
}

#[test]
#[ignore = "requires CUDA GPU; real launched direct operands survive error/unwind/Unknown"]
fn direct_failure_custody_and_unknown_are_sticky() {
    for mode in 0..4 {
        let mut fixture = DeltaFixture::new();
        if mode == 3 {
            fixture.owner.bounded_experts = Some(
                BoundedExpertCache::new(crate::transformer::ExpertCacheLimits {
                    max_experts: 1,
                    max_bytes: 64,
                })
                .unwrap(),
            );
            fixture.owner.lose_next_consumer_proof = true;
        }
        let mut states = [fixture.state(9)];
        let input = fixture.inputs(2);
        fixture.owner.set_diagnostic_trace(move |event| {
            if event.name == "gdn.gated" {
                if mode == 1 {
                    panic!("direct observer panic");
                }
                if mode == 2 {
                    event.operators.failpoints().arm_stream_sync();
                }
                return Err(cuda_error("direct observer failure"));
            }
            Ok(())
        });
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fixture.run(&input, &mut states, vec![0, 0])
        }));
        assert!(!matches!(result, Ok(Ok(_))));
        assert_eq!(fixture.owner.recurrent_scratch.direct_batches.get(), 1);
        assert_eq!(fixture.owner.recurrent_scratch.copy_calls.get(), 0);
        assert_eq!(states[0].position, 9);
        if mode == 0 {
            assert!(!fixture.owner.needs_quarantine());
            assert!(fixture.owner.recurrent_scratch.direct_hold.is_empty());
        } else {
            assert!(fixture.owner.needs_quarantine());
            assert_eq!(fixture.owner.recurrent_scratch.direct_hold.len(), 5);
            drop(input);
            fixture.owner.ops.sync_stream().unwrap();
            fixture.owner.ops.sync_upload_stream().unwrap();
            assert!(fixture.owner.quiesce().is_err());
            assert_eq!(fixture.owner.recurrent_scratch.direct_hold.len(), 5);
            let held = fixture.owner.ops.allocator_metrics().live_requested_bytes;
            let ops = Rc::clone(&fixture.owner.ops);
            drop(fixture);
            ops.trim_device_allocator().unwrap();
            assert!(ops.allocator_metrics().live_requested_bytes >= held);
        }
    }
}

#[test]
#[ignore = "requires CUDA GPU; direct path keeps real consumer and synchronous fences"]
fn direct_path_preserves_proof_and_sync_calls() {
    use crate::transformer::{ExpertCacheLimits, ExpertCachePolicy};
    for policy in [
        ExpertCachePolicy::KeepAll,
        ExpertCachePolicy::Bounded(ExpertCacheLimits {
            max_experts: 1,
            max_bytes: 64,
        }),
    ] {
        let mut fixture = DeltaFixture::new();
        fixture.owner.bounded_experts = match policy {
            ExpertCachePolicy::KeepAll => None,
            ExpertCachePolicy::Bounded(limits) => Some(BoundedExpertCache::new(limits).unwrap()),
        };
        let bounded = fixture.owner.bounded_experts.is_some();
        let mut states = [fixture.state(3)];
        let input = fixture.inputs(1);
        for general in [false, true] {
            fixture.owner.recurrent_scratch.force_general = general;
            let proofs = fixture.owner.consumer_proofs;
            fixture.owner.ops.reset_counters();
            fixture.run(&input, &mut states, vec![0]).unwrap();
            assert_eq!(fixture.owner.consumer_proofs - proofs, usize::from(bounded));
            assert_eq!(
                fixture.owner.ops.counters().stream_wide_syncs,
                if bounded { 3 } else { 2 }
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA GPU; actual lifecycle fork/checkout/rollback with general packed rows"]
fn forked_working_copies_fall_back_and_rollback_without_touching_committed_state() {
    use crate::decoder::{
        CudaHybridSequenceLifecycle, DecoderSequenceCheckout, DecoderSequenceLifecycle,
        HybridStateSchema, StandardSequenceState,
    };
    use crate::transformer::{
        Attention, DecoderLayer, DecoderModelParts, DecoderModelSpec, Embedding, FeedForward,
        Linear, Residual, RmsNorm, SwiGlu,
    };
    let mut fixture = DeltaFixture::new();
    let norm = RmsNorm::new(8, 1e-6).unwrap();
    let spec = DecoderModelSpec::new(DecoderModelParts {
        architecture: "delta-direct-test".into(),
        hidden_size: 8,
        vocab_size: 11,
        max_sequence_length: Some(32),
        token_embedding: Embedding::new(11, 8, None).unwrap(),
        layers: vec![
            DecoderLayer::new(
                0,
                norm.clone(),
                Attention::GatedDeltaNet(fixture.source.descriptor.clone()),
                Residual::Add,
                norm.clone(),
                FeedForward::SwiGlu(SwiGlu::new(8, 12, false).unwrap()),
                Residual::Add,
            )
            .unwrap(),
        ],
        final_norm: norm,
        output: Linear::new(8, 11, false).unwrap(),
        tie_word_embeddings: false,
    })
    .unwrap();
    let schema = HybridStateSchema::from_spec(&spec, 1).unwrap();
    let mut lifecycle = CudaHybridSequenceLifecycle::new(schema, fixture.device.clone()).unwrap();
    let mut source = lifecycle.create().unwrap();
    let GatedDeltaStateRef::Cuda(state) = source.linear_state(0).unwrap() else {
        panic!("CUDA state")
    };
    *state = fixture.state(7);
    let initial = snapshot(&fixture.owner.ops, state);
    let step = source.core().begin_step().unwrap();
    source.core_mut().commit_step(step, 7).unwrap();
    let mut fork = lifecycle.logical_fork(&source, 7).unwrap();
    assert_ne!(source.topology_id(), fork.topology_id());
    let bytes = fixture.device.live_state_bytes();
    let request = DecoderSequenceCheckout::new(ExecutionTransactionId::new(1).unwrap(), 0, 0);
    let mut first = lifecycle.checkout(request, &source).unwrap();
    let mut second = lifecycle.checkout(request, &fork).unwrap();
    let inputs = fixture.inputs(3);
    let metadata = GqaMetadata::new(2, vec![0, 1, 0], vec![7, 7, 8], vec![8, 8, 9]).unwrap();
    let mut refs = [
        first.linear_state(0).unwrap(),
        second.linear_state(0).unwrap(),
    ];
    let shape = match &refs[0] {
        GatedDeltaStateRef::Cuda(state) => state.shape,
        _ => panic!("CUDA"),
    };
    let copies = fixture.owner.recurrent_scratch.copy_calls.get();
    fixture
        .owner
        .delta_rows(GatedDeltaNetRequest {
            layer: 0,
            shape,
            qkv: &inputs[0],
            a: &inputs[1],
            b: &inputs[2],
            z: &inputs[3],
            conv: &fixture.source.conv,
            a_log: &fixture.source.a_log,
            dt_bias: &fixture.source.dt_bias,
            norm: &fixture.source.norm,
            metadata: &metadata,
            states: &mut refs,
        })
        .unwrap();
    assert_eq!(fixture.owner.recurrent_scratch.direct_batches.get(), 0);
    assert_eq!(
        fixture.owner.recurrent_scratch.copy_calls.get() - copies,
        15
    );
    for (state, position) in [(&mut first, 9), (&mut second, 8)] {
        let GatedDeltaStateRef::Cuda(state) = state.linear_state(0).unwrap() else {
            panic!("CUDA")
        };
        assert_eq!(state.position, position);
        assert_ne!(snapshot(&fixture.owner.ops, state).2, initial.2);
    }
    // Abort by releasing the real transaction checkouts, not by copying snapshots back.
    lifecycle.try_release(first).unwrap();
    lifecycle.try_release(second).unwrap();
    assert_eq!(fixture.device.live_state_bytes(), bytes);
    for committed in [&mut source, &mut fork] {
        let GatedDeltaStateRef::Cuda(state) = committed.linear_state(0).unwrap() else {
            panic!("CUDA")
        };
        assert_eq!(snapshot(&fixture.owner.ops, state), initial);
    }
    lifecycle.try_release(source).unwrap();
    lifecycle.try_release(fork).unwrap();
    assert_eq!(fixture.device.live_state_bytes(), 0);
}

#[test]
#[ignore = "requires CUDA GPU; cross-kernel state aliases must gather before any mutation"]
fn overlapping_state_inputs_fall_back_before_cross_kernel_writes() {
    let mut fixture = DeltaFixture::new();
    for use_recurrent in [false, true] {
        for input_index in 0..4 {
            let mut actual_state = [fixture.state(7)];
            let mut reference_state = [fixture.state(7)];
            let mut input = fixture.inputs(1);
            let mut reference = fixture.inputs(1);
            let source = if use_recurrent {
                &actual_state[0].recurrent
            } else {
                &actual_state[0].conv
            };
            let shape = input[input_index].shape();
            let alias = source.prefix(shape.elements()).unwrap();
            let values = fixture.owner.ops.download_f32_buffer(&alias).unwrap();
            input[input_index] = device_rows(shape, None, alias).unwrap();
            reference[input_index] = device_rows(
                shape,
                None,
                fixture.owner.ops.upload_f32_buffer(&values).unwrap(),
            )
            .unwrap();
            let calls = fixture.owner.recurrent_scratch.direct_batches.get();
            let copies = fixture.owner.recurrent_scratch.copy_calls.get();
            let actual = fixture.run(&input, &mut actual_state, vec![0]).unwrap();
            assert_eq!(fixture.owner.recurrent_scratch.direct_batches.get(), calls);
            assert_eq!(fixture.owner.recurrent_scratch.copy_calls.get() - copies, 5);
            let expected = fixture
                .run(&reference, &mut reference_state, vec![0])
                .unwrap();
            assert_eq!(
                fixture.owner.recurrent_scratch.direct_batches.get() - calls,
                1
            );
            assert_eq!(
                output_values(&fixture.owner, &actual),
                output_values(&fixture.owner, &expected)
            );
            assert_eq!(
                snapshot(&fixture.owner.ops, &actual_state[0]),
                snapshot(&fixture.owner.ops, &reference_state[0])
            );
        }
    }
}
