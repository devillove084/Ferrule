//! Production TP executor coverage for a rank-local SwiGLU stage.
//! Workers own model-layer CPU/CUDA shards; collective reduction, transaction
//! voting, post-decision ACK and publication remain production runtime behavior
//! in TensorParallelExecutor. CUDA tests are ignored and require an external
//! process timeout and --test-threads=1; transport is host staged, not NCCL.

use std::sync::Arc;
use std::thread::{self, ThreadId};

use ferrule_backend::cpu::{
    CpuExecutionPrecision, CpuOperatorProvider, HostRows, LinearRef, LinearWeight,
    NativeCpuProvider, RowsDType, RowsShape, SwiGluRef,
};
use ferrule_common::{
    ParallelGroupId, ParallelRankId, ParallelTopologyId, ParallelismPlan, ValidatedParallelTopology,
};
use ferrule_model::transformer::parallel::{
    CpuSwiGluShard, TensorParallelStagePlan, TensorParallelSwiGluPlan,
};
use ferrule_runtime::parallel::collective::HostCollectiveLimits;
use ferrule_runtime::parallel::data::{PanicQuiescence, ReplicaWorker, WorkRequest};
use ferrule_runtime::parallel::tensor::{
    TensorCommand, TensorParallelExecutor, TensorRank, TensorWork,
};
use ferrule_runtime::{ExecutionTransactionId, SessionId};

const HIDDEN: usize = 8;
const INTERMEDIATE: usize = 19;
const ROWS: usize = 3;

fn rank(value: usize) -> ParallelRankId {
    ParallelRankId::new(value as u32)
}

fn tx(value: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(value).unwrap()
}

fn topology(degree: usize) -> ValidatedParallelTopology {
    ValidatedParallelTopology::new(
        ParallelTopologyId::new(73),
        (2 * degree) as u32,
        rank(0),
        ParallelismPlan {
            data_parallel: 2,
            tensor_parallel: degree,
            ..ParallelismPlan::default()
        },
    )
    .unwrap()
}

fn weights(size: usize, salt: usize) -> Vec<f32> {
    (0..size)
        .map(|index| {
            let value = ((index * 17 + salt * 11) % 37) as f32 - 18.0;
            value / 23.0
        })
        .collect()
}

fn input(rows: usize) -> Arc<[f32]> {
    (0..rows * HIDDEN)
        .map(|index| ((index * 13 % 29) as f32 - 14.0) / 11.0)
        .collect::<Vec<_>>()
        .into()
}

fn limits(degree: usize) -> HostCollectiveLimits {
    HostCollectiveLimits {
        max_ranks: degree,
        max_elements_per_rank: ROWS * HIDDEN,
        max_host_bytes: 4 * degree * ROWS * HIDDEN * 3,
    }
}

struct SwiGluWorker {
    owner: ThreadId,
    rank: TensorRank,
    shard: CpuSwiGluShard,
}

impl ReplicaWorker<TensorWork> for SwiGluWorker {
    type Output = Vec<f32>;
    type Error = ferrule_common::Error;

    fn execute(&mut self, request: WorkRequest<TensorWork>) -> Result<Vec<f32>, Self::Error> {
        assert_eq!(thread::current().id(), self.owner);
        assert_eq!(request.rank, self.rank.local);
        assert_eq!(request.input.rank, self.rank);
        match request.input.command {
            TensorCommand::Compute { input, rows } => {
                self.shard
                    .execute(&input, rows)
                    .map_err(|error| ferrule_common::Error::Model {
                        message: error.to_string(),
                    })
            }
            TensorCommand::Apply { values, rows } => {
                assert_eq!(values.len(), rows * HIDDEN);
                Ok(values)
            }
        }
    }

    fn panic_quiescence(&mut self) -> PanicQuiescence {
        PanicQuiescence::Quiescent
    }
}

fn native_oracle(gate: &[f32], up: &[f32], down: &[f32], values: &[f32]) -> Vec<f32> {
    fn linear(weight: &[f32], out_features: usize, in_features: usize) -> LinearRef<'_> {
        LinearRef {
            weight: LinearWeight::F32(weight),
            out_features,
            in_features,
            bias: None,
        }
    }
    let rows = values.len() / HIDDEN;
    let input = HostRows::new(
        RowsShape::new(rows, HIDDEN).unwrap(),
        RowsDType::F32,
        None,
        values.to_vec(),
    )
    .unwrap();
    NativeCpuProvider
        .swiglu(
            SwiGluRef {
                gate: linear(gate, INTERMEDIATE, HIDDEN),
                up: linear(up, INTERMEDIATE, HIDDEN),
                down: linear(down, HIDDEN, INTERMEDIATE),
                activation_limit: None,
            },
            &input,
            None,
            CpuExecutionPrecision::F32,
        )
        .unwrap()
        .into_values()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 3e-5 * (1.0 + expected.abs()),
            "{actual} != {expected}"
        );
    }
}

#[test]
fn production_executor_reduces_ragged_swiglu_partials_once() {
    for degree in [2, 4, 8] {
        let stage = TensorParallelSwiGluPlan::new(HIDDEN, INTERMEDIATE, degree).unwrap();
        let stage_api = TensorParallelStagePlan::from(stage.clone());
        assert_eq!(stage_api.in_features(), HIDDEN);
        assert_eq!(stage_api.out_features(), HIDDEN);
        assert_eq!(stage_api.ranks(), degree);
        assert_eq!(
            stage_api.collective(),
            ferrule_model::transformer::parallel::TensorParallelCollective::Sum
        );
        assert_eq!(stage_api.collective_count(ROWS).unwrap(), ROWS * HIDDEN);

        let gate = Arc::new(weights(INTERMEDIATE * HIDDEN, 0));
        let up = Arc::new(weights(INTERMEDIATE * HIDDEN, 1));
        let down = Arc::new(weights(HIDDEN * INTERMEDIATE, 2));
        let factory_stage = stage.clone();
        let factory_gate = Arc::clone(&gate);
        let factory_up = Arc::clone(&up);
        let factory_down = Arc::clone(&down);
        let mut executor = TensorParallelExecutor::new(
            topology(degree),
            1,
            stage_api,
            ferrule_runtime::parallel::data::DataParallelConfig {
                replicas: degree,
                max_outstanding_per_replica: 1,
                session_capacity: degree,
            },
            move |rank: TensorRank| {
                let weights = factory_stage.shard_weights_f32(
                    rank.local,
                    &factory_gate,
                    &factory_up,
                    &factory_down,
                )?;
                CpuSwiGluShard::from_local(weights)
                    .map(|shard| SwiGluWorker {
                        owner: thread::current().id(),
                        rank,
                        shard,
                    })
                    .map_err(|error| ferrule_common::Error::Model {
                        message: error.to_string(),
                    })
            },
            ParallelGroupId::new(91),
            limits(degree),
        )
        .unwrap();

        let values = input(ROWS);
        let expected = native_oracle(&gate, &up, &down, &values);
        assert!(expected.iter().any(|v| v.abs() > 0.01));
        let output = executor
            .execute(
                tx(degree as u64),
                SessionId(1000 + degree as u64),
                values,
                ROWS,
            )
            .unwrap();
        assert_eq!(output.ranks.len(), degree);
        for (rank, result) in output.ranks {
            assert_eq!(
                rank.local.get() as usize,
                rank.global.get() as usize - degree
            );
            assert_eq!(result.len(), ROWS * HIDDEN);
            assert_close(&result, &expected);
        }
        assert_eq!(executor.outstanding(), 0);
        assert_eq!(executor.coordinator().in_use_credits(), 0);
        assert_eq!(executor.coordinator().retained_transaction_count(), 0);
        assert_eq!(executor.coordinator().publication_count(), 1);
        assert!(executor.collective().active_descriptor().is_none());
        assert_eq!(executor.collective().owned_host_bytes(), 0);
        executor.shutdown().unwrap();
    }
}

#[test]
fn stage_plan_metadata_is_the_only_runtime_swiglu_shape_contract() {
    let stage = TensorParallelStagePlan::from(
        TensorParallelSwiGluPlan::new(HIDDEN, INTERMEDIATE, 4).unwrap(),
    );
    assert_eq!(stage.in_features(), HIDDEN);
    assert_eq!(stage.out_features(), HIDDEN);
    assert_eq!(stage.ranks(), 4);
    assert_eq!(stage.collective_count(ROWS).unwrap(), ROWS * HIDDEN);
    assert!(stage.collective_count(0).is_err());
}

// serde_json is an existing Unix dependency; leave non-Unix CPU coverage above.
#[cfg(unix)]
mod checkpoint_stage {
    use super::{rank, tx};
    use std::collections::{BTreeMap, HashSet};
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread::{self, ThreadId};

    use ferrule_backend::cpu::{bf16_rne, bf16_rne_word};
    use ferrule_common::{
        ParallelGroupId, ParallelTopologyId, ParallelismPlan, ValidatedParallelTopology,
    };
    use ferrule_model::ModelFamily;
    use ferrule_model::checkpoint::{
        CheckpointDType, CheckpointSourceFileIdentity, CheckpointTensorReader,
        CheckpointTensorSlice, HfSafetensorsIndex, HfSafetensorsInventory,
    };
    use ferrule_model::transformer::parallel::{
        CpuSwiGluShard, TensorParallelCollective, TensorParallelStagePlan,
        TensorParallelSwiGluPlan, TensorParallelSwiGluWeights,
    };
    use ferrule_runtime::SessionId;
    use ferrule_runtime::parallel::collective::HostCollectiveLimits;
    use ferrule_runtime::parallel::data::{
        DataParallelConfig, PanicQuiescence, ReplicaWorker, WorkRequest,
    };
    use ferrule_runtime::parallel::tensor::{
        TensorCommand, TensorParallelExecutor, TensorRank, TensorWork,
    };

    const HIDDEN: usize = 32;
    const INTERMEDIATE: usize = 256;
    const ROWS: usize = 3;
    const DEVICES: usize = 8;
    const NAMES: [&str; 3] = [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ];
    // At most 256 accumulated F32 terms, without cancellation in each down row.
    // BF16 rounding is explicit in the oracle, not hidden in a larger tolerance.
    const ATOL: f32 = 2e-6;
    const RTOL: f32 = 8e-6;
    type Checked<T = ()> = Result<T, String>;

    fn text(error: impl std::fmt::Debug) -> String {
        format!("{error:?}")
    }
    fn require(value: bool, message: impl Into<String>) -> Checked {
        if value { Ok(()) } else { Err(message.into()) }
    }
    fn close(label: &str, actual: &[f32], expected: &[f32]) -> Checked {
        require(actual.len() == expected.len(), format!("{label}: shape"))?;
        let mut max_error = 0.0f32;
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            let error = (a - e).abs();
            require(
                a.is_finite() && e.is_finite() && error <= ATOL + RTOL * e.abs(),
                format!(
                    "{label}[{i}]: actual={a} expected={e} error={error}; atol={ATOL} rtol={RTOL}"
                ),
            )?;
            max_error = max_error.max(error);
        }
        eprintln!("{label}: max_abs_error={max_error:.8e}");
        Ok(())
    }
    fn weight(index: usize, matrix: usize) -> f32 {
        let width = if matrix == 2 { INTERMEDIATE } else { HIDDEN };
        let sign = if matrix == 2 && (index / width) % 3 == 1 {
            -1.0
        } else {
            1.0
        };
        sign * (0.019 + ((index * 7 + index / width + matrix * 13) % 31) as f32 * 0.0013)
    }
    fn input() -> Vec<f32> {
        (0..ROWS * HIDDEN)
            .map(|i| 0.23 + ((i * 11) % 37) as f32 * 0.027 + (i / HIDDEN) as f32 * 0.13)
            .collect()
    }
    fn encode(values: &[f32], dtype: &CheckpointDType) -> Vec<u8> {
        match dtype {
            CheckpointDType::Bf16 => values
                .iter()
                .flat_map(|v| bf16_rne_word(*v).to_le_bytes())
                .collect(),
            CheckpointDType::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            _ => unreachable!("dense fixture"),
        }
    }

    #[derive(Clone)]
    struct Catalog {
        tensors: [CheckpointTensorSlice; 3],
        sources: [CheckpointSourceFileIdentity; 3],
    }
    impl Catalog {
        fn load(
            &self,
            plan: &TensorParallelSwiGluPlan,
            local: ferrule_common::ParallelRankId,
        ) -> Checked<TensorParallelSwiGluWeights> {
            let range = plan.rank_range(local).map_err(text)?;
            let dtype = &self.tensors[0].dtype;
            let element_bytes = dtype.element_size_bytes().unwrap();
            let budget = HIDDEN * range.len() * element_bytes;
            // This limit excludes a full tensor on every TP>1 owner. The only
            // full reads belong to a separate TP=1 oracle, never the factory.
            let bundle = plan
                .read_weight_shards_with_sources(
                    &CheckpointTensorReader::new(budget as u64),
                    self.tensors.each_ref(),
                    local,
                    self.sources.each_ref(),
                )
                .map_err(text)?;
            for (matrix, shard) in [bundle.gate(), bundle.up(), bundle.down()]
                .into_iter()
                .enumerate()
            {
                require(
                    shard.rank() == local && shard.dtype() == dtype,
                    "rank/dtype changed",
                )?;
                require(shard.bytes().len() == budget, "nonlocal payload size")?;
                if plan.ranks() > 1 {
                    require(
                        (budget as u64) < self.tensors[matrix].bytes,
                        "full weight admitted",
                    )?;
                }
                let read = shard.provenance().ok_or("missing provenance")?;
                let (rows, columns, width) = if matrix == 2 {
                    (0..HIDDEN, range.clone(), INTERMEDIATE)
                } else {
                    (range.clone(), 0..HIDDEN, HIDDEN)
                };
                require(
                    read.rows() == rows && read.columns() == columns,
                    "wrong rank rectangle",
                )?;
                require(
                    read.read_plan().storage_bytes() == budget as u64,
                    "nonlocal storage size",
                )?;
                require(
                    read.read_plan().extents().len()
                        == if matrix == 2 && range.len() != INTERMEDIATE {
                            HIDDEN
                        } else {
                            1
                        },
                    "wrong read extents",
                )?;
                for (row, extent) in read.read_plan().extents().iter().enumerate() {
                    let offset = (rows.start + row) * width + columns.start;
                    let count = if columns.len() != width {
                        columns.len()
                    } else {
                        rows.len() * width
                    };
                    require(
                        extent.offset()
                            == self.tensors[matrix].offset + (offset * element_bytes) as u64
                            && extent.bytes() == (count * element_bytes) as u64,
                        "nonlocal file extent",
                    )?;
                }
                // Verify only the local rectangle; no full-weight decode/copy.
                let expected: Vec<_> = rows
                    .flat_map(|row| {
                        columns
                            .clone()
                            .map(move |col| weight(row * width + col, matrix))
                    })
                    .collect();
                require(
                    shard.bytes() == encode(&expected, dtype),
                    "native checkpoint bytes changed",
                )?;
            }
            Ok(bundle)
        }
    }

    struct Fixture {
        directory: PathBuf,
        catalog: Catalog,
    }
    impl Fixture {
        fn new(dtype: CheckpointDType) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let directory = std::env::temp_dir().join(format!(
                "ferrule-runtime-swiglu-{}-{nonce}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&directory).unwrap();
            let files = [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ];
            for (file, matrices) in [(files[0], vec![0, 2]), (files[1], vec![1])] {
                let mut header = serde_json::Map::new();
                let mut data = Vec::new();
                for matrix in matrices {
                    let start = data.len();
                    let values: Vec<_> = (0..HIDDEN * INTERMEDIATE)
                        .map(|i| weight(i, matrix))
                        .collect();
                    data.extend(encode(&values, &dtype));
                    let shape = if matrix == 2 {
                        [HIDDEN, INTERMEDIATE]
                    } else {
                        [INTERMEDIATE, HIDDEN]
                    };
                    header.insert(NAMES[matrix].into(), serde_json::json!({
                        "dtype": dtype.as_str(), "shape": shape, "data_offsets": [start, data.len()],
                    }));
                }
                let mut header = serde_json::to_vec(&header).unwrap();
                while !header.len().is_multiple_of(8) {
                    header.push(b' ');
                }
                let mut output = std::fs::File::create(directory.join(file)).unwrap();
                output
                    .write_all(&(header.len() as u64).to_le_bytes())
                    .unwrap();
                output.write_all(&header).unwrap();
                output.write_all(&data).unwrap();
            }
            let index_path = directory.join("model.safetensors.index.json");
            std::fs::write(&index_path, serde_json::to_vec(&serde_json::json!({
                "metadata": {"total_size": 3 * HIDDEN * INTERMEDIATE * dtype.element_size_bytes().unwrap()},
                "weight_map": BTreeMap::from([(NAMES[0], files[0]), (NAMES[1], files[1]), (NAMES[2], files[0])]),
            })).unwrap()).unwrap();
            let index = HfSafetensorsIndex::open(index_path).unwrap();
            let inventory =
                HfSafetensorsInventory::from_index(&directory, ModelFamily::Qwen3, &index).unwrap();
            assert_eq!((inventory.shard_count, inventory.tensor_count), (2, 3));
            let tensors = NAMES.map(|name| {
                CheckpointTensorSlice::from_hf_inventory(
                    &directory,
                    inventory.tensors.iter().find(|t| t.name == name).unwrap(),
                )
            });
            assert!(tensors[2].offset > tensors[0].offset);
            let sources = std::array::from_fn(|i| {
                CheckpointSourceFileIdentity::capture(&tensors[i].path).unwrap()
            });
            Self {
                directory,
                catalog: Catalog { tensors, sources },
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.directory);
        }
    }

    fn boundary_oracle(dtype: &CheckpointDType, cuda: bool, input: &[f32]) -> Vec<f32> {
        let bf16 = *dtype == CheckpointDType::Bf16;
        let round_weight = |v| if bf16 { bf16_rne(v) } else { v };
        let round_input = |v| if bf16 && cuda { bf16_rne(v) } else { v };
        let mut hidden = vec![0.0; ROWS * INTERMEDIATE];
        for row in 0..ROWS {
            for channel in 0..INTERMEDIATE {
                let mut gate = 0.0f32;
                let mut up = 0.0f32;
                for k in 0..HIDDEN {
                    let x = round_input(input[row * HIDDEN + k]);
                    gate = x.mul_add(round_weight(weight(channel * HIDDEN + k, 0)), gate);
                    up = x.mul_add(round_weight(weight(channel * HIDDEN + k, 1)), up);
                }
                // BF16 artifact linears round their inputs, not their F32
                // outputs. Round the activation at down's input only; partials
                // and the single runtime sum remain F32.
                let activation = gate * (1.0 / (1.0 + (-gate).exp())) * up;
                if bf16 && cuda {
                    // Keep libm/device exp differences away from RNE midpoints.
                    assert!(
                        (activation.to_bits() & 0xffff).abs_diff(0x8000) > 32,
                        "fixture too close to BF16 midpoint: {activation}"
                    );
                }
                hidden[row * INTERMEDIATE + channel] = round_input(activation);
            }
        }
        let mut output = vec![0.0; ROWS * HIDDEN];
        for row in 0..ROWS {
            for feature in 0..HIDDEN {
                let mut sum = 0.0f32;
                for k in 0..INTERMEDIATE {
                    sum = hidden[row * INTERMEDIATE + k]
                        .mul_add(round_weight(weight(feature * INTERMEDIATE + k, 2)), sum);
                }
                output[row * HIDDEN + feature] = sum;
            }
        }
        output
    }

    #[derive(Default)]
    struct Observations {
        owners: Mutex<Vec<(TensorRank, ThreadId)>>,
        computed: Mutex<Vec<(TensorRank, Vec<f32>)>>,
        applied: Mutex<Vec<(TensorRank, Vec<f32>)>>,
        stopped: Mutex<Vec<(TensorRank, ThreadId)>>,
    }
    #[derive(Clone, Copy, Debug)]
    enum Backend {
        Cpu,
        #[cfg(feature = "cuda")]
        Cuda,
    }
    enum Local {
        Cpu(CpuSwiGluShard),
        #[cfg(feature = "cuda")]
        Cuda(cuda::Resident),
    }
    struct Worker {
        rank: TensorRank,
        owner: ThreadId,
        local: Local,
        observations: Arc<Observations>,
    }
    impl ReplicaWorker<TensorWork> for Worker {
        type Output = Vec<f32>;
        type Error = String;
        fn execute(&mut self, request: WorkRequest<TensorWork>) -> Checked<Vec<f32>> {
            let work = |request: &WorkRequest<TensorWork>| {
                require(thread::current().id() == self.owner, "owner thread changed")?;
                require(
                    request.rank == self.rank.local && request.input.rank == self.rank,
                    "local/global rank mismatch",
                )?;
                require(
                    !request.cancellation.is_requested(),
                    "unexpected cancellation",
                )?;
                match &request.input.command {
                    TensorCommand::Compute { input, rows } => {
                        require(*rows == ROWS, "Compute rows")?;
                        let partial = match &self.local {
                            Local::Cpu(shard) => shard.execute(input, *rows).map_err(text)?,
                            #[cfg(feature = "cuda")]
                            Local::Cuda(resident) => {
                                require(
                                    resident.ops.device_ordinal()
                                        == self.rank.global.get() as usize,
                                    "Compute device ordinal",
                                )?;
                                resident.execute(input, *rows)?
                            }
                        };
                        self.observations
                            .computed
                            .lock()
                            .unwrap()
                            .push((self.rank, partial.clone()));
                        Ok(partial)
                    }
                    TensorCommand::Apply { values, rows } => {
                        require(
                            *rows == ROWS && values.len() == rows * HIDDEN,
                            "Apply shape",
                        )?;
                        let applied = match &self.local {
                            Local::Cpu(_) => values.clone(),
                            #[cfg(feature = "cuda")]
                            Local::Cuda(resident) => cuda::apply(&resident.ops, values)?,
                        };
                        self.observations
                            .applied
                            .lock()
                            .unwrap()
                            .push((self.rank, values.clone()));
                        Ok(applied)
                    }
                }
            };
            match &self.local {
                Local::Cpu(_) => work(&request),
                #[cfg(feature = "cuda")]
                Local::Cuda(resident) => {
                    // Keep the DMA source outside the unwind closure. A worker
                    // panic hook runs too late to protect a dropped request.
                    let retained = cuda::Resources::new(&resident.ops, request);
                    cuda::on_owner(&resident.ops, || work(retained.get()))
                }
            }
        }
        fn panic_quiescence(&mut self) -> PanicQuiescence {
            match &self.local {
                Local::Cpu(_) => PanicQuiescence::Quiescent,
                #[cfg(feature = "cuda")]
                Local::Cuda(resident) => {
                    if resident.shard.quiesce().is_ok() {
                        PanicQuiescence::Quiescent
                    } else {
                        PanicQuiescence::Unknown
                    }
                }
            }
        }
        fn shutdown(&mut self) -> Checked {
            #[cfg(feature = "cuda")]
            if let Local::Cuda(resident) = &self.local {
                cuda::on_owner(&resident.ops, || {
                    resident.shard.quiesce().map_err(cuda::shard_error)
                })?;
            }
            require(
                thread::current().id() == self.owner,
                "shutdown owner changed",
            )?;
            self.observations
                .stopped
                .lock()
                .unwrap()
                .push((self.rank, self.owner));
            Ok(())
        }
    }

    fn run_case(degree: usize, backend: Backend) {
        for dtype in [CheckpointDType::F32, CheckpointDType::Bf16] {
            let fixture = Fixture::new(dtype.clone());
            let plan = TensorParallelSwiGluPlan::new(HIDDEN, INTERMEDIATE, degree).unwrap();
            let stage = TensorParallelStagePlan::from(plan.clone());
            assert_eq!(stage.collective(), TensorParallelCollective::Sum);
            assert_eq!(
                (stage.in_features(), stage.out_features(), stage.ranks()),
                (HIDDEN, HIDDEN, degree)
            );
            assert_eq!(stage.collective_count(ROWS).unwrap(), ROWS * HIDDEN);
            for r in 0..degree {
                plan.validate_cuda(rank(r), &dtype).unwrap();
            }
            let values = input();
            let expected = match backend {
                Backend::Cpu => {
                    let full = TensorParallelSwiGluPlan::new(HIDDEN, INTERMEDIATE, 1).unwrap();
                    let full =
                        CpuSwiGluShard::from_local(fixture.catalog.load(&full, rank(0)).unwrap())
                            .unwrap();
                    full.execute(&values, ROWS).unwrap()
                }
                #[cfg(feature = "cuda")]
                Backend::Cuda => {
                    cuda::full_oracle(fixture.catalog.clone(), values.clone()).unwrap()
                }
            };
            let is_cuda = !matches!(backend, Backend::Cpu);
            close(
                "unsplit vs dtype-boundary oracle",
                &expected,
                &boundary_oracle(&dtype, is_cuda, &values),
            )
            .unwrap();
            assert!(expected.iter().all(|v| v.abs() > 0.01));
            let replica = if degree == DEVICES { 0 } else { 1 };
            let topology = ValidatedParallelTopology::new(
                ParallelTopologyId::new(74),
                DEVICES as u32,
                rank(0),
                ParallelismPlan {
                    data_parallel: DEVICES / degree,
                    tensor_parallel: degree,
                    ..ParallelismPlan::default()
                },
            )
            .unwrap();
            let observations = Arc::new(Observations::default());
            let observed = Arc::clone(&observations);
            let host = thread::current().id();
            // Only metadata/source snapshots cross into the factory. Each owner
            // reads its own gate/up rows and down columns from the actual files.
            let catalog = fixture.catalog.clone();
            let mut executor = TensorParallelExecutor::new(
                topology,
                replica,
                stage,
                DataParallelConfig {
                    replicas: degree,
                    max_outstanding_per_replica: 1,
                    session_capacity: degree,
                },
                move |r: TensorRank| {
                    let owner = thread::current().id();
                    require(owner != host, "factory ran on coordinator")?;
                    let weights = catalog.load(&plan, r.local)?;
                    let local = match backend {
                        Backend::Cpu => {
                            Local::Cpu(CpuSwiGluShard::from_local(weights).map_err(text)?)
                        }
                        #[cfg(feature = "cuda")]
                        Backend::Cuda => {
                            Local::Cuda(cuda::Resident::new(r.global.get() as usize, weights)?)
                        }
                    };
                    observed.owners.lock().unwrap().push((r, owner));
                    Ok(Worker {
                        rank: r,
                        owner,
                        local,
                        observations: Arc::clone(&observed),
                    })
                },
                ParallelGroupId::new(92),
                HostCollectiveLimits {
                    max_ranks: degree,
                    max_elements_per_rank: ROWS * HIDDEN,
                    max_host_bytes: 2 * degree * ROWS * HIDDEN * std::mem::size_of::<f32>(),
                },
            )
            .unwrap();
            // Always explicitly shut down/join even if execution/validation fails.
            let result = executor.execute(tx(1), SessionId(2000), values.into(), ROWS);
            let shutdown = executor.shutdown();
            assert!(shutdown.is_ok(), "shutdown={shutdown:?}; result={result:?}");
            let output = result.unwrap();
            assert_eq!(output.transaction, tx(1));
            assert_eq!(output.ranks.len(), degree);
            assert!(!executor.is_quarantined());
            assert_eq!(executor.outstanding(), 0);
            assert_eq!(executor.coordinator().publication_count(), 1);
            assert_eq!(executor.coordinator().in_use_credits(), 0);
            assert_eq!(executor.coordinator().retained_transaction_count(), 0);
            assert_eq!(executor.coordinator().retained_operation_count(), 0);
            assert!(executor.coordinator().pending_completions().is_empty());
            assert!(executor.collective().active_descriptor().is_none());
            assert_eq!(executor.collective().owned_host_bytes(), 0);
            assert_eq!(executor.collective().reserved_host_bytes(), 0);
            let owners = observations.owners.lock().unwrap();
            let stopped = observations.stopped.lock().unwrap();
            let computed = observations.computed.lock().unwrap();
            let applied = observations.applied.lock().unwrap();
            for count in [owners.len(), stopped.len(), computed.len(), applied.len()] {
                assert_eq!(count, degree);
            }
            assert_eq!(
                owners
                    .iter()
                    .map(|(_, owner)| *owner)
                    .collect::<HashSet<_>>()
                    .len(),
                degree
            );
            // Test-only sum of observed local partials detects a second reduction,
            // missing rank, or prematurely returning an unsummed local partial.
            let mut once = vec![0.0; ROWS * HIDDEN];
            for local in 0..degree {
                let r = TensorRank {
                    local: rank(local),
                    global: rank(replica as usize * degree + local),
                };
                let (_, partial) = computed.iter().find(|(rank, _)| *rank == r).unwrap();
                assert_eq!(partial.len(), once.len());
                assert!(partial.iter().any(|v| v.abs() > 0.01));
                for (sum, value) in once.iter_mut().zip(partial) {
                    *sum += value;
                }
            }
            for (local, (r, result)) in output.ranks.iter().enumerate() {
                assert_eq!(
                    *r,
                    TensorRank {
                        local: rank(local),
                        global: rank(replica as usize * degree + local)
                    }
                );
                let owner = owners.iter().find(|(rank, _)| rank == r).unwrap();
                assert_ne!(owner.1, host);
                assert!(stopped.contains(owner));
                let (_, applied) = applied.iter().find(|(rank, _)| rank == r).unwrap();
                close("Apply receives exactly one sum", applied, &once).unwrap();
                assert_eq!(result, applied, "Apply H2D/D2H must preserve F32 bits");
                close(
                    &format!("{backend:?} {dtype:?} TP={degree} rank={r:?}"),
                    result,
                    &expected,
                )
                .unwrap();
            }
        }
    }

    #[test]
    fn cpu_checkpoint_swiglu_executor_rank_matrix() {
        for degree in [2, 4, 8] {
            run_case(degree, Backend::Cpu);
        }
    }

    #[test]
    fn bf16_cuda_boundary_oracle_is_not_cpu_f32_activation_semantics() {
        let values = input();
        let cpu = boundary_oracle(&CheckpointDType::Bf16, false, &values);
        let cuda = boundary_oracle(&CheckpointDType::Bf16, true, &values);
        assert!(
            cpu.iter()
                .zip(&cuda)
                .any(|(a, b)| (a - b).abs() > ATOL + RTOL * b.abs())
        );
        assert_eq!(
            boundary_oracle(&CheckpointDType::F32, false, &values),
            boundary_oracle(&CheckpointDType::F32, true, &values)
        );
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;
        use ferrule_backend::cuda::operators::linear::CudaOperators;
        use ferrule_backend::cuda::providers::CudaContext;
        use ferrule_model::transformer::parallel::{CudaShardError, CudaSwiGluShard};
        use std::panic::{AssertUnwindSafe, catch_unwind};
        use std::rc::Rc;

        pub(super) fn shard_error(error: ferrule_common::Error) -> String {
            if CudaShardError::from_error(&error).is_some_and(CudaShardError::needs_quarantine) {
                eprintln!("CUDA shard requires owner quarantine: {error}");
                std::panic::panic_any(PanicQuiescence::Unknown);
            }
            text(error)
        }

        pub(super) fn sync_owner(ops: &CudaOperators) -> Checked {
            // Never short-circuit the upload fence when compute sync fails.
            let compute = ops.sync_stream();
            let upload = ops.sync_upload_stream();
            match (compute, upload) {
                (Ok(()), Ok(())) => Ok(()),
                (compute, upload) => {
                    Err(format!("compute sync={compute:?}; upload sync={upload:?}"))
                }
            }
        }

        pub(super) fn on_owner<T>(
            ops: &Rc<CudaOperators>,
            work: impl FnOnce() -> Checked<T>,
        ) -> Checked<T> {
            let result = catch_unwind(AssertUnwindSafe(work));
            let forced_unknown = result.as_ref().err().is_some_and(|payload| {
                payload.downcast_ref::<PanicQuiescence>() == Some(&PanicQuiescence::Unknown)
            });
            let synchronized = sync_owner(ops);
            if forced_unknown || synchronized.is_err() {
                // The factory has no worker quarantine hook yet. Preserve any
                // returned handles and the owner too, not just its host thread.
                std::mem::forget(result);
                std::mem::forget(Rc::clone(ops));
                eprintln!("CUDA SwiGLU owner quiescence unknown: {synchronized:?}");
                std::panic::panic_any(PanicQuiescence::Unknown);
            }
            match result {
                Ok(result) => result,
                Err(payload) => {
                    let message = payload
                        .downcast_ref::<String>()
                        .map(String::as_str)
                        .or_else(|| payload.downcast_ref::<&str>().copied())
                        .unwrap_or("non-string panic");
                    Err(format!(
                        "CUDA SwiGLU owner panic after successful fences: {message}"
                    ))
                }
            }
        }

        /// Request/DMA and locally returned device resources survive unwinding
        /// until both streams fence. Backend-internal temporaries still rely on
        /// its allocator event-retirement contract, as in parallel_cuda.
        pub(super) struct Resources<T> {
            ops: Rc<CudaOperators>,
            value: Option<T>,
        }
        impl<T> Resources<T> {
            pub(super) fn new(ops: &Rc<CudaOperators>, value: T) -> Self {
                Self {
                    ops: Rc::clone(ops),
                    value: Some(value),
                }
            }
            pub(super) fn get(&self) -> &T {
                self.value.as_ref().unwrap()
            }
        }
        impl<T> Drop for Resources<T> {
            fn drop(&mut self) {
                if sync_owner(&self.ops).is_err() {
                    if let Some(value) = self.value.take() {
                        std::mem::forget(value);
                    }
                    std::mem::forget(Rc::clone(&self.ops));
                    // Do not report an ordinary completion if the final guard
                    // fence fails. Unwinding here already carries on_owner's
                    // typed Unknown; a second panic would abort the process.
                    if !thread::panicking() {
                        std::panic::panic_any(PanicQuiescence::Unknown);
                    }
                }
            }
        }
        pub(super) struct Resident {
            pub(super) ops: Rc<CudaOperators>,
            pub(super) shard: CudaSwiGluShard,
        }
        impl Resident {
            pub(super) fn new(
                ordinal: usize,
                weights: TensorParallelSwiGluWeights,
            ) -> Checked<Self> {
                weights
                    .plan()
                    .validate_cuda(weights.rank(), weights.gate().dtype())
                    .map_err(text)?;
                let ops = Rc::new(CudaOperators::new_on_device(ordinal).map_err(text)?);
                let shard = on_owner(&ops, || {
                    require(ops.device_ordinal() == ordinal, "factory ordinal mismatch")?;
                    CudaSwiGluShard::from_local(Rc::clone(&ops), weights).map_err(shard_error)
                })?;
                let config = shard.transfer_config();
                require(
                    config.chunk_elements * size_of::<f32>() == 64 * 1024
                        && config.tx_slots == 1
                        && config.rx_slots == 1,
                    "default shard pinned budget changed",
                )?;
                Ok(Self { ops, shard })
            }

            pub(super) fn execute(&self, values: &[f32], rows: usize) -> Checked<Vec<f32>> {
                let before = self.shard.transfer_stats();
                let result = catch_unwind(AssertUnwindSafe(|| self.shard.execute(values, rows)));
                // Cover the transport control stream BEFORE on_owner is allowed
                // to turn an ordinary panic into an ordinary failure.
                self.shard.quiesce().map_err(shard_error)?;
                let output = match result {
                    Ok(result) => result.map_err(shard_error)?,
                    Err(payload) => std::panic::resume_unwind(payload),
                };
                let after = self.shard.transfer_stats();
                let chunk = self.shard.transfer_config().chunk_elements;
                let count = values.len().div_ceil(chunk) + output.len().div_ceil(chunk);
                require(
                    before.pinned_allocations == 2
                        && before.pinned_bytes == 128 * 1024
                        && after.pinned_allocations == before.pinned_allocations
                        && after.pinned_bytes == before.pinned_bytes
                        && after.submissions - before.submissions == count as u64
                        && after.completions - before.completions == count as u64
                        && after.tx_in_use == 0
                        && after.rx_in_use == 0
                        && after.device_holds_in_use == 0
                        && after.tx_high_water == 1
                        && after.rx_high_water == 1
                        && !after.quarantined
                        && self.shard.is_quiescent()
                        && !self.shard.needs_quarantine(),
                    format!(
                        "SwiGLU pinned transport did not retire: before={before:?} after={after:?}"
                    ),
                )?;
                Ok(output)
            }
        }

        pub(super) fn apply(ops: &Rc<CudaOperators>, values: &[f32]) -> Checked<Vec<f32>> {
            let buffer = Resources::new(ops, ops.upload_f32_buffer(values).map_err(text)?);
            // Apply is not a host-only acknowledgement: H2D -> fence -> D2H ->
            // both-stream fence completes before the production worker returns.
            on_owner(ops, || Ok(()))?;
            on_owner(ops, || ops.download_f32_buffer(buffer.get()).map_err(text))
        }

        pub(super) fn full_oracle(catalog: Catalog, values: Vec<f32>) -> Checked<Vec<f32>> {
            // Separate unsplit owner, completed/joined before starting TP owners.
            // No full weight is ever captured by a rank-local worker factory.
            thread::spawn(move || {
                let plan = TensorParallelSwiGluPlan::new(HIDDEN, INTERMEDIATE, 1).map_err(text)?;
                let resident = Resident::new(0, catalog.load(&plan, rank(0))?)?;
                let ops = Rc::clone(&resident.ops);
                let resident = Resources::new(&ops, resident);
                let values = Resources::new(&ops, values);
                on_owner(&ops, || resident.get().execute(values.get(), ROWS))
            })
            .join()
            .map_err(|_| "unsplit CUDA oracle owner panicked (may be quarantined)".to_string())?
        }

        fn run(degree: usize) {
            let visible = CudaContext::device_count().expect("enumerate CUDA devices");
            assert!(
                visible >= DEVICES,
                "requires eight visible CUDA GPUs, found {visible}"
            );
            run_case(degree, Backend::Cuda);
        }
        #[test]
        #[ignore = "requires eight visible CUDA GPUs; external process timeout and --test-threads=1"]
        fn cuda_checkpoint_swiglu_executor_tp2() {
            run(2);
        }
        #[test]
        #[ignore = "requires eight visible CUDA GPUs; external process timeout and --test-threads=1"]
        fn cuda_checkpoint_swiglu_executor_tp4() {
            run(4);
        }
        #[test]
        #[ignore = "requires eight visible CUDA GPUs; external process timeout and --test-threads=1"]
        fn cuda_checkpoint_swiglu_executor_tp8() {
            run(8);
        }
    }
}
