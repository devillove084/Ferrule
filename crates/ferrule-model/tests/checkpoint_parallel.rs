use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ferrule_common::ParallelRankId;
use ferrule_model::ModelFamily;
use ferrule_model::checkpoint::{
    CheckpointDType, CheckpointSourceFileIdentity, CheckpointTensorReader, CheckpointTensorSlice,
    HfSafetensorsIndex, HfSafetensorsInventory,
};
use ferrule_model::transformer::parallel::{
    CpuLinearShard, CpuSwiGluShard, TensorParallelLinearPartition as Partition,
    TensorParallelLinearPlan, TensorParallelSwiGluPlan, TensorParallelWeightShard,
};

const NAMES: [&str; 3] = [
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
];

fn rank(index: u32) -> ParallelRankId {
    ParallelRankId::new(index)
}

fn encode(values: &[f32], dtype: &CheckpointDType) -> Vec<u8> {
    match dtype {
        CheckpointDType::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        CheckpointDType::Bf16 => values
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect(),
        _ => panic!("fixture only supports dense weights"),
    }
}

struct Fixture {
    directory: PathBuf,
    tensors: [CheckpointTensorSlice; 3],
    values: [Vec<f32>; 3],
}
impl Fixture {
    fn new(dtype: CheckpointDType) -> Self {
        Self::with_shape(dtype, 3, 5)
    }
    fn with_shape(dtype: CheckpointDType, hidden: usize, intermediate: usize) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "ferrule-tp-checkpoint-{}-{nonce}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let values = [
            (0..hidden * intermediate)
                .map(|i| ((i % 15) as f32 - 7.0) / 16.0)
                .collect::<Vec<_>>(),
            (0..hidden * intermediate)
                .map(|i| (9.0 - (i % 15) as f32) / 8.0)
                .collect::<Vec<_>>(),
            (0..hidden * intermediate)
                .map(|i| ((i % 15) as f32 - 5.0) / 32.0)
                .collect::<Vec<_>>(),
        ];
        let first = "model-00001-of-00002.safetensors";
        let second = "model-00002-of-00002.safetensors";
        // Gate/down share a real shard, but down begins after another payload.
        write_shard(
            &directory.join(first),
            &[
                (NAMES[0], [intermediate, hidden], &values[0]),
                (NAMES[2], [hidden, intermediate], &values[2]),
            ],
            &dtype,
        );
        write_shard(
            &directory.join(second),
            &[(NAMES[1], [intermediate, hidden], &values[1])],
            &dtype,
        );
        let weight_map = BTreeMap::from([(NAMES[0], first), (NAMES[1], second), (NAMES[2], first)]);
        let index_path = directory.join("model.safetensors.index.json");
        std::fs::write(
            &index_path,
            serde_json::to_vec(&serde_json::json!({
                "metadata": {"total_size": 3 * hidden * intermediate * dtype.element_size_bytes().unwrap()},
                "weight_map": weight_map,
            }))
            .unwrap(),
        )
        .unwrap();
        let index = HfSafetensorsIndex::open(index_path).unwrap();
        let inventory =
            HfSafetensorsInventory::from_index(&directory, ModelFamily::Qwen3, &index).unwrap();
        assert_eq!(inventory.shard_count, 2);
        assert_eq!(inventory.tensor_count, 3);
        let tensors = NAMES.map(|name| {
            CheckpointTensorSlice::from_hf_inventory(
                &directory,
                inventory.tensors.iter().find(|t| t.name == name).unwrap(),
            )
        });
        Self {
            directory,
            tensors,
            values,
        }
    }
    fn tensors(&self) -> [&CheckpointTensorSlice; 3] {
        self.tensors.each_ref()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

fn write_shard(path: &Path, tensors: &[(&str, [usize; 2], &[f32])], dtype: &CheckpointDType) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let start = data.len();
        data.extend(encode(values, dtype));
        header.insert(
            (*name).into(),
            serde_json::json!({
                "dtype": dtype.as_str(), "shape": shape, "data_offsets": [start, data.len()],
            }),
        );
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&data).unwrap();
}

#[test]
fn real_shards_preserve_bf16_and_f32_bytes_for_both_partitions() {
    for dtype in [CheckpointDType::Bf16, CheckpointDType::F32] {
        let fixture = Fixture::new(dtype.clone());
        for partition in [Partition::Column, Partition::Row] {
            let tensor = &fixture.tensors[2];
            assert!(tensor.offset > fixture.tensors[0].offset);
            let plan = TensorParallelLinearPlan::new(3, 5, 2, partition).unwrap();
            for r in 0..2 {
                let (expected, range) = plan.shard_weight(rank(r), &fixture.values[2]).unwrap();
                let expected = encode(&expected, &dtype);
                assert!(expected.len() < tensor.bytes as usize);
                // The entire tensor exceeds this budget; only local extents fit.
                let reader = CheckpointTensorReader::new(expected.len() as u64);
                let local = plan.read_weight_shard(&reader, tensor, rank(r)).unwrap();
                assert_eq!(local.bytes(), expected);
                assert_eq!(local.dtype(), &dtype);
                assert_eq!(local.full_shape(), [3, 5]);
                let (out, width) = plan.local_shape(rank(r)).unwrap();
                assert_eq!(local.local_shape(), [out, width]);
                let provenance = local.provenance().unwrap();
                assert_eq!(provenance.tensor(), tensor);
                assert_eq!(
                    provenance.read_plan().storage_bytes(),
                    expected.len() as u64
                );
                assert_eq!(provenance.read_plan().source_files().len(), 1);
                if partition == Partition::Column {
                    assert_eq!(provenance.rows(), range);
                    assert_eq!(provenance.columns(), 0..5);
                    assert_eq!(provenance.read_plan().extents().len(), 1);
                } else {
                    assert_eq!(provenance.rows(), 0..3);
                    assert_eq!(provenance.columns(), range.clone());
                    let element = dtype.element_size_bytes().unwrap();
                    for (row, extent) in provenance.read_plan().extents().iter().enumerate() {
                        assert_eq!(
                            extent.offset(),
                            tensor.offset + ((row * 5 + range.start) * element) as u64
                        );
                        assert_eq!(extent.bytes(), (range.len() * element) as u64);
                    }
                }
                assert!(
                    plan.read_weight_shard(
                        &CheckpointTensorReader::new(expected.len() as u64 - 1),
                        tensor,
                        rank(r)
                    )
                    .is_err()
                );
            }
        }
    }
}

#[test]
fn rectangle_reads_keep_noncontiguous_provenance_and_validate_legacy_rows() {
    let fixture = Fixture::new(CheckpointDType::Bf16);
    let tensor = &fixture.tensors[2];
    let source = CheckpointSourceFileIdentity::capture(&tensor.path).unwrap();
    let reader = CheckpointTensorReader::new(12);
    let read = reader.plan_2d_range(tensor, 1..3, 1..4, &source).unwrap();
    let payload = reader.read_matrix(&read).unwrap();
    assert_eq!(payload.provenance().local_shape(), [2, 3]);
    assert_eq!(
        payload.bytes(),
        encode(
            &[
                fixture.values[2][6],
                fixture.values[2][7],
                fixture.values[2][8],
                fixture.values[2][11],
                fixture.values[2][12],
                fixture.values[2][13],
            ],
            &CheckpointDType::Bf16
        )
    );
    let columns = reader.read_2d_columns(tensor, 3, 2).unwrap();
    assert_eq!(columns.provenance().local_shape(), [3, 2]);
    assert_eq!(columns.provenance().read_plan().extents().len(), 3);
    let rows = CheckpointTensorReader::new(20)
        .read_2d_rows(tensor, 1, 2)
        .unwrap();
    assert_eq!(rows.slice.offset, tensor.offset + 10);
    assert_eq!(rows.slice.shape, [2, 5]);
    assert_eq!(
        rows.bytes,
        encode(&fixture.values[2][5..], &CheckpointDType::Bf16)
    );
    assert!(
        reader
            .read_matrix(
                &CheckpointTensorReader::new(30)
                    .plan_2d_range(tensor, 0..3, 0..5, &source)
                    .unwrap()
            )
            .is_err()
    );
}

#[test]
fn invalid_ranges_shapes_dtypes_and_file_extents_fail_closed() {
    let fixture = Fixture::new(CheckpointDType::Bf16);
    let tensor = &fixture.tensors[2];
    let source = CheckpointSourceFileIdentity::capture(&tensor.path).unwrap();
    let reader = CheckpointTensorReader::new(u64::MAX);
    for (rows, cols) in [
        (0..0, 0..1),
        (0..1, 2..2),
        (0..4, 0..1),
        (0..1, 0..6),
        (usize::MAX..usize::MAX, 0..1),
    ] {
        assert!(reader.plan_2d_range(tensor, rows, cols, &source).is_err());
    }
    assert!(reader.read_2d_rows(tensor, usize::MAX, 2).is_err());
    assert!(reader.read_2d_columns(tensor, usize::MAX, 2).is_err());
    for shape in [
        vec![15],
        vec![3, 5, 1],
        vec![0, 5],
        vec![4, 5],
        vec![usize::MAX, 5],
    ] {
        let mut bad = tensor.clone();
        bad.shape = shape;
        assert!(reader.plan_2d_range(&bad, 0..1, 0..1, &source).is_err());
    }
    for dtype in [
        CheckpointDType::F8E4M3,
        CheckpointDType::F8E8M0,
        CheckpointDType::I8,
        CheckpointDType::I32,
        CheckpointDType::Unknown("packed".into()),
    ] {
        let mut bad = tensor.clone();
        bad.dtype = dtype;
        let error = reader.plan_2d_range(&bad, 0..1, 0..1, &source).unwrap_err();
        assert!(error.to_string().contains("only dense BF16/F32"));
    }
    let mut bad = tensor.clone();
    bad.bytes -= 1;
    assert!(reader.plan_2d_range(&bad, 0..1, 0..1, &source).is_err());
    bad = tensor.clone();
    bad.offset = u64::MAX;
    assert!(reader.plan_2d_range(&bad, 0..1, 0..1, &source).is_err());
    // Even when this local piece fits, the original full tensor must fit too.
    bad = tensor.clone();
    bad.offset = source.length() - 4;
    assert!(reader.plan_2d_range(&bad, 0..1, 0..1, &source).is_err());
    let other = CheckpointSourceFileIdentity::capture(&fixture.tensors[1].path).unwrap();
    assert!(reader.plan_2d_range(tensor, 0..1, 0..1, &other).is_err());
    let plan = TensorParallelLinearPlan::new(3, 5, 2, Partition::Row).unwrap();
    assert!(
        plan.read_weight_shard(&reader, &fixture.tensors[0], rank(0))
            .is_err()
    );
    assert!(plan.read_weight_shard(&reader, tensor, rank(2)).is_err());
}

#[test]
fn catalog_snapshot_is_checked_before_read_and_before_payload_admission() {
    let fixture = Fixture::new(CheckpointDType::F32);
    let tensor = &fixture.tensors[2];
    let source = CheckpointSourceFileIdentity::capture(&tensor.path).unwrap();
    let reader = CheckpointTensorReader::new(24);
    let plan = TensorParallelLinearPlan::new(3, 5, 2, Partition::Row).unwrap();
    let read = reader.plan_2d_range(tensor, 0..3, 3..5, &source).unwrap();
    let payload = reader.read_matrix(&read).unwrap();
    std::fs::OpenOptions::new()
        .append(true)
        .open(&tensor.path)
        .unwrap()
        .write_all(&[0])
        .unwrap();
    assert!(!source.is_current());
    assert!(reader.read_matrix(&read).is_err());
    assert!(
        plan.read_weight_shard_with_source(&reader, tensor, rank(1), &source)
            .is_err()
    );
    assert!(TensorParallelWeightShard::from_checkpoint_payload(plan, rank(1), payload).is_err());
}

#[test]
fn source_replacement_and_wrong_rank_payload_are_rejected() {
    let fixture = Fixture::new(CheckpointDType::Bf16);
    let tensor = &fixture.tensors[0];
    let source = CheckpointSourceFileIdentity::capture(&tensor.path).unwrap();
    let reader = CheckpointTensorReader::new(30);
    let plan = TensorParallelLinearPlan::new(5, 3, 2, Partition::Column).unwrap();
    let read = reader.plan_2d_range(tensor, 0..3, 0..3, &source).unwrap();
    assert!(
        TensorParallelWeightShard::from_checkpoint_payload(
            plan.clone(),
            rank(1),
            reader.read_matrix(&read).unwrap()
        )
        .is_err()
    );
    let old_path = fixture.directory.join("old.safetensors");
    std::fs::rename(&tensor.path, &old_path).unwrap();
    std::fs::copy(&old_path, &tensor.path).unwrap();
    assert_eq!(
        std::fs::metadata(&tensor.path).unwrap().len(),
        source.length()
    );
    assert!(
        plan.read_weight_shard_with_source(&reader, tensor, rank(0), &source)
            .is_err()
    );
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() <= 3e-5 * (1.0 + b.abs()), "{a} != {b}");
    }
}

#[test]
fn two_rank_checkpoint_swiglu_matches_memory_path_and_unsplit_provider() {
    for dtype in [CheckpointDType::F32, CheckpointDType::Bf16] {
        let fixture = Fixture::new(dtype.clone());
        let plan = TensorParallelSwiGluPlan::new(3, 5, 2).unwrap();
        let shards: Vec<_> = (0..2)
            .map(|r| {
                let limit = plan.rank_range(rank(r)).unwrap().len()
                    * 3
                    * dtype.element_size_bytes().unwrap();
                let weights = plan
                    .read_weight_shards(
                        &CheckpointTensorReader::new(limit as u64),
                        fixture.tensors(),
                        rank(r),
                    )
                    .unwrap();
                assert_eq!(weights.gate().dtype(), &dtype);
                assert!(weights.down().provenance().is_some());
                CpuSwiGluShard::from_local(weights).unwrap()
            })
            .collect();
        let full = TensorParallelSwiGluPlan::new(3, 5, 1).unwrap();
        let full = CpuSwiGluShard::from_local(
            full.read_weight_shards(&CheckpointTensorReader::new(60), fixture.tensors(), rank(0))
                .unwrap(),
        )
        .unwrap();
        for rows in [1, 3, 2] {
            let input: Vec<_> = (0..rows * 3).map(|i| (i as f32 - 3.0) / 8.0).collect();
            let expected = full.execute(&input, rows).unwrap();
            let mut reduced = vec![0.0; rows * 3];
            for (r, shard) in shards.iter().enumerate() {
                let partial = shard.execute(&input, rows).unwrap();
                let memory = CpuSwiGluShard::from_local(
                    plan.shard_weights_f32(
                        rank(r as u32),
                        &fixture.values[0],
                        &fixture.values[1],
                        &fixture.values[2],
                    )
                    .unwrap(),
                )
                .unwrap();
                assert_close(&partial, &memory.execute(&input, rows).unwrap());
                // Test oracle only; production sum is exclusively runtime-owned.
                for (sum, value) in reduced.iter_mut().zip(partial) {
                    *sum += value;
                }
            }
            assert_close(&reduced, &expected);
        }
    }
}

#[test]
fn checkpoint_cpu_linear_uses_native_local_payload() {
    for dtype in [CheckpointDType::Bf16, CheckpointDType::F32] {
        let fixture = Fixture::new(dtype);
        let reader = CheckpointTensorReader::new(60);
        for partition in [Partition::Column, Partition::Row] {
            let plan = TensorParallelLinearPlan::new(3, 5, 2, partition).unwrap();
            for r in 0..2 {
                let shard = CpuLinearShard::from_local(
                    plan.read_weight_shard(&reader, &fixture.tensors[2], rank(r))
                        .unwrap(),
                )
                .unwrap();
                let memory = CpuLinearShard::from_local(
                    plan.shard_weight_f32(rank(r), &fixture.values[2]).unwrap(),
                )
                .unwrap();
                let input = [0.5; 10];
                assert_close(
                    &shard.execute(&input, 2).unwrap(),
                    &memory.execute(&input, 2).unwrap(),
                );
                assert!(shard.execute_local(&[], 1).is_err());
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA and explicit FERRULE_TP_TEST_DEVICE; local compute only, no runtime collective"]
fn cuda_checkpoint_local_linear_and_swiglu_rank_matrix() {
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use ferrule_model::transformer::parallel::{CudaLinearShard, CudaSwiGluShard};
    use std::rc::Rc;
    let ordinal: usize = std::env::var("FERRULE_TP_TEST_DEVICE")
        .expect("explicit device required")
        .parse()
        .unwrap();
    let ops = Rc::new(CudaOperators::new_on_device(ordinal).unwrap());
    let reader = CheckpointTensorReader::new(60);
    for dtype in [CheckpointDType::F32, CheckpointDType::Bf16] {
        let fixture = Fixture::new(dtype);
        for degree in [1, 2, 3] {
            let mlp = TensorParallelSwiGluPlan::new(3, 5, degree).unwrap();
            for r in 0..degree {
                let r = rank(r as u32);
                if fixture.tensors[0].dtype == CheckpointDType::Bf16 {
                    let error = CudaSwiGluShard::from_local(
                        Rc::clone(&ops),
                        mlp.read_weight_shards(&reader, fixture.tensors(), r)
                            .unwrap(),
                    )
                    .err()
                    .expect("tiny BF16 must fail before upload");
                    assert!(error.to_string().contains("K16"));
                    for partition in [Partition::Column, Partition::Row] {
                        let linear =
                            TensorParallelLinearPlan::new(3, 5, degree, partition).unwrap();
                        let error = CudaLinearShard::from_local(
                            Rc::clone(&ops),
                            linear
                                .read_weight_shard(&reader, &fixture.tensors[2], r)
                                .unwrap(),
                        )
                        .err()
                        .expect("tiny BF16 linear must fail before upload");
                        assert!(error.to_string().contains("K16"));
                    }
                    continue;
                }
                let gpu = CudaSwiGluShard::from_local(
                    Rc::clone(&ops),
                    mlp.read_weight_shards(&reader, fixture.tensors(), r)
                        .unwrap(),
                )
                .unwrap();
                let cpu = CpuSwiGluShard::from_local(
                    mlp.read_weight_shards(&reader, fixture.tensors(), r)
                        .unwrap(),
                )
                .unwrap();
                for rows in [1, 3] {
                    let input = vec![0.25; rows * 3];
                    assert_close(
                        &gpu.execute(&input, rows).unwrap(),
                        &cpu.execute(&input, rows).unwrap(),
                    );
                }
                for partition in [Partition::Column, Partition::Row] {
                    let linear = TensorParallelLinearPlan::new(3, 5, degree, partition).unwrap();
                    let gpu = CudaLinearShard::from_local(
                        Rc::clone(&ops),
                        linear
                            .read_weight_shard(&reader, &fixture.tensors[2], r)
                            .unwrap(),
                    )
                    .unwrap();
                    let cpu = CpuLinearShard::from_local(
                        linear
                            .read_weight_shard(&reader, &fixture.tensors[2], r)
                            .unwrap(),
                    )
                    .unwrap();
                    assert_close(
                        &gpu.execute(&[0.25; 10], 2).unwrap(),
                        &cpu.execute(&[0.25; 10], 2).unwrap(),
                    );
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA and FERRULE_TP_TEST_DEVICE; external timeout and --test-threads=1"]
fn async_shard_transfer_chunking() {
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use ferrule_backend::cuda::{CudaAsyncTransportStats, CudaTransferConfig};
    use ferrule_model::transformer::parallel::{CudaLinearShard, CudaSwiGluShard};
    use std::rc::Rc;

    const CONFIG: CudaTransferConfig = CudaTransferConfig {
        chunk_elements: 3,
        tx_slots: 1,
        rx_slots: 1,
    };
    fn retired(before: CudaAsyncTransportStats, after: CudaAsyncTransportStats, count: usize) {
        assert_eq!(before.pinned_allocations, 2);
        assert_eq!(before.pinned_bytes, 2 * 3 * size_of::<f32>());
        assert_eq!(after.pinned_allocations, before.pinned_allocations);
        assert_eq!(after.pinned_bytes, before.pinned_bytes);
        assert_eq!((after.tx_slots, after.rx_slots), (1, 1));
        assert_eq!(
            (after.tx_in_use, after.rx_in_use, after.device_holds_in_use),
            (0, 0, 0)
        );
        assert_eq!((after.tx_high_water, after.rx_high_water), (1, 1));
        assert_eq!(after.device_holds_high_water, 1);
        assert_eq!(after.submissions - before.submissions, count as u64);
        assert_eq!(after.completions - before.completions, count as u64);
        assert_eq!(after.failed_submissions, 0);
        assert_eq!(after.cancel_intents, 0);
        assert!(!after.quarantined);
    }
    fn activation(values: &[f32], dtype: &CheckpointDType) -> Vec<f32> {
        values
            .iter()
            .map(|&value| {
                if *dtype == CheckpointDType::Bf16 {
                    half::bf16::from_f32(value).to_f32()
                } else {
                    value
                }
            })
            .collect()
    }
    fn default_budget(config: CudaTransferConfig) {
        assert_eq!(config.chunk_elements * size_of::<f32>(), 64 * 1024);
        assert_eq!((config.tx_slots, config.rx_slots), (1, 1));
    }

    let ordinal: usize = std::env::var("FERRULE_TP_TEST_DEVICE")
        .expect("explicit device required")
        .parse()
        .unwrap();
    // All operators, device allocations, slots and tickets are constructed and
    // destroyed on this owner. Only the ordinal crosses the thread boundary.
    std::thread::spawn(move || {
        let ops = Rc::new(CudaOperators::new_on_device(ordinal).unwrap());
        let reader = CheckpointTensorReader::new(u64::MAX);
        let mut h2d_tail = false;
        let mut d2h_tail = false;
        for (dtype, hidden, intermediate, degrees) in [
            (CheckpointDType::F32, 5, 7, vec![1, 2, 3]),
            (CheckpointDType::Bf16, 16, 32, vec![1, 2]),
        ] {
            let fixture = Fixture::with_shape(dtype.clone(), hidden, intermediate);
            for degree in degrees {
                for r in 0..degree {
                    let r = rank(r as u32);
                    for partition in [Partition::Column, Partition::Row] {
                        let plan =
                            TensorParallelLinearPlan::new(hidden, intermediate, degree, partition)
                                .unwrap();
                        let load = || {
                            plan.read_weight_shard(&reader, &fixture.tensors[2], r)
                                .unwrap()
                        };
                        let gpu = CudaLinearShard::from_local_with_transfer_config(
                            Rc::clone(&ops),
                            load(),
                            CONFIG,
                        )
                        .unwrap();
                        let default = CudaLinearShard::from_local(Rc::clone(&ops), load()).unwrap();
                        let cpu = CpuLinearShard::from_local(load()).unwrap();
                        default_budget(default.transfer_config());
                        assert_eq!(gpu.transfer_config(), CONFIG);
                        let (out, width) = plan.local_shape(r).unwrap();
                        for rows in [1, 2, 4] {
                            let input: Vec<_> = (0..rows * intermediate)
                                .map(|i| ((i * 7 % 23) as f32 - 8.0) / 19.0)
                                .collect();
                            let local = plan.shard_input(r, &input, rows).unwrap();
                            let expected = cpu
                                .execute_local(&activation(&local, &dtype), rows)
                                .unwrap();
                            let before = gpu.transfer_stats();
                            assert_close(&gpu.execute(&input, rows).unwrap(), &expected);
                            assert_close(&default.execute(&input, rows).unwrap(), &expected);
                            let count = (rows * width).div_ceil(3) + (rows * out).div_ceil(3);
                            retired(before, gpu.transfer_stats(), count);
                            let before = gpu.transfer_stats();
                            assert_close(&gpu.execute_local(&local, rows).unwrap(), &expected);
                            retired(before, gpu.transfer_stats(), count);
                            h2d_tail |= !(rows * width).is_multiple_of(3);
                            d2h_tail |= !(rows * out).is_multiple_of(3);
                            assert!(gpu.is_quiescent() && !gpu.needs_quarantine());
                            let before = gpu.transfer_stats();
                            assert!(gpu.execute_local(&[], rows).is_err());
                            assert!(gpu.execute(&input, usize::MAX).is_err());
                            assert_eq!(gpu.transfer_stats(), before);
                        }
                        gpu.quiesce().unwrap();
                        default.quiesce().unwrap();
                    }
                    let plan = TensorParallelSwiGluPlan::new(hidden, intermediate, degree).unwrap();
                    let load = || {
                        plan.read_weight_shards(&reader, fixture.tensors(), r)
                            .unwrap()
                    };
                    let gpu = CudaSwiGluShard::from_local_with_transfer_config(
                        Rc::clone(&ops),
                        load(),
                        CONFIG,
                    )
                    .unwrap();
                    let default = CudaSwiGluShard::from_local(Rc::clone(&ops), load()).unwrap();
                    default_budget(default.transfer_config());
                    assert_eq!(gpu.transfer_config(), CONFIG);
                    // Independent CPU stages explicitly match the artifact BF16
                    // activation boundaries. No BF16 padding/fallback is added.
                    let gate = CpuLinearShard::from_local(
                        plan.gate_up_plan()
                            .read_weight_shard(&reader, &fixture.tensors[0], r)
                            .unwrap(),
                    )
                    .unwrap();
                    let up = CpuLinearShard::from_local(
                        plan.gate_up_plan()
                            .read_weight_shard(&reader, &fixture.tensors[1], r)
                            .unwrap(),
                    )
                    .unwrap();
                    let down = CpuLinearShard::from_local(
                        plan.down_plan()
                            .read_weight_shard(&reader, &fixture.tensors[2], r)
                            .unwrap(),
                    )
                    .unwrap();
                    for rows in [1, 2, 4] {
                        let input: Vec<_> = (0..rows * hidden)
                            .map(|i| ((i * 11 % 29) as f32 - 9.0) / 17.0)
                            .collect();
                        let rounded = activation(&input, &dtype);
                        let gated = gate.execute_local(&rounded, rows).unwrap();
                        let upd = up.execute_local(&rounded, rows).unwrap();
                        let intermediate: Vec<_> = gated
                            .iter()
                            .zip(upd)
                            .map(|(&g, u)| g / (1.0 + (-g).exp()) * u)
                            .collect();
                        let expected = down
                            .execute_local(&activation(&intermediate, &dtype), rows)
                            .unwrap();
                        let before = gpu.transfer_stats();
                        assert_close(&gpu.execute(&input, rows).unwrap(), &expected);
                        assert_close(&default.execute(&input, rows).unwrap(), &expected);
                        // Only input and down partial cross the host boundary.
                        retired(before, gpu.transfer_stats(), 2 * input.len().div_ceil(3));
                        assert!(gpu.is_quiescent() && !gpu.needs_quarantine());
                        let before = gpu.transfer_stats();
                        assert!(gpu.execute(&[], rows).is_err());
                        assert_eq!(gpu.transfer_stats(), before);
                    }
                    gpu.quiesce().unwrap();
                    default.quiesce().unwrap();
                }
            }
        }
        assert!(h2d_tail && d2h_tail);
    })
    .join()
    .expect("CUDA shard owner");
}

#[test]
fn aligned_checkpoint_swiglu_rank_matrix_preserves_native_local_payloads() {
    const HIDDEN: usize = 32;
    const INTERMEDIATE: usize = 256;
    for dtype in [CheckpointDType::Bf16, CheckpointDType::F32] {
        let fixture = Fixture::with_shape(dtype.clone(), HIDDEN, INTERMEDIATE);
        let input: Vec<_> = (0..3 * HIDDEN).map(|i| (i % 13) as f32 / 16.0).collect();
        let full_plan = TensorParallelSwiGluPlan::new(HIDDEN, INTERMEDIATE, 1).unwrap();
        let full = CpuSwiGluShard::from_local(
            full_plan
                .read_weight_shards(
                    &CheckpointTensorReader::new(fixture.tensors[0].bytes),
                    fixture.tensors(),
                    rank(0),
                )
                .unwrap(),
        )
        .unwrap();
        let expected = full.execute(&input, 3).unwrap();
        for degree in [2, 4, 8] {
            let plan = TensorParallelSwiGluPlan::new(HIDDEN, INTERMEDIATE, degree).unwrap();
            let mut sum = vec![0.0; 3 * HIDDEN];
            for r in 0..degree {
                let r = rank(r as u32);
                plan.validate_cuda(r, &dtype).unwrap();
                let range = plan.rank_range(r).unwrap();
                let bytes = HIDDEN * range.len() * dtype.element_size_bytes().unwrap();
                assert!((bytes as u64) < fixture.tensors[0].bytes);
                let weights = plan
                    .read_weight_shards(
                        &CheckpointTensorReader::new(bytes as u64),
                        fixture.tensors(),
                        r,
                    )
                    .unwrap();
                for (i, shard) in [weights.gate(), weights.up(), weights.down()]
                    .into_iter()
                    .enumerate()
                {
                    assert_eq!(shard.dtype(), &dtype);
                    assert_eq!(shard.bytes().len(), bytes);
                    let (memory, _) = shard.plan().shard_weight(r, &fixture.values[i]).unwrap();
                    assert_eq!(shard.bytes(), encode(&memory, &dtype));
                    let read = shard.provenance().unwrap();
                    assert_eq!(read.read_plan().storage_bytes(), bytes as u64);
                    assert_eq!(read.rows(), if i == 2 { 0..HIDDEN } else { range.clone() });
                    assert_eq!(
                        read.columns(),
                        if i == 2 { range.clone() } else { 0..HIDDEN }
                    );
                    assert_eq!(
                        read.read_plan().extents().len(),
                        if i == 2 { HIDDEN } else { 1 }
                    );
                }
                let partial = CpuSwiGluShard::from_local(weights)
                    .unwrap()
                    .execute(&input, 3)
                    .unwrap();
                // Oracle only: the production collective is tested in runtime.
                for (sum, value) in sum.iter_mut().zip(partial) {
                    *sum += value;
                }
            }
            assert_close(&sum, &expected);
        }
    }
}
