//! Numeric FP8 storage stays separate from the native E8M0 linear/MMA contract.
use std::ops::Range;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use ferrule_common::ParallelRankId;
use ferrule_model::TensorRole;
use ferrule_model::checkpoint::{
    CheckpointDType as DType, CheckpointPositionedReader, CheckpointSourceFileIdentity,
    CheckpointTensorPayload, CheckpointTensorReader, CheckpointTensorSlice, LinearWeight,
    NumericFp8Artifact, NumericFp8Encoding as Encoding, NumericFp8Source, decode_fp8_e4m3fn_byte,
};
use ferrule_model::nn::{
    ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec, StorageEncoding,
};
use ferrule_model::transformer::parallel::{
    TensorParallelLinearPartition as Partition, TensorParallelLinearPlan,
};
use ferrule_model::transformer::{
    BoundParameter, ExactNameMapper, NameMapping, StateDictBinder, StateDictMaterializer,
    StateDictSchema, TensorTransform,
};

const ENCODINGS: [Encoding; 2] = [Encoding::E4M3FnBlock128Bf16, Encoding::E4M3FnBlock128F32];

fn scales_bytes(encoding: Encoding, scales: &[f32]) -> Vec<u8> {
    scales
        .iter()
        .flat_map(|value| match encoding {
            Encoding::E4M3FnBlock128Bf16 => half::bf16::from_f32(*value)
                .to_bits()
                .to_le_bytes()
                .to_vec(),
            Encoding::E4M3FnBlock128F32 => value.to_le_bytes().to_vec(),
        })
        .collect()
}

struct Fixture {
    directory: PathBuf,
    weight: CheckpointTensorSlice,
    scale: CheckpointTensorSlice,
    encoding: Encoding,
    raw: Vec<u8>,
    scales: Vec<f32>,
}

impl Fixture {
    fn new(shape: &[usize], encoding: Encoding) -> Self {
        let elements = shape.iter().product();
        let raw = (0..elements)
            .map(|i| [0x38, 0xb8, 0x7e, 0xfe, 1, 0x81, 0, 0x80][i % 8])
            .collect();
        let rank = shape.len();
        let count = shape[..rank - 2].iter().product::<usize>()
            * shape[rank - 2].div_ceil(128)
            * shape[rank - 1].div_ceil(128);
        let scales = (0..count).map(|i| (i + 1) as f32 * 0.25).collect();
        Self::with_data(shape, encoding, raw, scales)
    }

    fn with_data(shape: &[usize], encoding: Encoding, raw: Vec<u8>, scales: Vec<f32>) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "ferrule-numeric-fp8-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let weight = CheckpointTensorSlice {
            name: "expert.weight".into(),
            role: TensorRole::RoutedExpertDown,
            path: directory.join("weight.bin"),
            offset: 19,
            bytes: raw.len() as u64,
            dtype: DType::F8E4M3,
            shape: shape.to_vec(),
        };
        let mut scale_shape = shape.to_vec();
        let n = shape.len();
        scale_shape[n - 2] = scale_shape[n - 2].div_ceil(128);
        scale_shape[n - 1] = scale_shape[n - 1].div_ceil(128);
        let scale_data = scales_bytes(encoding, &scales);
        let scale = CheckpointTensorSlice {
            name: "expert.weight_scale_inv".into(),
            role: TensorRole::RoutedExpertDown,
            path: directory.join("scale.bin"),
            offset: 7,
            bytes: scale_data.len() as u64,
            dtype: encoding.scale_dtype(),
            shape: scale_shape,
        };
        let mut weight_file = vec![0xaa; weight.offset as usize];
        weight_file.extend(&raw);
        std::fs::write(&weight.path, weight_file).unwrap();
        let mut scale_file = vec![0xbb; scale.offset as usize];
        scale_file.extend(scale_data);
        std::fs::write(&scale.path, scale_file).unwrap();
        Self {
            directory,
            weight,
            scale,
            encoding,
            raw,
            scales,
        }
    }

    fn source(&self) -> NumericFp8Source {
        self.source_with(self.weight.clone(), self.scale.clone())
            .unwrap()
    }

    fn source_with(
        &self,
        weight: CheckpointTensorSlice,
        scale: CheckpointTensorSlice,
    ) -> ferrule_common::Result<NumericFp8Source> {
        NumericFp8Source::new(
            weight,
            scale,
            self.encoding,
            CheckpointSourceFileIdentity::capture(&self.weight.path).unwrap(),
            CheckpointSourceFileIdentity::capture(&self.scale.path).unwrap(),
        )
    }

    fn tile(
        &self,
        expert: Option<usize>,
        rows: Range<usize>,
        columns: Range<usize>,
    ) -> NumericFp8Artifact {
        let reader = CheckpointTensorReader::new(1 << 20);
        self.source()
            .plan_tile(&reader, expert, rows, columns)
            .unwrap()
            .read(&reader)
            .unwrap()
    }

    fn expected(
        &self,
        expert: Option<usize>,
        rows: Range<usize>,
        columns: Range<usize>,
    ) -> Vec<f32> {
        let shape = self.source().matrix_shape();
        let matrix = expert.unwrap_or(0);
        let scale_cols = shape[1].div_ceil(128);
        let scale_base = matrix * shape[0].div_ceil(128) * scale_cols;
        rows.flat_map(|row| {
            columns.clone().map(move |column| {
                let weight = self.raw[matrix * shape[0] * shape[1] + row * shape[1] + column];
                decode_fp8_e4m3fn_byte(weight)
                    * self.scales[scale_base + row / 128 * scale_cols + column / 128]
            })
        })
        .collect()
    }

    fn binding(&self, transform: TensorTransform) -> BoundParameter {
        let path = ModulePath::new("projection").unwrap();
        let scale_dtype = match self.encoding {
            Encoding::E4M3FnBlock128Bf16 => ParameterDType::Bf16,
            Encoding::E4M3FnBlock128F32 => ParameterDType::F32,
        };
        let spec = ParameterSpec::new(
            ParameterId::new(1),
            path.clone(),
            ParameterDType::F8E4M3,
            self.weight.shape.clone(),
            ParameterResidency::Static,
        )
        .unwrap()
        .with_required_scale(scale_dtype, self.scale.shape.clone())
        .unwrap();
        let mut builder = StateDictSchema::builder();
        builder.register(spec).unwrap();
        let schema = builder.build().unwrap();
        let mut mapper = ExactNameMapper::new();
        let mut mapping = NameMapping::weight(path.clone());
        mapping.transform = transform;
        mapper.insert(&self.weight.name, mapping).unwrap();
        mapper
            .insert(&self.scale.name, NameMapping::scale(path.clone()))
            .unwrap();
        StateDictBinder::new(&schema, &mapper)
            .bind_slices(vec![self.weight.clone(), self.scale.clone()])
            .unwrap()
            .get(&path)
            .unwrap()
            .clone()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.directory).unwrap();
    }
}

#[test]
fn extremes_subnormals_signed_zero_and_scale_inv_are_multiplicative() {
    let bytes = vec![0, 0x80, 1, 0x81, 7, 8, 0x38, 0xb8, 0x7e, 0xfe];
    let expected = [
        0.0f32,
        -0.0,
        1.0 / 512.0,
        -1.0 / 512.0,
        7.0 / 512.0,
        1.0 / 64.0,
        1.0,
        -1.0,
        448.0,
        -448.0,
    ];
    for encoding in ENCODINGS {
        let f = Fixture::with_data(&[1, bytes.len()], encoding, bytes.clone(), vec![2.5]);
        let artifact = f.tile(None, 0..1, 0..bytes.len());
        assert_eq!(artifact.encoding(), encoding);
        assert_eq!(artifact.encoding().block_shape(), [128, 128]);
        assert_eq!(artifact.weight_bytes(), bytes);
        assert_eq!(artifact.scale_bytes(), scales_bytes(encoding, &[2.5]));
        for (actual, expected) in artifact.decode_f32(40).unwrap().iter().zip(expected) {
            assert_eq!(actual.to_bits(), (expected * 2.5).to_bits());
        }
        assert!(artifact.decode_f32(39).is_err());
    }
}

#[test]
fn partial_edges_ragged_rectangles_and_paired_physical_provenance() {
    for encoding in ENCODINGS {
        let f = Fixture::new(&[259, 391], encoding);
        for (rows, cols) in [
            (127..259, 125..391),
            (128..129, 256..257),
            (258..259, 390..391),
            (0..1, 0..391),
        ] {
            let a = f.tile(None, rows.clone(), cols.clone());
            let read = a.provenance();
            assert_eq!(read.source().weight(), &f.weight);
            assert_eq!(read.source().scale(), &f.scale);
            assert_eq!(read.weight_read().rows(), rows);
            assert_eq!(read.weight_read().columns(), cols);
            assert_eq!(
                read.scale_read().rows(),
                rows.start / 128..rows.end.div_ceil(128)
            );
            assert_eq!(
                read.scale_read().columns(),
                cols.start / 128..cols.end.div_ceil(128)
            );
            let scale_count = read.scale_read().local_shape().iter().product::<usize>();
            assert_eq!(
                a.storage_bytes(),
                (rows.len() * cols.len() + scale_count * encoding.scale_element_bytes()) as u64
            );
            assert_eq!(read.read_plan().source_files().len(), 2);
            assert_eq!(a.decode_f32(1 << 20).unwrap(), f.expected(None, rows, cols));
            for extent in read.weight_read().read_plan().extents() {
                assert!(
                    extent.offset() >= f.weight.offset && extent.end() <= f.weight.end_offset()
                );
            }
            for extent in read.scale_read().read_plan().extents() {
                assert!(extent.offset() >= f.scale.offset && extent.end() <= f.scale.end_offset());
            }
            let payloads = CheckpointPositionedReader::new(1 << 20)
                .read(read.read_plan())
                .unwrap();
            let split = read.weight_read().read_plan().extents().len();
            let reconstructed = read
                .materialize(payloads[..split].concat(), payloads[split..].concat())
                .unwrap();
            assert_eq!(reconstructed, a);
            assert!(read.materialize(vec![], a.scale_bytes().to_vec()).is_err());
            assert!(read.materialize(a.weight_bytes().to_vec(), vec![]).is_err());
        }
    }
}

#[test]
fn paired_read_budget_includes_scales_at_plan_and_read_time() {
    for encoding in ENCODINGS {
        let f = Fixture::new(&[129, 129], encoding);
        let source = f.source();
        let total = 4 + 4 * encoding.scale_element_bytes() as u64;
        let reader = CheckpointTensorReader::new(total);
        assert!(
            source
                .plan_tile(
                    &CheckpointTensorReader::new(total - 1),
                    None,
                    127..129,
                    127..129
                )
                .is_err()
        );
        let read = source.plan_tile(&reader, None, 127..129, 127..129).unwrap();
        assert_eq!(read.read_plan().storage_bytes(), total);
        assert!(read.read(&CheckpointTensorReader::new(total - 1)).is_err());
        assert_eq!(read.read(&reader).unwrap().weight_bytes().len(), 4);
    }
}

#[test]
fn expert_slices_do_not_read_neighboring_experts() {
    for encoding in ENCODINGS {
        let f = Fixture::new(&[3, 129, 257], encoding);
        let a = f.tile(Some(1), 126..129, 127..257);
        assert_eq!(a.provenance().expert(), Some(1));
        assert_eq!(a.provenance().source().expert_count(), Some(3));
        assert_eq!(a.provenance().source().weight().shape, [3, 129, 257]);
        assert_eq!(
            a.decode_f32(1 << 20).unwrap(),
            f.expected(Some(1), 126..129, 127..257)
        );
        assert_eq!(
            a.provenance().weight_read().tensor().offset,
            f.weight.offset + 129 * 257
        );
        assert_eq!(
            a.provenance().scale_read().tensor().offset,
            f.scale.offset + 6 * encoding.scale_element_bytes() as u64
        );
        let reader = CheckpointTensorReader::new(1024);
        for expert in [None, Some(3), Some(usize::MAX)] {
            assert!(f.source().plan_tile(&reader, expert, 0..1, 0..1).is_err());
        }
    }
}

#[test]
fn rejects_nan_weights_and_nonpositive_or_nonfinite_numeric_scales() {
    for encoding in ENCODINGS {
        for scale in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let f = Fixture::with_data(&[1, 1], encoding, vec![0x38], vec![scale]);
            let reader = CheckpointTensorReader::new(16);
            let read = f.source().plan_tile(&reader, None, 0..1, 0..1).unwrap();
            assert!(
                read.read(&reader)
                    .unwrap_err()
                    .to_string()
                    .contains("finite and strictly positive")
            );
        }
        for nan in [0x7f, 0xff] {
            assert!(decode_fp8_e4m3fn_byte(nan).is_nan());
            let f = Fixture::with_data(&[1, 1], encoding, vec![nan], vec![1.0]);
            let reader = CheckpointTensorReader::new(16);
            assert!(
                f.source()
                    .plan_tile(&reader, None, 0..1, 0..1)
                    .unwrap()
                    .read(&reader)
                    .is_err()
            );
        }
        let smallest = match encoding {
            Encoding::E4M3FnBlock128Bf16 => half::bf16::from_bits(1).to_f32(),
            Encoding::E4M3FnBlock128F32 => f32::from_bits(1),
        };
        let f = Fixture::with_data(&[1, 1], encoding, vec![0x38], vec![smallest]);
        assert_eq!(f.tile(None, 0..1, 0..1).decode_f32(4).unwrap(), [smallest]);
        let largest = match encoding {
            Encoding::E4M3FnBlock128Bf16 => half::bf16::MAX.to_f32(),
            Encoding::E4M3FnBlock128F32 => f32::MAX,
        };
        let f = Fixture::with_data(&[1, 1], encoding, vec![0x7e], vec![largest]);
        let reader = CheckpointTensorReader::new(16);
        let read = f.source().plan_tile(&reader, None, 0..1, 0..1).unwrap();
        for error in [
            read.read(&reader).unwrap_err(),
            read.materialize(f.raw.clone(), scales_bytes(encoding, &f.scales))
                .unwrap_err(),
        ] {
            assert!(matches!(error, ferrule_common::Error::Model { .. }));
            assert!(error.to_string().contains("overflow"));
        }
        let valid = Fixture::with_data(&[1, 1], encoding, vec![0x38], vec![largest]);
        assert_eq!(
            valid.tile(None, 0..1, 0..1).decode_f32(4).unwrap(),
            [largest]
        );
    }
}

#[test]
fn metadata_dtype_scale_layout_and_ranges_are_strict() {
    let f = Fixture::new(&[129, 257], Encoding::E4M3FnBlock128Bf16);
    for dtype in [
        DType::F32,
        DType::Bf16,
        DType::F8E8M0,
        DType::I8,
        DType::Unknown("F8_E4M3FNUZ".into()),
    ] {
        let mut weight = f.weight.clone();
        weight.dtype = dtype;
        assert!(f.source_with(weight, f.scale.clone()).is_err());
    }
    for dtype in [DType::F32, DType::F8E8M0, DType::I8] {
        let mut scale = f.scale.clone();
        scale.dtype = dtype;
        assert!(f.source_with(f.weight.clone(), scale).is_err());
    }
    for shape in [
        vec![3, 2],
        vec![6],
        vec![1, 2, 3],
        vec![1, 6],
        vec![2, 2],
        vec![0, 3],
    ] {
        let mut scale = f.scale.clone();
        scale.shape = shape;
        assert!(f.source_with(f.weight.clone(), scale).is_err());
    }
    for shape in [
        vec![0, 257],
        vec![129],
        vec![1, 1, 129, 257],
        vec![usize::MAX, 257],
    ] {
        let mut weight = f.weight.clone();
        weight.shape = shape;
        assert!(f.source_with(weight, f.scale.clone()).is_err());
    }
    for part in [false, true] {
        for offset in [u64::MAX, 1 << 40] {
            let mut weight = f.weight.clone();
            let mut scale = f.scale.clone();
            if part {
                scale.offset = offset;
            } else {
                weight.offset = offset;
            }
            assert!(f.source_with(weight, scale).is_err());
        }
        let mut weight = f.weight.clone();
        let mut scale = f.scale.clone();
        if part {
            scale.bytes -= 1;
        } else {
            weight.bytes -= 1;
        }
        assert!(f.source_with(weight, scale).is_err());
    }
    let mut weight = f.weight.clone();
    weight.path = f.scale.path.clone();
    assert!(f.source_with(weight, f.scale.clone()).is_err());
    let reader = CheckpointTensorReader::new(1 << 20);
    for rows in [0..0, 129..130, 0..usize::MAX, Range { start: 2, end: 1 }] {
        assert!(f.source().plan_tile(&reader, None, rows, 0..1).is_err());
    }
    for cols in [0..0, 257..258, 0..usize::MAX, Range { start: 2, end: 1 }] {
        assert!(f.source().plan_tile(&reader, None, 0..1, cols).is_err());
    }
    assert!(f.source().plan_tile(&reader, Some(0), 0..1, 0..1).is_err());
}

#[test]
fn same_file_pair_requires_nonoverlapping_ranges() {
    let f = Fixture::new(&[1, 8], Encoding::E4M3FnBlock128F32);
    let mut scale = f.scale.clone();
    scale.path = f.weight.path.clone();
    scale.offset = f.weight.offset;
    let source = CheckpointSourceFileIdentity::capture(&f.weight.path).unwrap();
    assert!(
        NumericFp8Source::new(
            f.weight.clone(),
            scale.clone(),
            f.encoding,
            source.clone(),
            source.clone()
        )
        .unwrap_err()
        .to_string()
        .contains("overlap")
    );
    scale.offset = 0;
    let pair =
        NumericFp8Source::new(f.weight.clone(), scale, f.encoding, source.clone(), source).unwrap();
    let plan = pair
        .plan_tile(&CheckpointTensorReader::new(32), None, 0..1, 0..8)
        .unwrap();
    assert_eq!(plan.read_plan().source_files().len(), 1);
}

#[test]
fn stale_weight_or_scale_is_rejected_before_read_decode_and_publication() {
    for change_scale in [false, true] {
        let f = Fixture::new(&[129, 129], Encoding::E4M3FnBlock128Bf16);
        let reader = CheckpointTensorReader::new(128);
        let source = f.source();
        let plan = source.plan_tile(&reader, None, 127..129, 127..129).unwrap();
        let artifact = plan.read(&reader).unwrap();
        let path = if change_scale {
            &f.scale.path
        } else {
            &f.weight.path
        };
        // Same-length replacement tests identity, not merely EOF checks.
        let replacement = f.directory.join("replacement");
        std::fs::copy(path, &replacement).unwrap();
        std::fs::rename(replacement, path).unwrap();
        assert!(source.validate_source_identity().is_err());
        assert!(source.plan_tile(&reader, None, 0..1, 0..1).is_err());
        assert!(plan.read(&reader).is_err());
        assert!(plan.read_plan().validate_source_identity().is_err());
        assert!(
            plan.materialize(
                artifact.weight_bytes().to_vec(),
                artifact.scale_bytes().to_vec()
            )
            .is_err()
        );
        assert!(artifact.decode_f32(16).is_err());
    }
}

#[test]
fn ragged_tensor_parallel_reads_match_global_block_origins() {
    for encoding in ENCODINGS {
        let f = Fixture::new(&[259, 391], encoding);
        for partition in [Partition::Column, Partition::Row] {
            let plan = TensorParallelLinearPlan::new(259, 391, 3, partition).unwrap();
            for rank in 0..3 {
                let rank = ParallelRankId::new(rank);
                let range = plan.rank_range(rank).unwrap();
                let (rows, cols) = match partition {
                    Partition::Column => (range, 0..391),
                    Partition::Row => (0..259, range),
                };
                let a = plan
                    .read_numeric_fp8_shard(
                        &CheckpointTensorReader::new(100_000),
                        &f.source(),
                        None,
                        rank,
                    )
                    .unwrap();
                assert_eq!(a.decode_f32(400_000).unwrap(), f.expected(None, rows, cols));
                assert!(
                    plan.read_weight_shard(&CheckpointTensorReader::new(100_000), &f.weight, rank)
                        .is_err()
                );
                assert!(plan.validate_cuda(rank, &f.weight.dtype).is_err());
            }
        }
        let plan = TensorParallelLinearPlan::new(258, 391, 3, Partition::Column).unwrap();
        assert!(
            plan.read_numeric_fp8_shard(
                &CheckpointTensorReader::new(100_000),
                &f.source(),
                None,
                ParallelRankId::new(0)
            )
            .is_err()
        );
    }
}

#[test]
fn state_dict_materialization_is_explicit_bounded_and_not_native_e8m0() {
    for encoding in ENCODINGS {
        let f = Fixture::new(&[129, 129], encoding);
        let binding = f.binding(TensorTransform::Identity);
        let total = 4 + 4 * encoding.scale_element_bytes() as u64;
        let materializer = StateDictMaterializer::new(total).unwrap();
        let a = materializer
            .numeric_fp8_tile(&binding, encoding, None, 127..129, 127..129)
            .unwrap();
        assert_eq!(a.storage_bytes(), total);
        assert_eq!(
            a.decode_f32(16).unwrap(),
            f.expected(None, 127..129, 127..129)
        );
        assert!(
            StateDictMaterializer::new(total - 1)
                .unwrap()
                .numeric_fp8_tile(&binding, encoding, None, 127..129, 127..129)
                .is_err()
        );
        assert!(
            f.binding(TensorTransform::transpose_2d())
                .numeric_fp8_source(encoding)
                .is_err()
        );
        let wrong = if encoding == ENCODINGS[0] {
            ENCODINGS[1]
        } else {
            ENCODINGS[0]
        };
        assert!(binding.numeric_fp8_source(wrong).is_err());
        assert!(
            LinearWeight::from_weight_and_scale(
                TensorRole::RoutedExpertDown,
                CheckpointTensorPayload {
                    slice: f.weight.clone(),
                    bytes: f.raw.clone()
                },
                Some(CheckpointTensorPayload {
                    slice: f.scale.clone(),
                    bytes: scales_bytes(encoding, &f.scales)
                })
            )
            .is_err()
        );
        // Native encoding still requires E8M0; the numeric API does not loosen it.
        for scale_dtype in [
            ParameterDType::Bf16,
            ParameterDType::F32,
            ParameterDType::F8E8M0,
        ] {
            let native = ParameterSpec::new_encoded(
                ParameterId::new(2),
                ModulePath::new("native").unwrap(),
                ParameterDType::F8E4M3,
                [129, 129],
                [129, 129],
                StorageEncoding::Fp8Block128,
                ParameterResidency::Static,
            )
            .unwrap()
            .with_required_scale(scale_dtype.clone(), [2, 2])
            .unwrap();
            let mut schema = StateDictSchema::builder();
            assert_eq!(
                schema.register(native).is_ok(),
                scale_dtype == ParameterDType::F8E8M0
            );
        }
    }
}

/// Requires only torch + the checkpoint, never vLLM or a full model invocation.
#[test]
#[ignore = "set FERRULE_NUMERIC_FP8_MODEL_DIR and FERRULE_FP8_PYTHON for a local 35B expert torch oracle"]
fn local_35b_expert_matches_torch_fp8_dequantization() {
    let model =
        std::env::var("FERRULE_NUMERIC_FP8_MODEL_DIR").expect("set FERRULE_NUMERIC_FP8_MODEL_DIR");
    let python = std::env::var("FERRULE_FP8_PYTHON").unwrap_or_else(|_| "python3".into());
    let output = std::process::Command::new(python)
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/numeric_fp8_oracle.py"
        ))
        .arg(model)
        .output()
        .expect("run Python torch oracle");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let data: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let tensor = |value: &serde_json::Value| CheckpointTensorSlice {
        name: value["name"].as_str().unwrap().into(),
        role: TensorRole::RoutedExpertDown,
        path: value["path"].as_str().unwrap().into(),
        offset: value["offset"].as_u64().unwrap(),
        bytes: value["bytes"].as_u64().unwrap(),
        dtype: DType::from_safetensors_dtype(value["dtype"].as_str().unwrap()),
        shape: serde_json::from_value(value["shape"].clone()).unwrap(),
    };
    let weight = tensor(&data["weight"]);
    let scale = tensor(&data["scale"]);
    let encoding = match scale.dtype {
        DType::Bf16 => Encoding::E4M3FnBlock128Bf16,
        DType::F32 => Encoding::E4M3FnBlock128F32,
        _ => panic!("expected numeric scale"),
    };
    let source = NumericFp8Source::new(
        weight.clone(),
        scale.clone(),
        encoding,
        CheckpointSourceFileIdentity::capture(&weight.path).unwrap(),
        CheckpointSourceFileIdentity::capture(&scale.path).unwrap(),
    )
    .unwrap();
    let rows: [usize; 2] = serde_json::from_value(data["rows"].clone()).unwrap();
    let cols: [usize; 2] = serde_json::from_value(data["columns"].clone()).unwrap();
    let limit = data["read_bytes"].as_u64().unwrap();
    let reader = CheckpointTensorReader::new(limit);
    let artifact = source
        .plan_tile(&reader, None, rows[0]..rows[1], cols[0]..cols[1])
        .unwrap()
        .read(&reader)
        .unwrap();
    assert_eq!(artifact.storage_bytes(), limit);
    let actual = artifact.decode_f32(100_000).unwrap();
    let expected: Vec<u32> = serde_json::from_value(data["expected_bits"].clone()).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.to_bits(), expected, "element {index}");
    }
    // The prepared path reads exactly one selected expert projection, retaining
    // compressed storage; it never materializes the full model or F32 matrix.
    let bound = prepared_storage::bind_pair(
        &weight,
        Some(&scale),
        ParameterResidency::expert(0, 0),
        false,
    );
    let binding = bound.get_by_id(ParameterId::new(1)).unwrap();
    let total = weight.bytes + scale.bytes;
    assert!(total <= 2 * 1024 * 1024, "local probe must stay small");
    let materializer = StateDictMaterializer::new(total).unwrap();
    let mut previous = None;
    for _ in 0..2 {
        let linear = materializer
            .prepared_linear(binding, binding.role().clone())
            .unwrap();
        assert_eq!(linear.global_shape(), [weight.shape[0], weight.shape[1]]);
        let ferrule_model::transformer::PreparedLinearStorage::NumericFp8(full) = linear.storage()
        else {
            panic!("expected compressed expert")
        };
        assert_eq!(full.storage_bytes(), total);
        assert!(linear.parameter().values_f32().is_err());
        let packed: Vec<u8> = (rows[0]..rows[1])
            .flat_map(|r| {
                full.weight_bytes()[r * weight.shape[1] + cols[0]..r * weight.shape[1] + cols[1]]
                    .iter()
                    .copied()
            })
            .collect();
        assert_eq!(packed, artifact.weight_bytes());
        let scale_rows = artifact.provenance().scale_read().rows();
        let scale_cols = artifact.provenance().scale_read().columns();
        let element_bytes = encoding.scale_element_bytes();
        let packed_scale: Vec<u8> = scale_rows
            .flat_map(|r| {
                full.scale_bytes()[(r * scale.shape[1] + scale_cols.start) * element_bytes
                    ..(r * scale.shape[1] + scale_cols.end) * element_bytes]
                    .iter()
                    .copied()
            })
            .collect();
        assert_eq!(packed_scale, artifact.scale_bytes());
        let weak = std::sync::Arc::downgrade(full);
        drop(linear);
        assert!(
            weak.upgrade().is_none(),
            "materializer retained expert host storage"
        );
        previous = Some(weak);
    }
    assert!(previous.unwrap().upgrade().is_none());
    eprintln!("prepared expert: {total} compressed bytes/read, 2 loads, no retained host artifact");
    // Torch also checks the decoder's entire 256-code E4M3FN table, including NaNs.
    let table: Vec<u32> = serde_json::from_value(data["e4m3_bits"].clone()).unwrap();
    assert_eq!(table.len(), 256);
    for (byte, bits) in table.into_iter().enumerate() {
        let decoded = decode_fp8_e4m3fn_byte(byte as u8);
        if f32::from_bits(bits).is_nan() {
            assert!(decoded.is_nan());
        } else {
            assert_eq!(decoded.to_bits(), bits, "E4M3FN code {byte}");
        }
    }
    eprintln!(
        "torch exact match: {}, shape {:?}, tile {:?}x{:?}, {} values, {} weight+scale bytes, {:?}",
        weight.name,
        weight.shape,
        rows,
        cols,
        actual.len(),
        limit,
        encoding
    );
}

#[test]
fn bounded_read_validates_only_selected_expert_and_scale_blocks() {
    for encoding in ENCODINGS {
        // Invalid payloads in an unselected expert must not trigger a full read.
        let f = Fixture::with_data(&[2, 1, 1], encoding, vec![0x7f, 0x38], vec![f32::NAN, 2.5]);
        let limit = 1 + encoding.scale_element_bytes() as u64;
        let reader = CheckpointTensorReader::new(limit);
        let selected = f.source().plan_tile(&reader, Some(1), 0..1, 0..1).unwrap();
        assert_eq!(
            selected.read(&reader).unwrap().decode_f32(4).unwrap(),
            [2.5]
        );
        assert!(
            f.source()
                .plan_tile(&reader, Some(0), 0..1, 0..1)
                .unwrap()
                .read(&reader)
                .is_err()
        );
    }
}

#[cfg(unix)]
#[test]
fn symlink_alias_cannot_hide_overlapping_pair_ranges() {
    let f = Fixture::new(&[1, 8], Encoding::E4M3FnBlock128F32);
    let alias = f.directory.join("alias.bin");
    std::os::unix::fs::symlink(&f.weight.path, &alias).unwrap();
    let mut scale = f.scale.clone();
    scale.path = alias.clone();
    scale.offset = f.weight.offset;
    let result = NumericFp8Source::new(
        f.weight.clone(),
        scale,
        f.encoding,
        CheckpointSourceFileIdentity::capture(&f.weight.path).unwrap(),
        CheckpointSourceFileIdentity::capture(&alias).unwrap(),
    );
    assert!(result.unwrap_err().to_string().contains("overlap"));
}

mod prepared_storage {
    use super::*;
    use ferrule_model::transformer::{
        BoundStateDict, CpuStandardDecoderOperators, HostRows, MemoryLayerWeightCache,
        PreparedLinear, PreparedLinearStorage, PreparedParameterStorage, Rows, RowsDType,
        RowsShape, StandardDecoderOperators, UnsupportedOperator,
    };
    use std::sync::Arc;

    fn dtype(dtype: &DType) -> ParameterDType {
        match dtype {
            DType::F32 => ParameterDType::F32,
            DType::Bf16 => ParameterDType::Bf16,
            DType::F8E4M3 => ParameterDType::F8E4M3,
            DType::F8E8M0 => ParameterDType::F8E8M0,
            DType::I8 => ParameterDType::I8,
            _ => panic!("unsupported fixture dtype"),
        }
    }

    pub(super) fn bind_pair(
        weight: &CheckpointTensorSlice,
        scale: Option<&CheckpointTensorSlice>,
        residency: ParameterResidency,
        alias: bool,
    ) -> BoundStateDict {
        let path = ModulePath::new("projection").unwrap();
        let mut spec = ParameterSpec::new(
            ParameterId::new(1),
            path.clone(),
            dtype(&weight.dtype),
            weight.shape.clone(),
            residency.clone(),
        )
        .unwrap();
        if let Some(scale) = scale {
            spec = spec
                .with_required_scale(dtype(&scale.dtype), scale.shape.clone())
                .unwrap();
        }
        let mut schema = StateDictSchema::builder();
        schema
            .register_with_role(spec, weight.role.clone())
            .unwrap();
        if alias {
            let mut spec = ParameterSpec::new(
                ParameterId::new(2),
                ModulePath::new("alias").unwrap(),
                dtype(&weight.dtype),
                weight.shape.clone(),
                residency,
            )
            .unwrap();
            if let Some(scale) = scale {
                spec = spec
                    .with_required_scale(dtype(&scale.dtype), scale.shape.clone())
                    .unwrap();
            }
            schema
                .register_with_role(spec.with_alias(ParameterId::new(1)), TensorRole::OutputHead)
                .unwrap();
        }
        let schema = schema.build().unwrap();
        let mut mapper = ExactNameMapper::new();
        mapper
            .insert(&weight.name, NameMapping::weight(path.clone()))
            .unwrap();
        let mut slices = vec![weight.clone()];
        if let Some(scale) = scale {
            mapper
                .insert(&scale.name, NameMapping::scale(path))
                .unwrap();
            slices.push(scale.clone());
        }
        StateDictBinder::new(&schema, &mapper)
            .bind_slices(slices)
            .unwrap()
    }

    fn compressed(linear: &PreparedLinear) -> &Arc<NumericFp8Artifact> {
        let PreparedLinearStorage::NumericFp8(artifact) = linear.storage() else {
            panic!("numeric storage must not build native LinearWeight");
        };
        artifact
    }

    fn typed_unsupported(error: ferrule_common::Error) {
        let ferrule_common::Error::ModelSource { source } = error else {
            panic!("not a typed model error")
        };
        let unsupported = source
            .downcast_ref::<UnsupportedOperator>()
            .expect("typed unsupported");
        assert!(unsupported.reason.contains("numeric FP8"));
    }

    #[test]
    fn prepared_numeric_payload_keeps_global_geometry_bytes_and_alias_custody() {
        for encoding in ENCODINGS {
            let f = Fixture::new(&[129, 257], encoding);
            let bound = bind_pair(&f.weight, Some(&f.scale), ParameterResidency::Static, true);
            let first = bound.get_by_id(ParameterId::new(1)).unwrap();
            let alias = bound.get_by_id(ParameterId::new(2)).unwrap();
            assert_eq!(first.numeric_fp8_encoding(), Some(encoding));
            let total = f.weight.bytes + f.scale.bytes;
            let materializer = StateDictMaterializer::new(total).unwrap();
            let linear = materializer
                .prepared_linear(first, first.role().clone())
                .unwrap();
            let tied = materializer
                .prepared_linear(alias, alias.role().clone())
                .unwrap();
            assert_eq!(linear.global_shape(), [129, 257]);
            assert_eq!(tied.role(), &TensorRole::OutputHead);
            assert_eq!(tied.parameter().binding().id(), alias.id());
            assert_eq!(tied.parameter().canonical_id(), first.id());
            assert!(tied.parameter().binding().shares_storage_with(first));
            assert!(Arc::ptr_eq(compressed(&linear), compressed(&tied)));
            let PreparedParameterStorage::NumericFp8(parameter_artifact) =
                linear.parameter().storage()
            else {
                panic!()
            };
            assert!(Arc::ptr_eq(compressed(&linear), parameter_artifact));
            let artifact = linear.numeric_fp8().unwrap();
            assert_eq!(artifact.encoding(), encoding);
            assert_eq!(artifact.local_shape(), linear.global_shape());
            assert_eq!(artifact.weight_bytes(), f.raw);
            assert_eq!(artifact.scale_bytes(), scales_bytes(encoding, &f.scales));
            assert_eq!(artifact.storage_bytes(), total);
            assert_eq!(artifact.provenance().source().weight(), &f.weight);
            assert_eq!(artifact.provenance().source().scale(), &f.scale);
            assert_eq!(artifact.provenance().weight_read().rows(), 0..129);
            assert_eq!(artifact.provenance().weight_read().columns(), 0..257);
            assert_eq!(artifact.provenance().scale_read().local_shape(), [2, 3]);
            assert!(linear.tensor_shard().is_none());
            typed_unsupported(linear.weight().unwrap_err());
            typed_unsupported(linear.parameter().weight().unwrap_err());
            typed_unsupported(linear.parameter().values_f32().unwrap_err());
            typed_unsupported(linear.parameter().values_bf16_words().unwrap_err());
            typed_unsupported(materializer.linear(first).unwrap_err());
            // Only an explicit bounded oracle call expands this artifact.
            assert!(artifact.decode_f32(f.weight.bytes * 4 - 1).is_err());
            assert_eq!(
                artifact.decode_f32(f.weight.bytes * 4).unwrap(),
                f.expected(None, 0..129, 0..257)
            );
        }
    }

    #[test]
    fn cpu_linear_fails_typed_without_implicit_numeric_fallback() {
        let f = Fixture::new(&[2, 3], ENCODINGS[0]);
        let binding = f.binding(TensorTransform::Identity);
        let linear = StateDictMaterializer::new(32)
            .unwrap()
            .prepared_linear(&binding, binding.role().clone())
            .unwrap();
        let mut cpu = CpuStandardDecoderOperators::new(
            ferrule_model::execution::ExecutionPrecisionPolicy::f32(),
        );
        let input = Rows::Host(
            HostRows::new(
                RowsShape::new(1, 3).unwrap(),
                RowsDType::F32,
                None,
                vec![1.; 3],
            )
            .unwrap(),
        );
        typed_unsupported(cpu.linear(&linear, &input, None).unwrap_err());
    }

    #[test]
    fn prepared_budget_counts_both_parts_and_errors_do_not_populate_cache() {
        for encoding in ENCODINGS {
            let f = Fixture::new(&[129, 129], encoding);
            let binding = f.binding(TensorTransform::Identity);
            let total = f.weight.bytes + f.scale.bytes;
            for limit in [f.weight.bytes, total - 1] {
                let materializer = StateDictMaterializer::new(limit).unwrap();
                for _ in 0..2 {
                    let error = materializer
                        .prepared_linear(&binding, binding.role().clone())
                        .unwrap_err();
                    assert!(error.to_string().contains("bounded read size"));
                }
            }
            let linear = StateDictMaterializer::new(total)
                .unwrap()
                .prepared_linear(&binding, binding.role().clone())
                .unwrap();
            assert_eq!(compressed(&linear).storage_bytes(), total);
        }
    }

    #[test]
    fn numeric_static_and_layer_caches_reject_stale_pairs_and_foreign_bindings() {
        for change_scale in [false, true] {
            for residency in [ParameterResidency::Static, ParameterResidency::layer(0)] {
                let f = Fixture::new(&[2, 3], ENCODINGS[0]);
                let bound = bind_pair(&f.weight, Some(&f.scale), residency.clone(), true);
                let binding = bound.get_by_id(ParameterId::new(1)).unwrap();
                let alias = bound.get_by_id(ParameterId::new(2)).unwrap();
                let materializer = StateDictMaterializer::new(32).unwrap();
                let mut cache = MemoryLayerWeightCache::new();
                let mut load = |binding: &BoundParameter| match residency {
                    ParameterResidency::Static => materializer.static_parameter(binding),
                    _ => materializer.layer_parameter(0, binding, &mut cache),
                };
                load(binding).unwrap();
                load(alias).unwrap();
                let other = bind_pair(&f.weight, Some(&f.scale), residency.clone(), false);
                assert!(
                    load(other.get_by_id(ParameterId::new(1)).unwrap())
                        .unwrap_err()
                        .to_string()
                        .contains("another state dict")
                );
                let path = if change_scale {
                    &f.scale.path
                } else {
                    &f.weight.path
                };
                let replacement = f.directory.join("replace");
                std::fs::copy(path, &replacement).unwrap();
                std::fs::rename(replacement, path).unwrap();
                assert!(load(binding).is_err());
                assert!(load(alias).is_err());
            }
        }
    }

    #[test]
    fn bounded_expert_reload_has_no_materializer_host_residency() {
        for encoding in ENCODINGS {
            let f = Fixture::new(&[129, 129], encoding);
            let bound = bind_pair(
                &f.weight,
                Some(&f.scale),
                ParameterResidency::expert(0, 7),
                false,
            );
            let binding = bound.get_by_id(ParameterId::new(1)).unwrap();
            let materializer = StateDictMaterializer::new(f.weight.bytes + f.scale.bytes).unwrap();
            let mut cache = MemoryLayerWeightCache::new();
            assert!(materializer.static_parameter(binding).is_err());
            assert!(
                materializer
                    .layer_parameter(0, binding, &mut cache)
                    .is_err()
            );
            assert!(materializer.expert_parameter(0, 6, binding).is_err());
            let first = materializer
                .prepared_linear(binding, binding.role().clone())
                .unwrap();
            let parameter = materializer.expert_parameter(0, 7, binding).unwrap();
            let second =
                PreparedLinear::from_parameter(parameter.clone(), binding.role().clone()).unwrap();
            assert!(!Arc::ptr_eq(compressed(&first), compressed(&second)));
            assert_eq!(
                compressed(&first).weight_bytes(),
                compressed(&second).weight_bytes()
            );
            let first_weak = Arc::downgrade(compressed(&first));
            let second_weak = Arc::downgrade(compressed(&second));
            drop(first);
            drop(second);
            drop(parameter);
            assert!(first_weak.upgrade().is_none());
            assert!(second_weak.upgrade().is_none());
            let third = materializer
                .prepared_linear(binding, binding.role().clone())
                .unwrap();
            assert_eq!(
                third.numeric_fp8().unwrap().storage_bytes(),
                f.weight.bytes + f.scale.bytes
            );
            drop(third);
            // Reload really consults the paired source again, not a retained expert cache.
            std::fs::remove_file(&f.scale.path).unwrap();
            assert!(
                materializer
                    .prepared_linear(binding, binding.role().clone())
                    .is_err()
            );
        }
    }

    #[test]
    fn dense_and_native_e8m0_prepared_storage_remain_legacy() {
        for (dtype, bytes, scale) in [
            (
                DType::F32,
                vec![1.0f32.to_le_bytes().to_vec(); 4].concat(),
                None,
            ),
            (
                DType::Bf16,
                vec![half::bf16::ONE.to_bits().to_le_bytes().to_vec(); 4].concat(),
                None,
            ),
            (DType::F8E4M3, vec![0x38; 4], Some(vec![127])),
            (DType::I8, vec![0x22; 4], Some(vec![127; 2])),
        ] {
            let f = Fixture::new(&[2, 2], ENCODINGS[0]);
            std::fs::write(&f.weight.path, &bytes).unwrap();
            let mut weight = f.weight.clone();
            weight.offset = 0;
            weight.bytes = bytes.len() as u64;
            weight.dtype = dtype.clone();
            let scale = scale.map(|bytes| {
                std::fs::write(&f.scale.path, &bytes).unwrap();
                let mut scale = f.scale.clone();
                scale.dtype = DType::F8E8M0;
                scale.offset = 0;
                scale.bytes = bytes.len() as u64;
                scale.shape = if dtype == DType::I8 {
                    vec![2, 1]
                } else {
                    vec![1, 1]
                };
                scale
            });
            let bound = bind_pair(&weight, scale.as_ref(), ParameterResidency::Static, false);
            let binding = bound.get_by_id(ParameterId::new(1)).unwrap();
            assert_eq!(binding.numeric_fp8_encoding(), None);
            let materializer = StateDictMaterializer::new(64).unwrap();
            let linear = materializer
                .prepared_linear(binding, binding.role().clone())
                .unwrap();
            assert!(matches!(linear.storage(), PreparedLinearStorage::Native(_)));
            assert!(matches!(
                linear.parameter().storage(),
                PreparedParameterStorage::Full(_)
            ));
            assert!(linear.numeric_fp8().is_none());
            assert_eq!(linear.weight().unwrap().weight.bytes, bytes);
            assert_eq!(
                linear.weight().unwrap(),
                &materializer.linear(binding).unwrap()
            );
            let width = if dtype == DType::I8 { 4 } else { 2 };
            assert_eq!(linear.global_shape(), [2, width]);
            assert_eq!(
                linear.weight().unwrap().reference_weights_f32().unwrap(),
                vec![1.; 2 * width]
            );
            if matches!(dtype, DType::Bf16 | DType::F32) {
                assert_eq!(linear.parameter().values_f32().unwrap(), vec![1.; 4]);
            }
        }
    }
    #[test]
    fn shared_expert_output_gate_is_feed_forward_and_limits_count_scales() {
        use ferrule_model::execution::TransformerStage;
        use ferrule_model::transformer::{
            BoundDecoderResources, DecoderRecipe, SyntheticDecoderRecipe, parameter_resource_id,
        };
        let f = Fixture::new(&[1, 2], ENCODINGS[0]);
        let mut weight = f.weight.clone();
        weight.role = TensorRole::SharedExpertOutputGate;
        let bound = bind_pair(&weight, Some(&f.scale), ParameterResidency::layer(0), false);
        let config = serde_json::json!({
            "vocab_size":4, "hidden_size":2, "num_attention_heads":1,
            "num_key_value_heads":1, "head_dim":2, "intermediate_size":4,
            "num_experts":1, "experts_per_token":1, "max_position_embeddings":16,
            "rms_norm_eps":1e-5, "rope_theta":10000., "tie_word_embeddings":false
        });
        let spec = SyntheticDecoderRecipe::new().build_spec(&config).unwrap();
        let resources = BoundDecoderResources::new(spec, Arc::new(bound)).unwrap();
        let bytes = weight.bytes + f.scale.bytes;
        assert!(
            resources
                .validate_parameter_limits(bytes - 1, bytes)
                .is_err()
        );
        resources.validate_parameter_limits(bytes, bytes).unwrap();
        let executable = resources.prepared_executable(1).unwrap();
        let uses = executable
            .stages()
            .iter()
            .filter(|stage| {
                stage
                    .resources()
                    .iter()
                    .any(|use_| use_.resource() == parameter_resource_id(ParameterId::new(1)))
            })
            .collect::<Vec<_>>();
        assert_eq!(uses.len(), 1);
        assert_eq!(
            uses[0].operation(),
            &TransformerStage::FeedForward { layer: 0 }
        );
    }
}
