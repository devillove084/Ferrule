//! Public API CPU tests; no private fault hooks or GPU dependencies.
use ferrule_common::SourceGeneration;
use ferrule_model::checkpoint::{
    CheckpointDType, CheckpointReadExtent, CheckpointReadPipeline, CheckpointReadPlan,
    CheckpointSourceFileIdentity, CheckpointTensorReader, CheckpointTensorSlice,
    NumericFp8Artifact, NumericFp8Encoding, NumericFp8Read, NumericFp8Source, ReadCompletion,
    ReadPipelineLimits, ReadPipelineOptions, ReadPoll, ReadTicket,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);
const GENERATION: SourceGeneration = SourceGeneration::new(12);
const OPTIONS: ReadPipelineOptions = ReadPipelineOptions {
    worker_stack_bytes: 256 * 1024,
    max_worker_stack_bytes: 512 * 1024,
};

struct Fixture {
    path: PathBuf,
    source: NumericFp8Source,
    reader: CheckpointTensorReader,
    read: NumericFp8Read,
}

impl Fixture {
    fn new(encoding: NumericFp8Encoding, weight: [u8; 4], scale: &[u8]) -> Self {
        let path = std::env::temp_dir().join(format!(
            "ferrule-packed-public-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut bytes = weight.to_vec();
        bytes.extend(scale);
        std::fs::write(&path, bytes).unwrap();
        let snapshot = CheckpointSourceFileIdentity::capture(&path).unwrap();
        let weight = CheckpointTensorSlice {
            name: "weight".into(),
            role: ferrule_model::TensorRole::AttentionQuery,
            path: path.clone(),
            offset: 0,
            bytes: 4,
            dtype: CheckpointDType::F8E4M3,
            shape: vec![2, 2],
        };
        let scale = CheckpointTensorSlice {
            name: "scale".into(),
            role: ferrule_model::TensorRole::AttentionQuery,
            path: path.clone(),
            offset: 4,
            bytes: scale.len() as u64,
            dtype: encoding.scale_dtype(),
            shape: vec![1, 1],
        };
        let source =
            NumericFp8Source::new(weight, scale, encoding, snapshot.clone(), snapshot).unwrap();
        let reader = CheckpointTensorReader::new(64);
        let read = source.plan_tile(&reader, None, 0..2, 0..2).unwrap();
        Self {
            path,
            source,
            reader,
            read,
        }
    }

    fn f32() -> Self {
        Self::new(
            NumericFp8Encoding::E4M3FnBlock128F32,
            [0x38, 0x40, 0x42, 0x44],
            &0.5f32.to_le_bytes(),
        )
    }

    fn complete(
        &self,
        plan: CheckpointReadPlan,
        generation: SourceGeneration,
    ) -> (CheckpointReadPipeline, ReadTicket, ReadCompletion) {
        let mut pipeline = CheckpointReadPipeline::with_options(
            self.reader.clone(),
            ReadPipelineLimits {
                max_host_bytes: 64,
                max_read_slots: 1,
            },
            OPTIONS,
        )
        .unwrap();
        let ticket = pipeline.submit(generation, plan).unwrap();
        assert!(matches!(
            pipeline.wait(&ticket).unwrap(),
            ReadPoll::Ready(_)
        ));
        let completion = pipeline.take(&ticket).unwrap();
        (pipeline, ticket, completion)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[test]
fn public_stack_options_and_combined_numeric_ranges_preserve_custody() {
    let fixture = Fixture::f32();
    let combined = CheckpointReadPlan::new(
        (0..3).flat_map(|_| fixture.read.read_plan().extents().iter().cloned()),
        Arc::<[CheckpointSourceFileIdentity]>::from(fixture.read.read_plan().source_files()),
    )
    .unwrap();
    let (mut pipeline, ticket, completion) = fixture.complete(combined, GENERATION);
    assert_eq!(pipeline.options(), OPTIONS);
    assert_eq!(
        pipeline.budget().worker_stack.configured_bytes,
        OPTIONS.worker_stack_bytes
    );
    assert_eq!(pipeline.budget().worker_stack.os_overhead_bytes, None);
    let artifacts = (0..3)
        .map(|index| {
            NumericFp8Artifact::from_verified_read_range(
                &fixture.read,
                &completion,
                &ticket,
                index * 8..(index + 1) * 8,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    for artifact in &artifacts {
        assert_eq!(artifact, &fixture.read.read(&fixture.reader).unwrap());
    }
    assert_eq!(pipeline.budget().reserved_bytes, 24);
    assert_eq!(
        pipeline.budget().requested_host_reservation_bytes(),
        24 + OPTIONS.worker_stack_bytes as u128
    );
    assert_eq!(pipeline.shutdown().budget.worker_stack.reserved_bytes, 0);
    assert_eq!(pipeline.budget().reserved_bytes, 24);
    drop(artifacts);
    assert_eq!(pipeline.budget().reserved_bytes, 24);
    drop(completion);
    assert_eq!(pipeline.budget().reserved_bytes, 0);
}

#[test]
fn public_plan_mismatch_cannot_be_relabelled_as_verified_numeric_storage() {
    let fixture = Fixture::f32();
    // Same bytes, but not the original paired extent identity/order.
    let plan = CheckpointReadPlan::new(
        [CheckpointReadExtent::new(fixture.path.clone(), 0, 8).unwrap()],
        Arc::<[CheckpointSourceFileIdentity]>::from(fixture.read.read_plan().source_files()),
    )
    .unwrap();
    let (pipeline, ticket, completion) = fixture.complete(plan, GENERATION);
    assert!(NumericFp8Artifact::from_verified_read(&fixture.read, &completion, &ticket).is_err());
    assert!(
        NumericFp8Artifact::from_verified_read_range(&fixture.read, &completion, &ticket, 0..8)
            .is_err()
    );
    assert_eq!(pipeline.budget().reserved_bytes, 8);
}

#[test]
fn foreign_generation_ticket_and_bad_ranges_do_not_release_custody() {
    let fixture = Fixture::f32();
    let (pipeline, ticket, completion) =
        fixture.complete(fixture.read.read_plan().clone(), GENERATION);
    let (_, foreign, held) =
        fixture.complete(fixture.read.read_plan().clone(), SourceGeneration::new(13));
    assert!(
        fixture
            .read
            .materialize_verified_read(&completion, &foreign)
            .is_err()
    );
    for range in [0..0, 0..7, 1..9, 0..9] {
        assert!(
            fixture
                .read
                .materialize_verified_read_range(&completion, &ticket, range)
                .is_err()
        );
    }
    assert_eq!(pipeline.budget().reserved_bytes, 8);
    drop(held);
}

#[test]
fn source_changed_after_worker_post_boundary_rejects_conversion() {
    let fixture = Fixture::f32();
    let (pipeline, ticket, completion) =
        fixture.complete(fixture.read.read_plan().clone(), GENERATION);
    std::fs::write(&fixture.path, [0; 16]).unwrap();
    assert!(
        fixture
            .read
            .materialize_verified_read(&completion, &ticket)
            .unwrap_err()
            .to_string()
            .contains("stale")
    );
    assert_eq!(pipeline.budget().reserved_bytes, 8);
}

#[test]
fn invalid_numeric_contents_match_normal_read_rejections() {
    for (weight, scale) in [
        (0x7f, 0.5f32),
        (0x38, f32::NAN),
        (0x38, f32::INFINITY),
        (0x38, -1.0),
        (0x38, 0.0),
        (0x7e, f32::MAX),
    ] {
        let fixture = Fixture::new(
            NumericFp8Encoding::E4M3FnBlock128F32,
            [weight; 4],
            &scale.to_le_bytes(),
        );
        let (pipeline, ticket, completion) =
            fixture.complete(fixture.read.read_plan().clone(), GENERATION);
        assert!(completion.payload().is_some());
        assert!(
            fixture
                .read
                .materialize_verified_read(&completion, &ticket)
                .is_err()
        );
        assert!(fixture.read.read(&fixture.reader).is_err());
        assert_eq!(pipeline.budget().reserved_bytes, 8);
    }
}

#[test]
fn bf16_and_f32_conversion_keep_the_same_content_proof_and_aliases() {
    for encoding in [
        NumericFp8Encoding::E4M3FnBlock128Bf16,
        NumericFp8Encoding::E4M3FnBlock128F32,
    ] {
        let scale = match encoding {
            NumericFp8Encoding::E4M3FnBlock128Bf16 => {
                half::bf16::from_f32(0.5).to_bits().to_le_bytes().to_vec()
            }
            NumericFp8Encoding::E4M3FnBlock128F32 => 0.5f32.to_le_bytes().to_vec(),
        };
        let fixture = Fixture::new(encoding, [0x38, 0x40, 0x42, 0x44], &scale);
        let (pipeline, ticket, completion) =
            fixture.complete(fixture.read.read_plan().clone(), GENERATION);
        let artifact =
            NumericFp8Artifact::from_verified_read(&fixture.read, &completion, &ticket).unwrap();
        assert_eq!(artifact, fixture.read.read(&fixture.reader).unwrap());
        assert_eq!(artifact.provenance().source(), &fixture.source);
        let alias = artifact.clone();
        assert_eq!(
            artifact.weight_bytes().as_ptr(),
            alias.weight_bytes().as_ptr()
        );
        assert_eq!(
            artifact.scale_bytes().as_ptr(),
            alias.scale_bytes().as_ptr()
        );
        assert_eq!(pipeline.budget().reserved_bytes, 4 + scale.len() as u64);
    }
}
