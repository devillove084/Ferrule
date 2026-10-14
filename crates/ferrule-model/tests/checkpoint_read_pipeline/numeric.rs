//! Completion-derived numeric proof tests; private descriptor corruption is
//! intentional fault injection, not a public unchecked proof constructor.
use super::*;
use crate::checkpoint::{
    CheckpointReadPipeline, ReadCompletion, ReadPipelineLimits, ReadPoll, ReadTicket,
};
use ferrule_common::SourceGeneration;
use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

thread_local! {
    static AFTER_VALIDATION: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
}

pub(super) fn after_validation() {
    if let Some(action) = AFTER_VALIDATION.with(|slot| slot.borrow_mut().take()) {
        action();
    }
}

static NEXT: AtomicU64 = AtomicU64::new(0);
const GENERATION: SourceGeneration = SourceGeneration::new(71);

struct Fixture {
    directory: PathBuf,
    paths: Vec<PathBuf>,
    reader: CheckpointTensorReader,
    reads: Vec<NumericFp8Read>,
    combined: CheckpointReadPlan,
}

impl Fixture {
    fn new(encoding: NumericFp8Encoding, weight: &[u8], scale: &[u8], shape: [usize; 2]) -> Self {
        let directory = std::env::temp_dir().join(format!(
            "ferrule-numeric-completion-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let reader = CheckpointTensorReader::new(128);
        let mut reads = Vec::new();
        let mut paths = Vec::new();
        let mut snapshots = Vec::new();
        for index in 0..3 {
            let path = directory.join(format!("projection-{index}"));
            let mut bytes = weight.to_vec();
            bytes.extend(scale);
            std::fs::write(&path, bytes).unwrap();
            let snapshot = CheckpointSourceFileIdentity::capture(&path).unwrap();
            let w = CheckpointTensorSlice {
                name: format!("p{index}.weight"),
                role: crate::TensorRole::AttentionQuery,
                path: path.clone(),
                offset: 0,
                bytes: weight.len() as u64,
                dtype: CheckpointDType::F8E4M3,
                shape: shape.to_vec(),
            };
            let s = CheckpointTensorSlice {
                name: format!("p{index}.weight_scale_inv"),
                role: crate::TensorRole::AttentionQuery,
                path: path.clone(),
                offset: weight.len() as u64,
                bytes: scale.len() as u64,
                dtype: encoding.scale_dtype(),
                shape: vec![shape[0].div_ceil(128), shape[1].div_ceil(128)],
            };
            let source =
                NumericFp8Source::new(w, s, encoding, snapshot.clone(), snapshot.clone()).unwrap();
            reads.push(
                source
                    .plan_tile(&reader, None, 0..shape[0], 0..shape[1])
                    .unwrap(),
            );
            paths.push(path);
            snapshots.push(snapshot);
        }
        let combined = CheckpointReadPlan::new(
            reads
                .iter()
                .flat_map(|read| read.read_plan().extents().iter().cloned()),
            Arc::<[CheckpointSourceFileIdentity]>::from(snapshots),
        )
        .unwrap();
        Self {
            directory,
            paths,
            reader,
            reads,
            combined,
        }
    }

    fn f32() -> Self {
        Self::new(
            NumericFp8Encoding::E4M3FnBlock128F32,
            &[0x38, 0x40, 0x42, 0x44],
            &0.5f32.to_le_bytes(),
            [2, 2],
        )
    }

    fn complete(
        &self,
        plan: &CheckpointReadPlan,
    ) -> (CheckpointReadPipeline, ReadTicket, ReadCompletion) {
        let mut pipeline = CheckpointReadPipeline::new(
            self.reader.clone(),
            ReadPipelineLimits {
                max_host_bytes: 128,
                max_read_slots: 2,
            },
        )
        .unwrap();
        let ticket = pipeline.submit(GENERATION, plan.clone()).unwrap();
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
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

#[test]
fn verified_slice_to_final_arc_matches_normal_read_for_both_scale_types() {
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
        let fixture = Fixture::new(encoding, &[0x38, 0x40, 0x42, 0x44], &scale, [2, 2]);
        let read = &fixture.reads[0];
        let normal = read.read(&fixture.reader).unwrap();
        let (pipeline, ticket, completion) = fixture.complete(read.read_plan());
        let before = proof_tests::counts();
        let artifact = NumericFp8Artifact::from_verified_read(read, &completion, &ticket).unwrap();
        assert_eq!(artifact, normal);
        assert_eq!(
            proof_tests::counts(),
            (before.0 + 1, before.1 + artifact.storage_bytes() as usize)
        );
        assert_eq!(
            artifact.validated_payload().weight_bytes().as_ptr(),
            artifact.weight_bytes().as_ptr()
        );
        assert_ne!(
            artifact.weight_bytes().as_ptr(),
            completion.payload().unwrap().as_ptr()
        );
        let alias = artifact.clone();
        assert_eq!(
            alias.weight_bytes().as_ptr(),
            artifact.weight_bytes().as_ptr()
        );
        assert_eq!(
            alias.scale_bytes().as_ptr(),
            artifact.scale_bytes().as_ptr()
        );
        assert_eq!(
            pipeline.budget().reserved_bytes,
            read.read_plan().storage_bytes()
        );
        drop(artifact);
        assert_eq!(
            pipeline.budget().reserved_bytes,
            read.read_plan().storage_bytes()
        );
        drop(alias);
        drop(completion);
        assert_eq!(pipeline.budget().reserved_bytes, 0);
    }
}

#[test]
fn three_projections_share_combined_completion_with_checked_packed_ranges() {
    let fixture = Fixture::f32();
    let (pipeline, ticket, completion) = fixture.complete(&fixture.combined);
    assert!(
        fixture.reads[0]
            .materialize_verified_read(&completion, &ticket)
            .is_err()
    );
    let before = proof_tests::counts().0;
    let artifacts = fixture
        .reads
        .iter()
        .enumerate()
        .map(|(index, read)| {
            NumericFp8Artifact::from_verified_read_range(
                read,
                &completion,
                &ticket,
                index * 8..(index + 1) * 8,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(proof_tests::counts().0, before + 3);
    for (artifact, read) in artifacts.iter().zip(&fixture.reads) {
        assert_eq!(artifact.provenance(), read);
        assert_eq!(artifact.weight_bytes(), [0x38, 0x40, 0x42, 0x44]);
        assert_eq!(artifact.scale_bytes(), &0.5f32.to_le_bytes());
    }
    assert_eq!(pipeline.budget().reserved_bytes, 24);
    drop(artifacts);
    assert_eq!(pipeline.budget().reserved_bytes, 24);
    drop(completion);
    assert_eq!(pipeline.budget().reserved_bytes, 0);
}

#[test]
fn wrong_unaligned_empty_reversed_and_out_of_bounds_ranges_fail_before_copy() {
    let fixture = Fixture::f32();
    let (pipeline, ticket, completion) = fixture.complete(&fixture.combined);
    let before = proof_tests::counts();
    for range in [0..0, Range { start: 8, end: 0 }, 1..9, 0..7, 0..25, 8..16] {
        assert!(
            fixture.reads[0]
                .materialize_verified_read_range(&completion, &ticket, range)
                .is_err()
        );
    }
    assert_eq!(proof_tests::counts(), before);
    assert_eq!(pipeline.budget().reserved_bytes, 24);
}

#[test]
fn nonfinite_weights_scales_and_dequantized_overflow_match_normal_rejection() {
    for (weight, scale) in [
        (0x7f, 0.5f32),
        (0xff, 0.5),
        (0x38, f32::NAN),
        (0x38, f32::INFINITY),
        (0x38, -1.0),
        (0x38, 0.0),
        (0x7e, f32::MAX),
    ] {
        let fixture = Fixture::new(
            NumericFp8Encoding::E4M3FnBlock128F32,
            &[weight; 4],
            &scale.to_le_bytes(),
            [2, 2],
        );
        let read = &fixture.reads[0];
        let (pipeline, ticket, completion) = fixture.complete(read.read_plan());
        assert!(completion.payload().is_some());
        assert!(read.read(&fixture.reader).is_err());
        assert!(NumericFp8Artifact::from_verified_read(read, &completion, &ticket).is_err());
        assert_eq!(pipeline.budget().reserved_bytes, 8);
        drop(completion);
        assert_eq!(pipeline.budget().reserved_bytes, 0);
    }
}

#[test]
fn bad_packed_lengths_type_and_layout_are_rejected_before_proof_construction() {
    let fixture = Fixture::f32();
    let read = &fixture.reads[0];
    let (pipeline, ticket, completion) = fixture.complete(read.read_plan());
    let packed = completion.payload().unwrap();
    let before = proof_tests::counts();
    for malformed in [&packed[..7], &[0x38; 9][..]] {
        assert!(read.materialize_packed(malformed).is_err());
    }
    let mut wrong_type = read.clone();
    wrong_type.source.encoding = NumericFp8Encoding::E4M3FnBlock128Bf16;
    assert!(
        wrong_type
            .materialize_verified_read(&completion, &ticket)
            .unwrap_err()
            .to_string()
            .contains("dtype")
    );
    let mut wrong_layout = read.clone();
    let mut scale = read.source.scale.clone();
    scale.offset = 0;
    scale.bytes = 8;
    scale.shape = vec![2, 1];
    wrong_layout.scale_read = fixture
        .reader
        .plan_2d_storage_range(&scale, 1..2, 0..1, &read.source.scale_source)
        .unwrap();
    assert!(
        wrong_layout
            .materialize_verified_read(&completion, &ticket)
            .unwrap_err()
            .to_string()
            .contains("global weight origins")
    );
    assert_eq!(proof_tests::counts(), before);
    assert_eq!(pipeline.budget().reserved_bytes, 8);
}

#[test]
fn freshness_is_rechecked_after_worker_boundary_and_after_numeric_validation() {
    for after_validation in [false, true] {
        let fixture = Fixture::f32();
        let read = &fixture.reads[0];
        let (pipeline, ticket, completion) = fixture.complete(read.read_plan());
        let path = fixture.paths[0].clone();
        let replace = move || {
            let replacement = path.with_extension("replacement");
            std::fs::copy(&path, &replacement).unwrap();
            std::fs::rename(replacement, path).unwrap();
        };
        let before = proof_tests::counts().0;
        if after_validation {
            AFTER_VALIDATION.with(|slot| *slot.borrow_mut() = Some(Box::new(replace)));
        } else {
            replace();
        }
        assert!(
            read.materialize_verified_read(&completion, &ticket)
                .unwrap_err()
                .to_string()
                .contains("stale")
        );
        assert_eq!(
            proof_tests::counts().0,
            before + usize::from(after_validation)
        );
        assert_eq!(pipeline.budget().reserved_bytes, 8);
    }
}

#[test]
fn sibling_source_change_invalidates_combined_conversion_before_or_after_copy() {
    for after_validation in [false, true] {
        let fixture = Fixture::f32();
        let (pipeline, ticket, completion) = fixture.complete(&fixture.combined);
        let path = fixture.paths[2].clone();
        let replace = move || {
            let replacement = path.with_extension("replacement");
            std::fs::copy(&path, &replacement).unwrap();
            std::fs::rename(replacement, path).unwrap();
        };
        if after_validation {
            AFTER_VALIDATION.with(|slot| *slot.borrow_mut() = Some(Box::new(replace)));
        } else {
            replace();
        }
        assert!(fixture.reads[0].source.validate_source_identity().is_ok());
        assert!(
            fixture.reads[0]
                .materialize_verified_read_range(&completion, &ticket, 0..8)
                .unwrap_err()
                .to_string()
                .contains("stale verified batch")
        );
        assert_eq!(pipeline.budget().reserved_bytes, 24);
    }
}

#[test]
fn failed_physical_read_is_not_a_numeric_verified_completion() {
    let fixture = Fixture::f32();
    std::fs::write(&fixture.paths[0], [0; 16]).unwrap();
    let read = &fixture.reads[0];
    let (pipeline, ticket, completion) = fixture.complete(read.read_plan());
    assert!(completion.payload().is_none());
    assert!(
        read.materialize_verified_read(&completion, &ticket)
            .unwrap_err()
            .to_string()
            .contains("not source verified")
    );
    assert_eq!(pipeline.budget().reserved_bytes, 8);
}

#[test]
fn ragged_noncontiguous_plan_keeps_original_layout_and_scale_origins() {
    let fixture = Fixture::new(
        NumericFp8Encoding::E4M3FnBlock128F32,
        &[0x38; 12],
        &0.5f32.to_le_bytes(),
        [3, 4],
    );
    let read = fixture.reads[0]
        .source
        .plan_tile(&fixture.reader, None, 1..3, 1..3)
        .unwrap();
    assert_eq!(read.read_plan().extents().len(), 3);
    let (_, ticket, completion) = fixture.complete(read.read_plan());
    let artifact = NumericFp8Artifact::from_verified_read(&read, &completion, &ticket).unwrap();
    assert_eq!(artifact.validated_payload().layout().row_origin, 1);
    assert_eq!(artifact.validated_payload().layout().column_origin, 1);
    assert_eq!(artifact.local_shape(), [2, 2]);
    assert_eq!(artifact, read.read(&fixture.reader).unwrap());
}

#[cfg(unix)]
use crate::checkpoint::metadata_test_support as metadata_support;

#[cfg(unix)]
#[test]
fn same_inode_restored_mtime_fails_before_conversion_and_after_numeric_validation() {
    for after_validation in [false, true] {
        let fixture = Fixture::f32();
        let read = &fixture.reads[0];
        let (pipeline, ticket, completion) = fixture.complete(read.read_plan());
        let path = fixture.paths[0].clone();
        let mutate = move || metadata_support::overwrite_restoring_mtime(&path);
        let before = proof_tests::counts().0;
        if after_validation {
            AFTER_VALIDATION.with(|slot| *slot.borrow_mut() = Some(Box::new(mutate)));
        } else {
            mutate();
        }
        assert!(
            read.materialize_verified_read(&completion, &ticket)
                .unwrap_err()
                .to_string()
                .contains("stale")
        );
        assert_eq!(
            proof_tests::counts().0,
            before + usize::from(after_validation)
        );
        assert_eq!(pipeline.budget().reserved_bytes, 8);
        assert_eq!(
            completion.receipt().source_generation,
            ticket.source_generation()
        );
        assert!(read.read(&fixture.reader).is_err());
    }
}

#[cfg(unix)]
#[test]
fn normal_numeric_read_cannot_publish_after_same_inode_restored_mtime_change() {
    let fixture = Fixture::f32();
    let read = &fixture.reads[0];
    let path = fixture.paths[0].clone();
    AFTER_VALIDATION.with(|slot| {
        *slot.borrow_mut() = Some(Box::new(move || {
            metadata_support::overwrite_restoring_mtime(&path);
        }))
    });
    let before = proof_tests::counts().0;
    assert!(
        read.read(&fixture.reader)
            .unwrap_err()
            .to_string()
            .contains("stale")
    );
    assert_eq!(proof_tests::counts().0, before + 1);
    assert!(read.source().validate_source_identity().is_err());
}
