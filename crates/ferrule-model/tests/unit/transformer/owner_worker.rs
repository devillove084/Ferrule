use super::*;
use crate::moe::ExpertId;

use ferrule_backend::cuda_plan::{NumericFp8Layout, NumericFp8LinearPlan, NumericFp8ScaleType};

const PACK_CACHE_CAP: usize = 4 * 1024 * 1024;

fn fixture_layouts(metadata: &ExpertMetadata) -> [NumericFp8Layout; 3] {
    metadata.parameters().each_ref().map(|p| NumericFp8Layout {
        n: p.spec().shape()[0],
        k: p.spec().shape()[1],
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::Bf16,
    })
}

fn reference_numeric_scratch(
    fixture: &Fixture,
    precision: NumericFp8Precision,
    max_group_rows: usize,
) -> usize {
    // Keep the small-fixture floor, but size the independent numeric workspace
    // for every projection's full tile and the largest group's conversion.
    let mut reservation = 4096;
    for id in 0..2 {
        let metadata = ExpertMetadata::new(0, id, fixture.parameters[id].clone(), None).unwrap();
        for layout in fixture_layouts(&metadata) {
            let bytes = match precision {
                NumericFp8Precision::Bf16RneF32Accumulate => max_group_rows
                    .checked_add(layout.n)
                    .and_then(|m| {
                        layout
                            .k
                            .div_ceil(8)
                            .checked_mul(8)
                            .and_then(|k| m.checked_mul(k))
                    })
                    .and_then(|n| n.checked_mul(2)),
                NumericFp8Precision::F32Tf32x3 => layout
                    .n
                    .checked_mul(layout.k)
                    .and_then(|n| n.checked_mul(4)),
                _ => panic!("fixture reference precision"),
            }
            .unwrap();
            let plan =
                NumericFp8LinearPlan::new(layout, max_group_rows, bytes, precision.backend())
                    .unwrap();
            assert_eq!(plan.workspace_requirements().bytes, bytes as u64);
            reservation =
                reservation.max(ferrule_backend::cuda_allocation_grant_bytes(bytes).unwrap());
        }
    }
    reservation
}

fn tokens(count: usize) -> Vec<ExpertToken> {
    (0..count)
        .map(|i| ExpertToken {
            transaction: ExecutionTransactionId::new(11).unwrap(),
            sequence: (i / 7) as u64,
            source_rank: ParallelRankId::new((i % 3 + 4) as u32),
            source_row: count - i - 1,
            route_slot: i % 2,
            expert: ExpertId::new(0, (i + 1) % 2),
            weight: if i % 2 == 0 { 0.0 } else { 0.75 },
            payload: (0..8)
                .map(|j| ((i * 7 + j * 3) % 31) as f32 * 0.013 - 0.12)
                .collect(),
        })
        .collect()
}

struct MixedProvider<'a> {
    inner: Provider<'a>,
    mixed: bool,
}

impl ExpertProvider for MixedProvider<'_> {
    fn expert_metadata(&mut self, layer: usize, id: usize) -> Result<Option<ExpertMetadata>> {
        if self.mixed && id == 0 {
            Ok(None)
        } else {
            self.inner.expert_metadata(layer, id)
        }
    }

    fn expert(&mut self, layer: usize, id: usize) -> Result<ExpertAvailability> {
        self.inner.expert(layer, id)
    }
}

fn placement() -> ExpertPlacement {
    ExpertPlacement::new([
        (0, 0, ParallelRankId::new(2)),
        (0, 1, ParallelRankId::new(2)),
    ])
    .unwrap()
}

#[test]
fn owner_worker_multichunk_fixture_workspace_budget_is_cpu_checked() {
    // Metadata/planning only: no CUDA owner, device, upload or tensor decoding.
    let fixture = Fixture::new();
    let mut provider = fixture.provider(true);
    let groups = [2051usize, 2052];
    let count = groups
        .iter()
        .try_fold(0usize, |n, &rows| n.checked_add(rows))
        .unwrap();
    let tokens = tokens(4103);
    assert_eq!(count, tokens.len());
    for (id, expected) in groups.into_iter().enumerate() {
        assert_eq!(
            tokens.iter().filter(|t| t.expert.expert == id).count(),
            expected
        );
    }
    let grant = |bytes| ferrule_backend::cuda_allocation_grant_bytes(bytes).unwrap();
    for precision in [
        NumericFp8Precision::F32Tf32x3,
        NumericFp8Precision::Bf16RneF32Accumulate,
    ] {
        let reservation = reference_numeric_scratch(&fixture, precision, 2052);
        assert_eq!(
            reservation,
            if precision == NumericFp8Precision::F32Tf32x3 {
                4096
            } else {
                131072
            }
        );
        for (id, rows) in groups.into_iter().enumerate() {
            let metadata = provider.expert_metadata(0, id).unwrap().unwrap();
            let layouts = fixture_layouts(&metadata);
            assert_eq!(layouts.map(|l| (l.n, l.k)), [(12, 8), (12, 8), (8, 12)]);
            for layout in layouts {
                let old = NumericFp8LinearPlan::new(layout, rows, 4096, precision.backend());
                if precision == NumericFp8Precision::Bf16RneF32Accumulate {
                    assert!(
                        old.unwrap_err()
                            .to_string()
                            .contains("scratch budget below activation bytes")
                    );
                } else {
                    old.unwrap();
                }
                let plan =
                    NumericFp8LinearPlan::new(layout, rows, reservation, precision.backend())
                        .unwrap();
                let workspace = plan
                    .activation_bytes()
                    .checked_add(plan.weight_tile_bytes())
                    .unwrap();
                assert_eq!(workspace as u64, plan.workspace_requirements().bytes);
                assert_eq!(plan.tile_rows(), layout.n);
                assert!(grant(workspace) <= reservation);
                if precision == NumericFp8Precision::Bf16RneF32Accumulate
                    && rows == 2052
                    && layout.k == 12
                {
                    assert_eq!(
                        (plan.activation_bytes(), plan.weight_tile_bytes(), workspace),
                        (65664, 256, 65920)
                    );
                }
            }
            let activation = SwiGluScratchPlan::for_metadata(&metadata, rows).unwrap();
            assert_eq!(
                activation.total_bytes().unwrap(),
                rows.checked_mul(208).unwrap()
            );
            let extra = activation.owner_bucket_extra_bytes(count, 8).unwrap();
            let reference = activation.total_with(0, reservation).unwrap();
            let packed = activation.total_with(extra, reservation).unwrap();
            let packed_grant = grant(count.checked_mul(8 * 4).unwrap());
            let group_grants = [8usize, 12, 12, 12, 8]
                .into_iter()
                .try_fold(0usize, |sum, width| {
                    sum.checked_add(grant(
                        rows.checked_mul(width).unwrap().checked_mul(4).unwrap(),
                    ))
                })
                .unwrap();
            assert_eq!((packed_grant, group_grants), (262144, 655360));
            let reference_grants = group_grants.checked_add(reservation).unwrap();
            assert!(reference <= reference_grants);
            assert_eq!(packed, reference_grants.checked_add(packed_grant).unwrap());
            assert_eq!(metadata.expected_device_bytes(), 294);
            let cache = crate::transformer::standard::expert_cache::Cache::<usize>::new(
                ExpertCacheLimits {
                    max_experts: 1,
                    max_bytes: PACK_CACHE_CAP,
                },
            )
            .unwrap();
            cache
                .preflight(metadata.expected_device_bytes(), reference)
                .unwrap();
            cache
                .preflight(metadata.expected_device_bytes(), packed)
                .unwrap();
            let charged = packed
                .checked_add(metadata.expected_device_bytes())
                .unwrap();
            assert_eq!(
                charged,
                if precision == NumericFp8Precision::F32Tf32x3 {
                    921894
                } else {
                    1048870
                }
            );
            // Cache weights use the existing compressed-payload contract. Even
            // their physical allocation grants fit, without increasing the cap.
            let weight_grants = layouts
                .into_iter()
                .try_fold(0usize, |sum, layout| {
                    let (weight, scale) = layout.storage_lengths().unwrap();
                    sum.checked_add(grant(weight))?.checked_add(grant(scale))
                })
                .unwrap();
            assert_eq!(weight_grants, 1536);
            assert!(packed.checked_add(weight_grants).unwrap() < PACK_CACHE_CAP);
        }
    }
    assert_eq!(provider.calls, 0);
}

// The existing pre-packing fixture schedule, shared by correctness and A/B.
// Timers wrap the whole call, never individual Boundary or CUDA operations.
fn per_expert_download_reference(
    reference: &mut CudaStandardDecoderOperators,
    provider: &mut dyn ExpertProvider,
    tokens: &[ExpertToken],
) -> (Vec<ExpertResult>, usize) {
    let mut expected = vec![Vec::new(); tokens.len()];
    let mut logical_exports = 0;
    for id in 0..2 {
        let slots = tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| t.expert.expert == id)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let shape = RowsShape::new(slots.len(), 8).unwrap();
        let input = reference
            .bind_rows(Rows::Host(
                HostRows::new(
                    shape,
                    RowsDType::F32,
                    None,
                    slots
                        .iter()
                        .flat_map(|&i| tokens[i].payload.iter().copied())
                        .collect(),
                )
                .unwrap(),
            ))
            .unwrap();
        let ExpertAvailability::Ready(expert) = provider.expert(0, id).unwrap() else {
            panic!("ready fixture")
        };
        let values = reference
            .synchronous(|this| {
                this.swiglu_consume(&expert, &input, None, |this, output| {
                    logical_exports += 1;
                    this.boundary.download(&this.ops, this.f32(&output)?)
                })
            })
            .unwrap();
        let (rows, remainder) = values.as_chunks::<8>();
        assert!(remainder.is_empty());
        assert_eq!(rows.len(), slots.len());
        for (&slot, row) in slots.iter().zip(rows) {
            expected[slot] = row.to_vec();
        }
    }
    let results = tokens
        .iter()
        .zip(expected)
        .map(|(token, output)| ExpertResult::from_token(token, ParallelRankId::new(2), output))
        .collect();
    (results, logical_exports)
}

#[test]
#[ignore = "requires CUDA; two same-owner experts, multi-chunk inputs/output, exact token order"]
fn owner_worker_packs_once_and_matches_per_expert_downloads() {
    let fixture = Fixture::new();
    let placement = placement();
    // Each expert input is >64 KiB; the combined output spans three chunks.
    let mut tokens = tokens(4103);
    // Arrival order differs from both expert order and source-row order.
    tokens.rotate_left(17);
    tokens.swap(3, 4099);
    let chunk = crate::transformer::parallel_transfer::DEFAULT_TRANSFER_CONFIG.chunk_elements;
    assert_eq!(chunk * 4, 64 * 1024);
    assert!(tokens.len() / 2 * 8 > chunk);
    assert_eq!((tokens.len() * 8).div_ceil(chunk), 3);
    let max_group_rows = (0..2)
        .map(|id| tokens.iter().filter(|t| t.expert.expert == id).count())
        .max()
        .unwrap();
    for precision in [
        NumericFp8Precision::F32Tf32x3,
        NumericFp8Precision::Bf16RneF32Accumulate,
    ] {
        let numeric_scratch = reference_numeric_scratch(&fixture, precision, max_group_rows);
        let mut reference = fixture.owner_with_scratch(precision, PACK_CACHE_CAP, numeric_scratch);
        let mut provider = fixture.provider(false);
        let (expected, exports) =
            per_expert_download_reference(&mut reference, &mut provider, &tokens);
        assert_eq!(exports, 2);
        for (metadata, mixed) in [(false, false), (true, false), (true, true)] {
            let mut owner = fixture.owner_with_scratch(precision, PACK_CACHE_CAP, numeric_scratch);
            let mut provider = MixedProvider {
                inner: fixture.provider(metadata),
                mixed,
            };
            let before = owner.consumer_proofs;
            let mut worker = CudaExpertWorker::new(
                ParallelRankId::new(2),
                &placement,
                &mut provider,
                &mut owner,
            );
            worker.pack_step_hook = Some(check_pack_grants);
            let actual = worker.compute(&tokens).unwrap();
            assert_eq!(worker.logical_downloads, 1);
            assert_eq!(worker.packed_groups, 2);
            assert_eq!(worker.pack_copies, 2);
            assert!(worker.compute(&[]).unwrap().is_empty());
            assert_eq!(worker.logical_downloads, 1, "empty buckets do not export");
            assert_eq!(actual.len(), tokens.len());
            for ((result, token), expected) in actual.iter().zip(&tokens).zip(&expected) {
                assert_eq!(
                    *result,
                    ExpertResult::from_token(token, ParallelRankId::new(2), result.output.clone())
                );
                close(&result.output, &expected.output);
            }
            // Retain all numeric/lease proofs. Only the two bind_rows boundaries
            // and one of the two old outer execution boundaries disappear.
            assert_eq!(
                owner.consumer_proofs - before,
                reference.consumer_proofs - 3
            );
            let stats = owner.expert_cache_stats().unwrap();
            assert_eq!(stats.resident_experts, 1);
            assert_eq!(stats.evictions, 1);
            assert_eq!(stats.uploads, 2);
            assert_eq!(stats.scratch_bytes, numeric_scratch);
            assert!(owner.route_hold.is_empty() && owner.expert_hold.is_empty());
            assert!(!owner.needs_quarantine());
        }
    }
}

#[test]
fn owner_worker_group_order_unpack_is_cpu_checked() {
    let owner = ParallelRankId::new(2);
    for count in [1, 18, 4103] {
        let mut input = tokens(count);
        input.reverse();
        let mut groups = BTreeMap::new();
        let expected = input
            .iter()
            .enumerate()
            .map(|(slot, token)| {
                groups
                    .entry(token.expert)
                    .or_insert_with(Vec::new)
                    .push(slot);
                let output = (0..8)
                    .map(|column| {
                        if column == 0 && slot % 2 == 0 {
                            -0.0
                        } else {
                            (slot * 8 + column) as f32
                        }
                    })
                    .collect::<Vec<_>>();
                ExpertResult::from_token(token, owner, output)
            })
            .collect::<Vec<_>>();
        let shape = RowsShape::new(count, 8).unwrap();
        // Model exactly one contiguous range copy for each group, including
        // ranges crossing transport chunk boundaries. The production restore
        // runs below; this does not pretend to exercise CUDA submission.
        let mut packed = vec![0.0; shape.elements()];
        let mut offset = 0;
        for slots in groups.values() {
            let source = slots
                .iter()
                .flat_map(|&slot| expected[slot].output.iter().copied())
                .collect::<Vec<_>>();
            let end = offset + source.len();
            packed[offset..end].copy_from_slice(&source);
            offset = end;
        }
        assert_eq!(offset, shape.elements());
        let actual = restore_owner_bucket_results(&input, &groups, &packed, shape, owner).unwrap();
        assert_eq!(actual, expected);
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(
                actual
                    .output
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .output
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        assert!(
            restore_owner_bucket_results(
                &input,
                &groups,
                &packed[..packed.len() - 1],
                shape,
                owner
            )
            .is_err()
        );
        packed[0] = f32::NAN;
        assert!(restore_owner_bucket_results(&input, &groups, &packed, shape, owner).is_err());
    }
}

const AB_TIMING_ENV: &str = "FERRULE_OWNER_BUCKET_AB_TIMING";
const AB_WARMUP: usize = 3;
const AB_MEASURE: usize = 10;

fn ab_tokens(count: usize) -> Vec<ExpertToken> {
    let mut input = tokens(count);
    input.rotate_left(17);
    input.swap(3, count - 4);
    input
}

// Activation-only transfer counts inferred from the existing chunk contract;
// not CUDA counter measurements (Boundary does not expose transport stats).
fn ab_transfer_layout(input: &[ExpertToken]) -> (usize, usize, usize) {
    let chunk = crate::transformer::parallel_transfer::DEFAULT_TRANSFER_CONFIG.chunk_elements;
    let group_chunks = (0..2)
        .map(|id| {
            input
                .iter()
                .filter(|t| t.expert.expert == id)
                .count()
                .checked_mul(8)
                .unwrap()
                .div_ceil(chunk)
        })
        .sum();
    let elements = input.len().checked_mul(8).unwrap();
    (
        elements.checked_mul(4).unwrap(),
        group_chunks,
        elements.div_ceil(chunk),
    )
}

#[test]
fn owner_worker_ab_workloads_are_cpu_checked() {
    let fixture = Fixture::new();
    let chunk = crate::transformer::parallel_transfer::DEFAULT_TRANSFER_CONFIG.chunk_elements;
    for count in [18, 4103] {
        let input = ab_tokens(count);
        let groups = [0, 1].map(|id| input.iter().filter(|t| t.expert.expert == id).count());
        assert_eq!(groups, if count == 18 { [9, 9] } else { [2051, 2052] });
        for (i, original) in tokens(count).iter().enumerate() {
            let matches = input
                .iter()
                .filter(|t| t.source_row == count - i - 1)
                .collect::<Vec<_>>();
            assert_eq!(matches.len(), 1);
            assert_eq!(matches[0].payload, original.payload);
            assert_eq!(matches[0].expert, original.expert);
        }
        let (bytes, before_chunks, after_chunks) = ab_transfer_layout(&input);
        assert_eq!(bytes, if count == 18 { 576 } else { 131296 });
        assert_eq!(
            before_chunks,
            groups
                .map(|n| (n * 8).div_ceil(chunk))
                .iter()
                .sum::<usize>()
        );
        assert_eq!(after_chunks, (count * 8).div_ceil(chunk));
        for precision in [
            NumericFp8Precision::F32Tf32x3,
            NumericFp8Precision::Bf16RneF32Accumulate,
        ] {
            let reservation =
                reference_numeric_scratch(&fixture, precision, *groups.iter().max().unwrap());
            assert_eq!(
                reservation,
                if count == 4103 && precision == NumericFp8Precision::Bf16RneF32Accumulate {
                    131072
                } else {
                    4096
                }
            );
            for (id, rows) in groups.into_iter().enumerate() {
                let metadata =
                    ExpertMetadata::new(0, id, fixture.parameters[id].clone(), None).unwrap();
                for layout in fixture_layouts(&metadata) {
                    NumericFp8LinearPlan::new(layout, rows, reservation, precision.backend())
                        .unwrap();
                }
                let plan = SwiGluScratchPlan::for_metadata(&metadata, rows).unwrap();
                let packed = plan
                    .total_with(
                        plan.owner_bucket_extra_bytes(count, 8).unwrap(),
                        reservation,
                    )
                    .unwrap();
                assert!(
                    packed
                        .checked_add(metadata.expected_device_bytes())
                        .unwrap()
                        < PACK_CACHE_CAP
                );
            }
        }
    }
}

fn ab_wall<T>(run: impl FnOnce() -> T) -> (T, std::time::Duration) {
    let start = std::time::Instant::now();
    let value = run();
    (value, start.elapsed())
}

#[test]
#[ignore = "requires CUDA; opt-in matched A/B via FERRULE_OWNER_BUCKET_AB_TIMING=1"]
fn owner_worker_matched_ab_correctness_and_timing() {
    // Even an unfiltered --ignored run must not opt CI into this timing loop.
    if std::env::var_os(AB_TIMING_ENV).as_deref() != Some(std::ffi::OsStr::new("1")) {
        println!("owner_bucket_ab skipped: set {AB_TIMING_ENV}=1 explicitly");
        return;
    }
    let fixture = Fixture::new();
    let placement = placement();
    for count in [18, 4103] {
        let input = ab_tokens(count);
        let groups = [0, 1].map(|id| input.iter().filter(|t| t.expert.expert == id).count());
        let (bytes, group_chunks, packed_chunks) = ab_transfer_layout(&input);
        let chunk_bytes =
            crate::transformer::parallel_transfer::DEFAULT_TRANSFER_CONFIG.chunk_elements * 4;
        for precision in [
            NumericFp8Precision::F32Tf32x3,
            NumericFp8Precision::Bf16RneF32Accumulate,
        ] {
            let numeric_scratch =
                reference_numeric_scratch(&fixture, precision, *groups.iter().max().unwrap());
            let mut reference =
                fixture.owner_with_scratch(precision, PACK_CACHE_CAP, numeric_scratch);
            let mut packed = fixture.owner_with_scratch(precision, PACK_CACHE_CAP, numeric_scratch);
            let mut before_provider = fixture.provider(false);
            let mut after_provider = fixture.provider(false);
            let mut samples = Vec::with_capacity(AB_MEASURE);
            let mut golden: Option<Vec<ExpertResult>> = None;

            println!(
                "owner_bucket_ab config profile={precision:?} tokens={count} groups={groups:?} owner=2 experts=2 max_experts=1 cache_cap={PACK_CACHE_CAP} numeric_reservation={numeric_scratch} warmup={AB_WARMUP} measure={AB_MEASURE} chunk_bytes={chunk_bytes} provider=legacy packing_layout=expert_contiguous_host_restore fixture_setup_and_result_comparison_outside_timer=true cpu_wall_includes_completion=true transfer_counts=layout_derived activation_h2d_bytes_total={bytes} activation_h2d_chunks_total={group_chunks} before_d2h_bytes_layout={bytes} before_d2h_chunks_layout={group_chunks} after_d2h_bytes_layout={bytes} after_d2h_chunks_layout={packed_chunks}"
            );
            for iteration in 0..AB_WARMUP + AB_MEASURE {
                let mut before = || {
                    ab_wall(|| {
                        per_expert_download_reference(&mut reference, &mut before_provider, &input)
                    })
                };
                let mut after = || {
                    ab_wall(|| {
                        let mut worker = CudaExpertWorker::new(
                            ParallelRankId::new(2),
                            &placement,
                            &mut after_provider,
                            &mut packed,
                        );
                        let result = worker.compute(&input).unwrap();
                        (result, worker.logical_downloads, worker.pack_copies)
                    })
                };

                // Alternate order to avoid always giving B the second call.
                let (a, b) = if iteration % 2 == 0 {
                    (before(), after())
                } else {
                    let b = after();
                    (before(), b)
                };

                let ((expected, before_exports), before_wall) = a;
                let ((actual, after_exports, pack_copies), after_wall) = b;
                assert_eq!(pack_copies, 2);

                assert_eq!((before_exports, after_exports), (2, 1));
                assert_eq!(expected.len(), input.len());
                assert_eq!(actual.len(), input.len());
                for ((a, b), token) in expected.iter().zip(&actual).zip(&input) {
                    assert_eq!(
                        *a,
                        ExpertResult::from_token(token, ParallelRankId::new(2), a.output.clone())
                    );
                    assert_eq!(
                        *b,
                        ExpertResult::from_token(token, ParallelRankId::new(2), b.output.clone())
                    );
                    close(&a.output, &b.output);
                }
                if let Some(golden) = &golden {
                    for (a, original) in expected.iter().zip(golden) {
                        close(&a.output, &original.output);
                    }
                } else {
                    golden = Some(expected);
                }

                for owner in [&reference, &packed] {
                    assert!(!owner.needs_quarantine());
                    assert!(owner.route_hold.is_empty() && owner.expert_hold.is_empty());
                    assert_eq!(
                        owner.expert_cache_stats().unwrap().scratch_bytes,
                        numeric_scratch
                    );
                }
                if iteration >= AB_WARMUP {
                    let sample = iteration - AB_WARMUP;
                    println!(
                        "owner_bucket_ab sample profile={precision:?} tokens={count} sample={sample} before_first={} before_cpu_wall_ns={} after_cpu_wall_ns={} before_logical_exports={before_exports} after_logical_exports={after_exports} after_d2d_submissions={pack_copies} before_d2h_bytes_layout={bytes} before_d2h_chunks_layout={group_chunks} after_d2h_bytes_layout={bytes} after_d2h_chunks_layout={packed_chunks} correctness=passed",
                        iteration % 2 == 0,
                        before_wall.as_nanos(),
                        after_wall.as_nanos()
                    );
                    samples.push([before_wall.as_nanos(), after_wall.as_nanos()]);
                }
            }
            assert_eq!(samples.len(), AB_MEASURE);
            let sums = samples
                .iter()
                .fold([0u128; 2], |s, t| [s[0] + t[0], s[1] + t[1]]);
            println!(
                "owner_bucket_ab summary profile={precision:?} tokens={count} samples={} before_cpu_wall_total_ns={} after_cpu_wall_total_ns={} before_logical_exports={} after_logical_exports={} fixture_only=true throughput_claim=false",
                samples.len(),
                sums[0],
                sums[1],
                AB_MEASURE * 2,
                AB_MEASURE
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA; allocator-grant admission rejects before payload IO or GPU work"]
fn owner_worker_budget_and_invalid_payload_reject_before_io() {
    let fixture = Fixture::new();
    let placement = placement();
    let mut tokens = tokens(18);
    // The larger group sorts last: admitting the first group is not sufficient.
    for token in &mut tokens {
        token.expert = ExpertId::new(0, 1);
    }
    tokens[7].expert = ExpertId::new(0, 0);
    let activation = SwiGluScratchPlan::for_descriptor(
        &crate::transformer::SwiGlu::new(8, 12, false).unwrap(),
        17,
    )
    .unwrap();
    let cap = 294
        + 4096
        + activation.total_bytes().unwrap()
        + activation
            .owner_bucket_extra_bytes(tokens.len(), 8)
            .unwrap();
    for (limit, admitted) in [(cap - 1, false), (cap, true)] {
        let mut owner = fixture.owner(NumericFp8Precision::F32Tf32x3, limit);
        let before = owner.ops.counters();
        let allocations = owner.ops.allocator_metrics();
        let stats = owner.expert_cache_stats();
        let mut provider = fixture.provider(true);
        let mut worker = CudaExpertWorker::new(
            ParallelRankId::new(2),
            &placement,
            &mut provider,
            &mut owner,
        );
        assert_eq!(worker.compute(&tokens).is_ok(), admitted);
        assert_eq!(worker.logical_downloads, usize::from(admitted));
        assert_eq!(worker.packed_groups, if admitted { 2 } else { 0 });
        assert_eq!(worker.pack_copies, if admitted { 2 } else { 0 });
        if !admitted {
            assert_eq!(provider.calls, 0);
            assert_eq!(owner.ops.counters(), before);
            assert_eq!(owner.ops.allocator_metrics(), allocations);
            assert_eq!(owner.expert_cache_stats(), stats);
        }
        assert!(!owner.needs_quarantine());
    }
    let mut owner = fixture.owner(NumericFp8Precision::F32Tf32x3, cap);
    for invalid in 0..3 {
        let mut bad = tokens.clone();
        match invalid {
            0 => bad[5].payload[0] = f32::NAN,
            1 => {
                bad[5].payload.pop();
            }
            _ => bad[5].expert = ExpertId::new(0, 7),
        }
        let before = owner.ops.counters();
        let mut provider = fixture.provider(true);
        assert!(
            CudaExpertWorker::new(
                ParallelRankId::new(2),
                &placement,
                &mut provider,
                &mut owner
            )
            .compute(&bad)
            .is_err()
        );
        assert_eq!(provider.calls, 0);
        assert_eq!(owner.ops.counters(), before);
    }
}

#[test]
#[ignore = "requires CUDA; lost post-D2D proof keeps exact lease and all operands quarantined"]
fn owner_worker_unknown_d2d_retains_lease_source_input_and_packed_output() {
    let fixture = Fixture::new();
    let placement = placement();
    let tokens = tokens(6);
    let mut owner = fixture.owner(NumericFp8Precision::F32Tf32x3, 32768);
    let mut provider = fixture.provider(true);
    let mut worker = CudaExpertWorker::new(
        ParallelRankId::new(2),
        &placement,
        &mut provider,
        &mut owner,
    );
    worker.lose_pack_proof = true;
    let error = worker.compute(&tokens).unwrap_err();
    assert_eq!(
        worker.packed_groups, 1,
        "second expert must not evict the unknown lease"
    );
    assert_eq!(worker.logical_downloads, 0);
    assert_eq!(worker.pack_copies, 1);
    assert!(
        crate::transformer::parallel_transfer::CudaShardError::from_error(&error)
            .unwrap()
            .needs_quarantine()
    );
    assert!(owner.needs_quarantine());
    let frozen = owner.expert_cache_stats().unwrap();
    assert_eq!(frozen.pending_upload_bytes, 294);
    assert_eq!(frozen.evictions, 0);
    assert_eq!(frozen.uploads, 0);
    assert_eq!(frozen.unknown_quarantine_bytes, frozen.charged_bytes());
    assert_eq!(
        owner.route_hold.len(),
        2,
        "packed destination and current input"
    );
    assert!(!owner.expert_hold.is_empty(), "D2D source custody");
    assert!(owner.bindings.expert_admission.is_some());
    let counters = owner.ops.counters();
    assert!(
        CudaExpertWorker::new(
            ParallelRankId::new(2),
            &placement,
            &mut provider,
            &mut owner
        )
        .compute(&tokens)
        .is_err()
    );
    assert_eq!(owner.ops.counters(), counters);
    assert!(owner.quiesce().is_err());
    assert_eq!(owner.expert_cache_stats().unwrap(), frozen);
    let ops = Rc::clone(owner.operators());
    let live = ops.allocator_metrics().live_granted_bytes;
    drop(owner);
    assert!(ops.allocator_metrics().live_granted_bytes >= live);
    partial_d2d_submission_failure_matrix(&fixture, &placement, &tokens);
}

fn check_pack_grants(
    owner: &mut CudaStandardDecoderOperators,
    _group: usize,
    row: usize,
) -> Result<()> {
    if row != 0 {
        return Ok(());
    }
    // Count allocation roots, not duplicate views. All six ordinary F32
    // allocations are live here; the numeric workspace has its own reservation.
    let mut roots = BTreeMap::new();
    for buffer in owner.route_hold.iter().chain(&owner.expert_hold) {
        let bytes = buffer.len().checked_mul(4).unwrap();
        roots
            .entry(buffer.cu_deviceptr())
            .and_modify(|n: &mut usize| *n = (*n).max(bytes))
            .or_insert(bytes);
    }
    assert_eq!(roots.len(), 6, "packed/input/gate/up/product/output roots");
    let grants = roots
        .values()
        .try_fold(0usize, |total, &bytes| {
            total.checked_add(ferrule_backend::cuda_allocation_grant_bytes(bytes).unwrap())
        })
        .unwrap();
    let reserved = owner.numeric_workspace_stats().unwrap().reserved_bytes;
    assert_eq!(
        owner.expert_cache_stats().unwrap().scratch_bytes,
        grants + reserved
    );
    Ok(())
}

#[derive(Debug)]
struct PackSubmissionFailure {
    // Exact allocation identities observed before the injected submission error.
    roots: [u64; 3],
}

impl std::fmt::Display for PackSubmissionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "injected owner D2D submission failure")
    }
}

impl std::error::Error for PackSubmissionFailure {}

fn fail_pack(
    owner: &mut CudaStandardDecoderOperators,
    group: usize,
    step: usize,
    target: usize,
    fail_step: usize,
    unknown: bool,
) -> Result<()> {
    check_pack_grants(owner, group, step)?;
    if group != target || step != fail_step {
        return Ok(());
    }
    // Step 0 fails before D2D; step 1 fails after the whole group is queued.
    // The previous group, if any, has its proof but shares the packed root.
    assert_eq!(owner.route_hold.len(), 2);
    assert!(owner.bindings.expert_admission.is_some());
    let source = owner
        .expert_hold
        .last()
        .expect("D2D source pinned before copy");
    assert_eq!(source.len(), 3 * 8);
    assert_eq!(owner.route_hold[0].len(), 6 * 8);
    assert_eq!(owner.route_hold[1].len(), 3 * 8);
    let roots = [
        owner.route_hold[0].cu_deviceptr(),
        owner.route_hold[1].cu_deviceptr(),
        source.cu_deviceptr(),
    ];
    owner.lose_next_consumer_proof = unknown;
    Err(ferrule_common::Error::Backend {
        source: Box::new(PackSubmissionFailure { roots }),
    })
}

fn partial_d2d_submission_failure_matrix(
    fixture: &Fixture,
    placement: &ExpertPlacement,
    tokens: &[ExpertToken],
) {
    let hooks: [(usize, usize, bool, PackStepHook); 8] = [
        (0, 0, false, |o, g, s| fail_pack(o, g, s, 0, 0, false)),
        (0, 0, true, |o, g, s| fail_pack(o, g, s, 0, 0, true)),
        (0, 1, false, |o, g, s| fail_pack(o, g, s, 0, 1, false)),
        (0, 1, true, |o, g, s| fail_pack(o, g, s, 0, 1, true)),
        (1, 0, false, |o, g, s| fail_pack(o, g, s, 1, 0, false)),
        (1, 0, true, |o, g, s| fail_pack(o, g, s, 1, 0, true)),
        (1, 1, false, |o, g, s| fail_pack(o, g, s, 1, 1, false)),
        (1, 1, true, |o, g, s| fail_pack(o, g, s, 1, 1, true)),
    ];
    for metadata in [false, true] {
        for (completed, submitted, unknown, hook) in hooks {
            let mut owner = fixture.owner(NumericFp8Precision::F32Tf32x3, 32768);
            let mut provider = fixture.provider(metadata);
            let mut worker =
                CudaExpertWorker::new(ParallelRankId::new(2), placement, &mut provider, &mut owner);
            worker.pack_step_hook = Some(hook);
            let error = worker.compute(tokens).unwrap_err();
            assert_eq!(worker.packed_groups, completed);
            assert_eq!(worker.pack_copies, completed + submitted);
            assert_eq!(
                worker.logical_downloads, 0,
                "never publish a partial bucket"
            );
            let mut primary: &(dyn std::error::Error + 'static) = &error;
            while let Some(source) = std::error::Error::source(primary) {
                primary = source;
            }
            let failure = primary
                .downcast_ref::<PackSubmissionFailure>()
                .expect("cleanup must not replace the typed submission error");
            let stats = owner.expert_cache_stats().unwrap();
            assert_eq!(stats.uploads, completed as u64);
            assert_eq!(stats.evictions, completed as u64);
            assert_eq!(owner.needs_quarantine(), unknown);
            if unknown {
                let shard =
                    crate::transformer::parallel_transfer::CudaShardError::from_error(&error)
                        .expect("typed Unknown result");
                assert!(shard.needs_quarantine());
                assert!(!shard.is_quiescent());
                assert_eq!(stats.pending_upload_bytes, 294);
                assert_eq!(stats.unknown_quarantine_bytes, stats.charged_bytes());
                assert!(owner.bindings.expert_admission.is_some());
                assert_eq!(owner.route_hold.len(), 2);
                assert_eq!(owner.route_hold[0].cu_deviceptr(), failure.roots[0]);
                assert_eq!(owner.route_hold[1].cu_deviceptr(), failure.roots[1]);
                assert!(
                    owner
                        .expert_hold
                        .iter()
                        .any(|b| b.cu_deviceptr() == failure.roots[2])
                );
                let counters = owner.ops.counters();
                assert!(owner.quiesce().is_err());
                assert_eq!(owner.ops.counters(), counters);
                assert_eq!(owner.expert_cache_stats().unwrap(), stats);
                let ops = Rc::clone(owner.operators());
                let live = ops.allocator_metrics().live_granted_bytes;
                drop(owner);
                assert!(ops.allocator_metrics().live_granted_bytes >= live);
            } else {
                assert_eq!(stats.pending_upload_bytes, 0);
                assert_eq!(stats.scratch_bytes, 4096);
                assert!(owner.bindings.expert_admission.is_none());
                assert!(owner.route_hold.is_empty() && owner.expert_hold.is_empty());
                let result = CudaExpertWorker::new(
                    ParallelRankId::new(2),
                    placement,
                    &mut provider,
                    &mut owner,
                )
                .compute(tokens)
                .unwrap();
                assert_eq!(result.len(), tokens.len(), "known failure permits retry");
            }
        }
    }
}
