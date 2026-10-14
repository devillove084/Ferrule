use super::*;
#[path = "hybrid_mixed_production_diagnostic.rs"]
mod production_diagnostic;
use ferrule_common::{ParallelGroupId, execution::*};
use ferrule_model::{
    ModelFamily, WeightSource, checkpoint::*, decoder::*, execution::ExecutionPrecisionPolicy,
    nn::*,
};
use ferrule_runtime::parallel::{
    collective::HostCollectiveLimits,
    pipeline::tensor_scope::*,
    tensor::decoder_collective::{
        DecoderTensorCollective, MixedRankCollective, MixedRankCollectivePlan,
    },
};
use std::{
    sync::{Condvar, Mutex, atomic::AtomicBool},
    time::{Duration, Instant},
};

// A failed peer must not leave the ignored gate blocked in an unbounded barrier.
struct Barrier {
    state: Mutex<(usize, u64)>,
    wake: Condvar,
}
impl Barrier {
    fn new(_: usize) -> Self {
        Self {
            state: Mutex::new((0, 0)),
            wake: Condvar::new(),
        }
    }
    fn wait(&self) {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut state = self.state.lock().unwrap();
        let generation = state.1;
        state.0 += 1;
        if state.0 == 2 {
            state.0 = 0;
            state.1 += 1;
            self.wake.notify_all();
            return;
        }
        while state.1 == generation {
            let left = deadline.saturating_duration_since(Instant::now());
            assert!(!left.is_zero(), "GPU fixture peer failed/timed out");
            let (next, _) = self.wake.wait_timeout(state, left).unwrap();
            state = next;
        }
    }
}

pub(super) fn native_resources(f: &base::Fixture) -> BoundDecoderResources {
    native_resources_from(f, f.resources())
}
fn native_resources_from(
    f: &base::Fixture,
    original: BoundDecoderResources,
) -> BoundDecoderResources {
    let path = f.dir.join("mixed-layer.bf16");
    let mut data = Vec::new();
    let mut slices = Vec::new();
    let mut schema = StateDictSchema::builder();
    let mut mapper = ExactNameMapper::new();
    let reader = CheckpointTensorReader::new(1 << 20);
    for (i, b) in original.state_dict().parameters().iter().enumerate() {
        let source = b.weight().slice();
        let offset = data.len();
        let f32_vector = matches!(
            b.role(),
            ferrule_model::TensorRole::LinearAttentionNorm
                | ferrule_model::TensorRole::LinearAttentionALog
        );
        for value in reader.read_slice(source).unwrap().bytes.as_chunks::<4>().0 {
            if f32_vector {
                data.extend(*value);
            } else {
                data.extend(
                    ferrule_model::execution::bf16_rne_word(f32::from_le_bytes(*value))
                        .to_le_bytes(),
                );
            }
        }
        let module = ModulePath::new(&source.name).unwrap();
        schema
            .register_with_role(
                ParameterSpec::new(
                    ParameterId::new(i as u64 + 1),
                    module.clone(),
                    if f32_vector {
                        ParameterDType::F32
                    } else {
                        ParameterDType::Bf16
                    },
                    source.shape.clone(),
                    b.residency().clone(),
                )
                .unwrap(),
                b.role().clone(),
            )
            .unwrap();
        mapper
            .insert(&source.name, NameMapping::weight(module))
            .unwrap();
        slices.push(CheckpointTensorSlice {
            name: source.name.clone(),
            role: source.role.clone(),
            path: path.clone(),
            offset: offset as u64,
            bytes: (data.len() - offset) as u64,
            dtype: if f32_vector {
                CheckpointDType::F32
            } else {
                CheckpointDType::Bf16
            },
            shape: source.shape.clone(),
        });
    }
    std::fs::write(path, data).unwrap();
    let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
        .bind_slices(slices)
        .unwrap();
    BoundDecoderResources::new(original.spec().clone(), Arc::new(bound)).unwrap()
}
#[test]
#[ignore = "requires explicit two-GPU grant and CUBLAS_WORKSPACE_CONFIG=:0:0; complete GQA+dense BF16 block only"]
fn frozen_tp2_complete_gqa_dense_bf16_journal_block() {
    run_complete_block(1..2, false, false);
}
#[test]
#[ignore = "requires explicit two-GPU grant and CUBLAS_WORKSPACE_CONFIG=:0:0; GQA+GDN pair only"]
fn frozen_tp2_complete_gqa_gdn_bf16_journal_pair() {
    run_complete_block(1..3, false, false);
}
#[test]
#[ignore = "requires explicit two-GPU grant; generic three-layer dense program, two compact KV layers"]
fn frozen_tp2_generic_gqa_gdn_gqa_bf16_program() {
    run_complete_block(1..4, false, false);
}
#[path = "hybrid_mixed_moe_gpu.rs"]
mod moe;
#[test]
#[ignore = "two GPUs; production runtime typed route/data/control adapter"]
fn frozen_tp2_generic_moe_top8_bf16_program() {
    run_complete_block(1..4, true, false);
}
#[test]
#[ignore = "two GPUs; replicated embedding, all generic layers, ragged BF16 vocab gather; production route adapter"]
fn frozen_tp2_tokens_moe_logits_bf16_program() {
    run_complete_block(0..3, true, true);
}
fn run_complete_block(range: std::ops::Range<usize>, routed: bool, boundaries: bool) {
    let recurrent = range.contains(&2);
    let suffix = range.end == 4 || boundaries;
    assert_eq!(
        std::env::var("FERRULE_RUN_MIXED_RANK_GPU").as_deref(),
        Ok("1")
    );
    let (fixture, r) = if boundaries {
        moe::boundary_resources()
    } else if routed {
        moe::resources()
    } else {
        let f = base::Fixture::for_degree(2, true);
        let r = native_resources(&f);
        (f, r)
    };
    let r = Arc::new(r);
    let p = parallel(&r, range.clone(), 2);
    let endpoints = DecoderTensorCollective::new_group_from_members(
        p.tensor_members(),
        ParallelTopologyId::new(907),
        ParallelGroupId::new(9),
        HostCollectiveLimits {
            max_ranks: 2,
            max_elements_per_rank: 1024,
            max_host_bytes: 1 << 20,
        },
        Duration::from_secs(30),
    )
    .unwrap();
    let done = Arc::new(Barrier::new(2));
    let identities = Arc::new(Mutex::new(vec![None, None]));
    let outputs = std::thread::scope(|scope| {
        let threads = endpoints
            .into_iter()
            .enumerate()
            .map(|(lane, endpoint)| {
                let r = r.clone();
                let p = p.clone();
                let done = done.clone();
                let identities = identities.clone();
                scope.spawn(move || {
                    let owner_id = p.tensor_members()[lane];
                    let layout = Arc::new(p.rank_layout(lane).unwrap());
                    let plan = HybridMixedRankProgramPlan::for_segment(layout.clone(),LayerSegmentPlan::new(r.spec().layers().len(),p.schema().layer_range(),boundaries,boundaries).unwrap(),2,1,32,2,3).unwrap();
                    let control = endpoint.control();
                    let collective_plan = Arc::new(
                        MixedRankCollectivePlan::from_hybrid_sites(
                            &p,
                            ParallelGroupId::new(9),
                            p.tensor_members()[0],
                            HYBRID_AGREEMENT_VERSION,
                            &plan.sites(),
                            HostCollectiveLimits {
                                max_ranks: 2,
                                max_elements_per_rank: 1024,
                                max_host_bytes: 1 << 20,
                            },
                            1,
                            2,
                        )
                        .unwrap().with_route_sites(plan.route_sites()).unwrap(),
                    );
                    let endpoint =
                        MixedRankCollective::prepare(endpoint, owner_id, collective_plan).unwrap();
                    let endpoint: Box<dyn HybridBf16Collective> = Box::new(endpoint);
                    let device = HybridCudaDevice::new_on_device(
                        lane,
                        HybridCudaMemoryBudget {
                            kv_pages: 16,
                            state_bytes: 1 << 20,
                            weight_bytes: 4 << 20,
                            workspace_bytes: 256 << 20,
                        },
                    )
                    .unwrap();
                    let mut owner = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
                        device.operators().clone(),
                        r.state_dict().parameters(),
                        ExpertCachePolicy::KeepAll,
                        0,
                        NumericFp8Precision::W8A16Bf16F32Accumulate,
                    )
                    .unwrap();
                    let mut life = owner
                        .prepare_mixed_rank_program(
                            &r,
                            plan,
                            device.clone(),
                            1 << 20,
                            [4 << 20, 64 << 20, 16 << 20],
                            1 << 20,
                            endpoint,
                            &[
                                ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionShape {
                                    rows: 2,
                                    tokens: 2,
                                },
                                ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionShape { rows:1,tokens:1 },
                                ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionShape { rows:1,tokens:2 },
                            ],
                            [64 << 20, 16 << 20],
                        )
                        .unwrap();
                    if routed {
                        assert!(owner.mixed_rank_all_expert_shards_resident(if boundaries {2} else {3}));
                        done.wait();
                        if lane == 0 {
                            for path in r.state_dict().parameters().iter().map(|p| p.weight().slice().path.clone()).collect::<std::collections::BTreeSet<_>>() {
                                std::fs::remove_file(path).unwrap();
                            }
                        }
                        done.wait();
                    }
                    let pool=TypedCudaPagedKvPool::<CudaHybridRankState>::from_strategy(device.operators().clone(),&layout.kv_planes(2,32).unwrap(),16).unwrap();
                    let mut backend=PagedKvBackend::new(pool);
                    let options=GenericDecoderOptions::standard_cpu(r.spec(),ModelFamily::Unknown("mixed-block".into()),WeightSource::Safetensors,2,32,2,1,1<<20,ExecutionPrecisionPolicy::f32()).unwrap();
                    let mut source=life.create().unwrap();
                    let input=(0..16).map(|i|ferrule_model::execution::bf16_rne_word(((i*7)%19) as f32/16.-9./16.)).collect::<Vec<_>>();
                    let frozen:serde_json::Value=serde_json::from_str(include_str!("hybrid_mixed_layer_oracle.json")).unwrap();
                    let gdn:serde_json::Value=serde_json::from_str(include_str!("hybrid_mixed_gdn_oracle.json")).unwrap();
                    let gdn_cuda:serde_json::Value=serde_json::from_str(include_str!("hybrid_mixed_gdn_cuda_oracle.json")).unwrap();
                    let program:serde_json::Value=serde_json::from_str(include_str!("hybrid_mixed_program_oracle.json")).unwrap();
                    let moe_oracle:serde_json::Value=serde_json::from_str(include_str!("hybrid_mixed_moe_cuda_oracle.json")).unwrap();
                    let boundary_oracle:serde_json::Value=serde_json::from_str(include_str!("hybrid_mixed_boundary_oracle.json")).unwrap();
                    let output_oracle=if boundaries { &boundary_oracle } else if routed { &moe_oracle } else if suffix { &program } else { &gdn };
                    let expected=words(if recurrent { &output_oracle["prefill"]["output"] } else { &frozen["output"] });
                    let expected_first=if recurrent { words(&output_oracle["fresh_one"]["output"]) } else { expected[..8].to_vec() };
                    let expected_decode=if recurrent { words(&output_oracle["cached_one"]["output"]) } else { expected[8..].to_vec() };
                    let mut run=StepRunner {lane,boundaries,owner:&mut owner,life:&mut life,backend:&mut backend,device:&device,
                        options:&options,control:&control,done:&done,identities:&identities,recurrent_oracle:recurrent.then_some(&gdn),recurrent_state_oracle:recurrent.then_some(&gdn_cuda),numerical_failures:0,output_words:0};
                    let full=run.step(71,&mut source,&input,false,false);
                    run.check_output(&full,&expected,"prefill");
                    let first=run.step(72,&mut source,&input[..8],true,false);
                    run.check_output(&first,&expected_first,"fresh_one");
                    assert_eq!(source.core().position(),1);
                    let mut fork=run.life.logical_fork(&source,1).unwrap();
                    if recurrent {
                        assert_eq!(state_words(&source),state_words(&fork),"fork preserves every native state bit");
                        assert_eq!(source.kv_state().resident_bytes(),108);
                        let third=run.life.create().unwrap();
                        assert!(run.life.create().is_err(),"rank state slots must be bounded");
                        run.life.try_release(third).unwrap();
                    }
                    let initial_slot=run.backend.pool().physical_slot(KvPageId(1));
                    let initial_free=run.backend.pool().free_slot_count();
                    let weights=run.owner.resident_parameter_bytes();
                    let mut stable=None;
                    for id in [73,74,75] {
                        let branch=run.step(id,&mut fork,&input[8..],false,true);
                        run.check_output(&branch,&expected_decode,"cached_one");
                        assert_eq!(source.core().position(),1);
                        assert_eq!(fork.core().position(),1);
                        assert_eq!(run.backend.pool().physical_slot(KvPageId(1)),initial_slot);
                        assert!(run.backend.pool().physical_slot(KvPageId(2)).is_none());
                        assert_eq!(run.backend.pool().free_slot_count(),initial_free);
                        assert_eq!(run.owner.resident_parameter_bytes(),weights);
                        let metrics=device.operators().allocator_metrics();
                        let current=(metrics.live_granted_bytes,metrics.reserved_bytes);
                        if let Some(before)=stable {assert_eq!(before,current,"COW abort warmed residency must stabilize");}
                        stable=Some(current);
                    }
                    let metrics=device.operators().allocator_metrics();
                    println!("BLOCK_GPU_METRICS {}",serde_json::json!({"lane":lane,"resident_weight_bytes":weights,
                        "live_granted_bytes":metrics.live_granted_bytes,"peak_live_granted_bytes":metrics.peak_live_granted_bytes,
                        "reserved_bytes":metrics.reserved_bytes,"peak_reserved_bytes":metrics.peak_reserved_bytes,
                        "live_pages":16-run.backend.pool().free_slot_count(),"active_transactions":run.backend.pool().active_transaction_count(),
                        "data_payload_bytes_per_element":2,"successful_invocations":5,"cow_append_aborts":3,"frozen_word_comparisons":run.output_words}));
                    assert_eq!(run.backend.pool().active_transaction_count(),0);
                    run.backend.release(&[KvPageId(1)]).unwrap();
                    assert_eq!(run.backend.pool().free_slot_count(),16);
                    if recurrent {
                        run.life.reset(&mut fork).unwrap();
                        assert_eq!(fork.core().position(),0);
                        let reset=state_words(&fork);
                        assert!(reset.0.iter().all(|v| *v==0));
                        assert!(reset.1.iter().all(|v| *v==0));
                        let reset_output=run.step(76,&mut fork,&input[..8],false,false);
                        run.check_output(&reset_output,&expected_first,"reset_one");
                        assert_eq!(state_words(&fork),reset,"aborted reset prefill must not mutate source");
                        assert_eq!(run.backend.pool().free_slot_count(),16);
                        println!("GDN_GPU_COUNTS lane={lane} invocations=6 output_words={} conv_words=180 ssm_words=72 cow_aborts=3 reset=1",run.output_words);
                    }
                    run.life.try_release(fork).unwrap();run.life.try_release(source).unwrap();
                    assert_eq!(run.life.live_state_grants(),0);
                    run.owner.quiesce().unwrap();assert!(!run.owner.needs_quarantine());
                    (full,run.numerical_failures,run.output_words)
                })
            })
            .collect::<Vec<_>>();
        threads
            .into_iter()
            .map(|t| t.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(outputs[0].0, outputs[1].0);
    assert_eq!(
        outputs.iter().map(|v| v.1).sum::<usize>(),
        0,
        "strict independent numerical comparisons failed; owners already drained"
    );
    if routed {
        println!(
            "MOE_GPU_COUNTS runtime_route_adapter=true invocations=12 output_words={} conv_words=360 ssm_words=144 experts_per_rank=16 top_k=8",
            outputs.iter().map(|v| v.2).sum::<usize>()
        );
    }
    drop(fixture);
    let oracle: serde_json::Value = serde_json::from_str(if boundaries {
        include_str!("hybrid_mixed_boundary_oracle.json")
    } else if routed {
        include_str!("hybrid_mixed_moe_cuda_oracle.json")
    } else if suffix {
        include_str!("hybrid_mixed_program_oracle.json")
    } else if recurrent {
        include_str!("hybrid_mixed_gdn_oracle.json")
    } else {
        include_str!("hybrid_mixed_layer_oracle.json")
    })
    .unwrap();
    let expected = words(if recurrent {
        &oracle["prefill"]["output"]
    } else {
        &oracle["output"]
    });
    assert_eq!(
        outputs[0].0, expected,
        "independent frozen TP2 words; no tolerance widening"
    );
}
fn words(v: &serde_json::Value) -> Vec<u16> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u16)
        .collect()
}
fn state_words(state: &CudaHybridRankState) -> (Vec<u16>, Vec<i32>) {
    let slots = state.kv_state().bf16_layers().unwrap();
    let recurrent = slots[1].as_ref().unwrap();
    let (conv, ssm) = recurrent.diagnostic_snapshot().unwrap();
    (
        conv.into_iter()
            .map(ferrule_model::execution::bf16_rne_word)
            .collect(),
        ssm.into_iter().map(|v| v.to_bits() as i32).collect(),
    )
}

struct StepRunner<'a> {
    lane: usize,
    boundaries: bool,
    owner: &'a mut CudaStandardDecoderOperators,
    life: &'a mut CudaHybridRankLifecycle,
    backend: &'a mut PagedKvBackend<TypedCudaPagedKvPool<CudaHybridRankState>>,
    device: &'a HybridCudaDevice,
    options: &'a GenericDecoderOptions,
    control:
        &'a ferrule_runtime::parallel::tensor::decoder_collective::DecoderTensorCollectiveControl,
    done: &'a Barrier,
    identities: &'a Mutex<Vec<Option<TensorBatchIdentity>>>,
    recurrent_oracle: Option<&'a serde_json::Value>,
    recurrent_state_oracle: Option<&'a serde_json::Value>,
    numerical_failures: usize,
    output_words: usize,
}
impl StepRunner<'_> {
    fn check_output(&mut self, actual: &[u16], expected: &[u16], case: &str) {
        self.output_words += actual.len();
        if self.recurrent_oracle.is_none() {
            assert_eq!(actual, expected);
            return;
        }
        assert_eq!(actual.len(), expected.len());
        let unequal = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
        self.numerical_failures += unequal;
        println!(
            "GDN_GPU_OUTPUT {}",
            serde_json::json!({"lane":self.lane,"case":case,"words":actual,"expected":expected,"unequal":unequal})
        );
    }
    fn step(
        &mut self,
        id: u64,
        source: &mut CudaHybridRankState,
        input: &[u16],
        publish: bool,
        cow: bool,
    ) -> Vec<u16> {
        let before = self.recurrent_oracle.map(|_| state_words(source));
        let tx = ExecutionTransactionId::new(id).unwrap();
        let start = source.core().position();
        let rows = input.len() / 8;
        let target = if cow { 2 } else { 1 };
        let mut states = vec![
            self.life
                .checkout(DecoderSequenceCheckout::new(tx, 0, 0), source)
                .unwrap(),
        ];
        let phase = if start == 0 {
            ForwardPhase::Prefill
        } else {
            ForwardPhase::Decode
        };
        let execution = ExecutionBatch::new(
            if start == 0 {
                ForwardMode::Prefill
            } else {
                ForwardMode::Decode
            },
            if self.boundaries {
                (start..start + rows).map(|i| i as u32 + 1).collect()
            } else {
                vec![1; rows]
            },
            (start..start + rows).map(|p| p as u32).collect(),
            (start..start + rows)
                .map(|p| Some(KvWriteSlot::new(target * 2 + (p % 2) as u32)))
                .collect(),
            vec![LogitsRequest::Full; rows],
            vec![ExecutionSequence::new(
                StateSlot::new(0),
                phase,
                0..rows as u32,
                start as u32,
                (start + rows) as u32,
                0..1,
            )],
            vec![KvBlockId::new(target)],
        );
        let reservations = [KvReservationView {
            state_slot: StateSlot::new(0),
            execution_state_slot: StateSlot::new(0),
            positions: start..start + rows,
            newly_allocated: if start == 0 {
                vec![KvPageId(target)]
            } else {
                vec![]
            },
            generation: source.core().generation(),
            execution_generation: source.core().generation(),
            cow_replacement: cow.then_some(KvCowReplacement {
                logical_page: 0,
                source: KvPageId(1),
                replacement: KvPageId(2),
            }),
        }];
        let batch = PackedDecoderBatch::lower(
            &execution,
            &reservations,
            &states,
            &self.options.capabilities(),
            2,
            self.backend,
        )
        .unwrap();
        let topology = ferrule_common::ValidatedParallelTopology::new(
            ParallelTopologyId::new(907),
            2,
            ParallelRankId::new(0),
            ferrule_common::ParallelismPlan::validated(1, 2, 1, 1, 1, 1).unwrap(),
        )
        .unwrap();
        let binding = KvCommitBinding::new(
            tx,
            ParallelTopologyId::new(907),
            topology.participants(),
            17,
        )
        .unwrap();
        let identity = TensorBatchIdentity::from_packed(&binding, &batch, &[900]).unwrap();
        self.identities.lock().unwrap()[self.lane] = Some(identity.clone());
        self.done.wait();
        if self.lane == 0 {
            let ids = self.identities.lock().unwrap();
            assert_eq!(ids[0], ids[1]);
            self.control
                .begin_sealed(identity.clone(), Arc::new(AtomicBool::new(false)))
                .unwrap();
        }
        self.done.wait();
        self.control
            .validate_owner_execution(ParallelRankId::new(self.lane as u32), &identity)
            .unwrap();
        self.done.wait();
        let seq = &batch.sequences()[0];
        let custody = [DecoderKvSequenceCustody {
            source_index: 0,
            topology_id: seq.topology_id(),
            page_state_slot: seq.page_state_slot(),
            page_generation: seq.page_generation(),
            execution_generation: seq.execution_generation(),
            context_len: start,
            query_len: rows,
        }];
        let pages = if cow {
            vec![KvPageId(1), KvPageId(2)]
        } else {
            vec![KvPageId(1)]
        };
        let snapshots = pages
            .iter()
            .map(|&page| DecoderKvPageSnapshot {
                page,
                status: DecoderKvPageView::page_status(self.backend, page),
            })
            .collect::<Vec<_>>();
        let mut handle = self
            .backend
            .prepare(DecoderKvPrepare {
                transaction: tx,
                sequences: &custody,
                new_pages: batch.new_pages(),
                writable_pages: batch.writable_pages(),
                cow_replacements: batch.cow_replacements(),
                protected_pages: batch.protected_pages(),
                capacity: self.backend.capacity(),
                page_statuses: &snapshots,
            })
            .unwrap();
        self.backend
            .enter(&mut handle, &batch, &mut states)
            .unwrap();
        let mut kv = self.backend.active_view(&mut handle).unwrap();
        let x = Rows::Cuda(
            CudaRows::bf16(
                RowsShape::new(rows, 8).unwrap(),
                None,
                self.device.operators().upload_bf16_words(input).unwrap(),
            )
            .unwrap(),
        );
        if self.recurrent_oracle.is_some()
            && let Ok(directory) = std::env::var("FERRULE_MIXED_GDN_CAPTURE_DIR")
        {
            let directory = std::path::PathBuf::from(directory);
            assert!(directory.is_dir());
            let lane = self.lane;
            self.owner.set_diagnostic_trace(move |event| {
                if event.name.starts_with("gdn.") {
                    let bits = event
                        .operators
                        .download_f32_buffer(event.buffer)?
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>();
                    let path = directory.join(format!("rank{lane}-tx{id}-{}.json", event.name));
                    assert!(!path.exists(), "duplicate GDN capture");
                    std::fs::write(
                        path,
                        serde_json::to_vec(&serde_json::json!({
                            "lane":lane,"transaction":id,"name":event.name,
                            "rows":event.shape.rows(),"width":event.shape.width(),"bits":bits
                        }))
                        .unwrap(),
                    )
                    .unwrap();
                }
                Ok(())
            });
        }
        self.device.operators().reset_counters();
        let output = if self.boundaries {
            match self
                .owner
                .execute_mixed_rank_segment(
                    &batch,
                    DeviceSegmentInput::Tokens,
                    &mut states,
                    &mut kv,
                    &[900],
                    &mut |_| Ok(()),
                )
                .unwrap()
            {
                DeviceSegmentOutput::Logits(rows) => rows,
                _ => panic!("owned output must return logits"),
            }
        } else {
            self.owner
                .execute_mixed_rank_program(
                    &batch,
                    x,
                    &mut states,
                    &mut kv,
                    &[900],
                    &mut |_| Ok(()),
                )
                .unwrap()
        };
        assert_eq!(self.device.operators().counters().artifact_uploads, 0);
        if self.recurrent_oracle.is_some() {
            let case = if start != 0 {
                "cached_one"
            } else if rows == 1 {
                "fresh_one"
            } else {
                "prefill"
            };
            let actual = state_words(&states[0]);
            let state_oracle = self
                .recurrent_state_oracle
                .expect("Mixed GDN requires the pinned HF CUDA F32 state oracle");
            let expected = &state_oracle[case]["ranks"][self.lane];
            let conv = words(&expected["conv"]);
            let ssm = expected["ssm"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect::<Vec<_>>();
            assert_eq!(actual.0.len(), conv.len());
            assert_eq!(actual.1.len(), ssm.len());
            let conv_unequal = actual.0.iter().zip(&conv).filter(|(a, b)| a != b).count();
            let ssm_unequal = actual.1.iter().zip(&ssm).filter(|(a, b)| a != b).count();
            self.numerical_failures += conv_unequal + ssm_unequal;
            println!(
                "GDN_GPU_STATE {}",
                serde_json::json!({"lane":self.lane,"transaction":id,"case":case,
                "conv":actual.0,"expected_conv":conv,"ssm":actual.1,"expected_ssm":ssm,
                "conv_unequal":conv_unequal,"ssm_unequal":ssm_unequal})
            );
            assert_eq!(
                state_words(source),
                before.unwrap(),
                "working recurrent state must not alias source"
            );
            assert_eq!(
                states[0].kv_state().bf16_layers().unwrap()[1]
                    .as_ref()
                    .unwrap()
                    .position(),
                start + rows
            );
        }
        let words = output
            .cuda()
            .unwrap()
            .bf16_buffer()
            .unwrap()
            .as_device_buffer()
            .to_host_vec(&self.device.operators().stream_clone())
            .unwrap();
        assert_eq!(states[0].core().position(), start);
        assert_eq!(source.core().position(), start);
        drop(output);
        drop(kv);
        self.backend.leave(&mut handle).unwrap();
        if publish {
            assert_eq!(
                self.backend.commit(&mut Some(handle)).unwrap(),
                KvEndProgress::Complete
            );
            let b = states[0].core().begin_step().unwrap();
            states[0].core_mut().commit_step(b, rows).unwrap();
            std::mem::swap(source, &mut states[0]);
        } else {
            assert_eq!(
                self.backend.rollback(&mut Some(handle)).unwrap(),
                KvEndProgress::Complete
            );
        }
        self.life.try_release(states.pop().unwrap()).unwrap();
        self.owner.quiesce().unwrap();
        self.done.wait();
        if self.lane == 0 {
            self.control.finish(tx).unwrap();
        }
        self.done.wait();
        words
    }
}
