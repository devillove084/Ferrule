//! Bounded selected prefill on one full resident image; observer abort is not acceptance.
use super::*;
use std::{cell::Cell, io::Write, path::PathBuf, rc::Rc};
#[allow(dead_code)]
#[path = "../../../ferrule-composition/tests/tp2_mixed_execution/oracle.rs"]
mod oracle;

const STOP: &str = "bounded Mixed layer-0 diagnostic complete";

#[test]
#[ignore = "sole TP2 GPU owner; original full image, Hello layer0 only, authenticated immutable oracle"]
fn full_image_hello_layer0_boundaries() {
    full_image_boundaries("hello", false);
}

#[test]
#[ignore = "sole TP2 GPU owner; original full image, Capital M5 layer0 only, authenticated immutable oracle"]
fn full_image_capital_layer0_boundaries() {
    full_image_boundaries("capital", false);
}

#[test]
#[ignore = "sole TP2 GPU owner; one complete Capital M5 diagnostic forward, no acceptance comparison"]
fn full_image_capital_all_layer_boundaries() {
    full_image_boundaries("capital", true);
}

fn full_image_boundaries(case: &str, complete: bool) {
    assert_eq!(std::env::var("FERRULE_MIXED_GPU_RUN").as_deref(), Ok("1"));
    assert_eq!(
        std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref(),
        Ok(":0:0")
    );
    let model = PathBuf::from(std::env::var("FERRULE_MIXED_MODEL").unwrap());
    let reference = PathBuf::from(std::env::var("FERRULE_MIXED_ORACLE_DIR").unwrap());
    let evidence = PathBuf::from(std::env::var("FERRULE_MIXED_EVIDENCE_DIR").unwrap());
    assert_ne!(
        reference.canonicalize().unwrap(),
        evidence.canonicalize().unwrap()
    );
    std::fs::write(evidence.join("capture-plan.json"),serde_json::to_vec_pretty(&serde_json::json!({
        "case":case,"complete":complete,"reference_manifest":std::env::var("FERRULE_MIXED_ORACLE_SHA256").unwrap(),
        "policy":"all semantic boundaries and projections; complete mode excludes chunk workspace; never compare inside collective schedule"
    })).unwrap()).unwrap();
    let oracle = oracle::Oracle::open(
        &reference,
        &std::env::var("FERRULE_MIXED_ORACLE_SHA256").unwrap(),
        &model,
    );
    let capture_cap = std::env::var("FERRULE_MIXED_CAPTURE_CAP_BYTES")
        .ok()
        .map(|n| n.parse::<usize>().unwrap())
        .unwrap_or(512 << 20);
    assert!((64 << 20..=512 << 20).contains(&capture_cap));
    let tokens = oracle.case(case).tokens("prefill");
    let rows = tokens.len();
    assert!(matches!(rows, 1 | 5));
    let (_, resources) =
        ferrule_model::models::qwen35::Qwen35Adapter::bind_hf_metadata(&model).unwrap();
    assert_eq!(resources.spec().layers().len(), 40);
    let resources = Arc::new(resources);
    let parallel = parallel(&resources, 0..40, 2);
    let limits = HostCollectiveLimits {
        max_ranks: 2,
        max_elements_per_rank: 8 * 124_160,
        max_host_bytes: 64 << 20,
    };
    let endpoints = DecoderTensorCollective::new_group_from_members(
        parallel.tensor_members(),
        ParallelTopologyId::new(907),
        ParallelGroupId::new(9),
        limits,
        Duration::from_secs(60),
    )
    .unwrap();
    let barrier = Arc::new(Barrier::new(2));
    std::thread::scope(|scope| {
        let joins = endpoints.into_iter().enumerate().map(|(lane, endpoint)| {
            let resources = resources.clone();
            let parallel = parallel.clone();
            let barrier = barrier.clone();
            let tokens = tokens.clone();
            let directory = evidence.join(format!("rank-{lane}"));
            std::fs::create_dir(&directory).unwrap();
            scope.spawn(move || {
                let layout = Arc::new(parallel.rank_layout(lane).unwrap());
                let program = HybridMixedRankProgramPlan::for_segment(layout.clone(), LayerSegmentPlan::new(40,0..40,true,true).unwrap(),8,2,128,16,5).unwrap();
                let shapes = [ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionShape { rows:rows as u32,tokens:rows as u32 }];
                let (estimate, context) = program.cuda_claim_inputs(&resources,64,lane,2 << 30,&shapes).unwrap();
                let device = HybridCudaDevice::new_on_device(lane,HybridCudaMemoryBudget {
                    kv_pages:64, state_bytes:estimate.per_sequence_state_bytes * 5,
                    weight_bytes:estimate.weight_bytes_upper_bound, workspace_bytes:estimate.workspace_bytes_upper_bound,
                }).unwrap();
                let control = endpoint.control();
                let plan = MixedRankCollectivePlan::from_hybrid_sites(&parallel,ParallelGroupId::new(9),parallel.tensor_members()[0],HYBRID_AGREEMENT_VERSION,&program.sites(),limits,2,8).unwrap().with_route_sites(program.route_sites()).unwrap();
                let endpoint = MixedRankCollective::prepare(endpoint,layout.owner(),Arc::new(plan)).unwrap();
                let claim = device.admit_with_context(estimate,context).unwrap();
                let mut owner = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(device.operators().clone(),resources.state_dict().parameters(),ExpertCachePolicy::KeepAll,0,NumericFp8Precision::W8A16Bf16F32Accumulate).unwrap();
                let mut life = owner.prepare_mixed_rank_program_with_claim(&resources,program,device.clone(),2 << 30,Box::new(endpoint),&shapes,&claim).unwrap();
                let pool = TypedCudaPagedKvPool::<CudaHybridRankState>::from_strategy(device.operators().clone(),&layout.kv_planes(16,128).unwrap(),64).unwrap();
                let state = life.create().unwrap();
                let (mut life, mut backend, mut states) = claim.finish(Ok((life,PagedKvBackend::new(pool),vec![state]))).unwrap();
                for layer in 0..40 { assert!(owner.mixed_rank_all_expert_shards_resident(layer)); }
                let options = GenericDecoderOptions::standard_cpu(resources.spec(),ModelFamily::Unknown("mixed-diagnostic".into()),WeightSource::Safetensors,16,128,8,2,2 << 30,ExecutionPrecisionPolicy::f32()).unwrap();
                let execution = ExecutionBatch::new(ForwardMode::Prefill,tokens,(0..rows as u32).collect(),(0..rows as u32).map(|p|Some(KvWriteSlot::new(16+p))).collect(),vec![LogitsRequest::Full;rows],vec![ExecutionSequence::new(StateSlot::new(0),ForwardPhase::Prefill,0..rows as u32,0,rows as u32,0..1)],vec![KvBlockId::new(1)]);
                let reservations = [KvReservationView { state_slot:StateSlot::new(0),execution_state_slot:StateSlot::new(0),positions:0..rows,newly_allocated:vec![KvPageId(1)],generation:states[0].core().generation(),execution_generation:states[0].core().generation(),cow_replacement:None }];
                let batch = PackedDecoderBatch::lower(&execution,&reservations,&states,&options.capabilities(),16,&backend).unwrap();
                let tx = ExecutionTransactionId::new(71).unwrap();
                let topology = ferrule_common::ValidatedParallelTopology::new(ParallelTopologyId::new(907),2,ParallelRankId::new(0),ferrule_common::ParallelismPlan::validated(1,2,1,1,1,1).unwrap()).unwrap();
                let binding = KvCommitBinding::new(tx,ParallelTopologyId::new(907),topology.participants(),17).unwrap();
                let identity = TensorBatchIdentity::from_packed(&binding,&batch,&[900]).unwrap();
                barrier.wait();
                if lane==0 { control.begin_sealed(identity.clone(),Arc::new(AtomicBool::new(false))).unwrap(); }
                barrier.wait();
                control.validate_owner_execution(layout.owner(),&identity).unwrap();
                barrier.wait();
                let seq=&batch.sequences()[0];
                let custody=[DecoderKvSequenceCustody { source_index:0,topology_id:seq.topology_id(),page_state_slot:seq.page_state_slot(),page_generation:seq.page_generation(),execution_generation:seq.execution_generation(),context_len:0,query_len:rows }];
                let snapshots=[DecoderKvPageSnapshot {page:KvPageId(1),status:DecoderKvPageView::page_status(&backend,KvPageId(1))}];
                let mut handle=backend.prepare(DecoderKvPrepare {transaction:tx,sequences:&custody,new_pages:batch.new_pages(),writable_pages:batch.writable_pages(),cow_replacements:batch.cow_replacements(),protected_pages:batch.protected_pages(),capacity:backend.capacity(),page_statuses:&snapshots}).unwrap();
                backend.enter(&mut handle,&batch,&mut states).unwrap();
                let mut kv=backend.active_view(&mut handle).unwrap();
                let stopped=Rc::new(Cell::new(false));
                let observer_stopped=stopped.clone();
                let stop_barrier=barrier.clone();
                let mut log=std::fs::OpenOptions::new().create_new(true).write(true).open(directory.join("events.jsonl")).unwrap();
                let output_directory=directory.clone();
                let skipped=Rc::new(Cell::new(0usize));
                let observer_skipped=skipped.clone();
                let mut bytes=0usize;
                let mut sequence=0usize;
                owner.set_diagnostic_trace(move |event| {
                    assert!(event.layer.is_none_or(|i| i < 40),"diagnostic escaped model layers");
                    // Chunk scratch is not a state boundary and dwarfs useful captures.
                    // Exclude it by metadata before any D2H; keep all QKV/state/FFN leaves.
                    if complete && event.name=="gdn.chunk.workspace" { return Ok(()); }
                    let cap = if complete { capture_cap/2 } else { 64 << 20 };
                    if bytes + event.shape.elements()*4 > cap {
                        observer_skipped.set(observer_skipped.get()+1);
                        return Ok(());
                    }
                    let values=event.operators.download_f32_buffer(event.buffer)?;
                    bytes+=values.len()*4;
                    let file=format!("{sequence:04}.f32");
                    std::fs::write(directory.join(&file),values.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
                    writeln!(log,"{}",serde_json::json!({"event":event.name,"layer":event.layer,"shape":[event.shape.rows(),event.shape.width()],"file":file,"parameter":event.parameter.map(|p|p.path().to_string()),"role":event.parameter.map(|p|format!("{:?}",p.role())),"ids":event.routes.map(|r|r.expert_ids()),"weights":event.routes.map(|r|r.weights()),"weight_bits":event.routes.map(|r|r.weights().iter().map(|v|v.to_bits()).collect::<Vec<_>>())}))?;
                    log.sync_data()?;
                    sequence+=1;
                    if event.name=="mixed.layer.output" && !complete {
                        observer_stopped.set(true);
                        stop_barrier.wait();
                        return Err(ferrule_common::Error::Model {message:STOP.into()});
                    }
                    Ok(())
                });
                let result=owner.execute_mixed_rank_segment(&batch,DeviceSegmentInput::Tokens,&mut states,&mut kv,&[900],&mut |_|Ok(()));
                owner.clear_diagnostic_trace();
                if complete {
                    let DeviceSegmentOutput::Logits(logits)=result.unwrap() else { panic!("missing final logits") };
                    let values=device.operators().download_bf16_buffer(logits.cuda().unwrap().bf16_buffer().unwrap()).unwrap();
                    std::fs::write(output_directory.join("logits.f32"),values.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()).unwrap();
                    assert_eq!(skipped.get(),0,"diagnostic cap exceeded; completed collectives but evidence incomplete");
                } else {
                    let error=result.expect_err("observer must stop after layer0");
                    assert!(stopped.get() && error.to_string().contains(STOP),"unexpected forward error: {error}");
                }
                drop(kv);
                backend.leave(&mut handle).unwrap();
                assert_eq!(backend.rollback(&mut Some(handle)).unwrap(),KvEndProgress::Complete);
                assert_eq!(backend.pool().active_transaction_count(),0);
                assert_eq!(backend.pool().free_slot_count(),64);
                life.try_release(states.pop().unwrap()).unwrap();
                owner.quiesce().unwrap();
                assert!(!owner.needs_quarantine() && !device.needs_quarantine());
                assert_eq!(device.live_state_bytes(),0);
                println!("BOUNDED_DIAGNOSTIC case={case} rows={rows} lane={lane} complete={complete} next_layer_not_executed={} state_bytes=0 active_transactions=0 metrics={:?}",!complete,device.operators().allocator_metrics());
                drop(owner); drop(life); drop(backend);
            })
        }).collect::<Vec<_>>();
        for join in joins {
            join.join().unwrap();
        }
    });
}
