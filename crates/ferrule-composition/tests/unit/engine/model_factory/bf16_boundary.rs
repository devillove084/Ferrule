//! Bounded producer capture on the existing full-depth factory/transaction owner.
//! Stops before layer5 compute; never presented as a full40 quality gate.
use super::*;
use ferrule_model::runner::{TransactionEndIntent, TransactionEndProgress};
use std::cell::Cell;

#[test]
#[ignore = "sole GPU owner; real factory Hello prefix through layer4 then intentional observer error/abort"]
fn mixed_layers234_actual_producer_capture_and_abort() {
    assert_eq!(
        std::env::var("FERRULE_NUMERIC_BF16_FACTORY_GPU").as_deref(),
        Ok("1")
    );
    let dir = required_path("FERRULE_NUMERIC_BF16_FIRST_TRACE_DIR");
    std::fs::create_dir(&dir).unwrap();
    let summary_path = dir.join("lifecycle.json");
    let plan = prepare_full40(
        &required_path("FERRULE_NUMERIC_BF16_MODEL"),
        NumericFp8Precision::Bf16ModelMixedF32Accumulate,
    );
    let stopped = Rc::new(Cell::new(false));
    let marker = Rc::clone(&stopped);
    let emitted = Rc::new(Cell::new(0usize));
    let rows_written = Rc::clone(&emitted);
    crate::model_factory::bf16_serving::NUMERIC_BF16_TEST_RUNNER.with(|slot| {
        assert!(slot.borrow().is_none());
        *slot.borrow_mut() = Some(Box::new(move |runner| {
            let mut file = std::fs::OpenOptions::new().create_new(true).write(true).open(dir.join("events.jsonl")).unwrap();
            let mut count = 0;
            let mut bytes = 0;
            let reached = Rc::clone(&marker);
            runner.forward_executor().module().set_diagnostic_trace(move |event| {
                if event.layer == Some(5) {
                    reached.set(true);
                    file.sync_all()?;
                    return Err(ferrule_common::Error::Model { message: "bounded layer234 diagnostic complete before layer5 compute".into() });
                }
                if event.layer.is_some_and(|i| (2..=4).contains(&i)) {
                    let values=event.operators.download_f32_buffer(event.buffer)?;
                    bytes += values.len()*4; assert!(bytes <= 64 << 20);
                    let name=format!("{count:04}.f32");
                    std::fs::write(dir.join(&name),values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
                    let tensor=|s: &ferrule_model::CheckpointTensorSlice|json!({"name":s.name,"path":s.path,"offset":s.offset,"bytes":s.bytes,"shape":s.shape,"dtype":format!("{:?}",s.dtype)});
                    writeln!(file,"{}",json!({"event":event.name,"layer":event.layer,"shape":[event.shape.rows(),event.shape.width()],"file":name,"parameter":event.parameter.map(|p|p.path().to_string()),"weight":event.parameter.map(|p|tensor(p.weight().slice())),"scale":event.parameter.and_then(|p|p.scale()).map(|p|tensor(p.slice())),"ids":event.routes.map(|r|r.expert_ids()),"weights":event.routes.map(|r|r.weights())}))?;
                    count+=1;rows_written.set(count);
                }
                Ok(())
            });
            let mut states=vec![runner.create_sequence_state().unwrap()];
            let mut pages=vec![Vec::new()];
            let caught=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::model_factory::qwen35::full_logits_step(runner,&mut states,&mut pages,&[9419],ForwardPhase::Prefill,1)
            }));
            let panic=caught.expect_err("must stop before layer5 compute");
            let message=panic.downcast_ref::<String>().map(String::as_str).or_else(||panic.downcast_ref::<&str>().copied()).unwrap_or("");
            assert!(marker.get() && message.contains("bounded layer234 diagnostic complete"),"unexpected failure: {message}");
            runner.forward_executor().module().clear_diagnostic_trace();
            let deadline=std::time::Instant::now()+std::time::Duration::from_secs(30);
            loop {
                match runner.end_transaction(ExecutionTransactionId::new(1).unwrap(),&mut states,TransactionEndIntent::Abort).unwrap() {
                    TransactionEndProgress::Complete=>break,
                    TransactionEndProgress::Pending=>assert!(std::time::Instant::now()<deadline),
                }
                std::thread::yield_now();
            }
            assert_eq!(states[0].core().position(),0);
            runner.try_release_sequence_state(states.remove(0)).unwrap();
            assert!(pages[0].is_empty(),"aborted pages never published");
        }));
    });
    let mut engine = plan.build().unwrap();
    let device = crate::model_factory::bf16_serving::NUMERIC_BF16_TEST_DEVICE
        .with(|s| s.borrow_mut().take())
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while engine.shutdown().unwrap() != InferenceShutdownProgress::Complete {
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    drop(engine);
    assert!(stopped.get() && emitted.get() > 0);
    assert_eq!(device.live_state_bytes(), 0);
    assert!(!device.needs_quarantine());
    let summary = json!({"scope":"Hello actual prefix through layer4; intentional abort before layer5 compute", "full40_quality_gate":false,"events":emitted.get(),"stopped_before_layer5":true,"aborted":true,"state_bytes_after_shutdown":device.live_state_bytes(),"quarantine":device.needs_quarantine()});
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(summary_path)
        .unwrap();
    serde_json::to_writer_pretty(&mut file, &summary).unwrap();
    file.sync_all().unwrap();
    eprintln!("{summary}");
}
