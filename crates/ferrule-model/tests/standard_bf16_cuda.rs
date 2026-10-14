//! Independent HF full-state acceptance; compile on CPU, run only explicitly.
#![cfg(feature = "cuda")]
#[path = "support/bf16_dense_fixture.rs"]
mod fixture;
use ferrule_backend::cuda::context::CudaOperators;
use ferrule_common::execution::{KvCowReplacement, KvElementType, KvPageId};
use ferrule_model::{decoder::*, execution::ExecutionPrecisionPolicy, transformer::*};
use fixture::*;
use std::{cell::RefCell, rc::Rc};

type Backend = PagedKvBackend<CudaPagedKvPool>;
fn snapshot(
    backend: &mut Backend,
    page: KvPageId,
) -> ferrule_backend::cuda::operators::kv::TypedKvHostSnapshot {
    backend.preempt(&[page]).unwrap();
    let snapshot = backend.pool().typed_host_snapshot(page).unwrap();
    backend.restore(&[page]).unwrap();
    snapshot
}
fn check_kv(backend: &mut Backend, pages: &[KvPageId], case: &serde_json::Value) {
    let snapshots = pages
        .iter()
        .map(|&p| snapshot(backend, p))
        .collect::<Vec<_>>();
    for (plane, name) in ["key", "value"].into_iter().enumerate() {
        for layer in 0..2 {
            let expected = values(&case[name][layer]);
            let mut actual = Vec::new();
            for page in &snapshots {
                assert_eq!(page.planes[plane].element_type(), KvElementType::Bf16);
                // 2 layers * 2 tokens * 8 elements * 2 bytes, no F32 mirror.
                assert_eq!(page.planes[plane].byte_len(), 64);
                assert!(page.planes[plane].f32_bits().is_none());
                let words = page.planes[plane].bf16_words().unwrap();
                actual.extend(
                    words[layer * 16..(layer + 1) * 16]
                        .iter()
                        .map(|&w| f32::from_bits(u32::from(w) << 16)),
                );
            }
            actual.truncate(expected.len());
            eprintln!("KV {name} layer {layer}");
            close(&actual, &expected);
        }
    }
}

#[test]
#[ignore = "requires CUDA SM80+ and CUTLASS BF16; independent HF CPU full-state oracle"]
fn dense_bf16_all_layers_prefill_decode_cow_abort_and_restore_match_hf() {
    let fixture = Fixture::new();
    let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
    let mut stage = StandardDecoderSegment::prepare_cuda(
        &fixture.resources,
        LayerSegmentPlan::new(2, 0..2, true, true).unwrap(),
        ExecutionPrecisionPolicy::bf16_compatibility(),
        POSITIONS,
        LIMIT,
        Rc::clone(&ops),
    )
    .unwrap();
    assert_eq!(stage.activation_dtype(), RowsDType::Bf16);
    assert_eq!(stage.kv_element_type(), KvElementType::Bf16);
    let weights: usize = fixture.oracle["weights"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w["values"].as_array().unwrap().len() * 2)
        .sum();
    // Shared rotary table: 16 positions * 4 pairs * cos/sin * 2 bytes.
    assert_eq!(stage.operators().resident_parameter_bytes(), weights + 256);
    let residuals = Rc::new(RefCell::new(Vec::<Vec<f32>>::new()));
    let captured = Rc::clone(&residuals);
    let trace = Rc::new(RefCell::new(Vec::new()));
    let trace_capture = Rc::clone(&trace);
    stage.operators().set_diagnostic_trace(move |event| {
        if std::env::var_os("FERRULE_BF16_TRACE_DIR").is_some() {
            trace_capture.borrow_mut().push(serde_json::json!({
                "name": event.name, "parameter": event.parameter.map(|p| p.path().as_str()),
                "shape": [event.shape.rows(), event.shape.width()],
                "values": event.operators.download_f32_buffer(event.buffer)?,
            }));
        }
        if event.name == "residual.output" {
            captured
                .borrow_mut()
                .push(event.operators.download_f32_buffer(event.buffer)?);
        }
        Ok(())
    });
    let mut backend = PagedKvBackend::new(
        CudaPagedKvPool::from_strategy(Rc::clone(&ops), &fixture.planes(2), 12).unwrap(),
    );
    let mut state = GenericDecoderSequenceState::new((), ());
    let mut fork = None;
    let mut aborted_fork = None;
    let first = KvPageId(700);
    let source = KvPageId(711);
    let main_tail = KvPageId(730);
    let fork_tail = KvPageId(740);
    let mut original = None;
    for (step, case_index) in [0usize, 1, 2, 3].into_iter().enumerate() {
        let case = &fixture.oracle["cases"][case_index];
        let tokens: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        let target = if case_index == 2 {
            fork.as_mut().unwrap()
        } else {
            &mut state
        };
        let tail = match case_index {
            1 => Some((
                main_tail,
                Some(KvCowReplacement {
                    logical_page: 1,
                    source,
                    replacement: main_tail,
                }),
            )),
            2 => Some((
                fork_tail,
                Some(KvCowReplacement {
                    logical_page: 1,
                    source,
                    replacement: fork_tail,
                }),
            )),
            3 => Some((main_tail, None)),
            _ => None,
        };
        let (batch, tx, mut view) = enter(
            &fixture,
            &mut backend,
            target,
            step as u64 + 1,
            &tokens,
            tail,
        );
        residuals.borrow_mut().clear();
        trace.borrow_mut().clear();
        let output = stage
            .execute_cuda_bound(&batch, DeviceSegmentInput::Tokens, &mut view)
            .unwrap();
        let DeviceSegmentOutput::Logits(ref rows) = output else {
            panic!("logits")
        };
        assert_eq!(rows.dtype(), RowsDType::Bf16);
        assert!(rows.cuda().unwrap().f32_buffer().is_none());
        let SegmentOutput::Logits(logits) = stage.download_output(output).unwrap() else {
            panic!("logits")
        };
        if let Some(dir) = std::env::var_os("FERRULE_BF16_TRACE_DIR") {
            std::fs::write(
                std::path::Path::new(&dir).join(format!("case-{case_index}.json")),
                serde_json::to_vec_pretty(&*trace.borrow()).unwrap(),
            )
            .unwrap();
        }
        eprintln!("CASE {case_index} logits");
        close(logits.values(), &values(&case["logits"]));
        assert_eq!(residuals.borrow().len(), 4);
        for layer in 0..2 {
            eprintln!("CASE {case_index} layer {layer} hidden");
            close(
                &residuals.borrow()[layer * 2 + 1],
                &values(&case["hidden"][layer]),
            );
        }
        drop(view);
        finish(&mut backend, target, tx, tokens.len());
        let pages = match case_index {
            0 => vec![first, source],
            1 => vec![first, main_tail],
            2 => vec![first, fork_tail],
            _ => vec![first, main_tail, KvPageId(722)],
        };
        check_kv(&mut backend, &pages, case);
        if case_index == 0 {
            fork = Some(state.logical_fork().unwrap());
            aborted_fork = Some(state.logical_fork().unwrap());
            original = Some(snapshot(&mut backend, source));
        } else {
            assert_eq!(
                snapshot(&mut backend, source),
                *original.as_ref().unwrap(),
                "COW changed shared source"
            );
        }
        assert_eq!(stage.operators().resident_parameter_bytes(), weights + 256);
    }
    // Abort an independently forked partial tail. No model or KV manager is rebuilt.
    let before = snapshot(&mut backend, source);
    let replacement = KvPageId(750);
    let mut aborted_fork = aborted_fork.unwrap();
    let cow = KvCowReplacement {
        logical_page: 1,
        source,
        replacement,
    };
    let (batch, mut tx, mut view) = enter(
        &fixture,
        &mut backend,
        &mut aborted_fork,
        5,
        &[7],
        Some((replacement, Some(cow))),
    );
    let _ = stage
        .execute_cuda_bound(&batch, DeviceSegmentInput::Tokens, &mut view)
        .unwrap();
    drop(view);
    backend.leave(&mut tx).unwrap();
    assert_eq!(
        backend.rollback(&mut Some(tx)).unwrap(),
        KvEndProgress::Complete
    );
    assert_eq!(snapshot(&mut backend, source), before);
    assert_eq!(
        DecoderKvBackend::page_status(&backend, replacement),
        DecoderKvPageStatus::Vacant
    );
    assert_eq!(aborted_fork.core().position(), 3);
    assert!(!stage.needs_quarantine());
}

#[test]
#[ignore = "requires CUDA; exact isolated HF BF16 boundary tensors"]
fn bf16_native_boundary_primitives_match_torch_and_reject_f32_rows() {
    let fixture = Fixture::new();
    let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
    let b = &fixture.oracle["boundaries"];
    let upload = |name: &str| {
        ops.upload_bf16_words(
            &values(&b[name])
                .into_iter()
                .map(|v| half::bf16::from_f32(v).to_bits())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let input = upload("input");
    let weight = upload("weight");
    let up = upload("up");
    let norm = ops.rms_norm_bf16(&input, &weight, 1, 1e-5).unwrap();
    assert_eq!(ops.download_bf16_buffer(&norm).unwrap(), values(&b["norm"]));
    let swiglu = ops.swiglu_bf16(&input, &up).unwrap();
    assert_eq!(
        ops.download_bf16_buffer(&swiglu).unwrap(),
        values(&b["swiglu"])
    );
    let residual = ops.residual_bf16(&input, &up).unwrap();
    assert_eq!(
        ops.download_bf16_buffer(&residual).unwrap(),
        values(&b["residual"])
    );
    let table = |name: &str| {
        ops.upload_bf16_words(
            &values(&b[name])[..2]
                .iter()
                .map(|&v| half::bf16::from_f32(v).to_bits())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let rope = ops
        .rope_bf16(
            &input,
            &table("cosine"),
            &table("sine"),
            &[0],
            ferrule_backend::cuda::operators::SplitHalfRopeLayout {
                rows: 1,
                heads: 1,
                head_dim: 4,
                rope_dim: 4,
                table_positions: 1,
                restore_bf16_boundary: true,
            },
        )
        .unwrap();
    assert_eq!(ops.download_bf16_buffer(&rope).unwrap(), values(&b["rope"]));
    let expanded = ops.bf16_to_f32(&input).unwrap();
    assert_eq!(
        ops.download_f32_buffer(&expanded).unwrap(),
        values(&b["input"])
    );
    let mut owner = CudaStandardDecoderOperators::new(
        Rc::clone(&ops),
        ExecutionPrecisionPolicy::bf16_compatibility(),
        &[],
    )
    .unwrap();
    let rows = Rows::Cuda(CudaRows::f32(RowsShape::new(1, 4).unwrap(), None, expanded).unwrap());
    assert!(owner.bind_rows(rows).is_err());
    let foreign = CudaOperators::new_on_device(0).unwrap();
    let rows = Rows::Cuda(
        CudaRows::bf16(
            RowsShape::new(1, 4).unwrap(),
            None,
            foreign.zero_bf16_buffer(4).unwrap(),
        )
        .unwrap(),
    );
    assert!(owner.bind_rows(rows).is_err());
    let rows = owner
        .bind_rows(Rows::Host(
            HostRows::new(
                RowsShape::new(1, 4).unwrap(),
                RowsDType::Bf16,
                None,
                values(&b["input"]),
            )
            .unwrap(),
        ))
        .unwrap();
    assert!(rows.cuda().unwrap().f32_buffer().is_none());
    assert_eq!(
        owner.download_rows(rows).unwrap().values(),
        values(&b["input"])
    );
}
