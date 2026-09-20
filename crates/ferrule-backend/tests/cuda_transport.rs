//! Real-driver coverage, deliberately ignored; run with an external process
//! timeout and --test-threads=1. No runtime/model/collective dependency, no
//! injected real CUDA failures, and no CUDA handles cross owner threads.
#![cfg(feature = "cuda")]

use std::mem::size_of;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use ferrule_backend::cuda::operators::linear::CudaOperators;
use ferrule_backend::cuda::providers::CudaContext;
use ferrule_backend::cuda::{
    CudaAsyncTransport, CudaTransferCompletion, CudaTransferConfig, CudaTransferDirection,
    CudaTransferError, CudaTransferTicket,
};

const CHUNK: usize = 4;
const TIMEOUT: Duration = Duration::from_secs(10);

fn config(tx_slots: usize, rx_slots: usize) -> CudaTransferConfig {
    CudaTransferConfig {
        chunk_elements: CHUNK,
        tx_slots,
        rx_slots,
    }
}

fn require_devices(count: usize) {
    let visible = CudaContext::device_count().expect("enumerate real CUDA devices");
    assert!(
        visible >= count,
        "requires {count} CUDA devices, found {visible}"
    );
}

fn poll_ready(
    transport: &mut CudaAsyncTransport,
    ticket: &CudaTransferTicket,
) -> CudaTransferCompletion {
    let deadline = Instant::now() + TIMEOUT;
    loop {
        if let Some(result) = transport.poll(ticket).expect("poll exact event") {
            return result;
        }
        assert!(Instant::now() < deadline, "CUDA transfer poll timed out");
        thread::sleep(Duration::from_micros(50));
    }
}

fn downloaded(completion: CudaTransferCompletion) -> Box<[f32]> {
    match completion {
        CudaTransferCompletion::D2H { values, .. } => values,
        other => panic!("expected D2H values, got {other:?}"),
    }
}

#[test]
#[ignore = "requires two real CUDA GPUs and an external timeout"]
fn two_owner_pinned_payload_relay_with_offsets_and_tail_chunks() {
    require_devices(2);
    let payload: Vec<f32> = (0..10).map(|i| i as f32 + 0.25).collect();
    let expected = payload.clone();
    let (send, receive) = mpsc::sync_channel::<(usize, Box<[f32]>)>(1);
    let producer = thread::spawn(move || {
        let ops = CudaOperators::new_on_device(0).unwrap();
        let mut transport = ops.new_async_transport(config(2, 2)).unwrap();
        let mut source = ops.zero_f32_buffer(payload.len() + 2).unwrap();
        let before = transport.stats();
        let allocations = ops.allocator_metrics().allocation_requests;
        for (chunk, values) in payload.chunks(CHUNK).enumerate() {
            let offset = chunk * CHUNK;
            let upload = transport
                .submit_h2d_f32(&mut source, offset + 1, values)
                .unwrap();
            transport.wait_h2d_on_compute(&upload).unwrap();
            let produced = ops.record_compute_event().unwrap();
            let download = transport
                .submit_d2h_f32_after(&source, offset + 1, values.len(), &produced)
                .unwrap();
            let values = downloaded(poll_ready(&mut transport, &download));
            assert!(matches!(
                poll_ready(&mut transport, &upload),
                CudaTransferCompletion::H2D { .. }
            ));
            // Only completed CPU-owned bytes cross the owner boundary.
            send.send((offset, values)).unwrap();
        }
        let after = transport.stats();
        assert_eq!(after.pinned_allocations, before.pinned_allocations);
        assert_eq!(after.pinned_bytes, before.pinned_bytes);
        assert_eq!(after.device_holds_high_water, 2);
        assert_eq!(after.device_holds_in_use, 0);
        assert_eq!(ops.allocator_metrics().allocation_requests, allocations);
        assert!(transport.drain().unwrap().is_empty());
    });
    let consumer = thread::spawn(move || {
        let ops = CudaOperators::new_on_device(1).unwrap();
        let mut transport = ops.new_async_transport(config(2, 2)).unwrap();
        let mut destination = ops.zero_f32_buffer(12).unwrap();
        let before = transport.stats();
        for expected_offset in [0, 4, 8] {
            let (offset, values) = receive.recv_timeout(TIMEOUT).expect("host payload relay");
            assert_eq!(offset, expected_offset);
            let upload = transport
                .submit_h2d_f32(&mut destination, offset + 1, &values)
                .unwrap();
            drop(values); // DMA must read TX pinned storage, never this Box.
            transport.wait_h2d_on_compute(&upload).unwrap();
            assert!(matches!(
                poll_ready(&mut transport, &upload),
                CudaTransferCompletion::H2D { .. }
            ));
        }
        let produced = ops.record_compute_event().unwrap();
        let mut output = Vec::new();
        for offset in [0, 4, 8] {
            let ticket = transport
                .submit_d2h_f32_after(&destination, offset, CHUNK, &produced)
                .unwrap();
            output.extend_from_slice(&downloaded(poll_ready(&mut transport, &ticket)));
        }
        assert_eq!(
            transport.stats().pinned_allocations,
            before.pinned_allocations
        );
        assert_eq!(transport.stats().device_holds_in_use, 0);
        output
    });
    // Always join both owners before reporting either owner's panic.
    let producer_result = producer.join();
    let consumer_result = consumer.join();
    producer_result.expect("producer owner");
    let output = consumer_result.expect("consumer owner");
    assert_eq!(output[0], 0.0);
    assert_eq!(&output[1..11], expected.as_slice());
    assert_eq!(output[11], 0.0);
}

#[test]
#[ignore = "requires two real CUDA GPUs and an external timeout"]
fn foreign_owners_and_tickets_are_rejected_without_restricting_expert_waits() {
    require_devices(2);
    let ops = CudaOperators::new_on_device(0).unwrap();
    let same_ordinal_foreign = CudaOperators::new_on_device(0).unwrap();
    let other_device = CudaOperators::new_on_device(1).unwrap();
    let mut transport = ops.new_async_transport(config(1, 1)).unwrap();
    let mut other_transport = ops.new_async_transport(config(1, 1)).unwrap();
    let mut local = ops.zero_f32_buffer(CHUNK).unwrap();
    for foreign in [&same_ordinal_foreign, &other_device] {
        let mut buffer = foreign.zero_f32_buffer(CHUNK).unwrap();
        let produced = foreign.record_compute_event().unwrap();
        let before = transport.stats();
        assert!(
            transport
                .submit_h2d_f32(&mut buffer, 0, &[1.0])
                .unwrap_err()
                .to_string()
                .contains("owner")
        );
        let local_produced = ops.record_compute_event().unwrap();
        assert!(
            transport
                .submit_d2h_f32_after(&buffer, 0, 1, &local_produced)
                .unwrap_err()
                .to_string()
                .contains("owner")
        );
        assert!(
            transport
                .submit_d2h_f32_after(&local, 0, 1, &produced)
                .unwrap_err()
                .to_string()
                .contains("owner")
        );
        assert_eq!(transport.stats(), before);
        // The existing expert fence boundary explicitly permits cross-owner
        // event waits. Transport validation must not change this global path.
        let upload_event = foreign.record_upload_event().unwrap();
        ops.wait_upload_event(&upload_event).unwrap();
        ops.wait_compute_event_on_upload_stream(&produced).unwrap();
        ops.sync_stream().unwrap();
        ops.sync_upload_stream().unwrap();
    }
    let ticket = transport.submit_h2d_f32(&mut local, 0, &[1.0]).unwrap();
    assert!(matches!(
        other_transport.poll(&ticket),
        Err(CudaTransferError::ForeignTicket)
    ));
    assert!(matches!(
        other_transport.cancel(&ticket),
        Err(CudaTransferError::ForeignTicket)
    ));
    assert!(matches!(
        other_transport.wait_h2d_on_compute(&ticket),
        Err(CudaTransferError::ForeignTicket)
    ));
    assert!(matches!(
        poll_ready(&mut transport, &ticket),
        CudaTransferCompletion::H2D { .. }
    ));
    let reused = transport.submit_h2d_f32(&mut local, 0, &[2.0]).unwrap();
    assert_eq!(ticket.id().slot(), reused.id().slot());
    assert!(reused.id().generation() > ticket.id().generation());
    assert!(matches!(
        transport.poll(&ticket),
        Err(CudaTransferError::StaleTicket)
    ));
    assert!(matches!(
        transport.cancel(&ticket),
        Err(CudaTransferError::StaleTicket)
    ));
    assert!(matches!(
        transport.wait_h2d_on_compute(&ticket),
        Err(CudaTransferError::StaleTicket)
    ));
    transport.drain().unwrap();
}

/// A bounded host callback makes NotReady deterministic without any invalid
/// CUDA operation. Drop releases it before the transport's potentially blocking
/// destructor; the callback watchdog also bounds assertion/unwind mistakes.
struct ComputeGate(Arc<AtomicBool>);
impl ComputeGate {
    fn enqueue(ops: &CudaOperators) -> Self {
        let release = Arc::new(AtomicBool::new(false));
        let worker = Arc::clone(&release);
        ops.notify_compute_stream(move || {
            let deadline = Instant::now() + TIMEOUT;
            while !worker.load(Ordering::Acquire) && Instant::now() < deadline {
                thread::sleep(Duration::from_micros(100));
            }
        })
        .unwrap();
        Self(release)
    }
    fn release(&self) {
        self.0.store(true, Ordering::Release);
    }
}
impl Drop for ComputeGate {
    fn drop(&mut self) {
        self.release();
    }
}

#[test]
#[ignore = "requires one real CUDA GPU and an external timeout"]
fn pending_d2h_cancel_retains_pinned_slot_and_source_allocation() {
    require_devices(1);
    let ops = CudaOperators::new().unwrap();
    let mut transport = ops.new_async_transport(config(1, 1)).unwrap();
    let source = ops.zero_f32_buffer(CHUNK).unwrap();
    let spare = ops.zero_f32_buffer(CHUNK).unwrap();
    ops.sync_stream().unwrap();
    let live = ops.allocator_metrics().live_requested_bytes;
    let allocations = ops.allocator_metrics().allocation_requests;
    let gate = ComputeGate::enqueue(&ops);
    let produced = ops.record_compute_event().unwrap();
    let ticket = transport
        .submit_d2h_f32_after(&source, 0, CHUNK, &produced)
        .unwrap();
    assert!(transport.poll(&ticket).unwrap().is_none());
    assert_eq!(transport.stats().device_holds_in_use, 1);
    drop(source);
    assert_eq!(ops.allocator_metrics().live_requested_bytes, live);
    transport.cancel(&ticket).unwrap();
    transport.cancel(&ticket).unwrap();
    assert_eq!(transport.stats().cancel_intents, 1);
    assert!(transport.poll(&ticket).unwrap().is_none());
    assert_eq!(transport.stats().rx_in_use, 1);
    assert!(matches!(
        transport.submit_d2h_f32_after(&spare, 0, 1, &produced),
        Err(CudaTransferError::Backpressure(CudaTransferDirection::D2H))
    ));
    gate.release();
    assert_eq!(
        poll_ready(&mut transport, &ticket),
        CudaTransferCompletion::Canceled { id: ticket.id() }
    );
    assert_eq!(transport.stats().rx_in_use, 0);
    assert_eq!(
        ops.allocator_metrics().live_requested_bytes,
        live - CHUNK * size_of::<f32>()
    );
    assert_eq!(ops.allocator_metrics().allocation_requests, allocations);
    let next = transport
        .submit_d2h_f32_after(&spare, 1, 2, &produced)
        .unwrap();
    assert!(next.id().generation() > ticket.id().generation());
    assert_eq!(
        downloaded(poll_ready(&mut transport, &next)).as_ref(),
        &[0.0, 0.0]
    );
}

#[test]
#[ignore = "requires one real CUDA GPU and an external timeout"]
fn tx_backpressure_ready_cancellation_and_fixed_allocation_counters() {
    require_devices(1);
    let ops = CudaOperators::new().unwrap();
    let mut transport = ops.new_async_transport(config(2, 1)).unwrap();
    let mut destination = ops.zero_f32_buffer(12).unwrap();
    let before = transport.stats();
    assert_eq!(before.pinned_allocations, 3);
    assert_eq!(before.pinned_bytes, 3 * CHUNK * size_of::<f32>());
    let first = transport
        .submit_h2d_f32(&mut destination, 1, &[1.0, 2.0, 3.0, 4.0])
        .unwrap();
    let second = transport
        .submit_h2d_f32(&mut destination, 6, &[5.0, 6.0])
        .unwrap();
    let busy = transport.stats();
    assert!(matches!(
        transport.submit_h2d_f32(&mut destination, 9, &[7.0]),
        Err(CudaTransferError::Backpressure(CudaTransferDirection::H2D))
    ));
    assert_eq!(transport.stats(), busy);
    assert_eq!(busy.tx_high_water, 2);
    assert_eq!(busy.device_holds_high_water, 2);
    assert_eq!(busy.pinned_allocations, before.pinned_allocations);
    assert_eq!(busy.pinned_bytes, before.pinned_bytes);
    // Even physical readiness does not return the slot before owner consumption.
    ops.sync_upload_stream().unwrap();
    transport.cancel(&first).unwrap();
    assert!(matches!(
        transport.wait_h2d_on_compute(&first),
        Err(CudaTransferError::Canceled)
    ));
    assert_eq!(transport.stats().tx_in_use, 2);
    assert_eq!(
        poll_ready(&mut transport, &first),
        CudaTransferCompletion::Canceled { id: first.id() }
    );
    transport.wait_h2d_on_compute(&second).unwrap();
    assert_eq!(
        poll_ready(&mut transport, &second),
        CudaTransferCompletion::H2D { id: second.id() }
    );
    assert_eq!(transport.stats().tx_in_use, 0);
}

#[test]
#[ignore = "requires one real CUDA GPU and an external timeout"]
fn drain_cancels_both_directions_and_drop_fences_abandoned_tickets() {
    require_devices(1);
    let ops = CudaOperators::new().unwrap();
    let mut transport = ops.new_async_transport(config(1, 1)).unwrap();
    let mut buffer = ops.zero_f32_buffer(CHUNK).unwrap();
    let upload = transport
        .submit_h2d_f32(&mut buffer, 0, &[1.0, 2.0, 3.0, 4.0])
        .unwrap();
    transport.wait_h2d_on_compute(&upload).unwrap();
    let producer = ops.record_compute_event().unwrap();
    let download = transport
        .submit_d2h_f32_after(&buffer, 0, CHUNK, &producer)
        .unwrap();
    assert!(matches!(
        transport.wait_h2d_on_compute(&download),
        Err(CudaTransferError::WrongDirection)
    ));
    drop(buffer);
    let completed = transport.drain().unwrap();
    assert_eq!(
        completed,
        vec![
            CudaTransferCompletion::Canceled { id: upload.id() },
            CudaTransferCompletion::Canceled { id: download.id() },
        ]
    );
    assert_eq!(transport.stats().device_holds_in_use, 0);
    assert_eq!(transport.stats().canceled_completions, 2);
    assert_eq!(ops.allocator_metrics().live_requested_bytes, 0);
    assert!(matches!(
        transport.poll(&download),
        Err(CudaTransferError::StaleTicket)
    ));
    assert!(transport.drain().unwrap().is_empty());

    let mut buffer = ops.zero_f32_buffer(CHUNK).unwrap();
    let abandoned = transport.submit_h2d_f32(&mut buffer, 0, &[7.0]).unwrap();
    drop(abandoned);
    drop(buffer);
    assert_eq!(transport.stats().tx_in_use, 1);
    assert_eq!(
        ops.allocator_metrics().live_requested_bytes,
        CHUNK * size_of::<f32>()
    );
    drop(transport);
    assert_eq!(ops.allocator_metrics().live_requested_bytes, 0);
}

#[test]
#[ignore = "requires one real CUDA GPU and an external timeout"]
fn invalid_ranges_and_lengths_do_not_admit_or_grow_slots() {
    require_devices(1);
    let ops = CudaOperators::new().unwrap();
    let mut transport = ops.new_async_transport(config(1, 1)).unwrap();
    let mut buffer = ops.zero_f32_buffer(CHUNK).unwrap();
    let produced = ops.record_compute_event().unwrap();
    let before = transport.stats();
    for (offset, len) in [(usize::MAX, 1), (CHUNK, 1), (0, CHUNK + 1), (0, 0)] {
        assert!(
            transport
                .submit_h2d_f32(&mut buffer, offset, &vec![1.0; len])
                .is_err()
        );
        assert!(
            transport
                .submit_d2h_f32_after(&buffer, offset, len, &produced)
                .is_err()
        );
        assert_eq!(transport.stats(), before);
    }
}
