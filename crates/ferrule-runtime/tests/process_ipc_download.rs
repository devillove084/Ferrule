//! Optional PR26 D2H segment microbaseline, NOT a process/GPU inference baseline.
//! Build with --features cuda; run the exact ignored test with an external 100s
//! timeout and FERRULE_PR26_ARTIFACT_DIR pointing at an ignored artifact folder.
#![cfg(feature = "cuda")]

use std::rc::Rc;
use std::time::Instant;

use ferrule_backend::cuda::context::CudaOperators;
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::transformer::{
    CudaRows, CudaStandardDecoderOperators, DeviceSegmentOutput, Rows, RowsShape, SegmentOutput,
};
use serde_json::json;

#[test]
#[ignore = "requires one CUDA GPU; actual pinned download boundary; external 100s timeout"]
fn activation_download_boundary_baseline() {
    let directory = std::path::PathBuf::from(
        std::env::var_os("FERRULE_PR26_ARTIFACT_DIR").expect("set ignored artifact directory"),
    );
    assert!(directory.is_dir());
    let ops = Rc::new(CudaOperators::new_on_device(0).expect("requires visible CUDA device 0"));
    let mut operators =
        CudaStandardDecoderOperators::new(ops.clone(), ExecutionPrecisionPolicy::f32(), &[])
            .unwrap();
    let mut cases = Vec::new();
    for (rows, width) in [(1, 4), (8, 4), (1, 4096), (8, 4096)] {
        let values: Vec<_> = (0..rows * width)
            .map(|i| (i % 257) as f32 / 32.0 - 4.0)
            .collect();
        let mut samples = Vec::new();
        for i in 0..36 {
            let buffer = ops.upload_f32_buffer(&values).unwrap();
            // Exclude input preparation/H2D; the measured boundary still does
            // its real producer-event, pinned staging, completion and host copy.
            operators.quiesce().unwrap();
            let output = DeviceSegmentOutput::Hidden {
                next_layer: 1,
                rows: Rows::Cuda(
                    CudaRows::f32(RowsShape::new(rows, width).unwrap(), None, buffer).unwrap(),
                ),
            };
            let start = Instant::now();
            // This is the exact implementation called by the process stage's
            // CudaStandardDecoderSegment::download_output, without altering it.
            let host = output.into_host(&mut operators).unwrap();
            let ns = start.elapsed().as_nanos();
            let SegmentOutput::Hidden { rows: host, .. } = host else {
                unreachable!()
            };
            assert_eq!(host.values(), values);
            if i >= 4 {
                samples.push(ns);
            }
        }
        samples.sort_unstable();
        cases.push(json!({"rows":rows,"width":width,"f32_bytes":values.len()*4,
            "iterations":32,"median_ns":samples[16],"p95_ns":samples[30],
            "total_ns":samples.iter().sum::<u128>()}));
    }
    operators.quiesce().unwrap();
    std::fs::write(directory.join("download.json"), serde_json::to_vec_pretty(&json!({
        "schema":"ferrule.pr26.download.v1", "warmup":4,"iterations":32,"device":0,
        "scope":"isolated DeviceSegmentOutput::into_host pinned D2H; no child inference; do not add to CPU roundtrip", "cases":cases,
    })).unwrap()).unwrap();
}
