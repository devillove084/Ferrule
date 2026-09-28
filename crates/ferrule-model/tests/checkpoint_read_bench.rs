//! CPU-only NAS I/O probe. No backend/device creation, no full-model execution.
//! Run with FERRULE_NUMERIC_FP8_MODEL_DIR and optionally
//! FERRULE_CHECKPOINT_BENCH_ITERS (default 64). Only the first 128 rows of one
//! expert projection per shard are read. FERRULE_CHECKPOINT_BENCH_MODE selects
//! all (default), preplanned, rebuild, or legacy for separate perf/strace runs.
//! Output is once per phase, never per read.
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Instant;

use ferrule_model::TensorRole;
use ferrule_model::checkpoint::{
    CheckpointDType, CheckpointSourceFileIdentity, CheckpointTensorReader, CheckpointTensorSlice,
    HfSafetensorsIndex, NumericFp8Encoding, NumericFp8Source,
};

fn tensor(
    path: &Path,
    name: &str,
    header: &serde_json::Value,
    data_start: u64,
) -> CheckpointTensorSlice {
    let info = &header[name];
    let offsets = info["data_offsets"].as_array().unwrap();
    let start = offsets[0].as_u64().unwrap();
    CheckpointTensorSlice {
        name: name.into(),
        role: TensorRole::RoutedExpertDown,
        path: path.into(),
        offset: data_start + start,
        bytes: offsets[1].as_u64().unwrap() - start,
        dtype: CheckpointDType::from_safetensors_dtype(info["dtype"].as_str().unwrap()),
        shape: serde_json::from_value(info["shape"].clone()).unwrap(),
    }
}

#[test]
#[ignore = "CPU-only NAS probe; set FERRULE_NUMERIC_FP8_MODEL_DIR"]
fn nas_small_expert_loop() {
    let directory =
        std::env::var("FERRULE_NUMERIC_FP8_MODEL_DIR").expect("set FERRULE_NUMERIC_FP8_MODEL_DIR");
    let iterations: u64 = std::env::var("FERRULE_CHECKPOINT_BENCH_ITERS")
        .map(|s| s.parse().unwrap())
        .unwrap_or(64);
    assert!(iterations > 0);
    let mode = std::env::var("FERRULE_CHECKPOINT_BENCH_MODE").unwrap_or_else(|_| "all".into());
    assert!(matches!(
        mode.as_str(),
        "all" | "preplanned" | "rebuild" | "legacy"
    ));
    let directory = Path::new(&directory);
    let index = HfSafetensorsIndex::open(directory.join("model.safetensors.index.json")).unwrap();
    let mut selected = BTreeMap::new();
    for (name, shard) in &index.weight_map {
        if name.contains(".mlp.experts.") && name.ends_with(".weight") {
            let scale = format!("{}_scale_inv", name);
            if index.weight_map.get(&scale) == Some(shard) {
                selected.entry(shard).or_insert(name);
            }
        }
    }
    assert!(!selected.is_empty());
    let reader = CheckpointTensorReader::new(16 << 20);
    assert!(selected.len() <= reader.max_shards());
    let mut sources = Vec::new();
    let mut snapshots = Vec::new();
    let mut legacy_metadata = Vec::new();
    for (shard, name) in selected {
        let path = directory.join(shard);
        let mut file = File::open(&path).unwrap();
        let mut size = [0; 8];
        file.read_exact(&mut size).unwrap();
        let size = u64::from_le_bytes(size);
        assert!(size <= 100 << 20);
        let mut bytes = vec![0; size as usize];
        file.read_exact(&mut bytes).unwrap();
        let header: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let weight = tensor(&path, name, &header, size + 8);
        let scale = tensor(&path, &format!("{name}_scale_inv"), &header, size + 8);
        assert_eq!(weight.shape.len(), 2);
        let encoding = match scale.dtype {
            CheckpointDType::Bf16 => NumericFp8Encoding::E4M3FnBlock128Bf16,
            CheckpointDType::F32 => NumericFp8Encoding::E4M3FnBlock128F32,
            _ => panic!("expected numeric BF16/F32 scales"),
        };
        let snapshot = CheckpointSourceFileIdentity::capture(&path).unwrap();
        legacy_metadata.push(std::fs::metadata(&path).unwrap());
        sources.push(
            NumericFp8Source::new(weight, scale, encoding, snapshot.clone(), snapshot.clone())
                .unwrap(),
        );
        snapshots.push(snapshot);
    }
    let plans: Vec<_> = sources
        .iter()
        .map(|source| {
            let [rows, cols] = source.matrix_shape();
            source
                .plan_tile(&reader, None, 0..rows.min(128), 0..cols)
                .unwrap()
        })
        .collect();
    let expected: Vec<_> = plans
        .iter()
        .map(|read| read.read(&reader).unwrap())
        .collect();
    if mode == "all" || mode == "preplanned" {
        let before = CheckpointSourceFileIdentity::counters();
        let start = Instant::now();
        for _ in 0..iterations {
            for (read, expected) in plans.iter().zip(&expected) {
                let actual = read.read(&reader).unwrap();
                assert_eq!(actual.weight_bytes(), expected.weight_bytes());
                assert_eq!(actual.scale_bytes(), expected.scale_bytes());
                std::hint::black_box(actual);
            }
        }
        let elapsed = start.elapsed();
        let after = CheckpointSourceFileIdentity::counters();
        assert_eq!(after.capture_calls, before.capture_calls);
        assert_eq!(after.canonicalize_calls, before.canonicalize_calls);
        assert_eq!(reader.read_counters().open_calls, sources.len() as u64);
        eprintln!(
            "preplanned: shards={} iterations={iterations} elapsed={elapsed:?} counters={:?}; source capture/canonicalize delta=0/0",
            sources.len(),
            reader.read_counters()
        );
    }

    // Include the old API's per-projection source construction/planning cost.
    if mode == "all" || mode == "rebuild" {
        let before = CheckpointSourceFileIdentity::counters();
        let start = Instant::now();
        for _ in 0..iterations {
            for ((source, snapshot), expected) in sources.iter().zip(&snapshots).zip(&expected) {
                let source = NumericFp8Source::new(
                    source.weight().clone(),
                    source.scale().clone(),
                    source.encoding(),
                    snapshot.clone(),
                    snapshot.clone(),
                )
                .unwrap();
                let [rows, cols] = source.matrix_shape();
                let actual = source
                    .plan_tile(&reader, None, 0..rows.min(128), 0..cols)
                    .unwrap()
                    .read(&reader)
                    .unwrap();
                assert_eq!(actual.weight_bytes(), expected.weight_bytes());
                assert_eq!(actual.scale_bytes(), expected.scale_bytes());
                std::hint::black_box(actual);
            }
        }
        let after = CheckpointSourceFileIdentity::counters();
        eprintln!(
            "construct_plan_read: elapsed={:?} path_metadata_delta={} captures_delta={} canonicalize_delta={}",
            start.elapsed(),
            after.path_metadata_calls - before.path_metadata_calls,
            after.capture_calls - before.capture_calls,
            after.canonicalize_calls - before.canonicalize_calls
        );
        assert_eq!(reader.read_counters().open_calls, sources.len() as u64);
        eprintln!(
            "reader counters (including warmup): {:?}",
            reader.read_counters()
        );
    }

    // Reproduce the diagnosed old I/O sequence: 16 canonicalize+stat captures,
    // two open/seek/read/close operations per pair. This baseline is I/O-only
    // (no NaN/scale scan), deliberately giving the old path a timing advantage.
    if mode == "all" || mode == "legacy" {
        let start = Instant::now();
        for _ in 0..iterations {
            for (((read, expected), snapshot), metadata) in plans
                .iter()
                .zip(&expected)
                .zip(&snapshots)
                .zip(&legacy_metadata)
            {
                for _ in 0..16 {
                    let canonical = std::fs::canonicalize(snapshot.catalog_path()).unwrap();
                    assert_eq!(canonical, snapshot.canonical_path());
                    let current = std::fs::metadata(canonical).unwrap();
                    assert_eq!(current.len(), metadata.len());
                    assert_eq!(current.modified().unwrap(), metadata.modified().unwrap());
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::MetadataExt;
                        assert_eq!(
                            (current.dev(), current.ino()),
                            (metadata.dev(), metadata.ino())
                        );
                    }
                }
                for (part, expected) in [
                    (read.weight_read(), expected.weight_bytes()),
                    (read.scale_read(), expected.scale_bytes()),
                ] {
                    let mut file = File::open(&part.tensor().path).unwrap();
                    let mut packed = vec![0; part.read_plan().storage_bytes() as usize];
                    let mut cursor = 0;
                    for extent in part.read_plan().extents() {
                        let end = cursor + extent.bytes() as usize;
                        file.seek(SeekFrom::Start(extent.offset())).unwrap();
                        file.read_exact(&mut packed[cursor..end]).unwrap();
                        cursor = end;
                    }
                    assert_eq!(packed, expected);
                    std::hint::black_box(packed);
                }
            }
        }
        eprintln!(
            "legacy_io_only: elapsed={:?} opens={} captures={} (16 canonicalize+stat per pair); selected bytes/pass={}",
            start.elapsed(),
            iterations * sources.len() as u64 * 2,
            iterations * sources.len() as u64 * 16,
            plans
                .iter()
                .map(|p| p.read_plan().storage_bytes())
                .sum::<u64>()
        );
    }
}
