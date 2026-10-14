//! Opt-in plain-CPU benchmark of the real process codec; no pipes or GPU.
use super::*;
use crate::parallel::expert_process::{NumericCommand, NumericReply, NumericToken};
use std::collections::BTreeMap;
use std::hint::black_box;
use std::path::Path;

fn measure<T>(mut operation: impl FnMut() -> T) -> u128 {
    for _ in 0..4 {
        black_box(operation());
    }
    let mut samples = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..8 {
            black_box(operation());
        }
        samples.push(start.elapsed().as_nanos() / 8);
    }
    samples.sort_unstable();
    samples[2]
}

fn tensor(path: &Path, name: &str, dtype: &str) -> Vec<u8> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).unwrap();
    let mut prefix = [0; 8];
    file.read_exact(&mut prefix).unwrap();
    let len = u64::from_le_bytes(prefix);
    assert!(len < 16 * 1024 * 1024);
    let mut header = vec![0; len as usize];
    file.read_exact(&mut header).unwrap();
    let header: Value = serde_json::from_slice(&header).unwrap();
    let spec = &header[name];
    assert_eq!(spec["dtype"], dtype, "{name}");
    let start = spec["data_offsets"][0].as_u64().unwrap();
    let end = spec["data_offsets"][1].as_u64().unwrap();
    file.seek(SeekFrom::Start(8 + len + start)).unwrap();
    let mut bytes = vec![0; (end - start) as usize];
    file.read_exact(&mut bytes).unwrap();
    bytes
}

fn identity() -> ProcessCommandIdentity {
    use super::super::*;
    ProcessCommandIdentity {
        owner: ProcessIdentity::new(
            ProcessGroupEpoch::new(1).unwrap(),
            ferrule_common::ParallelRankId::new(1),
            ProcessOwnerInstanceId::new(1).unwrap(),
        ),
        command: ProcessCommandId::new(1).unwrap(),
        transaction: ProcessTransactionId::new(1).unwrap(),
    }
}

#[test]
#[ignore = "release plain CPU only; FERRULE_NUMERIC_CPU_ORACLE and FERRULE_NUMERIC_CPU_REPORT required"]
fn numeric_process_plain_cpu() {
    assert!(!black_box(cfg!(debug_assertions)));
    assert!(!black_box(cfg!(feature = "cuda")));
    assert!(!black_box(timing::INSTRUMENTED));
    let directory =
        std::env::var_os("FERRULE_NUMERIC_CPU_ORACLE").expect("existing F32 oracle directory");
    let mut records = Vec::new();
    let limit = ProcessFrameLimits::default().max_frame_bytes;
    for case in ["hello", "capital"] {
        let path = Path::new(&directory).join(format!("{case}.safetensors"));
        for stage in ["prefill", "decode.0"] {
            let mut buckets = Vec::new();
            let mut histogram = BTreeMap::<usize, usize>::new();
            for layer in 0..40 {
                let prefix = format!("{stage}.layers.{layer:02}");
                let ids = tensor(&path, &format!("{prefix}.selected_expert_ids"), "I64");
                let ids = ids
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|v| i64::from_le_bytes(*v) as usize)
                    .collect::<Vec<_>>();
                let weights = tensor(&path, &format!("{prefix}.routing_weights"), "F32");
                let weights = weights
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| u32::from_le_bytes(*v))
                    .collect::<Vec<_>>();
                // Captured post-layer hidden values: real F32 magnitudes/bit
                // patterns, NOT claimed to be the unrecorded expert input/output.
                let values = tensor(&path, &format!("{prefix}.hidden"), "F32");
                let values = values
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| u32::from_le_bytes(*v))
                    .collect::<Vec<_>>();
                assert_eq!(values.len() / 2048 * 8, ids.len());
                assert_eq!(ids.len(), weights.len());
                for owner in 0..8 {
                    let tokens = ids
                        .iter()
                        .enumerate()
                        .filter(|(_, id)| **id % 8 == owner)
                        .map(|(i, &expert)| NumericToken {
                            transaction: 1,
                            sequence: 1,
                            source: 0,
                            row: i / 8,
                            route: i % 8,
                            layer,
                            expert,
                            weight: weights[i],
                            values: values[i / 8 * 2048..(i / 8 + 1) * 2048].to_vec(),
                        })
                        .collect::<Vec<_>>();
                    *histogram.entry(tokens.len()).or_default() += 1;
                    if !tokens.is_empty() {
                        buckets.push((layer, owner, tokens));
                    }
                }
            }
            for (layer, owner, tokens) in buckets {
                let n = tokens.len();
                let domain =
                    crate::parallel::expert_process::numeric_domain_cpu::costs(owner, &tokens);
                let command = NumericCommand::Compute {
                    transaction: 1,
                    source: 0,
                    layer,
                    tokens: tokens.clone(),
                };
                let reply = NumericReply::Results {
                    owner: owner as u32 + 1,
                    tokens,
                };
                let p = payload(&command, limit).unwrap();
                let wire = encode(
                    Message::Execute {
                        identity: identity(),
                        session: 0,
                        payload: p.clone(),
                    },
                    limit,
                )
                .unwrap();
                let r = payload(&reply, limit).unwrap();
                let reply_wire = encode(
                    Message::Complete {
                        identity: identity(),
                        session: 0,
                        payload: r.clone(),
                    },
                    limit,
                )
                .unwrap();
                let command_serialize = measure(|| serialize(black_box(&command), limit).unwrap());
                let payload_ns = measure(|| payload(black_box(&command), limit).unwrap());
                let envelope = Envelope {
                    version: PROCESS_PROTOCOL_VERSION,
                    message: Message::Execute {
                        identity: identity(),
                        session: 0,
                        payload: &p,
                    },
                };
                let envelope_encode = measure(|| serialize(black_box(&envelope), limit).unwrap());
                let frame_decode =
                    measure(|| serde_json::from_slice::<Envelope>(black_box(&wire)).unwrap());
                let size_validate = measure(|| serialize(black_box(&p), limit).unwrap());
                // Setup clones excluded: each value is consumed exactly once.
                let mut samples = Vec::new();
                for _ in 0..5 {
                    let values = (0..8).map(|_| p.clone()).collect::<Vec<_>>();
                    let start = Instant::now();
                    for v in values {
                        black_box(serde_json::from_value::<NumericCommand>(v).unwrap());
                    }
                    samples.push(start.elapsed().as_nanos() / 8);
                }
                samples.sort_unstable();
                let from_value = samples[2];
                let reply_encode = measure(|| {
                    let p = payload(black_box(&reply), limit).unwrap();
                    encode(
                        Message::Complete {
                            identity: identity(),
                            session: 0,
                            payload: p,
                        },
                        limit,
                    )
                    .unwrap()
                });
                let roundtrip = measure(|| {
                    let p = payload(black_box(&command), limit).unwrap();
                    let wire = encode(
                        Message::Execute {
                            identity: identity(),
                            session: 0,
                            payload: p,
                        },
                        limit,
                    )
                    .unwrap();
                    let env: Envelope = serde_json::from_slice(&wire).unwrap();
                    let Message::Execute { payload: p, .. } = env.message else {
                        unreachable!()
                    };
                    serialize(&p, limit).unwrap();
                    black_box(serde_json::from_value::<NumericCommand>(p).unwrap());
                    let p = payload(black_box(&reply), limit).unwrap();
                    let wire = encode(
                        Message::Complete {
                            identity: identity(),
                            session: 0,
                            payload: p,
                        },
                        limit,
                    )
                    .unwrap();
                    let env: Envelope = serde_json::from_slice(&wire).unwrap();
                    let Message::Complete { payload: p, .. } = env.message else {
                        unreachable!()
                    };
                    serialize(&p, limit).unwrap();
                    black_box(serde_json::from_value::<NumericReply>(p).unwrap());
                });
                let raw_command = serialize(&command, limit).unwrap();
                let raw = raw_payload(&raw_command).unwrap();
                let raw_wire = encode_message(
                    Message::Execute {
                        identity: identity(),
                        session: 0,
                        payload: raw,
                    },
                    limit,
                )
                .unwrap();
                assert_eq!(
                    serde_json::from_slice::<Value>(&wire).unwrap(),
                    serde_json::from_slice::<Value>(&raw_wire).unwrap()
                );
                assert_eq!(wire.len(), raw_wire.len());
                let raw_payload_ns = measure(|| raw_payload(black_box(&raw_command)).unwrap());
                let raw_frame_decode = measure(|| decode_frame(black_box(&raw_wire)).unwrap());
                let raw_size_validate =
                    measure(|| check_payload(black_box(raw), black_box(limit)).unwrap());
                let typed_decode = measure(|| {
                    serde_json::from_str::<NumericCommand>(black_box(raw.get())).unwrap()
                });
                let raw_reply_encode = measure(|| {
                    let bytes = serialize(black_box(&reply), limit).unwrap();
                    let p = raw_payload(&bytes).unwrap();
                    encode_message(
                        Message::Complete {
                            identity: identity(),
                            session: 0,
                            payload: p,
                        },
                        limit,
                    )
                    .unwrap()
                });
                let raw_roundtrip = measure(|| {
                    let bytes = serialize(black_box(&command), limit).unwrap();
                    let p = raw_payload(&bytes).unwrap();
                    let wire = encode_message(
                        Message::Execute {
                            identity: identity(),
                            session: 0,
                            payload: p,
                        },
                        limit,
                    )
                    .unwrap();
                    let Message::Execute { payload: p, .. } = decode_frame(&wire).unwrap() else {
                        unreachable!()
                    };
                    check_payload(p, limit).unwrap();
                    black_box(serde_json::from_str::<NumericCommand>(p.get()).unwrap());
                    let bytes = serialize(black_box(&reply), limit).unwrap();
                    let p = raw_payload(&bytes).unwrap();
                    let wire = encode_message(
                        Message::Complete {
                            identity: identity(),
                            session: 0,
                            payload: p,
                        },
                        limit,
                    )
                    .unwrap();
                    let Message::Complete { payload: p, .. } = decode_frame(&wire).unwrap() else {
                        unreachable!()
                    };
                    check_payload(p, limit).unwrap();
                    black_box(serde_json::from_str::<NumericReply>(p.get()).unwrap());
                });
                records.push(serde_json::json!({"case":case,"stage":stage,"layer":layer,"owner":owner,"tokens":n,"host_bytes":n*2048*4,
                    "command_bytes":serialize(&command,limit).unwrap().len(),"frame_bytes":wire.len(),"reply_frame_bytes":reply_wire.len(),
                    "domain_ns":domain,"ns":{"serialize":command_serialize,"payload_inclusive":payload_ns,"envelope_encode":envelope_encode,"frame_decode":frame_decode,"size_validate":size_validate,"from_value":from_value,"reply_encode_inclusive":reply_encode,"roundtrip_nonoverlap":roundtrip,"raw_payload":raw_payload_ns,"raw_frame_decode":raw_frame_decode,"raw_size_validate":raw_size_validate,"typed_decode":typed_decode,"raw_reply_encode":raw_reply_encode,"raw_roundtrip_nonoverlap":raw_roundtrip}}));
            }
            eprintln!("{case}/{stage} 40x8 owner-bucket histogram: {histogram:?}");
        }
    }
    let output = std::env::var_os("FERRULE_NUMERIC_CPU_REPORT").expect("report file");
    std::fs::write(
        output,
        serde_json::to_vec_pretty(
            &serde_json::json!({"plain_cpu":true,"instrumented":false,"records":records}),
        )
        .unwrap(),
    )
    .unwrap();
}
