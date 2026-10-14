use super::codec_tests::{boot, context, token};
use super::*;
use std::cell::Cell;

use super::super::expert::numeric_test_fixture as fixture;

fn unknown(message: impl std::fmt::Display) -> ProcessHandlerError {
    ProcessHandlerError::unknown(ProcessFailureKind::Handler, message.to_string())
}

thread_local! { static GROUP_CALLS: Cell<usize> = const { Cell::new(0) }; }
pub(super) fn record_group_validation() {
    GROUP_CALLS.with(|count| count.set(count.get() + 1));
}
fn group_calls() -> usize {
    GROUP_CALLS.with(Cell::get)
}
fn reset_calls() {
    GROUP_CALLS.with(|count| count.set(0));
}

fn identity(owner: u32) -> ProcessIdentity {
    ProcessIdentity::new(
        ProcessGroupEpoch::new(7).unwrap(),
        ParallelRankId::new(owner),
        ProcessOwnerInstanceId::new(9).unwrap(),
    )
}
fn validated(boot: NumericBoot) -> Arc<ValidatedNumericBoot> {
    let id = identity(boot.owner);
    ValidatedNumericBoot::build(boot, id, ProcessFrameLimits::default()).unwrap()
}
fn frames() -> Vec<NumericToken> {
    (0..2)
        .map(|row| NumericToken::encode(&token(row, vec![-0.0; 8])))
        .collect()
}
fn ready(v: &ValidatedNumericBoot) -> NumericProcessStats {
    NumericProcessStats {
        pid: 123,
        owner: v.boot.owner,
        slot: v.rank.slot.get(),
        device: v.device,
        owned_experts: v
            .boot
            .expected
            .iter()
            .map(|s| (s.layer, s.expert))
            .collect(),
        calls: 0,
        tokens: 0,
        host_payload_bytes: v.payload_bytes,
        host_read_bytes: v.payload_bytes,
        host_hits: 0,
        resident_experts: v.boot.expected.len(),
        resident_bytes: v.payload_bytes as usize,
        uploads: v.boot.expected.len() as u64,
        evictions: 0,
        pending_upload_bytes: 0,
        quarantined: false,
    }
}

#[test]
fn boot_cache_once_moves_catalog_and_reuses_group_for_compute_and_stats() {
    let boot = boot();
    let catalog_ptr = boot.expected.as_ptr();
    let placement_ptr = boot.placement.entries.as_ptr();
    reset_calls();
    let v = validated(boot);
    assert_eq!(group_calls(), 1);
    assert_eq!(catalog_ptr, v.boot.expected.as_ptr());
    assert_eq!(placement_ptr, v.boot.placement.entries.as_ptr());
    let frames = frames();
    let stats = ready(&v);
    for _ in 0..100 {
        v.validate_token_frames(context(), &frames).unwrap();
        let decoded = v.validate_tokens(context(), frames.clone()).unwrap();
        assert_eq!(decoded.len(), frames.len());
        assert!(valid_stats(&v, 123, &stats));
    }
    assert_eq!(group_calls(), 1, "request path rebuilt Boot proof");
    let weak = Arc::downgrade(&v);
    drop(v);
    assert!(weak.upgrade().is_none());
}

#[test]
fn boot_cache_concurrent_requests_share_only_immutable_authority() {
    let v = validated(boot());
    std::thread::scope(|scope| {
        for worker in 0..8 {
            let v = Arc::clone(&v);
            scope.spawn(move || {
                reset_calls();
                for iteration in 0..50 {
                    let mut context = context();
                    context.transaction =
                        ExecutionTransactionId::new(1 + worker * 50 + iteration).unwrap();
                    let mut frames = frames();
                    for f in &mut frames {
                        f.transaction = context.transaction.get();
                        f.sequence += iteration;
                    }
                    v.validate_token_frames(context, &frames).unwrap();
                    let decoded = v.validate_tokens(context, frames.clone()).unwrap();
                    for (t, f) in decoded.iter().zip(&frames) {
                        assert_eq!(t.sequence, f.sequence);
                    }
                }
                assert_eq!(group_calls(), 0);
            });
        }
    });
    assert_eq!(Arc::strong_count(&v), 1);
}

#[test]
fn boot_cache_rejects_boot_mutations_before_publication_and_never_reuses_old_token() {
    let v = validated(boot());
    for case in 0..18 {
        let mut candidate = v.boot.clone();
        let mut id = v.identity;
        let mut limits = ProcessFrameLimits::default();
        match case {
            0 => candidate.numeric_expert_boot = 1,
            1 => candidate.total_layers = 1,
            2 => candidate.layers = 2..3,
            3 => candidate.placement.source_scope = ExpertSourceScope::Member,
            4 => candidate.placement.source = candidate.owner,
            5 => candidate.expected[0].identity = [0; 32],
            6 => candidate.expected[0].content_hash = [0; 32],
            7 => candidate.expected[0].encoding = 0,
            8 => candidate.expected[0].generation = 0,
            9 => candidate.expected[0].bytes = 0,
            10 => candidate.expected.push(candidate.expected[0].clone()),
            11 => candidate.placement.members[1] = candidate.owner,
            12 => candidate.placement.devices.clear(),
            13 => candidate.owner = 999,
            14 => candidate.hidden = 0,
            15 => candidate.placement.timeout_ms = 0,
            16 => id.rank = ParallelRankId::new(42),
            17 => limits.max_command_bytes = 1,
            _ => unreachable!(),
        }
        assert!(
            ValidatedNumericBoot::build(candidate, id, limits).is_err(),
            "case {case}"
        );
    }
    let replacement = validated(v.boot.clone());
    assert!(!Arc::ptr_eq(&v, &replacement));
    assert_ne!(v.boot.expected.as_ptr(), replacement.boot.expected.as_ptr());
    v.validate_token_frames(context(), &frames()).unwrap();
}

#[test]
fn boot_cache_preserves_request_rejections_and_finite_bit_identity() {
    let v = validated(boot());
    for case in 0..17 {
        let mut frames = frames();
        let mut context = context();
        match case {
            0 => frames[0].transaction = 0,
            1 => frames[0].transaction += 1,
            2 => frames[0].source += 1,
            3 => frames[0].layer = 2,
            4 => frames[0].expert = 1,
            5 => frames[0].expert = 999,
            6 => frames[0].row = v.boot.max_rows,
            7 => frames[0].route = v.boot.placement.max_tokens,
            8 => {
                frames[1].row = frames[0].row;
                frames[1].route = frames[0].route;
            }
            9 => {
                frames[0].values.pop();
            }
            10 => frames[0].weight = f32::NAN.to_bits(),
            11 => frames[0].values[0] = f32::INFINITY.to_bits(),
            12 => context.source_rank = ParallelRankId::new(8),
            13 => context.layer = 3,
            14 => frames.resize(v.boot.placement.max_tokens + 1, frames[0].clone()),
            15 => {
                frames[0].values.clear();
                frames[1].weight = f32::NAN.to_bits();
            }
            16 => {
                frames[0].transaction = 0;
                frames[1].weight = f32::NAN.to_bits();
            }
            _ => unreachable!(),
        }
        let legacy = validate_token_frames(&v.boot, context, &frames)
            .unwrap_err()
            .to_string();
        assert_eq!(
            v.validate_token_frames(context, &frames)
                .unwrap_err()
                .to_string(),
            legacy,
            "proxy case {case}"
        );
        let legacy = validate_tokens(&v.boot, context, frames.clone())
            .unwrap_err()
            .to_string();
        assert_eq!(
            v.validate_tokens(context, frames).unwrap_err().to_string(),
            legacy,
            "child case {case}"
        );
    }
    let input = token(0, vec![-0.0; 8]);
    let encoded = NumericToken::encode(&input);
    let output = v.validate_tokens(context(), vec![encoded]).unwrap();
    assert!(NumericToken::encode(&output[0]).matches_input(&input, 8));
    let mut wrong = NumericToken::encode(&output[0]);
    wrong.sequence += 1;
    assert!(!wrong.matches_input(&input, 8));
}

fn resource_boot(path: &std::path::Path) -> (NumericBoot, BoundDecoderResources) {
    let resources = fixture::bind(path, 4, true, true);
    let mut boot = boot();
    boot.checkpoint = path.into();
    boot.total_layers = resources.spec().layers().len();
    boot.layers = 0..boot.total_layers;
    boot.placement.entries = boot
        .layers
        .clone()
        .flat_map(|l| (0..4).map(move |e| (l, e, if e % 2 == 0 { 109 } else { 42 })))
        .collect();
    boot.host_max_experts = boot.total_layers * 2;
    boot.host_max_bytes = 8 << 20;
    boot.device_bytes = 1 << 20;
    let group = boot.placement.decode(boot.layers.clone()).unwrap();
    boot.expected = owned_catalog(&resources, &group, ParallelRankId::new(109)).unwrap();
    (boot, resources)
}

#[test]
fn boot_cache_rebind_checks_content_encoding_generation_and_live_source_each_time() {
    use ferrule_model::transformer::host_experts::HostExpertWarmPlan;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("weights.bin");
    let (boot, resources) = resource_boot(&path);
    let v = validated(boot);
    v.validate_resources(&resources).unwrap();
    for case in 0..5 {
        let mut modified = v.boot.clone();
        match case {
            0 => modified.expected[0].identity[0] ^= 1,
            1 => modified.expected[0].content_hash[0] ^= 1,
            2 => modified.expected[0].encoding += 1,
            3 => modified.expected[0].generation += 1,
            4 => modified.expected[0].bytes += 1,
            _ => unreachable!(),
        }
        let candidate = validated(modified);
        assert!(
            candidate.validate_resources(&resources).is_err(),
            "case {case}"
        );
    }
    let warm = || {
        HostExpertWarmPlan::for_owned(
            resources.state_dict(),
            v.boot.host_options(),
            v.boot.max_parameter_bytes,
            v.boot.expected.iter().map(|s| (s.layer, s.expert)),
        )
        .unwrap()
        .warm(|_| {})
        .unwrap()
    };
    let cache = warm();
    let replacement = warm();
    preflight_numeric_source(&cache, cache.generation()).unwrap();
    assert!(preflight_numeric_source(&replacement, cache.generation()).is_err());
    let replaced = directory.path().join("replaced.bin");
    std::fs::copy(&path, &replaced).unwrap();
    std::fs::rename(&replaced, &path).unwrap();
    assert!(preflight_numeric_source(&cache, cache.generation()).is_err());
}

// Exercise the production child custody wrapper around the same private token.
// This is a host-only handler, not a numeric CPU fallback.
struct TokenHandler {
    validated: Arc<ValidatedNumericBoot>,
    drop_seen: Arc<AtomicBool>,
}
impl Drop for TokenHandler {
    fn drop(&mut self) {
        self.drop_seen.store(true, Ordering::Release);
    }
}
impl ProcessChildHandler for TokenHandler {
    type Command = NumericCommand;
    type Output = NumericReply;
    fn execute(
        &mut self,
        request: ProcessRequest<NumericCommand>,
    ) -> std::result::Result<NumericReply, ProcessHandlerError> {
        if request.identity.owner != self.validated.identity || request.session != 0 {
            return Err(unknown("numeric envelope mismatch"));
        }
        let NumericCommand::Compute {
            transaction,
            source,
            layer,
            tokens,
        } = request.command
        else {
            return Err(unknown("injected Unknown"));
        };
        if transaction != request.identity.transaction.get() {
            return Err(unknown("outer transaction mismatch"));
        }
        let context = ExpertDispatchContext {
            transaction: ExecutionTransactionId::new(transaction).map_err(unknown)?,
            source_rank: ParallelRankId::new(source),
            layer,
        };
        let tokens = self
            .validated
            .validate_tokens(context, tokens)
            .map_err(unknown)?;
        Ok(NumericReply::Results {
            owner: self.validated.boot.owner,
            tokens: tokens.iter().map(NumericToken::encode).collect(),
        })
    }
}

fn send_peer(stream: &mut std::os::unix::net::UnixStream, message: serde_json::Value) {
    use std::io::Write;
    let bytes = serde_json::to_vec(
        &serde_json::json!({"version":PROCESS_PROTOCOL_VERSION,"message":message}),
    )
    .unwrap();
    stream
        .write_all(&(bytes.len() as u32).to_be_bytes())
        .unwrap();
    stream.write_all(&bytes).unwrap();
}
fn receive_peer(stream: &mut std::os::unix::net::UnixStream) -> serde_json::Value {
    use std::io::Read;
    let mut prefix = [0; 4];
    stream.read_exact(&mut prefix).unwrap();
    let len = u32::from_be_bytes(prefix) as usize;
    assert!(len < 1 << 20);
    let mut bytes = vec![0; len];
    stream.read_exact(&mut bytes).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    value["message"].clone()
}

#[test]
fn boot_cache_child_shutdown_drops_but_unknown_retains_token_and_never_replays() {
    use serde_json::json;
    use std::os::unix::net::UnixStream;
    for mode in ["shutdown", "unknown", "replay", "invalid_boot"] {
        let (mut peer, reader) = UnixStream::pair().unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let writer = reader.try_clone().unwrap();
        let dropped = Arc::new(AtomicBool::new(false));
        let drop_seen = Arc::clone(&dropped);
        let (send, recv) = std::sync::mpsc::channel();
        let child = std::thread::spawn(move || {
            reset_calls();
            let result =
                child_serve_with(reader, writer, ProcessChildConfig::default(), move |boot| {
                    let v = ValidatedNumericBoot::build(
                        serde_json::from_value(boot.config).map_err(startup)?,
                        boot.identity,
                        boot.limits,
                    )
                    .map_err(startup)?;
                    send.send(Arc::downgrade(&v)).unwrap();
                    Ok(TokenHandler {
                        validated: v,
                        drop_seen,
                    })
                });
            (result, group_calls())
        });
        let mut config = boot();
        let id = identity(config.owner);
        if mode == "invalid_boot" {
            config.numeric_expert_boot = 1;
        }
        send_peer(
            &mut peer,
            json!({"type":"boot","identity":id,"limits":ProcessFrameLimits::default(),"config":config}),
        );
        let first = receive_peer(&mut peer);
        if mode == "invalid_boot" {
            assert_eq!(first["type"], "failure");
            let (result, calls) = child.join().unwrap();
            assert!(matches!(
                result,
                Err(ProcessError::RemoteFailure {
                    failure: ProcessHandlerError {
                        kind: ProcessFailureKind::Startup,
                        ..
                    }
                })
            ));
            assert_eq!(calls, 1);
            assert!(recv.recv().is_err());
            continue;
        }
        assert_eq!(first["type"], "ready");
        let weak = recv.recv().unwrap();
        let command = NumericCommand::Compute {
            transaction: context().transaction.get(),
            source: context().source_rank.get(),
            layer: context().layer,
            tokens: frames(),
        };
        let key = ProcessCommandIdentity {
            owner: id,
            command: ProcessCommandId::new(1).unwrap(),
            transaction: context().transaction.into(),
        };
        let execute = json!({"type":"execute","identity":key,"session":0,"payload":command});
        send_peer(&mut peer, execute.clone());
        assert_eq!(receive_peer(&mut peer)["type"], "complete");
        match mode {
            "shutdown" => {
                send_peer(&mut peer, json!({"type":"shutdown","owner":id}));
                assert_eq!(receive_peer(&mut peer)["type"], "shutdown_ack");
            }
            "unknown" => {
                let key = ProcessCommandIdentity {
                    command: ProcessCommandId::new(2).unwrap(),
                    ..key
                };
                send_peer(
                    &mut peer,
                    json!({"type":"execute","identity":key,"session":0,"payload":NumericCommand::Stats}),
                );
                let negative = receive_peer(&mut peer);
                assert_eq!(negative["type"], "failure");
                assert_eq!(negative["failure"]["quiescence"], "unknown");
            }
            "replay" => send_peer(&mut peer, execute),
            _ => unreachable!(),
        }
        let (result, calls) = child.join().unwrap();
        assert_eq!(calls, 1, "child compute rebuilt Boot validation");
        if mode == "shutdown" {
            assert!(result.is_ok());
            assert!(dropped.load(Ordering::Acquire));
            assert!(weak.upgrade().is_none());
        } else {
            assert!(result.is_err());
            assert!(!dropped.load(Ordering::Acquire));
            assert!(weak.upgrade().is_some());
        }
    }
}

/// Test-only guard also instruments the actual Proxy::execute hot path reached
/// by the existing real-child tests. Production builds contain no counters.
pub(super) struct RequestValidationGuard(usize);
impl RequestValidationGuard {
    pub(super) fn new() -> Self {
        Self(group_calls())
    }
}
impl Drop for RequestValidationGuard {
    fn drop(&mut self) {
        assert_eq!(
            group_calls(),
            self.0,
            "Compute/Stats rebuilt immutable Boot metadata"
        );
    }
}

#[test]
fn boot_cache_keeps_exact_dispatch_bytes_and_ready_identity_checks() {
    let mut b = boot();
    b.placement.max_bytes = 63;
    let v = validated(b);
    assert!(v.validate_token_frames(context(), &frames()).is_err());
    assert!(v.validate_tokens(context(), frames()).is_err());
    let v = validated(boot());
    let stats = ready(&v);
    for case in 0..9 {
        let mut s = stats.clone();
        match case {
            0 => s.pid += 1,
            1 => s.owner += 1,
            2 => s.slot += 1,
            3 => s.device = DecoderDevice::Cuda { ordinal: 1 },
            4 => s.owned_experts[0].1 += 1,
            5 => s.host_payload_bytes += 1,
            6 => s.pending_upload_bytes = 1,
            7 => s.evictions = 1,
            8 => s.quarantined = true,
            _ => unreachable!(),
        }
        assert!(!valid_stats(&v, 123, &s));
    }
}

#[test]
fn application_numeric_binding_is_not_called_before_boot_checks_and_mismatch_precedes_payload() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("weights.bin");
    let (mut boot, resources) = resource_boot(&path);
    let id = identity(boot.owner);
    let called = std::cell::Cell::new(false);
    boot.expected[0].identity[0] ^= 1;
    std::fs::remove_file(&path).unwrap();
    let result = NumericExpertChild::initialize(
        ProcessBoot {
            identity: id,
            limits: ProcessFrameLimits::default(),
            config: serde_json::to_value(&boot).unwrap(),
        },
        |_| {
            called.set(true);
            Ok(resources.clone())
        },
    );
    let failure = result
        .err()
        .expect("mismatch must not warm or initialize CUDA");
    if cfg!(feature = "cuda") {
        assert!(called.get());
        assert!(
            failure.message.contains("rebound source catalog/geometry"),
            "{failure:?}"
        );
    } else {
        assert!(!called.get());
        assert!(failure.message.contains("requires CUDA"), "{failure:?}");
    }
    boot.numeric_expert_boot = 0;
    called.set(false);
    assert!(
        NumericExpertChild::initialize(
            ProcessBoot {
                identity: id,
                limits: ProcessFrameLimits::default(),
                config: serde_json::to_value(&boot).unwrap(),
            },
            |_| {
                called.set(true);
                Ok(resources)
            }
        )
        .is_err()
    );
    assert!(!called.get());
}
