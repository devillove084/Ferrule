use super::super::process::decoder::{ExpertPlacementEntryFrame, ExpertPlacementFrame};
use super::*;

pub(super) fn boot() -> NumericBoot {
    NumericBoot {
        numeric_expert_boot: NUMERIC_EXPERT_WIRE_VERSION,
        numeric_profile: ferrule_model::transformer::NumericFp8Precision::expert_wire_default(),
        checkpoint: PathBuf::from("unused-cpu-codec-fixture"),
        total_layers: 3,
        layers: 1..3,
        placement: ExpertPlacementFrame {
            source_scope: ExpertSourceScope::ExternalStage,
            source: 7,
            members: vec![109, 42],
            entries: vec![(1, 0, 109), (1, 1, 42), (2, 0, 109)],
            route_entries: None,
            placement_epoch: 0,
            max_tokens: 16,
            max_bytes: 16 * 8 * 4,
            devices: vec![DecoderDevice::Cuda { ordinal: 0 }; 2],
            timeout_ms: 2000,
        },
        owner: 109,
        hidden: 8,
        max_rows: 16,
        max_parameter_bytes: 4096,
        host_max_bytes: 8192,
        host_max_experts: 2,
        host_workers: 1,
        host_headroom_bytes: 0,
        device_bytes: 8192,
        workspace_bytes: 4096,
        expected: [1, 2]
            .map(|layer| NumericSource {
                layer,
                expert: 0,
                identity: [1; 32],
                content_hash: [2; 32],
                encoding: 1,
                generation: 1,
                bytes: 128,
            })
            .to_vec(),
    }
}

#[test]
fn numeric_boot_uses_explicit_route_entries_and_rejects_unowned_replicas() {
    let mut versioned = boot();
    let entries = std::mem::take(&mut versioned.placement.entries);
    versioned.placement.placement_epoch = 7;
    versioned.placement.route_entries = Some(
        entries
            .into_iter()
            .enumerate()
            .map(
                |(physical_slot, (layer, expert, owner))| ExpertPlacementEntryFrame {
                    layer,
                    expert,
                    physical_slot: 20 + physical_slot,
                    replica: 0,
                    owner,
                },
            )
            .collect(),
    );
    let group = versioned.group().unwrap();
    assert_eq!(group.placement.epoch().get(), 7);
    assert_eq!(
        group
            .placement
            .entry(ferrule_model::moe::ExpertId::new(1, 0), 20)
            .unwrap()
            .owner_rank,
        ParallelRankId::new(109)
    );

    let mut replicated = versioned.clone();
    replicated
        .placement
        .route_entries
        .as_mut()
        .unwrap()
        .push(ExpertPlacementEntryFrame {
            layer: 1,
            expert: 0,
            physical_slot: 99,
            replica: 1,
            owner: 42,
        });
    let error = replicated.group().unwrap_err().to_string();
    assert!(error.contains("route unsupported"), "{error}");
    assert!(error.contains("replica=1"), "{error}");
}

pub(super) fn context() -> ExpertDispatchContext {
    ExpertDispatchContext {
        transaction: ExecutionTransactionId::new(99).unwrap(),
        source_rank: ParallelRankId::new(7),
        layer: 1,
    }
}

pub(super) fn token(row: usize, payload: Vec<f32>) -> ExpertToken {
    let c = context();
    ExpertToken {
        transaction: c.transaction,
        sequence: 81 + row as u64,
        source_rank: c.source_rank,
        source_row: row,
        route_slot: row % 2,
        expert: ferrule_model::moe::ExpertId::new(c.layer, 0),
        weight: -0.0,
        payload,
    }
}

// Local accounting of the removed Vec allocations/copies, not a global
// allocator or a claim about serde, BTreeSet, or allocator bookkeeping.
#[derive(Debug, Default, PartialEq, Eq)]
struct Copies {
    allocations: usize,
    payload_bytes: usize,
}
impl Copies {
    fn payload<T>(&mut self, from: &[T], to: &[T]) {
        assert_eq!(from.len(), to.len());
        if !from.is_empty() {
            assert_ne!(from.as_ptr(), to.as_ptr());
            self.allocations += 1;
            self.payload_bytes += std::mem::size_of_val(from);
        }
    }
    fn frames(&mut self, frames: &[NumericToken]) -> Vec<NumericToken> {
        let copy = frames.to_vec();
        if !frames.is_empty() {
            assert_ne!(frames.as_ptr(), copy.as_ptr());
            self.allocations += 1;
        }
        for (a, b) in frames.iter().zip(&copy) {
            self.payload(&a.values, &b.values);
        }
        copy
    }
    fn tokens(&mut self, tokens: &[ExpertToken]) -> Vec<ExpertToken> {
        let copy = tokens.to_vec();
        if !tokens.is_empty() {
            assert_ne!(tokens.as_ptr(), copy.as_ptr());
            self.allocations += 1;
        }
        for (a, b) in tokens.iter().zip(&copy) {
            self.payload(&a.payload, &b.payload);
        }
        copy
    }
}

// Frozen pre-optimization validator, including its group checks, so new and
// legacy paths do not silently agree because they share the changed helper.
fn legacy_validate(
    boot: &NumericBoot,
    context: ExpertDispatchContext,
    frames: Vec<NumericToken>,
    copies: &mut Copies,
) -> Result<Vec<ExpertToken>> {
    if frames.len() > boot.placement.max_tokens {
        return Err(error("numeric token limit"));
    }
    let tokens = frames
        .into_iter()
        .map(NumericToken::decode)
        .collect::<Result<Vec<_>>>()?;
    let mut seen = BTreeSet::new();
    for t in &tokens {
        if t.payload.len() != boot.hidden
            || t.source_row >= boot.max_rows
            || t.route_slot >= boot.placement.max_tokens
            || !seen.insert((t.source_row, t.route_slot))
        {
            return Err(error("numeric route/width bounds"));
        }
    }
    let group = boot.group()?;
    let bucket = ExpertTokenBucket {
        owner_rank: ParallelRankId::new(boot.owner),
        tokens: copies.tokens(&tokens),
    };
    let fail = |message: &str| Error::Execution {
        message: format!("expert runtime: {message}"),
    };
    if context.source_rank != group.source_rank || !group.layers.contains(&context.layer) {
        return Err(fail("unknown expert source or layer"));
    }
    if !group.members.contains(&bucket.owner_rank) {
        return Err(fail("unknown expert owner"));
    }
    if bucket.tokens.len() > group.limits.max_tokens {
        return Err(fail("routed token limit exceeded"));
    }
    let mut bytes = 0usize;
    for t in &bucket.tokens {
        if t.transaction != context.transaction
            || t.source_rank != context.source_rank
            || t.expert.layer != context.layer
        {
            return Err(fail("unknown token transaction, source or layer"));
        }
        if group.placement.owner(t.expert.layer, t.expert.expert) != Some(bucket.owner_rank) {
            return Err(fail("unknown or non-local expert"));
        }
        bytes = t
            .payload
            .len()
            .checked_mul(4)
            .and_then(|n| bytes.checked_add(n))
            .ok_or_else(|| fail("activation byte count overflow"))?;
        if bytes > group.limits.max_bytes {
            return Err(fail("dispatch byte limit exceeded"));
        }
    }
    Ok(tokens)
}

fn command(tokens: Vec<NumericToken>) -> NumericCommand {
    NumericCommand::Compute {
        transaction: context().transaction.get(),
        source: context().source_rank.get(),
        layer: context().layer,
        tokens,
    }
}

#[test]
fn numeric_codec_finite_bit_patterns_preserve_command_and_reply_bytes() {
    let boot = boot();
    let edges = [
        0,
        0x8000_0000,
        1,
        0x8000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
    ];
    let mut seed = 0x1234_5678u32;
    for case in 0..256 {
        let count = case % 17;
        let inputs = (0..count)
            .map(|row| {
                let payload = (0..boot.hidden)
                    .map(|column| {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        let bits = if case == 1 {
                            edges[column]
                        } else if seed & 0x7f80_0000 == 0x7f80_0000 {
                            seed ^ 0x0080_0000
                        } else {
                            seed
                        };
                        f32::from_bits(bits)
                    })
                    .collect();
                let mut t = token(row, payload);
                t.weight = f32::from_bits(edges[case % edges.len()]);
                t
            })
            .collect::<Vec<_>>();
        let frames = inputs.iter().map(NumericToken::encode).collect::<Vec<_>>();
        let before = serde_json::to_vec(&command(frames.clone())).unwrap();
        let ptr = frames.as_ptr();
        let payloads = frames.iter().map(|f| f.values.as_ptr()).collect::<Vec<_>>();
        validate_token_frames(&boot, context(), &frames).unwrap();
        assert_eq!(frames.as_ptr(), ptr);
        assert_eq!(
            payloads,
            frames.iter().map(|f| f.values.as_ptr()).collect::<Vec<_>>()
        );
        let new_command = command(frames);
        assert_eq!(serde_json::to_vec(&new_command).unwrap(), before);
        let NumericCommand::Compute { tokens: frames, .. } = new_command else {
            unreachable!()
        };
        assert_eq!(frames.as_ptr(), ptr);
        let old =
            legacy_validate(&boot, context(), frames.clone(), &mut Copies::default()).unwrap();
        let new = validate_tokens(&boot, context(), frames).unwrap();
        let mut old_reply = Vec::new();
        let mut new_reply = Vec::new();
        for (old, new) in old.iter().zip(&new) {
            assert_eq!(
                old.payload.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                new.payload.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            let output = new
                .payload
                .iter()
                .rev()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>();
            let mut frame = NumericToken::encode(old);
            frame.values = output.clone();
            old_reply.push(frame);
            let output_ptr = output.as_ptr();
            let frame = NumericToken::encode_values(new, output);
            assert_eq!(frame.values.as_ptr(), output_ptr);
            assert!(frame.matches_input(new, boot.hidden));
            new_reply.push(frame);
        }
        assert_eq!(
            serde_json::to_vec(&NumericReply::Results {
                owner: boot.owner,
                tokens: old_reply
            })
            .unwrap(),
            serde_json::to_vec(&NumericReply::Results {
                owner: boot.owner,
                tokens: new_reply
            })
            .unwrap()
        );
    }
}

#[test]
fn numeric_codec_invalid_input_matches_legacy_rejection_order() {
    for case in 0..29 {
        let mut boot = boot();
        let mut context = context();
        let mut frames = (0..2)
            .map(|row| NumericToken::encode(&token(row, vec![0.0; 8])))
            .collect::<Vec<_>>();
        match case {
            0 => frames[0].transaction = 0,
            1 => frames[0].transaction += 1,
            2 => frames[0].source += 1,
            3 => frames[0].layer = 2,
            4 => frames[0].expert = 1,
            5 => frames[0].expert = 999,
            6 => frames[0].row = boot.max_rows,
            7 => frames[0].route = boot.placement.max_tokens,
            8 => {
                frames[1].row = frames[0].row;
                frames[1].route = frames[0].route;
            }
            9 => {
                frames[0].values.pop();
            }
            10 => frames[0].weight = f32::NAN.to_bits(),
            11 => frames[0].weight = f32::INFINITY.to_bits(),
            12 => frames[1].values[0] = f32::NEG_INFINITY.to_bits(),
            13 => boot.placement.max_tokens = 1,
            14 => boot.placement.max_bytes = 63,
            15 => context.source_rank = ParallelRankId::new(8),
            16 => context.layer = 3,
            17 => boot.owner = 999,
            18 => boot.placement.members[1] = boot.owner,
            19 => boot.layers = 2..3,
            20 => boot.expected[0].identity = [0; 32],
            // Multiple simultaneous defects lock down rejection precedence.
            21 => {
                boot.placement.max_tokens = 1;
                frames[0].weight = f32::NAN.to_bits();
            }
            22 => {
                frames[0].transaction = 0;
                frames[0].weight = f32::NAN.to_bits();
            }
            23 => {
                frames[0].values.clear();
                frames[1].weight = f32::NAN.to_bits();
            }
            24 => {
                frames[0].transaction = 0;
                frames[1].weight = f32::NAN.to_bits();
            }
            25 => {
                boot.owner = 999;
                frames[0].row = boot.max_rows;
            }
            26 => {
                context.layer = 3;
                boot.placement.max_bytes = 1;
            }
            27 => {
                frames[0].expert = 1;
                boot.placement.max_bytes = 1;
            }
            28 => {
                frames[1].transaction += 1;
                boot.placement.max_bytes = 1;
            }
            _ => unreachable!(),
        }
        let old = legacy_validate(&boot, context, frames.clone(), &mut Copies::default())
            .unwrap_err()
            .to_string();
        assert_eq!(
            validate_token_frames(&boot, context, &frames)
                .unwrap_err()
                .to_string(),
            old,
            "borrowed case {case}"
        );
        assert_eq!(
            validate_tokens(&boot, context, frames)
                .unwrap_err()
                .to_string(),
            old,
            "consumed case {case}"
        );
    }
    let mut boot = boot();
    boot.placement.max_bytes = 64;
    let frames = (0..2)
        .map(|row| NumericToken::encode(&token(row, vec![0.0; 8])))
        .collect::<Vec<_>>();
    validate_token_frames(&boot, context(), &frames).unwrap();
    validate_tokens(&boot, context(), frames).unwrap();
}

#[test]
fn numeric_codec_reply_correlation_precedes_finite_decode() {
    let input = token(0, vec![0.0; 8]);
    for case in 0..9 {
        let mut frame = NumericToken::encode(&input);
        match case {
            0 => frame.transaction += 1,
            1 => frame.sequence += 1,
            2 => frame.source += 1,
            3 => frame.row += 1,
            4 => frame.route += 1,
            5 => frame.layer += 1,
            6 => frame.expert += 1,
            7 => frame.weight = 0.0f32.to_bits(), // -0.0 is not +0.0 identity
            8 => {
                frame.values.pop();
            }
            _ => unreachable!(),
        }
        frame.values[0] = f32::NAN.to_bits();
        assert!(!frame.matches_input(&input, 8), "case {case}");
    }
    for bits in [
        f32::NAN.to_bits(),
        f32::INFINITY.to_bits(),
        f32::NEG_INFINITY.to_bits(),
    ] {
        let mut frame = NumericToken::encode(&input);
        frame.values[0] = bits;
        assert!(frame.matches_input(&input, 8));
        assert!(
            frame
                .decode()
                .unwrap_err()
                .to_string()
                .contains("nonfinite")
        );
    }
}

#[test]
fn numeric_codec_removed_copy_allocations_are_locally_counted() {
    let boot = boot();
    for count in [1, 2, 16] {
        let inputs = (0..count)
            .map(|row| token(row, vec![0.0; boot.hidden]))
            .collect::<Vec<_>>();
        let frames = inputs.iter().map(NumericToken::encode).collect::<Vec<_>>();
        let mut copies = Copies::default();
        let validation_frames = copies.frames(&frames);
        legacy_validate(&boot, context(), validation_frames, &mut copies).unwrap();
        let command_frames = copies.frames(&frames);
        let child_tokens = legacy_validate(&boot, context(), command_frames, &mut copies).unwrap();
        for t in &child_tokens {
            let frame = NumericToken::encode(t);
            // This independently allocated input encoding was immediately
            // discarded by the legacy output replacement.
            assert_ne!(t.payload.as_ptr().cast::<u32>(), frame.values.as_ptr());
            copies.allocations += 1;
            copies.payload_bytes += frame.values.len() * 4;
        }
        assert_eq!(
            copies,
            Copies {
                allocations: 5 * count + 4,
                payload_bytes: 5 * count * boot.hidden * 4,
            }
        );
        eprintln!(
            "N={count} H={}: removed Vec allocations={}, copied/encoded payload bytes={}",
            boot.hidden, copies.allocations, copies.payload_bytes
        );
    }
}

#[test]
fn numeric_codec_shared_metadata_rejects_overflow_membership_and_bounds() {
    let boot = boot();
    let mut group = boot.group().unwrap();
    let context = context();
    let owner = ParallelRankId::new(boot.owner);
    let metadata = |width| {
        (
            context.transaction.get(),
            context.source_rank,
            context.layer,
            0,
            width,
        )
    };
    group.limits.max_bytes = usize::MAX;
    for widths in [vec![usize::MAX], vec![usize::MAX / 4, 1]] {
        let error = group
            .validate_token_metadata(context, owner, widths.into_iter().map(metadata))
            .unwrap_err()
            .to_string();
        assert!(error.contains("activation byte count overflow"), "{error}");
    }
    for (owner, tokens, message) in [
        (ParallelRankId::new(999), vec![], "unknown expert owner"),
        (
            owner,
            vec![metadata(8); boot.placement.max_tokens + 1],
            "routed token limit exceeded",
        ),
    ] {
        assert!(
            group
                .validate_token_metadata(context, owner, tokens.into_iter())
                .unwrap_err()
                .to_string()
                .contains(message)
        );
    }
    group.limits.max_bytes = 32;
    group
        .validate_token_metadata(context, owner, [metadata(8)].into_iter())
        .unwrap();
    assert!(
        group
            .validate_token_metadata(context, owner, [metadata(9)].into_iter())
            .unwrap_err()
            .to_string()
            .contains("dispatch byte limit exceeded")
    );
}
