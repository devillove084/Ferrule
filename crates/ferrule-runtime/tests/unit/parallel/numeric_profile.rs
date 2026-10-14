use super::*;
use ferrule_model::transformer::NumericFp8Precision;

#[test]
fn numeric_profile_boot_roundtrip_and_legacy_default() {
    for profile in [
        NumericFp8Precision::F32Tf32x3,
        NumericFp8Precision::W8A16Bf16F32Accumulate,
    ] {
        let mut boot = codec_tests::boot();
        boot.numeric_profile = profile;
        boot.group().unwrap();
        let json = serde_json::to_value(&boot).unwrap();
        assert_eq!(
            json["numeric_profile"],
            match profile {
                NumericFp8Precision::F32Tf32x3 => "f32",
                NumericFp8Precision::W8A16Bf16F32Accumulate => "w8a16-bf16",
                NumericFp8Precision::Bf16RneF32Accumulate
                | NumericFp8Precision::Bf16ModelF32Accumulate
                | NumericFp8Precision::Bf16ModelTiledF32Accumulate
                | NumericFp8Precision::Bf16ModelSlicedK2F32Accumulate
                | NumericFp8Precision::Bf16ModelMixedF32Accumulate => unreachable!(),
            }
        );
        let restored: NumericBoot = serde_json::from_value(json).unwrap();
        assert_eq!(restored.numeric_profile, profile);
        assert_eq!(restored.workspace_bytes, boot.workspace_bytes);
        assert_eq!(restored.device_bytes, boot.device_bytes);
        assert_eq!(restored.group().unwrap(), boot.group().unwrap());
    }
    let mut legacy = serde_json::to_value(codec_tests::boot()).unwrap();
    legacy.as_object_mut().unwrap().remove("numeric_profile");
    assert_eq!(
        serde_json::from_value::<NumericBoot>(legacy.clone())
            .unwrap()
            .numeric_profile,
        NumericFp8Precision::F32Tf32x3
    );
    for unknown in ["bf16", "auto", "unknown"] {
        legacy["numeric_profile"] = serde_json::json!(unknown);
        assert!(serde_json::from_value::<NumericBoot>(legacy.clone()).is_err());
    }
}

#[test]
fn numeric_profile_mixed_owner_boots_are_rejected_before_spawn() {
    let first = codec_tests::boot();
    let mut second = first.clone();
    second.owner = 42;
    second.expected.truncate(1);
    second.expected[0].expert = 1;
    second.numeric_profile = NumericFp8Precision::W8A16Bf16F32Accumulate;
    let boots = [first, second]
        .into_iter()
        .enumerate()
        .map(|(slot, boot)| {
            let identity = ProcessIdentity::new(
                ProcessGroupEpoch::new(1).unwrap(),
                ParallelRankId::new(boot.owner),
                ProcessOwnerInstanceId::new(slot as u64 + 1).unwrap(),
            );
            (identity, boot)
        })
        .collect();
    let error = ExpertParallelExecutor::new_numeric_process(
        ProcessLaunch::new("must-not-spawn-mixed-numeric-profiles"),
        boots,
        ProcessOwnerConfig::default(),
    )
    .err()
    .expect("mixed profiles must fail before launching a process");
    assert!(
        error
            .to_string()
            .contains("inconsistent scoped numeric boots"),
        "{error}"
    );
}

#[test]
fn numeric_wire_rejects_model_only_profile_without_cuda_or_payload() {
    for profile in [
        NumericFp8Precision::Bf16RneF32Accumulate,
        NumericFp8Precision::Bf16ModelF32Accumulate,
        NumericFp8Precision::Bf16ModelTiledF32Accumulate,
        NumericFp8Precision::Bf16ModelSlicedK2F32Accumulate,
        NumericFp8Precision::Bf16ModelMixedF32Accumulate,
    ] {
        let mut boot = codec_tests::boot();
        boot.numeric_profile = profile;
        assert!(boot.group().is_err());
        assert!(super::super::expert::validate_numeric_profile(boot.numeric_profile).is_err());
        assert!(serde_json::to_value(boot).is_err());
    }
    for profile in [
        NumericFp8Precision::F32Tf32x3,
        NumericFp8Precision::W8A16Bf16F32Accumulate,
    ] {
        assert_eq!(
            super::super::expert::validate_numeric_profile(profile).unwrap(),
            profile
        );
    }
}
