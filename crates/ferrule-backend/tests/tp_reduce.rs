//! CPU-only contracts and pinned artifact inventory. No CUDA initialization.
#[allow(dead_code)]
#[path = "../src/cuda/operators/tp_reduce_contract.rs"]
mod contract;
#[path = "../src/cuda/providers/nccl_sha256.rs"]
mod sha256;
use contract::*;
use std::{fs::File, io::BufReader, path::Path};

#[test]
fn original_member_order_exact_allocator_grant_and_unknown_private_bounds() {
    let p = TpReducePlan::new([37, 9], STORAGE_GRANT_BYTES).unwrap();
    assert_eq!(p.members(), [37, 9]);
    assert!(TpReducePlan::new([9, 9], STORAGE_GRANT_BYTES).is_err());
    assert!(TpReducePlan::new([37, 9], STORAGE_GRANT_BYTES - 1).is_err());
    assert_eq!(p.count(8).unwrap(), 16384);
    assert!(p.count(0).is_err());
    assert!(p.count(9).is_err());
    assert_eq!(STORAGE_BYTES, 131076);
    assert_eq!(
        Some(STORAGE_GRANT_BYTES),
        ferrule_backend::cuda_allocation_grant_bytes(STORAGE_BYTES)
    );
    let e = p.estimate();
    assert_eq!(e.nccl_private_device, PrivateAllocationBound::Unknown);
    assert_eq!(e.nccl_private_host, PrivateAllocationBound::Unknown);
    assert_eq!(e.configured_sliding_buffer_bytes, None);
    assert!(e.hard_cap_bytes().is_err());
}
#[test]
fn first_member_not_plus_zero_and_finite_host_oracle() {
    assert_eq!(ordered_host_word(0x8000, 0x8000), Ok(0x8000));
    assert_eq!(ordered_host_word(0x0000, 0x8000), Ok(0x0000));
    assert_eq!(ordered_host_word(0x0001, 0x0001), Ok(0x0002));
    assert_eq!(ordered_host_word(0x0001, 0x8001), Ok(0));
    assert_eq!(ordered_host_word(0x3f80, 0x3b80), Ok(0x3f80));
    assert_eq!(ordered_host_word(0x3f81, 0x3b80), Ok(0x3f82));
    for value in [0x7f80, 0xff80, 0x7fc1, 0xffc1] {
        assert_eq!(ordered_host_word(value, 0), Err(INPUT_NONFINITE));
        assert_eq!(ordered_host_word(0, value), Err(INPUT_NONFINITE));
    }
    assert_eq!(ordered_host_word(0x7f7f, 0x7f7f), Err(OUTPUT_NONFINITE));
    // F32 is still finite here; BF16 RNE alone overflows.
    assert_eq!(ordered_host_word(0x7f7f, 0x7b00), Err(OUTPUT_NONFINITE));
}
#[test]
fn unknown_never_authorizes_retirement_even_after_later_success() {
    for phase in [Phase::Initializing, Phase::Submitting, Phase::Unknown] {
        assert!(!phase.can_release(false));
        assert!(!phase.can_release(true));
    }
    assert!(Phase::InFlight.can_release(true));
    assert!(!Phase::InFlight.can_release(false));
    assert!(Phase::Rejected.can_release(true));
}
#[test]
fn preallocated_input_custody_budget_has_no_zero_or_overflow_escape() {
    assert_eq!(checked_input_custody(2, 48, 96).unwrap(), 96);
    assert!(checked_input_custody(2, 48, 95).is_err());
    assert!(checked_input_custody(0, 48, 96).is_err());
    assert!(checked_input_custody(2, 0, 96).is_err());
    assert!(checked_input_custody(usize::MAX, 48, usize::MAX).is_err());
}

#[test]
fn sha256_standard_vectors_and_padding_boundaries() {
    assert_eq!(
        sha256::digest(&b""[..]).unwrap(),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256::digest(&b"abc"[..]).unwrap(),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256::digest(&b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"[..]).unwrap(),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
    assert_eq!(
        sha256::digest(&vec![b'a'; 1000000][..]).unwrap(),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}
#[test]
fn local_pinned_cuda12_inventory_header_library_and_metadata() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/vendor/vllm-baseline/lib/python3.10/site-packages");
    if !root.join("nvidia/nccl/include/nccl.h").exists() {
        return;
    }
    let metadata =
        std::fs::read_to_string(root.join("nvidia_nccl_cu12-2.27.5.dist-info/METADATA")).unwrap();
    assert!(metadata.lines().any(|l| l == "Name: nvidia-nccl-cu12"));
    assert!(metadata.lines().any(|l| l == "Version: 2.27.5"));
    for (file, expected) in [
        ("nvidia/nccl/include/nccl.h", NCCL_HEADER_SHA256),
        ("nvidia/nccl/lib/libnccl.so.2", NCCL_LIBRARY_SHA256),
    ] {
        assert_eq!(
            sha256::digest(BufReader::new(File::open(root.join(file)).unwrap())).unwrap(),
            expected
        );
    }
}
#[test]
fn native_api_has_no_sum_env_mutation_payload_copy_or_sync() {
    let source = include_str!("../native/cuda/core/tp_reduce.cuh");
    assert!(source.contains("__fadd_rn"));
    assert!(source.contains("config.blocking = 0"));
    for forbidden in [
        "ncclAllReduce(",
        "ncclSum",
        "setenv(",
        "cudaStreamSynchronize(",
        "cudaDeviceSynchronize(",
    ] {
        assert!(!source.contains(forbidden), "{forbidden}");
    }
    assert!(source.contains("cudaMemcpy(out, flag(c), sizeof(*out), cudaMemcpyDeviceToHost)"));
}

#[cfg(feature = "cuda")]
#[test]
fn exact_native_library_load_is_cpu_only_and_rejects_relative_paths_and_host_budget() {
    use ferrule_backend::cuda::operators::tp_reduce::NcclLibrary;
    let bytes = NcclLibrary::required_native_handle_bytes().unwrap();
    let custody =
        ferrule_backend::cuda::operators::tp_reduce::InternalTpReduceTwo::input_custody_estimate(2)
            .unwrap();
    println!(
        "Measured native library+two-comm metadata minimum: {bytes} bytes; two-site retained input views: {} bytes; NCCL-private host/device bounds remain Unknown",
        custody.input_view_bytes
    );
    assert!(NcclLibrary::load(Path::new("libnccl.so.2"), bytes).is_err());
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/vendor/vllm-baseline/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2")
        .canonicalize().unwrap();
    assert!(NcclLibrary::load(&path, bytes - 1).is_err());
    let lib = NcclLibrary::load(&path, bytes).unwrap();
    assert_eq!(lib.source(), path);
    assert_eq!(lib.sha256(), NCCL_LIBRARY_SHA256);
}
