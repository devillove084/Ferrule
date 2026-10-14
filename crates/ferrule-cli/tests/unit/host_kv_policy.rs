use super::*;
use ferrule_runtime::{HostKvDecodeConfig, KvExecutionPolicy};

#[test]
fn host_kv_cli_leaves_policy_installation_to_composition() {
    let default = serve(&["ferrule", "serve", "model"]);
    assert_eq!(
        worker_config(&default).unwrap().kv_execution_policy,
        KvExecutionPolicy::Resident
    );
    let optin = serve(&[
        "ferrule",
        "serve",
        "model",
        "--kv-execution-policy",
        "host-kv-decode",
        "--max-active-sequences",
        "1",
    ]);
    assert_eq!(
        serving_options(&optin).unwrap().kv_execution_policy,
        KvExecutionPolicy::HostKvDecode(HostKvDecodeConfig::default())
    );
    assert_eq!(
        worker_config(&optin).unwrap().kv_execution_policy,
        WorkerConfig::default().kv_execution_policy
    );
    let options = model_factory_options(&optin).unwrap();
    assert_eq!(options.scheduler_config.max_active_sequences, 1);
    assert_eq!(options.scheduler_config.max_decode_batch, 1);
    assert_eq!(
        options.scheduler_config.prefill_chunk_size,
        optin.prefill_chunk_size
    );
}

#[test]
fn host_kv_cli_public_model_plan_is_reachable_without_cuda_initialization() {
    let fixture = PlanFixture::new(false);
    let args = fixture.args(&[
        "--backend",
        "cuda",
        "--kv-execution-policy",
        "host-kv-decode",
        "--max-active-sequences",
        "1",
    ]);
    let plan = prepare_model(&args).unwrap();
    assert_eq!(plan.backend_profile(), "cuda-f32-host-kv-decode");
    assert_eq!(
        plan.support().arithmetic,
        ferrule_composition::model_factory::ArithmeticPrecision::F32
    );
}
