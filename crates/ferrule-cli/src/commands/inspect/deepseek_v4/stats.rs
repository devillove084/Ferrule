use ferrule_model::models::deepseek_v4::DeepSeekV4ReferenceRunner;

pub(super) fn print_deepseek_v4_runtime_stats(runner: &DeepSeekV4ReferenceRunner) {
    let stats = runner.layer_runtime_stats();
    if stats.is_empty() {
        return;
    }
    println!("layer stats:");
    for stat in stats {
        println!(
            "  L{:>2}: window_kv={} compressed_kv={} indexer_kv={} resident_experts={} resident_bytes={}",
            stat.layer,
            stat.window_kv_len,
            stat.compressed_kv_len,
            stat.indexer_compressed_kv_len,
            stat.resident_experts,
            stat.resident_expert_bytes,
        );
    }
}
