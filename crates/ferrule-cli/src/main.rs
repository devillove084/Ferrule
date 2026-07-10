use clap::Parser;

mod args;
mod bench;
mod commands;

pub(crate) use args::SamplingArgs;
use args::{Cli, Command};
use commands::bench_interactive::cmd_bench_interactive;
use commands::chat::cmd_chat;
use commands::cuda::cmd_cuda;
use commands::info::cmd_info;
use commands::inspect::{
    cmd_deepseek_v4_generate, cmd_deepseek_v4_prefill_parity, cmd_deepseek_v4_probe,
    cmd_expert_stream_smoke, cmd_inspect_weightpack,
};

fn main() -> anyhow::Result<()> {
    ferrule_common::observability::init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Command::Info { model } => cmd_info(&model),
        Command::Cuda => cmd_cuda(),
        Command::Chat {
            model,
            max_tokens,
            quant,
            sampling,
            chat_template,
        } => cmd_chat(
            &model,
            max_tokens,
            &quant,
            &sampling,
            chat_template.as_deref(),
        ),
        Command::BenchInteractive {
            model,
            prompts,
            max_tokens,
            chat_template,
            runtime_driver,
            warmup_tokens,
            max_layers,
            prefill_chunk_size,
            golden,
            json,
        } => cmd_bench_interactive(
            &model,
            &prompts,
            max_tokens,
            chat_template.as_deref(),
            runtime_driver,
            warmup_tokens,
            max_layers,
            prefill_chunk_size,
            golden.as_deref(),
            json,
        ),
        Command::InspectWeightPack { path } => cmd_inspect_weightpack(&path),
        Command::ExpertStreamSmoke {
            model,
            layer,
            expert,
            max_slice_mb,
        } => cmd_expert_stream_smoke(&model, layer, expert, max_slice_mb),
        Command::DeepSeekV4Generate {
            model,
            prompt,
            max_tokens,
            max_layers,
            output_head_chunk_rows,
            max_tensor_mb,
            expert_reader_max_slice_mb,
            backend,
            no_stop_eos,
            verbose_tokens,
            chat,
            json,
            warmup_tokens,
            moe_prefetch_experts,
            moe_hotset_experts,
        } => cmd_deepseek_v4_generate(
            &model,
            &prompt,
            max_tokens,
            max_layers,
            output_head_chunk_rows,
            max_tensor_mb,
            expert_reader_max_slice_mb,
            &backend,
            !no_stop_eos,
            verbose_tokens,
            chat,
            json,
            warmup_tokens,
            moe_prefetch_experts,
            moe_hotset_experts,
        ),
        Command::DeepSeekV4Probe {
            model,
            prompt,
            max_layers,
            start_row,
            row_count,
            top_k,
            full_vocab_topk,
            output_head_chunk_rows,
            max_tensor_mb,
            expert_reader_max_slice_mb,
            backend,
            reference_json,
            reference_atol,
        } => cmd_deepseek_v4_probe(
            &model,
            &prompt,
            max_layers,
            start_row,
            row_count,
            top_k,
            full_vocab_topk,
            output_head_chunk_rows,
            max_tensor_mb,
            expert_reader_max_slice_mb,
            &backend,
            reference_json.as_deref(),
            reference_atol,
        ),
        Command::DeepSeekV4PrefillParity {
            model,
            prompt,
            max_layers,
            max_tensor_mb,
            expert_reader_max_slice_mb,
            backend,
            chat,
            atol,
            cuts,
            json,
        } => cmd_deepseek_v4_prefill_parity(
            &model,
            &prompt,
            max_layers,
            max_tensor_mb,
            expert_reader_max_slice_mb,
            &backend,
            chat,
            atol,
            &cuts,
            json,
        ),
    }
}
