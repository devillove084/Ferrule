use clap::Parser;

mod args;
mod bench;
mod commands;

pub(crate) use args::GenerationConfig;
pub(crate) use args::SamplingArgs;
use args::{Cli, Command};
use commands::bench_interactive::cmd_bench_interactive;
use commands::chat::cmd_chat;
use commands::cuda::cmd_cuda;
use commands::info::cmd_info;
use commands::inspect::cmd_inspect_weightpack;
use commands::serve::cmd_serve;

fn main() -> anyhow::Result<()> {
    ferrule_common::observability::init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Command::Info { model } => cmd_info(&model),
        Command::Cuda => cmd_cuda(),
        Command::Serve(args) => cmd_serve(args),
        Command::Chat {
            model,
            max_tokens,
            sampling,
            backend,
            chat_template,
        } => cmd_chat(
            &model,
            max_tokens,
            &sampling,
            backend.as_deref(),
            chat_template.as_deref(),
        ),
        Command::BenchInteractive {
            model,
            prompts,
            max_tokens,
            chat_template,
            warmup_tokens,
            max_layers,
            prefill_chunk_size,
            output_head_chunk_rows,
            moe_hotset_experts,
            golden,
            json,
        } => cmd_bench_interactive(
            &model,
            &prompts,
            max_tokens,
            chat_template.as_deref(),
            warmup_tokens,
            max_layers,
            prefill_chunk_size,
            output_head_chunk_rows,
            moe_hotset_experts,
            golden.as_deref(),
            json,
        ),
        Command::InspectWeightPack { path } => cmd_inspect_weightpack(&path),
    }
}
