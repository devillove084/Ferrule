mod deepseek_v4;
mod expert_stream;
mod weightpack;

pub use deepseek_v4::{
    cmd_deepseek_v4_generate, cmd_deepseek_v4_prefill_parity, cmd_deepseek_v4_probe,
};
pub use expert_stream::cmd_expert_stream_smoke;
pub use weightpack::cmd_inspect_weightpack;
