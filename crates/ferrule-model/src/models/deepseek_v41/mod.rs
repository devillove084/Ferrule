//! DeepSeek-V4.1 typed metadata-only artifact support.

mod metadata;

pub use metadata::{
    DEEPSEEK_V41_REAL_SHARD_COUNT, DEEPSEEK_V41_REAL_TOTAL_SIZE, DeepSeekV41Config,
    DeepSeekV41DType, DeepSeekV41EngramConfig, DeepSeekV41IndexSummary, DeepSeekV41Metadata,
    DeepSeekV41QuantizationContract, DeepSeekV41TextConfig, DeepSeekV41VisionConfig,
    validate_header_inventory, validate_index, validate_real_index,
};
