use thiserror::Error;

pub type FerruleResult<T> = Result<T, FerruleError>;

#[derive(Debug, Error)]
pub enum FerruleError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("toml decode error: {0}")]
    TomlDe(#[from] toml::de::Error),

    #[error("address parse error: {0}")]
    AddrParse(#[from] std::net::AddrParseError),

    #[error("config error: {0}")]
    Config(String),

    #[error("setup error: {0}")]
    Setup(String),

    #[error("model error: {0}")]
    Model(String),

    #[error("runtime error: {0}")]
    Runtime(String),

    #[error("environment error: {0}")]
    Env(String),

    #[error("reward error: {0}")]
    Reward(String),
}
