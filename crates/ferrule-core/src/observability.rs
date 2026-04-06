use crate::{FerruleError, FerruleResult, LogFormat, ObservabilityConfig};
use metrics_exporter_prometheus::PrometheusBuilder;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

pub fn init_observability(cfg: &ObservabilityConfig) -> FerruleResult<()> {
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(cfg.log_level.clone()))
        .map_err(|e| FerruleError::Setup(format!("failed to build log filter: {e}")))?;

    match cfg.log_format {
        LogFormat::Json => {
            tracing_subscriber::registry()
                .with(filter)
                .with(
                    fmt::layer()
                        .json()
                        .with_target(true)
                        .with_thread_ids(true)
                        .with_thread_names(true),
                )
                .try_init()
                .map_err(|e| FerruleError::Setup(format!("failed to init tracing: {e}")))?;
        }
        LogFormat::Pretty => {
            tracing_subscriber::registry()
                .with(filter)
                .with(
                    fmt::layer()
                        .compact()
                        .with_target(true)
                        .with_thread_ids(true)
                        .with_thread_names(true),
                )
                .try_init()
                .map_err(|e| FerruleError::Setup(format!("failed to init tracing: {e}")))?;
        }
    }

    if cfg.metrics_enabled {
        let addr: std::net::SocketAddr = cfg.metrics_bind.parse()?;
        PrometheusBuilder::new()
            .with_http_listener(addr)
            .install()
            .map_err(|e| FerruleError::Setup(format!("failed to install metrics exporter: {e}")))?;
    }

    tracing::info!(
        service = %cfg.service_name,
        metrics_enabled = cfg.metrics_enabled,
        metrics_bind = %cfg.metrics_bind,
        "observability initialized"
    );

    Ok(())
}
