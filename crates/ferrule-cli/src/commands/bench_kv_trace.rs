//! CLI-owned collection only. Runtime remains the sole trace validator/projector.
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use ferrule_runtime::engine::{
    LocalSessionInferenceEngine, RuntimeKvProjectionLimits, RuntimeKvTraceConfig,
    RuntimeKvTraceSnapshot, project_runtime_kv_trace_with_limits, replay_runtime_prefix_lru,
};
use serde::Serialize;

use crate::args::BenchTraceArgs;

#[derive(Default)]
pub(crate) struct Capture {
    config: Option<RuntimeKvTraceConfig>,
    batches: Vec<RuntimeKvTraceSnapshot>,
    events: usize,
    token_bytes: usize,
    failed: bool,
}

impl Capture {
    pub(crate) fn new(options: &BenchTraceArgs) -> Result<Self> {
        let Some(path) = options.kv_trace.as_deref() else {
            return Ok(Self::default());
        };
        ensure!(
            options.independent_prompts,
            "KV trace requires independent prompts"
        );
        ensure!(
            path != Path::new("-"),
            "KV trace requires an explicit private file, not stdout"
        );
        ensure!(
            !path.try_exists()?,
            "KV trace output already exists; refusing to overwrite"
        );
        let defaults = RuntimeKvTraceConfig::default();
        let max_events = options.kv_trace_max_events.unwrap_or(defaults.max_events);
        let max_token_bytes = options
            .kv_trace_max_token_bytes
            .unwrap_or(defaults.max_token_bytes);
        let ceiling = RuntimeKvProjectionLimits::default();
        ensure!(
            max_events <= ceiling.max_events && max_token_bytes / 4 <= ceiling.max_tokens,
            "KV trace limits exceed offline validation ceilings"
        );
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        Ok(Self {
            config: Some(RuntimeKvTraceConfig {
                run_scope: format!("bench-{}-{nonce}", std::process::id()),
                max_events,
                max_token_bytes,
                ..defaults
            }),
            ..Default::default()
        })
    }

    pub(crate) fn enable(&self, engine: &mut LocalSessionInferenceEngine) -> Result<()> {
        if let Some(config) = &self.config {
            engine.enable_kv_trace(config.clone())?;
        }
        Ok(())
    }

    fn keep(&mut self, batch: RuntimeKvTraceSnapshot) -> Result<()> {
        self.events = self
            .events
            .checked_add(batch.lifecycle_events.len())
            .context("trace event count overflow")?;
        self.token_bytes = self
            .token_bytes
            .checked_add(
                batch
                    .tokens
                    .len()
                    .checked_mul(4)
                    .context("trace token bytes overflow")?,
            )
            .context("trace token bytes overflow")?;
        // Preserve the triggering batch for diagnosis. Stop execution on overflow, so
        // at most one recorder-sized batch exceeds the collection budget; final is empty.
        let intact = batch.status.is_complete();
        self.batches.push(batch);
        ensure!(
            intact,
            "KV trace recorder overflow or unsupported/missing observation"
        );
        let config = self.config.as_ref().context("trace was not enabled")?;
        ensure!(
            self.events <= config.max_events && self.token_bytes <= config.max_token_bytes,
            "collected KV trace resource limit exceeded"
        );
        Ok(())
    }

    /// Called only between requests, never from a token callback or engine step.
    pub(crate) fn drain(&mut self, engine: &mut LocalSessionInferenceEngine) -> Result<()> {
        if self.config.is_some() {
            let batch = engine
                .drain_kv_trace()
                .context("enabled engine omitted a KV trace drain")?;
            self.keep(batch)?;
        }
        Ok(())
    }

    /// No additional cleanup is attempted to make an unsuccessful run look complete.
    pub(crate) fn finalize(
        &mut self,
        engine: &mut LocalSessionInferenceEngine,
        succeeded: bool,
    ) -> Result<()> {
        if self.config.is_none() {
            return Ok(());
        }
        self.failed |= !succeeded;
        let mut batch = engine
            .finalize_kv_trace()
            .context("enabled engine omitted final KV trace")?;
        if !succeeded {
            batch.complete = false;
        }
        self.keep(batch)
    }

    /// Must run after the local inference owner has returned/dropped.
    pub(crate) fn export(
        &mut self,
        path: &Path,
        owner_succeeded: bool,
    ) -> Result<serde_json::Value> {
        let valid = if owner_succeeded && !self.failed {
            project_runtime_kv_trace_with_limits(
                &self.batches,
                RuntimeKvProjectionLimits::default(),
            )
            .and_then(|_| replay_runtime_prefix_lru(&self.batches).map(|_| ()))
            .map_err(anyhow::Error::from)
        } else {
            Err(anyhow::anyhow!("benchmark owner failed"))
        };
        if valid.is_err() {
            for batch in &mut self.batches {
                batch.complete = false;
            }
        }
        #[derive(Serialize)]
        struct Diagnostic<'a> {
            schema: &'static str,
            schema_version: u32,
            complete: bool,
            physical_shutdown_proof: bool,
            reason: &'static str,
            batches: &'a [RuntimeKvTraceSnapshot],
        }
        super::trace_output::publish(path, |file| {
            if valid.is_err() {
                serde_json::to_writer(
                    file,
                    &Diagnostic {
                        schema: "ferrule-runtime-kv-trace-diagnostic",
                        schema_version: 1,
                        complete: false,
                        physical_shutdown_proof: false,
                        reason: "owner_failed_or_trace_validation_failed",
                        batches: &self.batches,
                    },
                )?;
            } else if self.batches.len() == 1 {
                serde_json::to_writer(file, &self.batches[0])?;
            } else {
                serde_json::to_writer(file, &self.batches)?;
            }
            Ok(())
        })?;
        valid.context("KV trace incomplete; diagnostic saved, no successful benchmark report")?;
        Ok(serde_json::json!({
            "schema": "ferrule-runtime-kv-trace", "schema_version": 2,
            "path": path, "batches": self.batches.len(), "complete": true,
            "physical_shutdown_proof": false,
            "privacy": "Contains sensitive prompt/generated token IDs",
        }))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/bench_kv_trace.rs"]
mod tests;
