//! Competitive benchmark manifest schema.
//!
//! Benchmark claims should be generated from recorded manifests and raw results,
//! not screenshots or ad-hoc commands. This schema captures the minimum context
//! needed to reproduce Ferrule vs llama.cpp/vLLM/SGLang comparisons.

use ferrule_core::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PkManifestId(String);

impl PkManifestId {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        non_empty_newtype(value, "PK manifest id").map(Self)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PkModelId(String);

impl PkModelId {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        non_empty_newtype(value, "PK model id").map(Self)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PkQuantizationId(String);

impl PkQuantizationId {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        non_empty_newtype(value, "PK quantization id").map(Self)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PkPromptSetId(String);

impl PkPromptSetId {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        non_empty_newtype(value, "PK prompt set id").map(Self)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PkRuntimeKind {
    Ferrule,
    LlamaCpp,
    Vllm,
    SgLang,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PkMetricKind {
    LoadTimeMs,
    TimeToFirstTokenMs,
    PromptTokensPerSecond,
    DecodeTokensPerSecond,
    RequestsPerSecond,
    TokensPerSecond,
    PeakRssBytes,
    GpuMemoryBytes,
    FirstTokenCorrect,
    ExpertActivationDistribution,
    ExpertResidencyHitRate,
    ExpertBatchingEfficiency,
    ExpertBytesMoved,
    SpeculationAcceptanceRate,
    SpeculationSpeedupRatio,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompetitivePkManifest {
    pub id: PkManifestId,
    pub model: PkModelId,
    pub quantization: PkQuantizationId,
    pub context_length: usize,
    pub prompt_set: PkPromptSetId,
    pub hardware: HardwareSpec,
    pub runs: Vec<PkRunSpec>,
}

impl CompetitivePkManifest {
    pub fn validate(&self) -> Result<()> {
        if self.context_length == 0 {
            return Err(Error::Model(format!(
                "PK manifest '{}' context_length must be > 0",
                self.id.as_str()
            )));
        }
        self.hardware.validate(&self.id)?;
        if self.runs.is_empty() {
            return Err(Error::Model(format!(
                "PK manifest '{}' must contain at least one run",
                self.id.as_str()
            )));
        }
        for run in &self.runs {
            run.validate(&self.id)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareSpec {
    pub cpu: String,
    pub gpu: Option<String>,
    pub ram_bytes: Option<u64>,
    pub vram_bytes: Option<u64>,
    pub os: String,
}

impl HardwareSpec {
    fn validate(&self, manifest_id: &PkManifestId) -> Result<()> {
        if self.cpu.trim().is_empty() || self.os.trim().is_empty() {
            return Err(Error::Model(format!(
                "PK manifest '{}' must record CPU and OS",
                manifest_id.as_str()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PkCommand {
    pub program: String,
    pub args: Vec<String>,
}

impl PkCommand {
    pub fn new(
        program: impl Into<String>,
        args: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            program: program.into(),
            args: args.into_iter().map(Into::into).collect(),
        }
    }

    pub fn argv(&self) -> Vec<&str> {
        std::iter::once(self.program.as_str())
            .chain(self.args.iter().map(String::as_str))
            .collect()
    }

    fn validate(&self, manifest_id: &PkManifestId) -> Result<()> {
        if self.program.trim().is_empty() || self.args.iter().any(|part| part.trim().is_empty()) {
            return Err(Error::Model(format!(
                "PK manifest '{}' has a run with an empty command part",
                manifest_id.as_str()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PkRunSpec {
    pub runtime: PkRuntimeKind,
    pub command: PkCommand,
    pub concurrency: usize,
    pub max_new_tokens: usize,
    pub measured_metrics: Vec<PkMetricKind>,
    pub raw_result_json: Option<String>,
    pub speculation: Option<PkSpeculationConfig>,
}

impl PkRunSpec {
    fn validate(&self, manifest_id: &PkManifestId) -> Result<()> {
        self.command.validate(manifest_id)?;
        if self.concurrency == 0 || self.max_new_tokens == 0 {
            return Err(Error::Model(format!(
                "PK manifest '{}' has invalid concurrency/max_new_tokens",
                manifest_id.as_str()
            )));
        }
        if self.measured_metrics.is_empty() {
            return Err(Error::Model(format!(
                "PK manifest '{}' has a run without measured metrics",
                manifest_id.as_str()
            )));
        }
        if let Some(speculation) = &self.speculation {
            speculation.validate(manifest_id)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PkSpeculationConfig {
    pub target_model: PkModelId,
    pub draft_model: PkModelId,
    pub draft_block_size: usize,
    pub expected_acceptance_rate: Option<f32>,
}

impl PkSpeculationConfig {
    fn validate(&self, manifest_id: &PkManifestId) -> Result<()> {
        if self.draft_block_size == 0 {
            return Err(Error::Model(format!(
                "PK manifest '{}' speculation draft_block_size must be > 0",
                manifest_id.as_str()
            )));
        }
        if let Some(rate) = self.expected_acceptance_rate {
            if !(0.0..=1.0).contains(&rate) {
                return Err(Error::Model(format!(
                    "PK manifest '{}' speculation acceptance rate must be in [0, 1]",
                    manifest_id.as_str()
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PkResultRecord {
    pub runtime: PkRuntimeKind,
    pub metrics: Vec<PkMetricValue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PkMetricValue {
    pub kind: PkMetricKind,
    pub value: f64,
}

pub fn render_pk_markdown_summary(
    manifest: &CompetitivePkManifest,
    results: &[PkResultRecord],
) -> String {
    let mut out = String::new();
    out.push_str("| runtime | metric | value |\n");
    out.push_str("| --- | --- | ---: |\n");
    for result in results {
        for metric in &result.metrics {
            out.push_str(&format!(
                "| {:?} | {:?} | {:.4} |\n",
                result.runtime, metric.kind, metric.value
            ));
        }
    }
    if results.is_empty() {
        out.push_str(&format!(
            "| {:?} | {:?} | {:.4} |\n",
            PkRuntimeKind::Other(manifest.id.as_str().into()),
            PkMetricKind::FirstTokenCorrect,
            0.0
        ));
    }
    out
}

fn non_empty_newtype(value: impl Into<String>, label: &str) -> Result<String> {
    let value = value.into();
    if value.trim().is_empty() {
        return Err(Error::Model(format!("{label} is empty")));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pk_manifest_records_reproducible_ferrule_vs_llamacpp_commands() {
        let manifest = CompetitivePkManifest {
            id: PkManifestId::new("dsv4-local-first-token").unwrap(),
            model: PkModelId::new("DeepSeek-V4-Flash-DSpark").unwrap(),
            quantization: PkQuantizationId::new("source-fp4/fp8").unwrap(),
            context_length: 4096,
            prompt_set: PkPromptSetId::new("golden/dsv4-smoke.jsonl").unwrap(),
            hardware: HardwareSpec {
                cpu: "recorded-by-runner".into(),
                gpu: Some("recorded-by-runner".into()),
                ram_bytes: None,
                vram_bytes: None,
                os: "recorded-by-runner".into(),
            },
            runs: vec![
                PkRunSpec {
                    runtime: PkRuntimeKind::Ferrule,
                    command: PkCommand::new(
                        "ferrule",
                        ["bench-infer", "models/DeepSeek-V4-Flash-DSpark", "-n", "1"],
                    ),
                    concurrency: 1,
                    max_new_tokens: 1,
                    measured_metrics: vec![
                        PkMetricKind::LoadTimeMs,
                        PkMetricKind::TimeToFirstTokenMs,
                        PkMetricKind::FirstTokenCorrect,
                    ],
                    raw_result_json: Some("results/ferrule-dsv4-first-token.json".into()),
                    speculation: Some(PkSpeculationConfig {
                        target_model: PkModelId::new("DeepSeek-V4-Flash-DSpark").unwrap(),
                        draft_model: PkModelId::new("DSpark").unwrap(),
                        draft_block_size: 5,
                        expected_acceptance_rate: Some(0.6),
                    }),
                },
                PkRunSpec {
                    runtime: PkRuntimeKind::LlamaCpp,
                    command: PkCommand::new(
                        "llama-cli",
                        ["-m", "models/deepseek-v4.gguf", "-n", "1"],
                    ),
                    concurrency: 1,
                    max_new_tokens: 1,
                    measured_metrics: vec![
                        PkMetricKind::LoadTimeMs,
                        PkMetricKind::TimeToFirstTokenMs,
                        PkMetricKind::FirstTokenCorrect,
                    ],
                    raw_result_json: Some("results/llamacpp-dsv4-first-token.json".into()),
                    speculation: None,
                },
            ],
        };
        manifest.validate().unwrap();
        assert_eq!(manifest.runs.len(), 2);
        assert!(manifest.runs[0].command.argv().contains(&"bench-infer"));
    }

    #[test]
    fn pk_manifest_renders_markdown_summary_from_raw_results() {
        let manifest = minimal_manifest();
        let markdown = render_pk_markdown_summary(
            &manifest,
            &[PkResultRecord {
                runtime: PkRuntimeKind::Ferrule,
                metrics: vec![PkMetricValue {
                    kind: PkMetricKind::TimeToFirstTokenMs,
                    value: 12.5,
                }],
            }],
        );
        assert!(markdown.contains("TimeToFirstTokenMs"));
        assert!(markdown.contains("12.5000"));
    }

    #[test]
    fn pk_manifest_rejects_missing_hardware_context() {
        let mut manifest = minimal_manifest();
        manifest.hardware.cpu.clear();
        let err = manifest.validate().unwrap_err();
        assert!(err.to_string().contains("CPU and OS"));
    }

    fn minimal_manifest() -> CompetitivePkManifest {
        CompetitivePkManifest {
            id: PkManifestId::new("minimal").unwrap(),
            model: PkModelId::new("m").unwrap(),
            quantization: PkQuantizationId::new("q").unwrap(),
            context_length: 1,
            prompt_set: PkPromptSetId::new("p").unwrap(),
            hardware: HardwareSpec {
                cpu: "cpu".into(),
                gpu: None,
                ram_bytes: None,
                vram_bytes: None,
                os: "linux".into(),
            },
            runs: vec![PkRunSpec {
                runtime: PkRuntimeKind::Ferrule,
                command: PkCommand::new("ferrule", ["bench-infer"]),
                concurrency: 1,
                max_new_tokens: 1,
                measured_metrics: vec![PkMetricKind::LoadTimeMs],
                raw_result_json: None,
                speculation: None,
            }],
        }
    }
}
