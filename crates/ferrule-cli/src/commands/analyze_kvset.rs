//! Bounded transport and reporting only; common/runtime own all replay semantics.
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use ferrule_common::kvset::{
    AnalysisReport, Analyzer, AnalyzerConfig, CoverageDenominator, DeferredAnalyzer,
    MeasurementWindow, RecencyModel, ResourceLimits, TargetBasis, TraceEvent, TraceHeader,
    TraceInput, analyze,
};
use ferrule_runtime::engine::{
    RuntimeKvProjectionLimits, RuntimeKvTraceSnapshot, RuntimePrefixReplay,
    project_runtime_kv_trace_with_limits, replay_runtime_prefix_lru,
};
use serde::{Serialize, de::DeserializeOwned};

use crate::args::{AnalyzeKvsetArgs, KvsetDenominator, KvsetFormat, KvsetTargetBasis};

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    schema_version: u32,
    trace_schema_version: u32,
    trace_semantics_version: u32,
    page_size_tokens: usize,
    capacity_unit: &'static str,
    config: AnalyzerConfig,
    effective_resource_limits: ResourceLimits,
    analysis: AnalysisReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime_actual_entry_lru: Option<RuntimeReport>,
    replay: ReplayStats,
}

#[derive(Serialize)]
struct RuntimeReport {
    source_schema: String,
    source_schema_version: u32,
    observed_prefix_capacity_pages: usize,
    comparison_scope: &'static str,
    actual_entry_lru: RuntimePrefixReplay,
}

#[derive(Serialize)]
struct ReplayStats {
    elapsed_ns: u128,
    input_bytes: usize,
    largest_input_record_bytes: usize,
    process_peak_rss_bytes: Option<u64>,
    elapsed_scope: &'static str,
    rss_scope: &'static str,
    retained_token_payload_scope: &'static str,
}

/// Check encoded byte ceilings before growing a buffer, not after read_line/read_to_end.
struct Input {
    reader: Box<dyn BufRead>,
    limit: usize,
    bytes: usize,
    largest_record: usize,
}

impl Input {
    fn open(path: &Path, limit: usize) -> Result<Self> {
        let reader: Box<dyn BufRead> = if path == Path::new("-") {
            Box::new(BufReader::new(io::stdin()))
        } else {
            Box::new(BufReader::new(File::open(path).context("open input")?))
        };
        Ok(Self {
            reader,
            limit,
            bytes: 0,
            largest_record: 0,
        })
    }

    fn record(&mut self, line_limit: Option<usize>) -> Result<Option<Vec<u8>>> {
        let mut record = Vec::new();
        loop {
            let available = self.reader.fill_buf().context("read input")?;
            if available.is_empty() {
                break;
            }
            let newline = line_limit.and_then(|_| available.iter().position(|b| *b == b'\n'));
            let count = newline.map_or(available.len(), |index| index + 1);
            let total = self
                .bytes
                .checked_add(count)
                .context("input byte count overflow")?;
            let length = record
                .len()
                .checked_add(count)
                .context("record byte count overflow")?;
            ensure!(total <= self.limit, "input byte limit exceeded");
            if let Some(limit) = line_limit {
                ensure!(length <= limit, "JSONL line byte limit exceeded");
            }
            record
                .try_reserve_exact(count)
                .context("input buffer allocation failed")?;
            record.extend_from_slice(&available[..count]);
            self.reader.consume(count);
            self.bytes = total;
            if newline.is_some() {
                break;
            }
        }
        self.largest_record = self.largest_record.max(record.len());
        Ok((!record.is_empty()).then_some(record))
    }

    fn json<T: DeserializeOwned>(&mut self) -> Result<T> {
        decode(&self.record(None)?.context("empty JSON input")?)
    }
}

fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    serde_json::from_slice(bytes).map_err(|e| {
        // Serde's Display may include caller-controlled identifiers or token strings.
        anyhow::anyhow!(
            "JSON {:?} error at line {}, column {}",
            e.classify(),
            e.line(),
            e.column()
        )
    })
}

fn config(args: &AnalyzeKvsetArgs) -> Result<AnalyzerConfig> {
    ensure!(
        !(args.trace == Path::new("-") && args.config.as_deref() == Some(Path::new("-"))),
        "trace and config cannot both read stdin"
    );
    let mut config = match &args.config {
        Some(path) => Input::open(path, args.max_config_bytes)?.json()?,
        None => AnalyzerConfig::default(),
    };
    if !args.capacities.is_empty() {
        config.capacities = Some(args.capacities.clone());
    }
    if !args.alphas.is_empty() {
        config.alphas = args.alphas.clone();
    }
    if !args.coverage.is_empty() {
        config.p_values = args.coverage.clone();
    }
    if let Some(basis) = args.target_basis {
        config.target_basis = match basis {
            KvsetTargetBasis::TheoreticalReusable => TargetBasis::TheoreticalReusable,
            KvsetTargetBasis::EligibleTokens => TargetBasis::EligibleTokens,
            KvsetTargetBasis::FullPrompt => TargetBasis::FullPrompt,
        };
    }
    if let Some(denominator) = args.denominator {
        config.coverage_denominator = match denominator {
            KvsetDenominator::Opportunity => CoverageDenominator::Opportunity,
            KvsetDenominator::AllDemands => CoverageDenominator::AllDemands,
        };
    }
    if let Some(budget) = args.capacity_budget {
        config.capacity_budget = Some(budget);
    }
    if let (Some(start_seq), Some(end_seq)) = (args.window_start, args.window_end) {
        config.measurement_window = Some(MeasurementWindow { start_seq, end_seq });
    }
    Ok(config)
}

fn cap_limits(limits: &mut ResourceLimits) {
    let ceiling = ResourceLimits::default();
    // Imported limits may tighten the caller's ceiling, never authorize larger allocations.
    macro_rules! cap {
        ($($field:ident),+ $(,)?) => { $(limits.$field = limits.$field.min(ceiling.$field);)+ };
    }
    cap!(
        max_events,
        max_publish_positions,
        max_unique_keys,
        max_identity_token_bytes,
        max_pending_requests,
        max_pending_tokens,
        max_tokens_per_event,
        max_namespaces,
        max_namespace_bytes,
        max_request_id_bytes,
        max_request_history_bytes,
        max_histogram_entries,
        max_capacity_points,
        max_total_capacity_entries,
        max_target_parameters,
        max_report_points,
        max_demand_summary_entries
    );
}

/// Dispatch to the two official streaming implementations without duplicating analysis.
enum Replay {
    Snapshot(Analyzer),
    Deferred(DeferredAnalyzer),
}
impl Replay {
    fn new(header: TraceHeader, config: AnalyzerConfig) -> Result<Self> {
        Ok(match header.recency_model {
            RecencyModel::PublishRecencySnapshotV1 => {
                Self::Snapshot(Analyzer::new(header, config)?)
            }
            RecencyModel::DeferredAdmissionPrefixLruV1 => {
                Self::Deferred(DeferredAnalyzer::new(header, config)?)
            }
        })
    }
    fn ingest(&mut self, event: &TraceEvent) -> Result<()> {
        match self {
            Self::Snapshot(a) => {
                a.ingest(event)?;
            }
            Self::Deferred(a) => {
                a.ingest(event)?;
            }
        }
        Ok(())
    }
    fn finish(self) -> Result<AnalysisReport> {
        Ok(match self {
            Self::Snapshot(a) => a.finish()?,
            Self::Deferred(a) => a.finish()?,
        })
    }
}

fn projection_limits(args: &AnalyzeKvsetArgs) -> Result<RuntimeKvProjectionLimits> {
    let ceiling = RuntimeKvProjectionLimits::default();
    ensure!(
        args.max_projected_events <= ceiling.max_events
            && args.max_projected_tokens <= ceiling.max_tokens
            && args.max_projected_pages <= ceiling.max_publish_pages,
        "projection limits may only lower the runtime defaults"
    );
    Ok(RuntimeKvProjectionLimits {
        max_events: args.max_projected_events,
        max_tokens: args.max_projected_tokens,
        max_publish_pages: args.max_projected_pages,
    })
}

pub(crate) fn run(args: AnalyzeKvsetArgs) -> Result<()> {
    let config = config(&args)?;
    let projection_limits = projection_limits(&args)?;
    let start = Instant::now();
    let mut input = Input::open(&args.trace, args.max_input_bytes)?;
    let mut runtime = None;
    let (analysis, limits) = match args.format {
        KvsetFormat::Json => {
            let mut trace: TraceInput = input.json()?;
            cap_limits(&mut trace.header.limits);
            (analyze(&trace, config)?, trace.header.limits)
        }
        KvsetFormat::Jsonl => {
            let bytes = input
                .record(Some(args.max_line_bytes))?
                .context("missing JSONL header")?;
            let mut header: TraceHeader = decode(&bytes).context("JSONL header")?;
            cap_limits(&mut header.limits);
            let limits = header.limits.clone();
            let mut replay = Replay::new(header, config)?;
            let mut line = 1usize;
            while let Some(bytes) = input.record(Some(args.max_line_bytes))? {
                line = line.checked_add(1).context("line count overflow")?;
                let event: TraceEvent =
                    decode(&bytes).with_context(|| format!("JSONL record {line}"))?;
                replay.ingest(&event)?;
            }
            (replay.finish()?, limits)
        }
        KvsetFormat::RuntimeJson => {
            let bytes = input.record(None)?.context("empty JSON input")?;
            // Select the wire shape without a Value/untagged intermediate, preserving
            // strict duplicate-field rejection and useful sanitized serde positions.
            let snapshots: Vec<RuntimeKvTraceSnapshot> =
                if bytes.iter().find(|b| !b.is_ascii_whitespace()) == Some(&b'[') {
                    decode(&bytes)?
                } else {
                    vec![decode(&bytes)?]
                };
            let batches = snapshots.as_slice();
            // This official API charges cumulative tokens/pages BEFORE each full-prefix copy.
            // Entry replay validates without constructing another projected TraceInput.
            let trace = project_runtime_kv_trace_with_limits(batches, projection_limits)
                .context("runtime trace projection rejected")?;
            let actual_entry_lru =
                replay_runtime_prefix_lru(batches).context("runtime trace replay rejected")?;
            runtime = Some(RuntimeReport {
                source_schema: batches[0].header.schema.clone(),
                source_schema_version: batches[0].header.schema_version,
                observed_prefix_capacity_pages: batches[0].header.prefix_capacity_pages,
                comparison_scope: "Observed entry-LRU trajectory only; ideal page-LRU curves are not counterfactual runtime predictions at other capacities. Entry charge is not physical release or shutdown proof.",
                actual_entry_lru,
            });
            (analyze(&trace, config)?, trace.header.limits)
        }
    };
    let report = Report {
        schema: "ferrule-kvset-report",
        schema_version: 1,
        trace_schema_version: analysis.schema_version,
        trace_semantics_version: analysis.semantics_version,
        page_size_tokens: analysis.page_size,
        capacity_unit: "equal_charge_logical_pages",
        config: analysis.config.clone(),
        effective_resource_limits: limits,
        analysis,
        runtime_actual_entry_lru: runtime,
        replay: ReplayStats {
            elapsed_ns: start.elapsed().as_nanos(),
            input_bytes: input.bytes,
            largest_input_record_bytes: input.largest_record,
            process_peak_rss_bytes: peak_rss(),
            elapsed_scope: "Trace read/decode, projection/replay, analyzer construction/ingest/finish; excludes config and output.",
            rss_scope: "Process lifetime high-water RSS sampled after replay, before output; not analyzer heap.",
            retained_token_payload_scope: "Common retained_token_payload_bytes is final retained token payload only; excludes allocator, maps, Fenwick, input and report; not peak heap.",
        },
    };
    match args.output.as_deref() {
        Some(path) if path != Path::new("-") => {
            super::trace_output::publish(path, |output| render(output, &report, args.json))
        }
        _ => {
            let mut output = io::stdout().lock();
            render(&mut output, &report, args.json)?;
            output.flush().context("flush stdout")
        }
    }
}

fn render(output: &mut impl Write, report: &Report, json: bool) -> Result<()> {
    if json {
        serde_json::to_writer_pretty(&mut *output, report).context("write JSON report")?;
        writeln!(output)?;
    } else {
        writeln!(
            output,
            "KVSET {:?}; page size {} tokens",
            report.analysis.recency_model, report.page_size_tokens
        )?;
        writeln!(output, "Capacity unit: {}", report.capacity_unit)?;
        writeln!(
            output,
            "Demands: {}; opportunities: {}",
            report.analysis.workload.demands, report.analysis.workload.opportunity_requests
        )?;
        writeln!(
            output,
            "capacity_pages\tpage_reference_hit_rate\tcontiguous_prefix_tokens\ttheoretical\teligible\tprompt"
        )?;
        for point in &report.analysis.capacity_points {
            let fraction = |v: Option<ferrule_common::kvset::Ratio>| {
                v.map_or_else(
                    || "undefined".to_owned(),
                    |r| format!("{}/{}", r.numerator, r.denominator),
                )
            };
            writeln!(
                output,
                "{}\t{}\t{}\t{}\t{}\t{}",
                point.capacity_pages,
                fraction(point.page_reference_hit_rate),
                point.contiguous_prefix_tokens,
                fraction(point.contiguous_prefix_token_hit_rate_theoretical),
                fraction(point.contiguous_prefix_token_hit_rate_eligible),
                fraction(point.contiguous_prefix_token_hit_rate_prompt)
            )?;
        }
        for target in &report.analysis.target_capacities {
            writeln!(
                output,
                "alpha={}/{} coverage={}/{} {:?}: {:?}",
                target.alpha.numerator,
                target.alpha.denominator,
                target.p.numerator,
                target.p.denominator,
                target.denominator,
                target.requirement
            )?;
        }
        if let Some(runtime) = &report.runtime_actual_entry_lru {
            writeln!(
                output,
                "Actual entry-LRU at {} pages: {:?}",
                runtime.observed_prefix_capacity_pages, runtime.actual_entry_lru
            )?;
            writeln!(output, "{}", runtime.comparison_scope)?;
        }
        writeln!(
            output,
            "Replay: {} ns; {} input bytes; peak RSS bytes: {:?}",
            report.replay.elapsed_ns,
            report.replay.input_bytes,
            report.replay.process_peak_rss_bytes
        )?;
        writeln!(
            output,
            "{}\n{}",
            report.replay.rss_scope, report.replay.retained_token_payload_scope
        )?;
    }
    Ok(())
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios"
))]
#[allow(unsafe_code)]
fn peak_rss() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes this writable rusage on success; it retains no pointer.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: the successful call above initialized usage.
    let bytes = u64::try_from(unsafe { usage.assume_init() }.ru_maxrss).ok()?;
    if cfg!(any(target_os = "linux", target_os = "android")) {
        bytes.checked_mul(1024)
    } else {
        Some(bytes)
    }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios"
)))]
fn peak_rss() -> Option<u64> {
    None
}
