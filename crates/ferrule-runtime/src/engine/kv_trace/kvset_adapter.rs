//! Converts the shared validated stream into common KVSET DTOs.
use super::validation::{Projected, validate_semantics};
use super::*;
use ferrule_common::kvset::{
    ActualObservation, ContentNamespace, DemandEvent, NamespaceDefinition, OutcomeEvent,
    PublishEvent, RawPage, RequestId, RequestStatus, ReuseLimit, TraceEvent, TraceHeader,
    TraceInput,
};

pub fn project_runtime_kv_trace(batches: &[RuntimeKvTraceSnapshot]) -> TraceResult<TraceInput> {
    project_runtime_kv_trace_with_limits(batches, RuntimeKvProjectionLimits::default())
}
pub fn project_runtime_kv_trace_with_limits(
    batches: &[RuntimeKvTraceSnapshot],
    limits: RuntimeKvProjectionLimits,
) -> TraceResult<TraceInput> {
    let mut events = Vec::new();
    let validated = validate_semantics(batches, limits, |header, seq, projected| {
        let event = match projected {
            Projected::Demand {
                observation,
                request,
                tokens,
            } => TraceEvent::Demand(DemandEvent {
                seq,
                request: RequestId {
                    id: request.to_string(),
                    generation: observation,
                },
                namespace_id: "runtime".into(),
                prompt_tokens: tokens.to_vec(),
                reuse_limit: ReuseLimit::BeforeLastPromptToken,
            }),
            Projected::Publish {
                observation,
                request,
                tokens,
            } => TraceEvent::Publish(PublishEvent {
                seq,
                request: RequestId {
                    id: request.to_string(),
                    generation: observation,
                },
                namespace_id: "runtime".into(),
                pages: tokens
                    .chunks(header.page_size)
                    .map(|chunk| RawPage {
                        tokens: chunk.to_vec(),
                        valid_count: chunk.len(),
                    })
                    .collect(),
            }),
            Projected::Outcome {
                observation,
                request,
                status,
                actual,
            } => TraceEvent::Outcome(OutcomeEvent {
                seq,
                request: RequestId {
                    id: request.to_string(),
                    generation: observation,
                },
                status: match status {
                    TerminalStatus::Success => RequestStatus::Success,
                    TerminalStatus::Failed => RequestStatus::Failed,
                    TerminalStatus::Cancelled => RequestStatus::Cancelled,
                },
                actual: actual.map(|(reused_prefix_tokens, reused_pages)| ActualObservation {
                    reused_prefix_tokens,
                    reused_pages,
                    cache_capacity_pages: header.prefix_capacity_pages,
                }),
            }),
        };
        events.push(event);
        Ok(())
    })?;
    let header = validated.header;
    let ns = header
        .config
        .stable_content_namespace
        .clone()
        .unwrap_or_else(|| ContentNamespace {
            model: format!("runtime-model-{}", header.namespace[0]),
            adapter: "encoded-in-plan".into(),
            position: "absolute-origin-zero".into(),
            layout: format!("owner-{}-page-{}", header.namespace[4], header.page_size),
            backend: format!(
                "backend-{}-device-{}",
                header.namespace[1], header.namespace[2]
            ),
            plan: format!("plan-{}", header.namespace[3]),
            run_scope: header.config.run_scope.clone(),
        });
    Ok(TraceInput {
        header: TraceHeader::new(
            header.page_size,
            vec![NamespaceDefinition {
                id: "runtime".into(),
                namespace: ns,
            }],
        ),
        events,
    })
}
