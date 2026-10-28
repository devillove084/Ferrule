//! Physical numeric CUDA child; no CPU implementation or stats defaults.
use std::collections::BTreeMap;

use ferrule_model::transformer::ExpertPlacementEpoch;
use ferrule_model::transformer::host_experts::HostExpertWarmPlan;

use super::*;
use crate::parallel::expert::ExpertRankWorker;

fn unknown(message: impl std::fmt::Display) -> ProcessHandlerError {
    ProcessHandlerError::unknown(ProcessFailureKind::Handler, message.to_string())
}

pub struct NumericExpertChild {
    validated: Arc<ValidatedNumericBoot>,
    worker: ExpertRankWorker,
    host: Arc<ferrule_model::transformer::host_experts::HostExpertCache>,
    host_generation: u64,
}

impl NumericExpertChild {
    pub(super) fn initialize_checked(
        validated: Arc<ValidatedNumericBoot>,
        bind: impl FnOnce(&NumericBoot) -> Result<BoundDecoderResources>,
    ) -> std::result::Result<Self, ProcessHandlerError> {
        let resources = bind(&validated.boot).map_err(startup)?;
        Self::initialize_validated(validated, resources)
    }

    /// Boot/source/profile checks precede payload warmup and device allocation.
    /// Available through `crate::parallel::expert_process::NumericExpertChild`
    /// only when the `cuda` feature is enabled.
    pub fn initialize_bound(
        boot: ProcessBoot,
        resources: BoundDecoderResources,
    ) -> std::result::Result<Self, ProcessHandlerError> {
        Self::initialize(boot, |_| Ok(resources))
    }

    fn initialize_validated(
        validated: Arc<ValidatedNumericBoot>,
        resources: BoundDecoderResources,
    ) -> std::result::Result<Self, ProcessHandlerError> {
        validated.validate_resources(&resources).map_err(startup)?;
        let config = &validated.boot;
        let host = Arc::new(
            HostExpertWarmPlan::for_owned(
                resources.state_dict(),
                config.host_options(),
                config.max_parameter_bytes,
                config.expected.iter().map(|s| (s.layer, s.expert)),
            )
            .map_err(startup)?
            .warm(|_| {})
            .map_err(startup)?,
        );
        let DecoderDevice::Cuda { ordinal } = validated.device else {
            unreachable!()
        };
        let ops = std::rc::Rc::new(
            ferrule_backend::cuda::operators::linear::CudaOperators::new_on_device(ordinal)
                .map_err(startup)?,
        );
        let (free, _) = ops.memory_info().map_err(startup)?;
        if config.device_bytes > free {
            return Err(startup(
                "numeric child device budget exceeds live free memory",
            ));
        }
        let worker = ExpertRankWorker::prepare_cuda_numeric_with_profile(
            &resources,
            validated.rank,
            &validated.group,
            &host,
            crate::parallel::expert::NumericExpertConfig {
                max_parameter_bytes: config.max_parameter_bytes,
                max_device_bytes: config.device_bytes,
                workspace_bytes: config.workspace_bytes,
                dispatch_timeout: Duration::from_millis(config.placement.timeout_ms),
            },
            ops,
            config.numeric_profile,
        )
        .map_err(startup)?;
        let host_generation = host.generation();
        Ok(Self {
            validated,
            worker,
            host,
            host_generation,
        })
    }
    fn stats(&mut self) -> Result<NumericProcessStats> {
        let host = self.host.stats();
        let boot = &self.validated.boot;
        let rank = self.validated.rank;
        let cache = self
            .worker
            .numeric_cache_stats()
            .ok_or_else(|| error("numeric child lacks CUDA cache"))?;
        Ok(NumericProcessStats {
            pid: std::process::id(),
            owner: boot.owner,
            slot: rank.slot.get(),
            device: self.validated.device,
            owned_experts: boot.expected.iter().map(|s| (s.layer, s.expert)).collect(),
            calls: self.worker.counters().0,
            tokens: self.worker.counters().1,
            host_payload_bytes: host.payload_bytes,
            host_read_bytes: host.io.bytes_read,
            host_hits: self.host.hits(),
            resident_experts: cache.resident_experts,
            resident_bytes: cache.resident_bytes,
            uploads: cache.uploads,
            evictions: cache.evictions,
            pending_upload_bytes: cache.pending_upload_bytes,
            quarantined: cache.quarantined,
        })
    }
}
impl ProcessChildHandler for NumericExpertChild {
    type Command = NumericCommand;
    type Output = NumericReply;
    fn execute(
        &mut self,
        request: ProcessRequest<NumericCommand>,
    ) -> std::result::Result<NumericReply, ProcessHandlerError> {
        #[cfg(test)]
        let _proof_reuse = super::boot_cache_tests::RequestValidationGuard::new();
        if request.identity.owner != self.validated.identity || request.session != 0 {
            return Err(invalid("numeric envelope mismatch"));
        }
        enum Work {
            Legacy(Vec<NumericToken>),
            Routes(Vec<NumericRouteToken>),
        }
        let (transaction, source, layer, work) = match request.command {
            NumericCommand::Stats => return self.stats().map(NumericReply::Stats).map_err(unknown),
            NumericCommand::Compute {
                transaction,
                source,
                layer,
                tokens,
            } => (transaction, source, layer, Work::Legacy(tokens)),
            NumericCommand::ComputeRoutes {
                transaction,
                source,
                layer,
                tokens,
            } => (transaction, source, layer, Work::Routes(tokens)),
        };
        if transaction != request.identity.transaction.get() {
            return Err(invalid(format!(
                "outer transaction mismatch: transaction={} envelope={} source={} layer={}",
                transaction,
                request.identity.transaction.get(),
                source,
                layer
            )));
        }
        let context = ExpertDispatchContext {
            transaction: ExecutionTransactionId::new(transaction).map_err(invalid)?,
            source_rank: ParallelRankId::new(source),
            layer,
        };
        match work {
            Work::Legacy(tokens) => {
                if self.validated.group.placement.is_replicated()
                    || self.validated.group.placement.epoch() != ExpertPlacementEpoch::INITIAL
                {
                    return Err(invalid(
                        "versioned or replicated placement requires route-aware numeric command",
                    ));
                }
                let tokens = self
                    .validated
                    .validate_tokens(context, tokens)
                    .map_err(invalid)?;
                preflight_numeric_source(&self.host, self.host_generation).map_err(invalid)?;
                let results = self.worker.compute(&tokens).map_err(unknown)?;
                let mut by_route = BTreeMap::new();
                for r in results {
                    if r.owner_rank.get() != self.validated.boot.owner
                        || r.transaction != context.transaction
                        || r.source_rank != context.source_rank
                        || by_route.insert((r.source_row, r.route_slot), r).is_some()
                    {
                        return Err(unknown("invalid numeric worker result"));
                    }
                }
                let mut output = Vec::with_capacity(tokens.len());
                for t in &tokens {
                    let r = by_route
                        .remove(&(t.source_row, t.route_slot))
                        .ok_or_else(|| unknown("missing numeric result"))?;
                    if r.sequence != t.sequence
                        || r.expert != t.expert
                        || r.output.len() != self.validated.boot.hidden
                        || r.output.iter().any(|v| !v.is_finite())
                    {
                        return Err(unknown("invalid numeric output identity/shape"));
                    }
                    output.push(NumericToken::encode_values(
                        t,
                        r.output.into_iter().map(f32::to_bits).collect(),
                    ));
                }
                if !by_route.is_empty() {
                    return Err(unknown("extra numeric result"));
                }
                self.worker.record_process_call(tokens.len());
                Ok(NumericReply::Results {
                    owner: self.validated.boot.owner,
                    tokens: output,
                })
            }
            Work::Routes(frames) => {
                self.validated
                    .validate_route_frames(context, &frames)
                    .map_err(invalid)?;
                let routes = frames
                    .into_iter()
                    .map(|frame| {
                        let token = frame.token.decode().map_err(invalid)?;
                        let metadata = validate_expert_route(
                            &self.validated.group.placement,
                            self.validated.rank.owner,
                            context,
                            token.expert,
                            frame.route,
                        )
                        .map_err(invalid)?;
                        Ok(ExpertRouteToken { metadata, token })
                    })
                    .collect::<std::result::Result<Vec<_>, ProcessHandlerError>>()?;
                preflight_numeric_source(&self.host, self.host_generation).map_err(invalid)?;
                let results = self.worker.compute_routes(&routes).map_err(unknown)?;
                let output =
                    encode_numeric_route_results(&routes, results, self.validated.rank.owner)
                        .map_err(unknown)?;
                self.worker.record_process_call(routes.len());
                Ok(NumericReply::RouteResults {
                    owner: self.validated.boot.owner,
                    tokens: output,
                })
            }
        }
    }
    fn shutdown(&mut self) -> std::result::Result<(), ProcessHandlerError> {
        self.worker.shutdown().map_err(unknown)
    }
}
