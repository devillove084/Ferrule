//! Prepared synchronous model EP seam. Existing dispatch/combine/CPU math only;
//! no runtime workers, authentication claims or child lifecycle simulation.
use ferrule_common::{ParallelRankId, Result, execution::ExecutionTransactionId};
use ferrule_model::{
    TensorRole, moe::ExpertId, transformer::expert_parallel::*,
    transformer::hybrid_parallel::HybridExpertAttachment, transformer::*,
};
use std::{collections::BTreeMap, sync::Arc};
#[derive(Debug)]
pub struct Dispatch {
    pub transaction: ExecutionTransactionId,
    pub source: ParallelRankId,
    pub layer: usize,
    pub sequences: Vec<u64>,
    pub routes: Vec<usize>,
    pub weights: Vec<f32>,
}
struct Provider(BTreeMap<ExpertId, Arc<PreparedSwiGlu>>);
impl ExpertProvider for Provider {
    fn expert(&mut self, layer: usize, expert: usize) -> Result<ExpertAvailability> {
        Ok(ExpertAvailability::Ready(Arc::clone(
            self.0
                .get(&ExpertId::new(layer, expert))
                .expect("owned expert"),
        )))
    }
}
pub struct Experts {
    attachment: HybridExpertAttachment,
    providers: Vec<(ParallelRankId, Provider)>,
    pub calls: Vec<Dispatch>,
}
impl Experts {
    pub fn prepare(
        resources: &BoundDecoderResources,
        attachment: &HybridExpertAttachment,
        layers: std::ops::Range<usize>,
    ) -> Self {
        let materializer = StateDictMaterializer::new(1 << 20).unwrap();
        let providers = attachment
            .members
            .iter()
            .map(|&owner| {
                let mut experts = BTreeMap::new();
                for layer in layers.clone() {
                    let FeedForward::Moe(moe) = resources.spec().layers()[layer].feed_forward()
                    else {
                        continue;
                    };
                    for expert in 0..moe.router_spec().num_experts() {
                        if attachment.placement.owner(layer, expert) != Some(owner) {
                            continue;
                        }
                        let linear = |role: TensorRole, shape: [usize; 2]| {
                            let binding = resources
                                .experts()
                                .require_shape(layer, expert, role.clone(), &shape)
                                .unwrap();
                            PreparedLinear::from_parameter(
                                materializer
                                    .expert_parameter(layer, expert, binding)
                                    .unwrap(),
                                role,
                            )
                            .unwrap()
                        };
                        experts.insert(
                            ExpertId::new(layer, expert),
                            Arc::new(
                                PreparedSwiGlu::new(
                                    linear(
                                        TensorRole::RoutedExpertGate,
                                        moe.expert().gate().weight_shape(),
                                    ),
                                    linear(
                                        TensorRole::RoutedExpertUp,
                                        moe.expert().up().weight_shape(),
                                    ),
                                    linear(
                                        TensorRole::RoutedExpertDown,
                                        moe.expert().down().weight_shape(),
                                    ),
                                    moe.expert().activation_limit(),
                                )
                                .unwrap(),
                            ),
                        );
                    }
                }
                (owner, Provider(experts))
            })
            .collect();
        Self {
            attachment: attachment.clone(),
            providers,
            calls: vec![],
        }
    }
}
impl RoutedSwiGluExecutor for Experts {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        assert_eq!(request.context.source_rank, self.attachment.source_rank);
        self.calls.push(Dispatch {
            transaction: request.context.transaction,
            source: request.context.source_rank,
            layer: request.context.layer,
            sequences: request.sequences.to_vec(),
            routes: request.routes.expert_ids().to_vec(),
            weights: request.routes.weights().to_vec(),
        });
        let mut workers = self
            .providers
            .iter_mut()
            .map(|(owner, provider)| {
                CpuReferenceExpertWorker::new(*owner, &self.attachment.placement, provider)
            })
            .collect::<Vec<_>>();
        let mut results = CpuExpertResultExecutor::new(
            workers
                .iter_mut()
                .map(|w| w as &mut dyn ExpertWorker)
                .collect(),
        )?;
        ExpertParallelRoutedExecutor::new(
            self.attachment.members.clone(),
            &self.attachment.placement,
            ExpertDispatchLimits {
                max_tokens: 64,
                max_bytes: 8192,
            },
            &mut results,
        )
        .routed_swiglu(request, active)
    }
}
