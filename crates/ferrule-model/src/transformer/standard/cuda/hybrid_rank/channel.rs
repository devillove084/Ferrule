//! Ordered V2 schedule consumer. Host values are transport buffers, never an
//! alternate implementation of linear, attention, recurrence or SwiGLU.
use super::*;

pub(super) struct RankChannel {
    pub schedule: HybridCollectiveSchedule,
    collective: Box<dyn StandardTensorCollective>,
    transaction: Option<ExecutionTransactionId>,
    rows: usize,
    sequences: usize,
    next: usize,
    failed: bool,
}
impl RankChannel {
    pub fn new(
        schedule: HybridCollectiveSchedule,
        collective: Box<dyn StandardTensorCollective>,
    ) -> Self {
        Self {
            schedule,
            collective,
            transaction: None,
            rows: 0,
            sequences: 0,
            next: 0,
            failed: false,
        }
    }
    pub fn begin(
        &mut self,
        transaction: ExecutionTransactionId,
        rows: usize,
        sequences: usize,
    ) -> Result<()> {
        if self.failed || self.transaction.is_some() {
            return Err(cuda_error("rank collective lifetime unavailable"));
        }
        self.transaction = Some(transaction);
        self.rows = rows;
        self.sequences = sequences;
        self.next = 0;
        Ok(())
    }
    pub fn exchange(
        &mut self,
        semantic: HybridCollectiveSemantic,
        layer: Option<usize>,
        values: Vec<f32>,
    ) -> Result<Vec<f32>> {
        let id = self
            .transaction
            .ok_or_else(|| cuda_error("rank collective outside transaction"))?;
        let site = self
            .schedule
            .sites()
            .get(self.next)
            .ok_or_else(|| cuda_error("rank collective exceeds schedule"))?;
        if self.failed
            || site.semantic != semantic
            || site.global_layer != layer
            || values.len() != site.shape.elements(self.sequences, self.rows)?
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(cuda_error("rank collective semantic/shape/order mismatch"));
        }
        let expected = values
            .len()
            .checked_mul(if site.kind == HybridCollectiveKind::AllGather {
                self.schedule.degree()
            } else {
                1
            })
            .ok_or_else(|| cuda_error("rank collective result overflow"))?;
        let result = if self.schedule.degree() == 1 {
            values
        } else {
            self.collective.exchange(
                id,
                site.ordinal,
                match site.kind {
                    HybridCollectiveKind::AllGather => TensorParallelCollective::AllGather,
                    HybridCollectiveKind::ReduceSum => TensorParallelCollective::Sum,
                },
                values,
            )?
        };
        if result.len() != expected || result.iter().any(|v| !v.is_finite()) {
            return Err(cuda_error("rank collective returned invalid extent/values"));
        }
        self.next += 1;
        Ok(result)
    }
    pub fn finish(&mut self) -> Result<()> {
        if self.next != self.schedule.sites().len() {
            return Err(cuda_error("incomplete rank schedule"));
        }
        self.transaction = None;
        Ok(())
    }
    pub fn abort(&mut self) {
        self.transaction = None;
        if self.schedule.degree() > 1 && !self.failed {
            self.failed = true;
            let _ =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.collective.abort()));
        }
    }
}

impl Drop for RankChannel {
    fn drop(&mut self) {
        self.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::*;
    use ferrule_common::ParallelTopologyId;
    #[derive(Default)]
    struct Trace {
        calls: usize,
        aborts: usize,
        malformed: bool,
    }
    struct Endpoint {
        members: Vec<ParallelRankId>,
        trace: Rc<RefCell<Trace>>,
    }
    impl StandardTensorCollective for Endpoint {
        fn owner(&self) -> ParallelRankId {
            self.members[0]
        }
        fn members(&self) -> &[ParallelRankId] {
            &self.members
        }
        fn exchange(
            &mut self,
            _: ExecutionTransactionId,
            _: u64,
            kind: TensorParallelCollective,
            values: Vec<f32>,
        ) -> Result<Vec<f32>> {
            self.trace.borrow_mut().calls += 1;
            if self.trace.borrow().malformed {
                return Ok(vec![f32::NAN]);
            }
            Ok(match kind {
                TensorParallelCollective::AllGather => values.repeat(self.members.len()),
                TensorParallelCollective::Sum => values
                    .into_iter()
                    .map(|v| v * self.members.len() as f32)
                    .collect(),
            })
        }
        fn abort(&mut self) {
            self.trace.borrow_mut().aborts += 1;
        }
    }
    fn channel(degree: usize) -> (RankChannel, Rc<RefCell<Trace>>) {
        let norm = RmsNorm::new(8, 1e-6).unwrap();
        let rope = RotaryEmbedding::new(
            2,
            10000.0,
            RotaryPairing::SplitHalf,
            RotaryRegion::Prefix { dimensions: 2 },
            RotaryScaling::None,
        )
        .unwrap();
        let spec = DecoderModelSpec::new(DecoderModelParts {
            architecture: "rank-channel".into(),
            hidden_size: 8,
            vocab_size: 11,
            max_sequence_length: Some(8),
            token_embedding: Embedding::new(11, 8, None).unwrap(),
            layers: vec![
                DecoderLayer::new(
                    0,
                    norm.clone(),
                    Attention::Gqa(GqaAttention::new(8, 4, 1, 2, false, rope).unwrap()),
                    Residual::Add,
                    norm.clone(),
                    FeedForward::SwiGlu(SwiGlu::new(8, 12, false).unwrap()),
                    Residual::Add,
                )
                .unwrap(),
            ],
            final_norm: norm,
            output: Linear::new(8, 11, false).unwrap(),
            tie_word_embeddings: false,
        })
        .unwrap();
        let members = (0..degree)
            .map(|n| ParallelRankId::new((n * 7 + 3) as u32))
            .collect::<Vec<_>>();
        let plan = HybridParallelPlan::new(
            &spec,
            0..1,
            HybridExpertScope {
                topology: ParallelTopologyId::new(8),
                replica: 0,
                stage: 0,
            },
            members.clone(),
            None,
        )
        .unwrap();
        let schedule = plan
            .collective_schedule(&LayerSegmentPlan::new(1, 0..1, true, true).unwrap())
            .unwrap();
        let trace = Rc::new(RefCell::new(Trace::default()));
        (
            RankChannel::new(
                schedule,
                Box::new(Endpoint {
                    members,
                    trace: trace.clone(),
                }),
            ),
            trace,
        )
    }
    #[test]
    fn rank_v2_schedule_consumes_all_sites_and_tp1_makes_no_wire_calls() {
        for degree in [1, 2, 4] {
            let (mut c, t) = channel(degree);
            let tx = ExecutionTransactionId::new(1).unwrap();
            assert_eq!(c.schedule.agreement_version(), 2);
            c.begin(tx, 3, 2).unwrap();
            assert!(c.begin(tx, 3, 2).is_err());
            assert!(c.finish().is_err());
            let sites = c.schedule.sites().to_vec();
            for site in &sites {
                let n = site.shape.elements(2, 3).unwrap();
                let result = c
                    .exchange(site.semantic, site.global_layer, vec![1.; n])
                    .unwrap();
                assert_eq!(
                    result.len(),
                    n * if site.kind == HybridCollectiveKind::AllGather {
                        degree
                    } else {
                        1
                    }
                );
            }
            c.finish().unwrap();
            assert_eq!(t.borrow().calls, if degree == 1 { 0 } else { sites.len() });
        }
    }
    #[test]
    fn rank_channel_rejects_wrong_order_geometry_nonfinite_and_malformed_peer() {
        let (mut c, t) = channel(2);
        c.begin(ExecutionTransactionId::new(1).unwrap(), 2, 1)
            .unwrap();
        assert!(
            c.exchange(HybridCollectiveSemantic::Completion, None, vec![1.])
                .is_err()
        );
        assert!(
            c.exchange(HybridCollectiveSemantic::Agreement, None, vec![1.])
                .is_err()
        );
        let n = c.schedule.sites()[0].shape.elements(1, 2).unwrap();
        assert!(
            c.exchange(HybridCollectiveSemantic::Agreement, None, vec![f32::NAN; n])
                .is_err()
        );
        assert_eq!(t.borrow().calls, 0);
        t.borrow_mut().malformed = true;
        assert!(
            c.exchange(HybridCollectiveSemantic::Agreement, None, vec![1.; n])
                .is_err()
        );
        c.abort();
        assert_eq!(t.borrow().aborts, 1);
        assert!(
            c.begin(ExecutionTransactionId::new(2).unwrap(), 2, 1)
                .is_err()
        );
    }
    #[test]
    fn rank_channel_tp1_abort_has_no_peer_lifetime_to_poison() {
        let (mut c, t) = channel(1);
        c.begin(ExecutionTransactionId::new(1).unwrap(), 2, 1)
            .unwrap();
        c.abort();
        assert_eq!(t.borrow().aborts, 0);
        c.begin(ExecutionTransactionId::new(2).unwrap(), 2, 1)
            .unwrap();
    }
}
