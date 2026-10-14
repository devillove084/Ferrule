//! Owner-local Mixed data schedule; no rendezvous, device or KV authority.
//! Shared by CUDA execution and CPU fault tests.
use super::*;
use crate::decoder::CompletionUnknown;
use ferrule_common::{error::ErrorChain, execution::ExecutionTransactionId};

pub(crate) struct Bf16Channel {
    pub(crate) transport: Box<dyn HybridBf16Collective>,
    sites: Vec<HybridCollectiveSite>,
    pub(crate) active: Option<(ExecutionTransactionId, usize, usize)>,
    watermark: Option<ExecutionTransactionId>,
    next: usize,
    agreed_route: Option<HybridMixedRouteSite>,
    pub(crate) failed: bool,
    unknown: bool,
    device_permit: Option<DevicePayloadIdentity>,
    device_uncompleted: bool,
}
impl Bf16Channel {
    pub(crate) fn new(
        transport: Box<dyn HybridBf16Collective>,
        sites: Vec<HybridCollectiveSite>,
    ) -> Self {
        Self {
            transport,
            sites,
            active: None,
            watermark: None,
            next: 0,
            agreed_route: None,
            failed: false,
            unknown: false,
            device_permit: None,
            device_uncompleted: false,
        }
    }
    pub(crate) fn completion_unknown(&self) -> bool {
        self.unknown || (self.failed && (self.device_uncompleted || self.device_permit.is_some()))
    }
    fn ensure_known(&self) -> Result<()> {
        if self.unknown {
            Err(CompletionUnknown::wrap(invalid(
                "Mixed completion remains unknown; custody retained",
            )))
        } else {
            Ok(())
        }
    }
    /// Cleanup errors mean attention completion was not proven. Poison the same
    /// channel and preserve both branches for the existing owner completion boundary.
    pub(crate) fn fail_cleanup(
        &mut self,
        primary: Error,
        cleanup: Result<()>,
        quarantined: bool,
    ) -> Error {
        self.unknown |= self.device_uncompleted
            || self.device_permit.is_some()
            || quarantined
            || cleanup.is_err()
            || ErrorChain::new(&primary)
                .downcast_ref::<CompletionUnknown>()
                .is_some();
        self.abort();
        let source = Error::with_cleanup("Mixed block attention cleanup", primary, cleanup);
        if self.unknown {
            CompletionUnknown::wrap(source)
        } else {
            source
        }
    }
    /// Keep the original panic payload in the error custody when cleanup is
    /// unknown. Resuming an ordinary panic would erase the stronger evidence.
    pub(crate) fn panic_cleanup(
        &mut self,
        panic: Box<dyn std::any::Any + Send>,
        cleanup: Result<()>,
        quarantined: bool,
    ) -> Result<Box<dyn std::any::Any + Send>> {
        if cleanup.is_err() || quarantined || self.unknown {
            Err(self.fail_cleanup(
                Error::ModelSource {
                    source: Box::new(RetainedPanic(std::sync::Mutex::new(panic))),
                },
                cleanup,
                true,
            ))
        } else {
            self.abort();
            Ok(panic)
        }
    }

    pub(crate) fn begin(
        &mut self,
        id: ExecutionTransactionId,
        rows: usize,
        sequences: usize,
    ) -> Result<()> {
        self.ensure_known()?;
        if self.failed || self.active.is_some() || self.watermark.is_some_and(|v| v >= id) {
            return Err(invalid("Mixed block collective lifetime is stale/failed"));
        }
        self.watermark = Some(id);
        self.active = Some((id, rows, sequences));
        self.next = 0;
        Ok(())
    }
    pub(crate) fn agree_routes(&mut self, agreement: &HybridMixedRouteAgreement) -> Result<()> {
        self.ensure_known()?;
        let (id, rows, _) = self
            .active
            .ok_or_else(|| invalid("route agreement outside invocation"))?;
        let site = agreement.site();
        let data = self
            .sites
            .get(self.next)
            .ok_or_else(|| invalid("Mixed route schedule exhausted"))?;
        if self.failed
            || self.agreed_route.is_some()
            || rows != agreement.rows()
            || data.semantic != HybridCollectiveSemantic::RoutedExpertOutput
            || data.global_layer != Some(site.layer)
            || data.ordinal != site.output_site
            || !self.transport.sealed_route_sites()?.contains(site)
        {
            return Err(invalid("Mixed route agreement owner/site/order mismatch"));
        }
        self.transport.agree_routes(id, agreement)?;
        self.agreed_route = Some(site.clone());
        Ok(())
    }
    pub(crate) fn validate_routed_output(&self, layer: usize) -> Result<()> {
        self.ensure_known()?;
        if self.failed
            || self.active.is_none()
            || self.agreed_route.as_ref().is_none_or(|s| {
                s.layer != layer
                    || self
                        .sites
                        .get(self.next)
                        .is_none_or(|data| s.output_site != data.ordinal)
            })
        {
            return Err(invalid("Mixed FFN output lacks exact route agreement"));
        }
        Ok(())
    }
    pub(crate) fn exchange(
        &mut self,
        semantic: HybridCollectiveSemantic,
        values: Vec<u16>,
    ) -> Result<Vec<u16>> {
        self.ensure_known()?;
        let (id, rows, sequences) = self
            .active
            .ok_or_else(|| invalid("Mixed collective outside invocation"))?;
        let site = self
            .sites
            .get(self.next)
            .ok_or_else(|| invalid("Mixed schedule exhausted"))?;
        let n = site.shape.elements(sequences, rows)?;
        if self.failed
            || self.device_permit.is_some()
            || site.semantic != semantic
            || values.len() != n
            || values.iter().any(|v| v & 0x7f80 == 0x7f80)
        {
            return Err(invalid("Mixed BF16 site/order/extent mismatch"));
        }
        if semantic == HybridCollectiveSemantic::RoutedExpertOutput {
            self.validate_routed_output(
                site.global_layer
                    .ok_or_else(|| invalid("Mixed routed site lost layer"))?,
            )?;
        }
        let expected = n
            .checked_mul(if site.kind == HybridCollectiveKind::AllGather {
                2
            } else {
                1
            })
            .ok_or_else(|| invalid("Mixed collective result overflow"))?;
        let result = self
            .transport
            .exchange_bf16(id, site.ordinal, site.kind, values)?;
        if result.len() != expected || result.iter().any(|v| v & 0x7f80 == 0x7f80) {
            return Err(invalid("Mixed BF16 collective returned invalid result"));
        }
        if semantic == HybridCollectiveSemantic::RoutedExpertOutput {
            self.agreed_route = None;
        }
        self.next += 1;
        Ok(result)
    }
    pub(crate) fn finish(&mut self) -> Result<()> {
        self.ensure_known()?;
        if self.device_permit.is_some() || self.device_uncompleted {
            self.unknown = true;
            self.abort();
            return Err(CompletionUnknown::wrap(invalid(
                "Mixed device consumers are not completed",
            )));
        }
        if self.failed || self.next != self.sites.len() {
            return Err(invalid("incomplete Mixed block schedule"));
        }
        self.active = None;
        Ok(())
    }
    pub(crate) fn abort(&mut self) {
        self.active = None;
        if !self.failed {
            self.failed = true;
            let _ =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.transport.abort()));
        }
    }
}
impl Drop for Bf16Channel {
    fn drop(&mut self) {
        self.abort();
    }
}

// The payload is deliberately opaque (including in diagnostics) and is retained
// with the typed Unknown error until the runtime completion boundary takes custody.
struct RetainedPanic(std::sync::Mutex<Box<dyn std::any::Any + Send>>);
impl std::fmt::Debug for RetainedPanic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let _ = &self.0;
        f.write_str("Mixed execution panicked; original payload retained")
    }
}
impl std::fmt::Display for RetainedPanic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}
impl std::error::Error for RetainedPanic {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::PackedDecoderBatch;
    use std::{cell::RefCell, rc::Rc};
    struct Endpoint {
        members: Vec<ParallelRankId>,
        calls: Rc<RefCell<usize>>,
        malformed: bool,
    }
    impl HybridBf16Collective for Endpoint {
        fn owner(&self) -> ParallelRankId {
            self.members[0]
        }
        fn members(&self) -> &[ParallelRankId] {
            &self.members
        }
        fn validate_execution(
            &mut self,
            _: ExecutionTransactionId,
            _: &PackedDecoderBatch,
            _: &[u64],
        ) -> Result<()> {
            Ok(())
        }
        fn complete_owner(&mut self, _: ExecutionTransactionId) -> Result<()> {
            Ok(())
        }
        fn exchange_bf16(
            &mut self,
            _: ExecutionTransactionId,
            _: u64,
            kind: HybridCollectiveKind,
            v: Vec<u16>,
        ) -> Result<Vec<u16>> {
            *self.calls.borrow_mut() += 1;
            if self.malformed {
                return Ok(vec![0x7fc1]);
            }
            Ok(if kind == HybridCollectiveKind::AllGather {
                v.repeat(2)
            } else {
                v
            })
        }
        fn abort(&mut self) {}
    }
    fn channel(malformed: bool) -> (Bf16Channel, Rc<RefCell<usize>>) {
        let calls = Rc::new(RefCell::new(0));
        let sites = [
            (
                HybridCollectiveSemantic::OutputHead,
                HybridCollectiveKind::AllGather,
            ),
            (
                HybridCollectiveSemantic::AttentionOutput,
                HybridCollectiveKind::ReduceSum,
            ),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, (semantic, kind))| HybridCollectiveSite {
            ordinal: i as u64,
            global_layer: None,
            semantic,
            kind,
            shape: HybridCollectiveShape {
                fixed: 2,
                per_sequence: 0,
                per_row: 0,
            },
        })
        .collect();
        (
            Bf16Channel::new(
                Box::new(Endpoint {
                    members: vec![ParallelRankId::new(0), ParallelRankId::new(1)],
                    calls: calls.clone(),
                    malformed,
                }),
                sites,
            ),
            calls,
        )
    }
    #[test]
    fn mixed_complete_block_channel_preserves_bf16_words_order_and_watermark() {
        let (mut c, calls) = channel(false);
        let tx = ExecutionTransactionId::new(1).unwrap();
        assert!(
            c.exchange(HybridCollectiveSemantic::OutputHead, vec![0x3f81; 2])
                .is_err()
        );
        c.begin(tx, 2, 1).unwrap();
        assert!(c.finish().is_err());
        assert!(
            c.exchange(HybridCollectiveSemantic::AttentionOutput, vec![0x3f81; 2])
                .is_err()
        );
        assert_eq!(*calls.borrow(), 0);
        assert_eq!(
            c.exchange(HybridCollectiveSemantic::OutputHead, vec![0x8000, 0x3f81])
                .unwrap(),
            [0x8000, 0x3f81, 0x8000, 0x3f81]
        );
        c.exchange(HybridCollectiveSemantic::AttentionOutput, vec![0x3f81; 2])
            .unwrap();
        c.finish().unwrap();
        assert!(c.begin(tx, 2, 1).is_err());
        c.begin(ExecutionTransactionId::new(2).unwrap(), 2, 1)
            .unwrap();
        c.abort();
        assert!(
            c.begin(ExecutionTransactionId::new(3).unwrap(), 2, 1)
                .is_err()
        );
    }
    #[test]
    fn mixed_complete_block_channel_rejects_nonfinite_and_wrong_output_extent() {
        let (mut c, calls) = channel(true);
        c.begin(ExecutionTransactionId::new(1).unwrap(), 1, 1)
            .unwrap();
        assert!(
            c.exchange(HybridCollectiveSemantic::OutputHead, vec![0x7f80; 2])
                .is_err()
        );
        assert_eq!(*calls.borrow(), 0);
        assert!(
            c.exchange(HybridCollectiveSemantic::OutputHead, vec![0x3f80; 2])
                .is_err()
        );
        assert_eq!(c.next, 0);
    }
    #[test]
    fn cleanup_unknown_keeps_both_sources_and_permanently_rejects_retry() {
        for typed in [false, true] {
            let (mut channel, calls) = channel(false);
            let tx = ExecutionTransactionId::new(1).unwrap();
            channel.begin(tx, 1, 1).unwrap();
            let cleanup: Error =
                std::io::Error::new(std::io::ErrorKind::TimedOut, "attention fence").into();
            let failure = channel.fail_cleanup(
                std::io::Error::new(std::io::ErrorKind::InvalidData, "primary math failure").into(),
                Err(if typed {
                    CompletionUnknown::wrap(cleanup)
                } else {
                    cleanup
                }),
                false,
            );
            assert!(
                ErrorChain::new(&failure)
                    .downcast_ref::<CompletionUnknown>()
                    .is_some()
            );
            assert_eq!(
                ErrorChain::new(&failure)
                    .filter_map(|e| e.downcast_ref::<std::io::Error>().map(std::io::Error::kind))
                    .collect::<Vec<_>>(),
                vec![
                    std::io::ErrorKind::InvalidData,
                    std::io::ErrorKind::TimedOut
                ]
            );
            assert!(channel.completion_unknown());
            assert_eq!(channel.watermark, Some(tx));
            for failure in [
                channel
                    .begin(ExecutionTransactionId::new(2).unwrap(), 1, 1)
                    .unwrap_err(),
                channel.finish().unwrap_err(),
                channel
                    .exchange(HybridCollectiveSemantic::OutputHead, vec![0x3f80; 2])
                    .unwrap_err(),
            ] {
                assert!(
                    ErrorChain::new(&failure)
                        .downcast_ref::<CompletionUnknown>()
                        .is_some()
                );
            }
            assert_eq!(
                *calls.borrow(),
                0,
                "failed lifetime cannot communicate or finish"
            );
            let retry = channel.fail_cleanup(invalid("retry"), Ok(()), false);
            assert!(
                ErrorChain::new(&retry)
                    .downcast_ref::<CompletionUnknown>()
                    .is_some()
            );
        }
    }
    #[test]
    fn primary_unknown_and_quarantine_cannot_be_cleared_by_successful_cleanup() {
        for quarantined in [false, true] {
            let (mut channel, _) = channel(false);
            let source = invalid("lost owner");
            let source = if quarantined {
                source
            } else {
                CompletionUnknown::wrap(source)
            };
            let failure = channel.fail_cleanup(source, Ok(()), quarantined);
            assert!(channel.completion_unknown());
            assert!(
                ErrorChain::new(&failure)
                    .downcast_ref::<CompletionUnknown>()
                    .is_some()
            );
        }
        let (mut channel, _) = channel(false);
        let ordinary = channel.fail_cleanup(invalid("ordinary"), Ok(()), false);
        assert!(!channel.completion_unknown());
        assert!(
            ErrorChain::new(&ordinary)
                .downcast_ref::<CompletionUnknown>()
                .is_none()
        );
        assert!(
            channel
                .begin(ExecutionTransactionId::new(2).unwrap(), 1, 1)
                .is_err()
        );
    }
    #[test]
    fn panic_cleanup_keeps_original_payload_in_typed_unknown_custody() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        struct Payload(Arc<AtomicUsize>);
        impl Drop for Payload {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let drops = Arc::new(AtomicUsize::new(0));
        let (mut channel, _) = channel(false);
        let failure = channel
            .panic_cleanup(
                Box::new(Payload(Arc::clone(&drops))),
                Err(invalid("lost cleanup")),
                false,
            )
            .unwrap_err();
        assert!(
            ErrorChain::new(&failure)
                .downcast_ref::<CompletionUnknown>()
                .is_some()
        );
        assert!(
            ErrorChain::new(&failure)
                .downcast_ref::<RetainedPanic>()
                .is_some()
        );
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0,
            "payload stays with error custody"
        );
        assert!(channel.finish().is_err());
        drop(failure);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        let (mut channel, _) = self::channel(false);
        let original = channel
            .panic_cleanup(Box::new(17_u32), Ok(()), false)
            .unwrap();
        assert_eq!(*original.downcast::<u32>().unwrap(), 17);
        assert!(!channel.completion_unknown());
    }
    #[test]
    fn mixed_route_control_is_required_once_at_exact_output_site() {
        use crate::transformer::RouterRoutes;
        struct Routes {
            members: Vec<ParallelRankId>,
            site: HybridMixedRouteSite,
        }
        impl HybridBf16Collective for Routes {
            fn owner(&self) -> ParallelRankId {
                self.members[0]
            }
            fn members(&self) -> &[ParallelRankId] {
                &self.members
            }
            fn sealed_route_sites(&self) -> Result<Vec<HybridMixedRouteSite>> {
                Ok(vec![self.site.clone()])
            }
            fn agree_routes(
                &mut self,
                _: ExecutionTransactionId,
                _: &HybridMixedRouteAgreement,
            ) -> Result<()> {
                Ok(())
            }
            fn exchange_bf16(
                &mut self,
                _: ExecutionTransactionId,
                _: u64,
                _: HybridCollectiveKind,
                v: Vec<u16>,
            ) -> Result<Vec<u16>> {
                Ok(v)
            }
            fn abort(&mut self) {}
        }
        let site = HybridMixedRouteSite {
            layer: 5,
            output_site: 9,
            experts: 16,
            top_k: 2,
            max_rows: 1,
            accumulation: HybridRoutedAccumulation::HfEagerBf16ExpertOrder,
            combine: HybridMoeCombineOrder::LocalBf16AddThenReduce,
        };
        let mut c = Bf16Channel::new(
            Box::new(Routes {
                members: vec![ParallelRankId::new(0), ParallelRankId::new(1)],
                site: site.clone(),
            }),
            vec![HybridCollectiveSite {
                ordinal: 9,
                global_layer: Some(5),
                semantic: HybridCollectiveSemantic::RoutedExpertOutput,
                kind: HybridCollectiveKind::ReduceSum,
                shape: HybridCollectiveShape {
                    fixed: 0,
                    per_sequence: 0,
                    per_row: 2,
                },
            }],
        );
        let routes = RouterRoutes::new(1, 2, vec![15, 0], vec![0.5, 0.5]).unwrap();
        let agreement = HybridMixedRouteAgreement::new(site.clone(), &routes).unwrap();
        assert!(c.agree_routes(&agreement).is_err());
        c.begin(ExecutionTransactionId::new(1).unwrap(), 1, 1)
            .unwrap();
        assert!(
            c.exchange(HybridCollectiveSemantic::RoutedExpertOutput, vec![0; 2])
                .is_err()
        );
        assert!(c.finish().is_err());
        for changed in [
            HybridMixedRouteSite {
                layer: 6,
                ..site.clone()
            },
            HybridMixedRouteSite {
                output_site: 8,
                ..site.clone()
            },
            HybridMixedRouteSite {
                experts: 17,
                ..site.clone()
            },
        ] {
            assert!(
                c.agree_routes(&HybridMixedRouteAgreement::new(changed, &routes).unwrap())
                    .is_err()
            );
        }
        c.agree_routes(&agreement).unwrap();
        assert!(c.agree_routes(&agreement).is_err());
        assert!(c.validate_routed_output(6).is_err());
        c.exchange(HybridCollectiveSemantic::RoutedExpertOutput, vec![0; 2])
            .unwrap();
        assert!(c.validate_routed_output(5).is_err());
        assert!(c.agree_routes(&agreement).is_err());
        c.finish().unwrap();
    }
}

// The current leaf uses host exchange; the unbound adapter is test-only.
#[cfg(test)]
#[path = "mixed_channel_device_tests.rs"]
mod device_tests;
