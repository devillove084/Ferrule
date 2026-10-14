//! Real runtime owners and the existing parent transaction protocol. The shared
//! fixture supplies checkpoint bytes only, never a test-owned execution loop.
use super::*;
#[path = "../../../../../ferrule-model/tests/support/hybrid_tp_ep_fixture.rs"]
mod checkpoint_fixture;
use ferrule_common::execution::ForwardPhase;
use ferrule_common::{ParallelTopologyId, ParallelismPlan};
use ferrule_model::decoder::GatedDeltaNetState;
use ferrule_model::transformer::hybrid_parallel::{
    HybridExpertAttachment, HybridExpertScope, HybridLocalAttention,
};
use ferrule_model::transformer::{ExpertPlacement, KvHistory, KvView};
use std::sync::atomic::{AtomicUsize, Ordering};

mod fixture;
use fixture::*;
mod probe;
use probe::*;
mod numerics;
use numerics::close;
mod admission;
mod child_lifecycle;
mod custody;
mod late_fault;
mod protocol;
mod retry;
mod schedule;
mod startup;
mod topology;
mod tracing;

type ExpertStartupHook =
    Arc<dyn Fn(crate::parallel::expert::ExpertRank) -> Result<()> + Send + Sync>;
std::thread_local! {pub(super) static EXPERT_STARTUP:RefCell<Option<ExpertStartupHook>>=RefCell::new(None);}
