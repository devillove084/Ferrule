use std::collections::{BTreeMap, BTreeSet};

use ferrule_common::ParallelRankId;
use ferrule_common::execution::ExecutionTransactionId;
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::{
    ProcessError, ProcessGroupEpoch, ProcessIdentity, ProcessLaunch, ProcessOwnerInstanceId,
    ProcessOwnerState, ProcessRankOwner, ProcessSupervisorConfig, ProcessTerminationReport,
    ReapOutcome, invalid,
};

/// One recovery domain (a singleton DP rank or one TP group). This synchronous
/// control object owns no transaction/collective state. A transport failure
/// immediately invalidates the domain; callers must explicitly restart with a
/// larger epoch and explicitly resubmit only replay-safe work.
pub struct ProcessRankSupervisor<C, I, O> {
    config: ProcessSupervisorConfig,
    epoch: ProcessGroupEpoch,
    valid: bool,
    next_owner: u64,
    restarts: usize,
    attempted: BTreeSet<ParallelRankId>,
    owners: BTreeMap<ParallelRankId, ProcessRankOwner<C, I, O>>,
}
impl<C: Serialize, I: Serialize, O: DeserializeOwned> ProcessRankSupervisor<C, I, O> {
    pub fn new(
        epoch: ProcessGroupEpoch,
        config: ProcessSupervisorConfig,
    ) -> Result<Self, ProcessError> {
        config.owner.validate()?;
        if config.max_ranks == 0 || config.max_ranks > config.max_quarantined {
            return Err(invalid("max_ranks must be nonzero and <= max_quarantined"));
        }
        Ok(Self {
            config,
            epoch,
            valid: true,
            next_owner: 1,
            restarts: 0,
            attempted: BTreeSet::new(),
            owners: BTreeMap::new(),
        })
    }
    pub const fn epoch(&self) -> ProcessGroupEpoch {
        self.epoch
    }
    pub const fn is_valid(&self) -> bool {
        self.valid
    }
    pub const fn restart_attempts(&self) -> usize {
        self.restarts
    }
    pub fn owner_count(&self) -> usize {
        self.owners.len()
    }
    pub fn quarantined_count(&self) -> usize {
        self.owners
            .values()
            .filter(|owner| owner.state() == ProcessOwnerState::Quarantined)
            .count()
    }
    pub fn owner(&self, rank: ParallelRankId) -> Option<&ProcessRankOwner<C, I, O>> {
        self.owners.get(&rank)
    }

    pub fn spawn_rank(
        &mut self,
        rank: ParallelRankId,
        launch: ProcessLaunch,
        config: &C,
    ) -> Result<ProcessIdentity, ProcessError> {
        if !self.valid {
            return Err(ProcessError::GroupInvalidated);
        }
        if self.attempted.contains(&rank) {
            return Err(ProcessError::RankAlreadySpawned { rank });
        }
        if self.attempted.len() == self.config.max_ranks {
            return Err(ProcessError::Capacity {
                resource: "supervisor ranks",
                limit: self.config.max_ranks,
            });
        }
        let owner_instance = ProcessOwnerInstanceId::new(self.next_owner)?;
        self.next_owner = self
            .next_owner
            .checked_add(1)
            .ok_or(ProcessError::IdentityExhausted)?;
        let identity = ProcessIdentity::new(self.epoch, rank, owner_instance);
        self.attempted.insert(rank);
        match ProcessRankOwner::spawn(launch, identity, config, self.config.owner) {
            Ok(owner) => {
                self.owners.insert(rank, owner);
                Ok(identity)
            }
            Err(error) => {
                if let Some(owner) = error.owner {
                    self.owners.insert(rank, *owner);
                }
                // A failed startup is an attempt too. It cannot be retried via
                // spawn_rank repeatedly without consuming the restart budget.
                self.valid = false;
                Err(error.source)
            }
        }
    }

    pub fn execute(
        &mut self,
        rank: ParallelRankId,
        transaction: ExecutionTransactionId,
        session: u64,
        input: &I,
    ) -> Result<O, ProcessError> {
        if !self.valid {
            return Err(ProcessError::GroupInvalidated);
        }
        let result = self
            .owners
            .get_mut(&rank)
            .ok_or(ProcessError::RankNotFound { rank })?
            .execute(transaction, session, input);
        if result.as_ref().is_err_and(|error| {
            error.is_quiescence_unknown() || matches!(error, ProcessError::OwnerUnavailable)
        }) {
            // Admission/publication gating changes BEFORE returning the failure.
            // Peer termination is explicit and does not fabricate their fences.
            self.valid = false;
        }
        result
    }

    /// Invalidate admission immediately, then terminate every owner. Reports
    /// are OS evidence only; old transaction custody remains caller-owned.
    pub fn invalidate_group(&mut self) -> Vec<(ProcessIdentity, ProcessTerminationReport)> {
        self.valid = false;
        self.owners
            .values_mut()
            .map(|owner| (owner.identity(), owner.terminate()))
            .collect()
    }
    pub fn shutdown(&mut self) -> Vec<(ProcessIdentity, ProcessTerminationReport)> {
        self.invalidate_group()
    }

    /// Every attempt consumes budget, including one blocked by D-state children.
    /// No same-epoch replacement, owner-ID reuse, or command replay is allowed.
    /// An unreaped child blocks replacement (device migration is caller policy).
    pub fn restart_group(&mut self, epoch: ProcessGroupEpoch) -> Result<(), ProcessError> {
        if epoch <= self.epoch {
            return Err(invalid("restart requires a strictly newer group epoch"));
        }
        if self.restarts == self.config.max_restarts {
            return Err(ProcessError::RestartBudgetExhausted);
        }
        self.restarts += 1;
        let reports = self.invalidate_group();
        let unreaped = reports
            .iter()
            .filter(|(_, report)| report.reap != ReapOutcome::Reaped)
            .count();
        if unreaped != 0 {
            return Err(ProcessError::RestartBlocked { unreaped });
        }
        self.owners.clear();
        self.attempted.clear();
        self.epoch = epoch;
        self.valid = true;
        Ok(())
    }

    /// Nonblocking later reap attempts. This never revalidates a failed epoch.
    pub fn poll_reap(&mut self) -> Result<usize, ProcessError> {
        let mut reaped = 0;
        for owner in self.owners.values_mut() {
            if owner.state() == ProcessOwnerState::Quarantined
                && owner.poll_reap()? == ReapOutcome::Reaped
            {
                reaped += 1;
            }
        }
        Ok(reaped)
    }
}
