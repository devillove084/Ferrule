use ferrule_common::{ParallelTopologyId, ParallelismPlan, ValidatedParallelTopology};
use ferrule_runtime::distributed::{CommunicationOperation, PendingCompletion};
use ferrule_runtime::{
    Decision, DistributedTransaction, DistributedTransactionError, ExecutionTransactionId,
    FinalizeOutcome, ParallelRankId, TransactionState,
};
fn topology(world_size: u32, local_rank: u32) -> ValidatedParallelTopology {
    ValidatedParallelTopology::new(
        ParallelTopologyId::new(1),
        world_size,
        ParallelRankId::new(local_rank),
        ParallelismPlan {
            data_parallel: usize::try_from(world_size).unwrap(),
            ..ParallelismPlan::default()
        },
    )
    .unwrap()
}

fn id(n: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(n).unwrap()
}
fn prepared(transaction: &mut DistributedTransaction, tx: ExecutionTransactionId) {
    let participants = transaction.topology().participants();
    for rank in participants.iter() {
        transaction.prepare(tx, rank).unwrap();
    }
}
#[test]
fn success_publishes_once_after_all_reports() {
    let mut s = DistributedTransaction::new(topology(2, 0), 4);
    let tx = id(1);
    s.begin(tx).unwrap();
    prepared(&mut s, tx);
    s.reserve(tx, 2).unwrap();
    assert_eq!(s.commit_decision(tx), Ok(Decision::Commit));
    assert!(s.publish(tx).is_err());
    assert_eq!(s.in_use_credits(), 2);
    for rank in 0..2 {
        s.finalize(tx, ParallelRankId::new(rank), FinalizeOutcome::Success)
            .unwrap();
    }
    s.publish(tx).unwrap();
    assert_eq!(s.state(tx), Ok(TransactionState::Published));
    assert_eq!(s.publication_count(), 1);
    assert_eq!(s.in_use_credits(), 0);
    assert!(s.publish(tx).is_err());
    assert_eq!(s.publication_count(), 1);
}
#[test]
fn failure_before_decision_aborts() {
    let mut s = DistributedTransaction::new(topology(2, 0), 1);
    let tx = id(2);
    s.begin(tx).unwrap();
    prepared(&mut s, tx);
    s.finalize(tx, ParallelRankId::new(1), FinalizeOutcome::Failure)
        .unwrap();
    assert_eq!(s.commit_decision(tx), Ok(Decision::Abort));
}
#[test]
fn pending_and_cancel_are_observable() {
    let mut s = DistributedTransaction::new(topology(2, 0), 2);
    let tx = id(3);
    s.begin(tx).unwrap();
    prepared(&mut s, tx);
    s.reserve(tx, 2).unwrap();
    assert_eq!(s.in_use_credits(), 2);
    s.finalize(tx, ParallelRankId::new(0), FinalizeOutcome::Success)
        .unwrap();
    assert_eq!(s.pending_ranks(tx).unwrap(), vec![ParallelRankId::new(1)]);
    assert!(s.publish(tx).is_err());
    s.cancel(tx).unwrap();
    assert_eq!(s.in_use_credits(), 0);
}
#[test]
fn bad_reports_and_backpressure_are_rejected() {
    let mut s = DistributedTransaction::new(topology(2, 0), 1);
    let tx = id(4);
    s.begin(tx).unwrap();
    s.reserve(tx, 1).unwrap();
    assert_eq!(
        s.reserve(tx, 1),
        Err(DistributedTransactionError::Backpressure)
    );
    assert_eq!(
        s.finalize(tx, ParallelRankId::new(9), FinalizeOutcome::Success),
        Err(DistributedTransactionError::InvalidRank)
    );
    prepared(&mut s, tx);
    s.commit_decision(tx).unwrap();
    s.finalize(tx, ParallelRankId::new(0), FinalizeOutcome::Success)
        .unwrap();
    assert_eq!(
        s.finalize(tx, ParallelRankId::new(0), FinalizeOutcome::Success),
        Err(DistributedTransactionError::DuplicateReport)
    );
    assert_eq!(
        s.finalize(tx, ParallelRankId::new(1), FinalizeOutcome::Success),
        Ok(())
    );
}
#[test]
fn duplicate_report_cannot_replace_original_outcome() {
    for (original, duplicate, decision) in [
        (
            FinalizeOutcome::Success,
            FinalizeOutcome::Failure,
            Decision::Commit,
        ),
        (
            FinalizeOutcome::Failure,
            FinalizeOutcome::Success,
            Decision::Abort,
        ),
    ] {
        let mut s = DistributedTransaction::new(topology(2, 0), 1);
        let tx = id(1);
        s.begin(tx).unwrap();
        prepared(&mut s, tx);
        s.finalize(tx, ParallelRankId::new(0), original).unwrap();
        assert_eq!(
            s.finalize(tx, ParallelRankId::new(0), duplicate),
            Err(DistributedTransactionError::DuplicateReport)
        );
        assert_eq!(s.commit_decision(tx), Ok(decision));
    }
}
#[test]
fn communication_blocks_decision_until_completion_is_drained() {
    let mut s = DistributedTransaction::new(topology(2, 0), 2);
    let tx = id(1);
    s.begin(tx).unwrap();
    prepared(&mut s, tx);
    let operation = s.communicate(tx, ParallelRankId::new(0)).unwrap();
    assert_eq!(
        s.pending_completions(),
        vec![PendingCompletion {
            operation,
            outcome: None
        }]
    );
    assert_eq!(s.drain(), 0);
    assert_eq!(s.in_use_credits(), 1);
    assert_eq!(
        s.commit_decision(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    for rank in 0..2 {
        s.finalize(tx, ParallelRankId::new(rank), FinalizeOutcome::Success)
            .unwrap();
    }
    assert_eq!(
        s.publish(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    s.complete(operation, FinalizeOutcome::Success).unwrap();
    assert_eq!(
        s.pending_completions(),
        vec![PendingCompletion {
            operation,
            outcome: Some(FinalizeOutcome::Success)
        }]
    );
    assert_eq!(s.in_use_credits(), 1);
    assert_eq!(
        s.commit_decision(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    assert_eq!(s.drain(), 1);
    assert_eq!(s.drain(), 0);
    assert!(s.pending_completions().is_empty());
    assert_eq!(s.in_use_credits(), 0);
    assert_eq!(s.commit_decision(tx), Ok(Decision::Commit));
    s.publish(tx).unwrap();
    assert_eq!(
        s.complete(operation, FinalizeOutcome::Failure),
        Err(DistributedTransactionError::DuplicateOperation)
    );
    assert_eq!(
        s.communicate(tx, ParallelRankId::new(0)),
        Err(DistributedTransactionError::InvalidState)
    );
    assert_eq!(s.publication_count(), 1);
}
#[test]
fn out_of_order_completions_drain_independently_and_failure_aborts() {
    let mut s = DistributedTransaction::new(topology(2, 0), 2);
    let tx = id(1);
    s.begin(tx).unwrap();
    prepared(&mut s, tx);
    let first = s.communicate(tx, ParallelRankId::new(0)).unwrap();
    let second = s.communicate(tx, ParallelRankId::new(1)).unwrap();
    s.complete(second, FinalizeOutcome::Failure).unwrap();
    assert_eq!(
        s.complete(second, FinalizeOutcome::Success),
        Err(DistributedTransactionError::DuplicateOperation)
    );
    assert_eq!(s.drain(), 1);
    assert_eq!(
        s.pending_completions(),
        vec![PendingCompletion {
            operation: first,
            outcome: None
        }]
    );
    assert_eq!(s.in_use_credits(), 1);
    assert_eq!(
        s.commit_decision(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    s.complete(first, FinalizeOutcome::Success).unwrap();
    assert_eq!(s.drain(), 1);
    assert_eq!(s.commit_decision(tx), Ok(Decision::Abort));
    for rank in 0..2 {
        s.finalize(tx, ParallelRankId::new(rank), FinalizeOutcome::Success)
            .unwrap();
    }
    s.publish(tx).unwrap();
    assert_eq!(s.publication_count(), 1);
    assert_eq!(s.in_use_credits(), 0);
}
#[test]
fn stale_identity_rejection_has_no_side_effects() {
    let mut s = DistributedTransaction::new(topology(2, 0), 2);
    s.begin(id(1)).unwrap();
    s.begin(id(2)).unwrap();
    let operation = s.communicate(id(1), ParallelRankId::new(0)).unwrap();
    let other = s.communicate(id(2), ParallelRankId::new(0)).unwrap();
    assert_ne!(operation.identity, other.identity);
    let pending = s.pending_completions();
    for stale in [
        CommunicationOperation {
            transaction: id(2),
            ..operation
        },
        CommunicationOperation {
            transaction: id(99),
            ..operation
        },
        CommunicationOperation {
            rank: ParallelRankId::new(1),
            ..operation
        },
        CommunicationOperation {
            rank: ParallelRankId::new(99),
            ..operation
        },
        CommunicationOperation {
            identity: 0,
            ..operation
        },
        CommunicationOperation {
            identity: other.identity,
            ..operation
        },
    ] {
        assert_eq!(
            s.complete(stale, FinalizeOutcome::Failure),
            Err(DistributedTransactionError::StaleOperation)
        );
        assert_eq!(s.pending_completions(), pending);
        assert_eq!(s.in_use_credits(), 2);
        assert_eq!(s.drain(), 0);
    }
    s.complete(operation, FinalizeOutcome::Success).unwrap();
    assert_eq!(s.drain(), 1);
    assert_eq!(
        s.complete(operation, FinalizeOutcome::Failure),
        Err(DistributedTransactionError::DuplicateOperation)
    );
    assert_eq!(
        s.pending_completions(),
        vec![PendingCompletion {
            operation: other,
            outcome: None
        }]
    );
}
#[test]
fn cancellation_keeps_operation_credits_until_late_completion_drains() {
    let mut s = DistributedTransaction::new(topology(2, 0), 2);
    let tx = id(1);
    s.begin(tx).unwrap();
    s.begin(id(2)).unwrap();
    s.reserve(tx, 1).unwrap();
    let operation = s.communicate(tx, ParallelRankId::new(0)).unwrap();
    assert_eq!(
        s.communicate(id(2), ParallelRankId::new(0)),
        Err(DistributedTransactionError::Backpressure)
    );
    s.cancel(tx).unwrap();
    assert_eq!(s.state(tx), Ok(TransactionState::Cancelled));
    assert_eq!(s.in_use_credits(), 1);
    assert_eq!(
        s.release(tx, 1),
        Err(DistributedTransactionError::InvalidState)
    );
    assert_eq!(s.cancel(tx), Err(DistributedTransactionError::InvalidState));
    assert_eq!(
        s.begin(tx),
        Err(DistributedTransactionError::InvalidTransaction)
    );
    assert_eq!(s.drain(), 0);
    s.reserve(id(2), 1).unwrap();
    assert_eq!(
        s.reserve(id(2), 1),
        Err(DistributedTransactionError::Backpressure)
    );
    s.complete(operation, FinalizeOutcome::Success).unwrap();
    assert_eq!(s.in_use_credits(), 2);
    assert_eq!(s.drain(), 1);
    assert_eq!(s.in_use_credits(), 1);
    assert_eq!(s.drain(), 0);
    assert_eq!(
        s.publish(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    assert_eq!(
        s.finalize(tx, ParallelRankId::new(0), FinalizeOutcome::Success),
        Err(DistributedTransactionError::InvalidState)
    );
    assert_eq!(
        s.communicate(tx, ParallelRankId::new(0)),
        Err(DistributedTransactionError::InvalidState)
    );
    assert_eq!(
        s.reserve(tx, 1),
        Err(DistributedTransactionError::InvalidState)
    );
    assert_eq!(s.publication_count(), 0);
    let next = s.communicate(id(2), ParallelRankId::new(0)).unwrap();
    assert_ne!(next.identity, operation.identity);
    assert_eq!(
        s.complete(operation, FinalizeOutcome::Success),
        Err(DistributedTransactionError::DuplicateOperation)
    );
    assert_eq!(s.in_use_credits(), 2);
    s.complete(next, FinalizeOutcome::Success).unwrap();
    s.cancel(id(2)).unwrap();
    assert_eq!(s.in_use_credits(), 1);
    assert_eq!(s.drain(), 1);
    assert_eq!(s.in_use_credits(), 0);
}
#[test]
fn credits_are_bounded_and_operation_credits_cannot_be_released_manually() {
    let mut s = DistributedTransaction::new(topology(2, 0), usize::MAX);
    let tx = id(1);
    s.begin(tx).unwrap();
    s.reserve(tx, usize::MAX).unwrap();
    assert_eq!(
        s.reserve(tx, 1),
        Err(DistributedTransactionError::Backpressure)
    );
    assert_eq!(
        s.communicate(tx, ParallelRankId::new(0)),
        Err(DistributedTransactionError::Backpressure)
    );
    assert_eq!(s.in_use_credits(), usize::MAX);
    s.release(tx, usize::MAX).unwrap();
    let operation = s.communicate(tx, ParallelRankId::new(0)).unwrap();
    assert_eq!(
        s.reserve(tx, usize::MAX),
        Err(DistributedTransactionError::Backpressure)
    );
    assert_eq!(
        s.release(tx, 1),
        Err(DistributedTransactionError::InvalidState)
    );
    assert_eq!(s.in_use_credits(), 1);
    s.complete(operation, FinalizeOutcome::Success).unwrap();
    assert_eq!(s.drain(), 1);
    assert_eq!(s.in_use_credits(), 0);
}
#[test]
fn invalid_submissions_and_zero_capacity_do_not_create_operations() {
    let mut s = DistributedTransaction::new(topology(2, 0), 0);
    let tx = id(1);
    s.begin(tx).unwrap();
    assert_eq!(
        s.communicate(tx, ParallelRankId::new(2)),
        Err(DistributedTransactionError::InvalidRank)
    );
    assert_eq!(
        s.communicate(id(2), ParallelRankId::new(0)),
        Err(DistributedTransactionError::InvalidTransaction)
    );
    assert_eq!(
        s.communicate(tx, ParallelRankId::new(0)),
        Err(DistributedTransactionError::Backpressure)
    );
    assert_eq!(
        s.reserve(id(2), 0),
        Err(DistributedTransactionError::InvalidTransaction)
    );
    assert!(s.pending_completions().is_empty());
    assert_eq!(s.in_use_credits(), 0);
    assert_eq!(s.drain(), 0);
}

#[test]
fn commit_rejects_cancel_and_can_finish_publication() {
    let mut s = DistributedTransaction::new(topology(2, 1), 3);
    let tx = id(10);
    s.begin(tx).unwrap();
    s.reserve(tx, 2).unwrap();
    let operation = s.communicate(tx, ParallelRankId::new(0)).unwrap();
    s.complete(operation, FinalizeOutcome::Success).unwrap();
    assert_eq!(s.drain(), 1);
    prepared(&mut s, tx);
    assert_eq!(s.commit_decision(tx), Ok(Decision::Commit));
    s.finalize(tx, ParallelRankId::new(0), FinalizeOutcome::Success)
        .unwrap();
    let pending = s.pending_completions();
    assert_eq!(s.pending_ranks(tx).unwrap(), vec![ParallelRankId::new(1)]);
    assert_eq!(s.in_use_credits(), 2);
    assert_eq!(s.cancel(tx), Err(DistributedTransactionError::InvalidState));
    assert_eq!(s.state(tx), Ok(TransactionState::Decided(Decision::Commit)));
    assert_eq!(s.pending_ranks(tx).unwrap(), vec![ParallelRankId::new(1)]);
    assert_eq!(s.pending_completions(), pending);
    assert_eq!(s.in_use_credits(), 2);
    assert_eq!(s.publication_count(), 0);
    assert_eq!(
        s.commit_decision(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    assert_eq!(
        s.publish(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    s.finalize(tx, ParallelRankId::new(1), FinalizeOutcome::Success)
        .unwrap();
    s.publish(tx).unwrap();
    assert_eq!(s.state(tx), Ok(TransactionState::Published));
    assert_eq!(s.in_use_credits(), 0);
    assert_eq!(s.publication_count(), 1);
    assert_eq!(s.cancel(tx), Err(DistributedTransactionError::InvalidState));
    assert_eq!(
        s.publish(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    assert_eq!(s.publication_count(), 1);
}

#[test]
fn failure_after_commit_is_retryable_and_blocks_publication() {
    let mut s = DistributedTransaction::new(topology(2, 0), 2);
    let tx = id(11);
    s.begin(tx).unwrap();
    s.reserve(tx, 2).unwrap();
    prepared(&mut s, tx);
    assert_eq!(s.commit_decision(tx), Ok(Decision::Commit));
    let retry_rank = ParallelRankId::new(0);
    s.finalize(tx, ParallelRankId::new(1), FinalizeOutcome::Success)
        .unwrap();
    for _ in 0..2 {
        s.finalize(tx, retry_rank, FinalizeOutcome::Failure)
            .unwrap();
        assert_eq!(s.pending_ranks(tx).unwrap(), vec![retry_rank]);
        assert_eq!(
            s.publish(tx),
            Err(DistributedTransactionError::InvalidState)
        );
        assert_eq!(s.cancel(tx), Err(DistributedTransactionError::InvalidState));
        assert_eq!(s.state(tx), Ok(TransactionState::Decided(Decision::Commit)));
        assert_eq!(s.in_use_credits(), 2);
        assert_eq!(s.publication_count(), 0);
    }
    s.finalize(tx, retry_rank, FinalizeOutcome::Success)
        .unwrap();
    assert!(s.pending_ranks(tx).unwrap().is_empty());
    for duplicate in [FinalizeOutcome::Success, FinalizeOutcome::Failure] {
        assert_eq!(
            s.finalize(tx, retry_rank, duplicate),
            Err(DistributedTransactionError::DuplicateReport)
        );
        assert!(s.pending_ranks(tx).unwrap().is_empty());
        assert_eq!(s.state(tx), Ok(TransactionState::Decided(Decision::Commit)));
        assert_eq!(s.in_use_credits(), 2);
    }
    s.publish(tx).unwrap();
    assert_eq!(s.state(tx), Ok(TransactionState::Published));
    assert_eq!(s.publication_count(), 1);
    assert_eq!(s.in_use_credits(), 0);
    assert_eq!(
        s.publish(tx),
        Err(DistributedTransactionError::InvalidState)
    );
}

#[test]
fn topology_membership_drives_all_protocol_paths() {
    for world in [1, 3] {
        let topology = topology(world, world - 1);
        let participants = topology.participants();
        let last = ParallelRankId::new(world - 1);
        let mut s = DistributedTransaction::new(topology.clone(), 1);
        assert_eq!(s.topology(), &topology);
        let tx: ferrule_common::execution::ExecutionTransactionId = id(12);
        let local: ferrule_common::ParallelRankId = last;
        s.begin(tx).unwrap();
        for rank in participants.iter().filter(|rank| *rank != local) {
            s.prepare(tx, rank).unwrap();
        }
        assert_eq!(
            s.commit_decision(tx),
            Err(DistributedTransactionError::InvalidState)
        );
        let outsider = ParallelRankId::new(world);
        assert!(!participants.contains(outsider));
        assert_eq!(
            s.prepare(tx, outsider),
            Err(DistributedTransactionError::InvalidRank)
        );
        assert_eq!(
            s.communicate(tx, outsider),
            Err(DistributedTransactionError::InvalidRank)
        );
        assert_eq!(
            s.finalize(tx, outsider, FinalizeOutcome::Success),
            Err(DistributedTransactionError::InvalidRank)
        );
        assert_eq!(s.in_use_credits(), 0);
        assert!(s.pending_completions().is_empty());
        assert_eq!(
            s.pending_ranks(tx).unwrap(),
            participants.iter().collect::<Vec<_>>()
        );
        s.prepare(tx, local).unwrap();
        assert_eq!(
            s.prepare(tx, local),
            Err(DistributedTransactionError::InvalidState)
        );
        let operation = s.communicate(tx, local).unwrap();
        s.complete(operation, FinalizeOutcome::Success).unwrap();
        assert_eq!(s.drain(), 1);
        assert_eq!(s.commit_decision(tx), Ok(Decision::Commit));
        for rank in participants.iter().filter(|rank| *rank != local) {
            s.finalize(tx, rank, FinalizeOutcome::Success).unwrap();
        }
        assert_eq!(s.pending_ranks(tx).unwrap(), vec![local]);
        assert_eq!(
            s.publish(tx),
            Err(DistributedTransactionError::InvalidState)
        );
        s.finalize(tx, local, FinalizeOutcome::Success).unwrap();
        s.publish(tx).unwrap();
        assert_eq!(s.publication_count(), 1);
        assert_eq!(s.in_use_credits(), 0);
    }
}

#[test]
fn terminal_ids_and_stale_reports_cannot_target_new_transactions() {
    let mut s = DistributedTransaction::new(topology(2, 0), 2);
    let old = id(20);
    let current = id(21);
    let rank = ParallelRankId::new(0);
    s.begin(old).unwrap();
    let old_operation = s.communicate(old, rank).unwrap();
    s.cancel(old).unwrap();
    s.begin(current).unwrap();
    let current_operation = s.communicate(current, rank).unwrap();
    assert_ne!(old_operation.identity, current_operation.identity);
    let pending = s.pending_completions();
    assert_eq!(
        s.begin(old),
        Err(DistributedTransactionError::InvalidTransaction)
    );
    assert_eq!(
        s.begin(current),
        Err(DistributedTransactionError::InvalidTransaction)
    );
    for outcome in [FinalizeOutcome::Failure, FinalizeOutcome::Success] {
        assert_eq!(
            s.finalize(old, rank, outcome),
            Err(DistributedTransactionError::InvalidState)
        );
        assert_eq!(
            s.finalize(id(99), rank, outcome),
            Err(DistributedTransactionError::InvalidTransaction)
        );
        assert_eq!(
            s.complete(
                CommunicationOperation {
                    transaction: current,
                    ..old_operation
                },
                outcome
            ),
            Err(DistributedTransactionError::StaleOperation)
        );
        assert_eq!(s.pending_completions(), pending);
        assert_eq!(s.in_use_credits(), 2);
        assert_eq!(
            s.pending_ranks(current).unwrap(),
            vec![rank, ParallelRankId::new(1)]
        );
        assert_eq!(s.state(current), Ok(TransactionState::Preparing));
    }
    // A legitimate late completion releases only the cancelled transaction's credit.
    s.complete(old_operation, FinalizeOutcome::Failure).unwrap();
    assert_eq!(s.in_use_credits(), 2);
    assert_eq!(s.drain(), 1);
    assert_eq!(s.in_use_credits(), 1);
    assert_eq!(
        s.pending_completions(),
        vec![PendingCompletion {
            operation: current_operation,
            outcome: None
        }]
    );
    assert_eq!(
        s.complete(old_operation, FinalizeOutcome::Success),
        Err(DistributedTransactionError::DuplicateOperation)
    );
    s.complete(current_operation, FinalizeOutcome::Success)
        .unwrap();
    assert_eq!(s.drain(), 1);
    prepared(&mut s, current);
    assert_eq!(s.commit_decision(current), Ok(Decision::Commit));
    for rank in s.topology().participants().iter() {
        s.finalize(current, rank, FinalizeOutcome::Success).unwrap();
    }
    s.publish(current).unwrap();
    assert_eq!(
        s.begin(current),
        Err(DistributedTransactionError::InvalidTransaction)
    );
    let next = id(22);
    s.begin(next).unwrap();
    s.reserve(next, 1).unwrap();
    for stale in [old, current] {
        assert_eq!(
            s.finalize(stale, rank, FinalizeOutcome::Failure),
            Err(DistributedTransactionError::InvalidState)
        );
        assert_eq!(s.in_use_credits(), 1);
        assert_eq!(s.state(next), Ok(TransactionState::Preparing));
        assert_eq!(
            s.pending_ranks(next).unwrap(),
            vec![rank, ParallelRankId::new(1)]
        );
        assert_eq!(s.publication_count(), 1);
    }
    s.cancel(next).unwrap();
    assert_eq!(s.drain(), 0);
    assert!(s.pending_completions().is_empty());
    assert_eq!(s.in_use_credits(), 0);
}

#[test]
fn abort_can_be_cancelled_or_published_only_after_all_reports() {
    for cancel in [false, true] {
        let mut s = DistributedTransaction::new(topology(2, 0), 1);
        let tx = id(30);
        s.begin(tx).unwrap();
        s.reserve(tx, 1).unwrap();
        prepared(&mut s, tx);
        s.finalize(tx, ParallelRankId::new(0), FinalizeOutcome::Failure)
            .unwrap();
        assert_eq!(s.commit_decision(tx), Ok(Decision::Abort));
        assert_eq!(
            s.publish(tx),
            Err(DistributedTransactionError::InvalidState)
        );
        assert_eq!(s.in_use_credits(), 1);
        if cancel {
            s.cancel(tx).unwrap();
            assert_eq!(s.state(tx), Ok(TransactionState::Cancelled));
            assert_eq!(s.publication_count(), 0);
        } else {
            s.finalize(tx, ParallelRankId::new(1), FinalizeOutcome::Success)
                .unwrap();
            s.publish(tx).unwrap();
            assert_eq!(s.state(tx), Ok(TransactionState::Published));
            assert_eq!(s.publication_count(), 1);
        }
        assert_eq!(s.in_use_credits(), 0);
    }
}
