use ferrule_common::{ParallelTopologyId, ParallelismPlan, ValidatedParallelTopology};
use ferrule_runtime::distributed::{
    CommunicationOperation, DistributedTransactionLimits, PendingCompletion,
};
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
fn ready(transaction: &mut DistributedTransaction, tx: ExecutionTransactionId) {
    let participants = transaction.participants(tx).unwrap().clone();
    for rank in participants.iter() {
        transaction
            .prepare_vote(tx, rank, FinalizeOutcome::Success)
            .unwrap();
    }
}
fn post_acked(transaction: &mut DistributedTransaction, tx: ExecutionTransactionId) {
    let participants = transaction.participants(tx).unwrap().clone();
    for rank in participants.iter() {
        transaction
            .finalize(tx, rank, FinalizeOutcome::Success)
            .unwrap();
    }
}
fn cancelled(transaction: &mut DistributedTransaction, tx: ExecutionTransactionId) {
    transaction.cancel(tx).unwrap();
    post_acked(transaction, tx);
    transaction.cancel(tx).unwrap();
    assert_eq!(transaction.state(tx), Ok(TransactionState::Cancelled));
}
#[test]
fn success_publishes_once_after_all_reports() {
    let mut s = DistributedTransaction::new(topology(2, 0), 4);
    let tx = id(1);
    s.begin(tx).unwrap();
    ready(&mut s, tx);
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
fn pre_vote_failure_aborts_without_becoming_a_finalize_ack() {
    let mut s = DistributedTransaction::new(topology(2, 0), 1);
    let tx = id(2);
    s.begin(tx).unwrap();
    s.prepare(tx, ParallelRankId::new(0)).unwrap();
    s.prepare_vote(tx, ParallelRankId::new(0), FinalizeOutcome::Success)
        .unwrap();
    s.prepare_vote(tx, ParallelRankId::new(1), FinalizeOutcome::Failure)
        .unwrap();
    assert_eq!(s.commit_decision(tx), Ok(Decision::Abort));
    assert_eq!(
        s.pending_ranks(tx).unwrap(),
        vec![ParallelRankId::new(0), ParallelRankId::new(1)]
    );
}
#[test]
fn pending_and_cancel_are_observable() {
    let mut s = DistributedTransaction::new(topology(2, 0), 2);
    let tx = id(3);
    s.begin(tx).unwrap();
    ready(&mut s, tx);
    s.reserve(tx, 2).unwrap();
    assert_eq!(s.in_use_credits(), 2);
    assert_eq!(
        s.finalize(tx, ParallelRankId::new(0), FinalizeOutcome::Success),
        Err(DistributedTransactionError::FinalizeBeforeDecision)
    );
    assert_eq!(s.pending_ranks(tx).unwrap().len(), 2);
    assert!(s.publish(tx).is_err());
    s.cancel(tx).unwrap();
    assert_eq!(s.state(tx), Ok(TransactionState::Decided(Decision::Abort)));
    assert_eq!(s.in_use_credits(), 2);
    assert_eq!(s.retire(tx), Err(DistributedTransactionError::InvalidState));
    post_acked(&mut s, tx);
    s.cancel(tx).unwrap();
    assert_eq!(s.state(tx), Ok(TransactionState::Cancelled));
    assert_eq!(s.in_use_credits(), 0);
}
#[test]
fn missing_ready_vote_blocks_decision_even_after_admission_and_drain() {
    for outcome in [FinalizeOutcome::Success, FinalizeOutcome::Failure] {
        let mut s = DistributedTransaction::new(topology(2, 0), 2);
        let tx = id(1);
        s.begin(tx).unwrap();
        for rank in 0..2 {
            s.prepare(tx, ParallelRankId::new(rank)).unwrap();
        }
        let operations = s.communicate_all(tx).unwrap();
        for operation in operations {
            s.complete(operation, FinalizeOutcome::Success).unwrap();
        }
        assert_eq!(s.drain(), 2);
        assert_eq!(
            s.commit_decision(tx),
            Err(DistributedTransactionError::InvalidState)
        );
        s.prepare_vote(tx, ParallelRankId::new(0), outcome).unwrap();
        for _ in 0..2 {
            assert_eq!(
                s.commit_decision(tx),
                Err(DistributedTransactionError::InvalidState)
            );
            assert_eq!(s.state(tx), Ok(TransactionState::Preparing));
            assert_eq!(s.retire(tx), Err(DistributedTransactionError::InvalidState));
            assert_eq!(s.publication_count(), 0);
        }
        s.prepare_vote(tx, ParallelRankId::new(1), FinalizeOutcome::Success)
            .unwrap();
        assert_eq!(
            s.commit_decision(tx),
            Ok(if outcome == FinalizeOutcome::Success {
                Decision::Commit
            } else {
                Decision::Abort
            })
        );
        assert_eq!(s.pending_ranks(tx).unwrap().len(), 2);
        assert_eq!(
            s.publish(tx),
            Err(DistributedTransactionError::InvalidState)
        );
    }
}

#[test]
fn successful_ready_votes_never_satisfy_post_decision_acks() {
    let mut s = DistributedTransaction::new(topology(2, 0), 1);
    let tx = id(1);
    s.begin(tx).unwrap();
    for rank in 0..2 {
        s.prepare_vote(tx, ParallelRankId::new(rank), FinalizeOutcome::Success)
            .unwrap();
        for outcome in [FinalizeOutcome::Success, FinalizeOutcome::Failure] {
            assert_eq!(
                s.finalize(tx, ParallelRankId::new(rank), outcome),
                Err(DistributedTransactionError::FinalizeBeforeDecision)
            );
        }
    }
    assert_eq!(s.pending_ranks(tx).unwrap().len(), 2);
    assert_eq!(
        s.publish(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    assert_eq!(s.retire(tx), Err(DistributedTransactionError::InvalidState));
    assert_eq!(s.commit_decision(tx), Ok(Decision::Commit));
    assert_eq!(s.pending_ranks(tx).unwrap().len(), 2);
    assert_eq!(
        s.publish(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    assert_eq!(s.publication_count(), 0);
    for outcome in [FinalizeOutcome::Success, FinalizeOutcome::Failure] {
        assert_eq!(
            s.prepare_vote(tx, ParallelRankId::new(0), outcome),
            Err(DistributedTransactionError::InvalidState)
        );
        assert_eq!(s.state(tx), Ok(TransactionState::Decided(Decision::Commit)));
        assert_eq!(s.pending_ranks(tx).unwrap().len(), 2);
    }
    post_acked(&mut s, tx);
    s.publish(tx).unwrap();
    s.retire(tx).unwrap();
    s.begin(id(2)).unwrap();
    for outcome in [FinalizeOutcome::Success, FinalizeOutcome::Failure] {
        assert_eq!(
            s.finalize(tx, ParallelRankId::new(0), outcome),
            Err(DistributedTransactionError::InvalidTransaction)
        );
        assert_eq!(
            s.prepare_vote(tx, ParallelRankId::new(0), outcome),
            Err(DistributedTransactionError::InvalidTransaction)
        );
    }
    assert_eq!(s.state(id(2)), Ok(TransactionState::Preparing));
    assert_eq!(s.pending_ranks(id(2)).unwrap().len(), 2);
    assert_eq!(s.publication_count(), 1);
}

#[test]
fn pre_vote_failure_cannot_be_erased_by_prepare_or_late_votes() {
    for prepared_first in [false, true] {
        let mut s = DistributedTransaction::new(topology(2, 0), 0);
        let tx = id(1);
        let rank = ParallelRankId::new(0);
        s.begin(tx).unwrap();
        if prepared_first {
            s.prepare(tx, rank).unwrap();
        }
        s.prepare_vote(tx, rank, FinalizeOutcome::Failure).unwrap();
        if !prepared_first {
            s.prepare(tx, rank).unwrap();
        }
        assert_eq!(
            s.prepare_vote(tx, rank, FinalizeOutcome::Success),
            Err(DistributedTransactionError::DuplicateReport)
        );
        assert_eq!(s.retire(tx), Err(DistributedTransactionError::InvalidState));
        // Admission and a peer's failure do not manufacture the missing vote.
        assert_eq!(
            s.commit_decision(tx),
            Err(DistributedTransactionError::InvalidState)
        );
        s.prepare_vote(tx, ParallelRankId::new(1), FinalizeOutcome::Failure)
            .unwrap();
        assert_eq!(s.commit_decision(tx), Ok(Decision::Abort));
        for outcome in [FinalizeOutcome::Success, FinalizeOutcome::Failure] {
            assert_eq!(
                s.prepare_vote(tx, rank, outcome),
                Err(DistributedTransactionError::InvalidState)
            );
        }
        assert_eq!(s.pending_ranks(tx).unwrap().len(), 2);
        assert_eq!(
            s.publish(tx),
            Err(DistributedTransactionError::InvalidState)
        );
        assert_eq!(s.retire(tx), Err(DistributedTransactionError::InvalidState));
        assert_eq!(s.state(tx), Ok(TransactionState::Decided(Decision::Abort)));
    }
}

#[test]
fn explicit_abort_needs_cleanup_acks_and_drain_even_if_acks_arrive_first() {
    let mut s = DistributedTransaction::new(topology(2, 0), 1);
    let tx = id(1);
    s.begin(tx).unwrap();
    let operation = s.communicate(tx, ParallelRankId::new(0)).unwrap();
    assert_eq!(s.abort_decision(tx), Ok(Decision::Abort));
    post_acked(&mut s, tx);
    for duplicate in [FinalizeOutcome::Success, FinalizeOutcome::Failure] {
        assert_eq!(
            s.finalize(tx, ParallelRankId::new(0), duplicate),
            Err(DistributedTransactionError::DuplicateReport)
        );
    }
    assert_eq!(
        s.publish(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    s.cancel(tx).unwrap();
    assert_eq!(s.state(tx), Ok(TransactionState::Decided(Decision::Abort)));
    assert_eq!(s.retire(tx), Err(DistributedTransactionError::InvalidState));
    s.complete(operation, FinalizeOutcome::Failure).unwrap();
    assert_eq!(
        s.publish(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    assert_eq!(s.drain(), 1);
    s.publish(tx).unwrap();
    assert_eq!(s.publication_count(), 1);
    assert_eq!(
        s.publish(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    s.retire(tx).unwrap();
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
    ready(&mut s, tx);
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
        s.prepare_vote(tx, ParallelRankId::new(0), original)
            .unwrap();
        s.prepare_vote(tx, ParallelRankId::new(1), FinalizeOutcome::Success)
            .unwrap();
        assert_eq!(
            s.prepare_vote(tx, ParallelRankId::new(0), duplicate),
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
    ready(&mut s, tx);
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
        assert_eq!(
            s.finalize(tx, ParallelRankId::new(rank), FinalizeOutcome::Success),
            Err(DistributedTransactionError::FinalizeBeforeDecision)
        );
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
    assert_eq!(s.pending_ranks(tx).unwrap().len(), 2);
    assert_eq!(
        s.publish(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    post_acked(&mut s, tx);
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
    ready(&mut s, tx);
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
fn cancellation_keeps_credits_until_late_completion_and_cleanup_acks() {
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
    assert_eq!(s.state(tx), Ok(TransactionState::Decided(Decision::Abort)));
    assert_eq!(s.in_use_credits(), 2);
    assert_eq!(s.cancel(tx), Ok(()));
    assert_eq!(s.retire(tx), Err(DistributedTransactionError::InvalidState));
    assert_eq!(
        s.begin(tx),
        Err(DistributedTransactionError::InvalidTransaction)
    );
    assert_eq!(s.drain(), 0);
    assert_eq!(
        s.reserve(id(2), 1),
        Err(DistributedTransactionError::Backpressure)
    );
    s.complete(operation, FinalizeOutcome::Success).unwrap();
    assert_eq!(s.in_use_credits(), 2);
    assert_eq!(s.drain(), 1);
    assert_eq!(s.in_use_credits(), 1);
    assert_eq!(s.drain(), 0);
    assert_eq!(s.retire(tx), Err(DistributedTransactionError::InvalidState));
    post_acked(&mut s, tx);
    s.cancel(tx).unwrap();
    assert_eq!(s.state(tx), Ok(TransactionState::Cancelled));
    assert_eq!(s.in_use_credits(), 0);
    s.reserve(id(2), 1).unwrap();
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
    assert_eq!(s.in_use_credits(), 2);
    assert_eq!(s.drain(), 1);
    assert_eq!(s.in_use_credits(), 1);
    post_acked(&mut s, id(2));
    s.cancel(id(2)).unwrap();
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
    ready(&mut s, tx);
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
    ready(&mut s, tx);
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
        assert_eq!(
            s.abort_decision(tx),
            Err(DistributedTransactionError::InvalidState)
        );
        assert_eq!(s.retire(tx), Err(DistributedTransactionError::InvalidState));
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
        ready(&mut s, tx);
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
    post_acked(&mut s, old);
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
            Err(DistributedTransactionError::DuplicateReport)
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
    // A legitimate late completion releases only the aborting transaction's credit.
    s.complete(old_operation, FinalizeOutcome::Failure).unwrap();
    assert_eq!(s.in_use_credits(), 2);
    assert_eq!(s.drain(), 1);
    assert_eq!(s.in_use_credits(), 1);
    s.cancel(old).unwrap();
    assert_eq!(s.state(old), Ok(TransactionState::Cancelled));
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
    ready(&mut s, current);
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
    cancelled(&mut s, next);
    assert_eq!(s.drain(), 0);
    assert!(s.pending_completions().is_empty());
    assert_eq!(s.in_use_credits(), 0);
}

#[test]
fn abort_can_be_cancelled_or_published_only_after_all_cleanup_acks() {
    for cancel in [false, true] {
        let mut s = DistributedTransaction::new(topology(2, 0), 1);
        let tx = id(30);
        s.begin(tx).unwrap();
        s.reserve(tx, 1).unwrap();
        s.prepare_vote(tx, ParallelRankId::new(0), FinalizeOutcome::Failure)
            .unwrap();
        s.prepare_vote(tx, ParallelRankId::new(1), FinalizeOutcome::Success)
            .unwrap();
        assert_eq!(s.commit_decision(tx), Ok(Decision::Abort));
        assert_eq!(
            s.publish(tx),
            Err(DistributedTransactionError::InvalidState)
        );
        assert_eq!(s.in_use_credits(), 1);
        s.cancel(tx).unwrap();
        assert_eq!(s.state(tx), Ok(TransactionState::Decided(Decision::Abort)));
        assert_eq!(s.retire(tx), Err(DistributedTransactionError::InvalidState));
        s.finalize(tx, ParallelRankId::new(1), FinalizeOutcome::Success)
            .unwrap();
        for _ in 0..2 {
            s.finalize(tx, ParallelRankId::new(0), FinalizeOutcome::Failure)
                .unwrap();
            assert_eq!(s.pending_ranks(tx).unwrap(), vec![ParallelRankId::new(0)]);
            assert_eq!(
                s.publish(tx),
                Err(DistributedTransactionError::InvalidState)
            );
            s.cancel(tx).unwrap();
            assert_eq!(s.state(tx), Ok(TransactionState::Decided(Decision::Abort)));
            assert_eq!(s.retire(tx), Err(DistributedTransactionError::InvalidState));
            assert_eq!(s.in_use_credits(), 1);
            assert_eq!(s.publication_count(), 0);
        }
        s.finalize(tx, ParallelRankId::new(0), FinalizeOutcome::Success)
            .unwrap();
        if cancel {
            s.cancel(tx).unwrap();
            assert_eq!(s.state(tx), Ok(TransactionState::Cancelled));
            assert_eq!(s.publication_count(), 0);
        } else {
            s.publish(tx).unwrap();
            assert_eq!(s.state(tx), Ok(TransactionState::Published));
            assert_eq!(s.publication_count(), 1);
        }
        assert_eq!(s.in_use_credits(), 0);
    }
}

fn mesh(dp: u32, tp: u32) -> ValidatedParallelTopology {
    ValidatedParallelTopology::new(
        ParallelTopologyId::new(1),
        dp.checked_mul(tp).unwrap(),
        ParallelRankId::new(0),
        ParallelismPlan::validated(dp as usize, tp as usize, 1, 1, 1, 1).unwrap(),
    )
    .unwrap()
}

#[test]
fn slow_or_cancelled_replica_does_not_block_other_replica() {
    for cancel_slow in [false, true] {
        let topology = mesh(2, 2);
        let slow_scope = topology.tensor_participants(0).unwrap();
        let fast_scope = topology.tensor_participants(1).unwrap();
        let mut s = DistributedTransaction::new(topology, 5);
        let slow = id(40);
        let fast = id(41);
        s.begin_scoped(slow, slow_scope.clone()).unwrap();
        s.begin_scoped(fast, fast_scope.clone()).unwrap();
        s.reserve(slow, 1).unwrap();
        // The slow replica is not even fully prepared; its operations arrive later.
        s.prepare(slow, ParallelRankId::new(0)).unwrap();
        let slow_operations = s.communicate_all(slow).unwrap();
        for rank in fast_scope.iter() {
            s.prepare(fast, rank).unwrap();
        }
        let fast_operations = s.communicate_all(fast).unwrap();
        assert_eq!(s.in_use_credits(), 5);
        if cancel_slow {
            s.cancel(slow).unwrap();
            assert_eq!(s.in_use_credits(), 5);
        }
        for operation in fast_operations.into_iter().rev() {
            s.complete(operation, FinalizeOutcome::Success).unwrap();
        }
        assert_eq!(s.drain(), 2);
        ready(&mut s, fast);
        assert_eq!(s.commit_decision(fast), Ok(Decision::Commit));
        assert_eq!(
            s.pending_ranks(fast).unwrap(),
            fast_scope.iter().collect::<Vec<_>>()
        );
        for rank in fast_scope.iter() {
            s.finalize(fast, rank, FinalizeOutcome::Success).unwrap();
        }
        s.publish(fast).unwrap();
        assert_eq!(s.publication_count(), 1);
        assert_eq!(s.state(fast), Ok(TransactionState::Published));
        assert_eq!(
            s.pending_ranks(slow).unwrap(),
            slow_scope.iter().collect::<Vec<_>>()
        );
        assert_eq!(
            s.pending_completions()
                .iter()
                .map(|p| p.operation)
                .collect::<Vec<_>>(),
            slow_operations
        );
        assert_eq!(s.in_use_credits(), 3);
        for operation in slow_operations {
            s.complete(operation, FinalizeOutcome::Failure).unwrap();
        }
        assert_eq!(s.drain(), 2);
        if cancel_slow {
            assert_eq!(
                s.state(slow),
                Ok(TransactionState::Decided(Decision::Abort))
            );
            assert_eq!(
                s.commit_decision(slow),
                Err(DistributedTransactionError::InvalidState)
            );
        } else {
            s.prepare(slow, ParallelRankId::new(1)).unwrap();
            ready(&mut s, slow);
            assert_eq!(s.commit_decision(slow), Ok(Decision::Abort));
        }
        post_acked(&mut s, slow);
        s.cancel(slow).unwrap();
        assert_eq!(s.state(slow), Ok(TransactionState::Cancelled));
        assert_eq!(s.state(fast), Ok(TransactionState::Published));
        assert_eq!(s.publication_count(), 1);
        assert_eq!(s.in_use_credits(), 0);
        assert!(s.pending_completions().is_empty());
    }
}

#[test]
fn tensor_scope_strictly_controls_prepare_communication_finalize_and_publication() {
    let topology = mesh(2, 2);
    let scope = topology.tensor_participants(1).unwrap();
    let mut s = DistributedTransaction::new(topology, 2);
    let tx = id(42);
    s.begin_scoped(tx, scope.clone()).unwrap();
    assert_eq!(s.participants(tx), Ok(&scope));
    for outsider in [0, 1, 4, u32::MAX].map(ParallelRankId::new) {
        assert_eq!(
            s.prepare(tx, outsider),
            Err(DistributedTransactionError::InvalidRank)
        );
        assert_eq!(
            s.communicate(tx, outsider),
            Err(DistributedTransactionError::InvalidRank)
        );
        assert_eq!(
            s.finalize(tx, outsider, FinalizeOutcome::Failure),
            Err(DistributedTransactionError::InvalidRank)
        );
    }
    assert_eq!(s.in_use_credits(), 0);
    assert!(s.pending_completions().is_empty());
    assert_eq!(
        s.pending_ranks(tx).unwrap(),
        scope.iter().collect::<Vec<_>>()
    );
    s.prepare(tx, ParallelRankId::new(2)).unwrap();
    assert_eq!(
        s.commit_decision(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    s.prepare(tx, ParallelRankId::new(3)).unwrap();
    let operations = s.communicate_all(tx).unwrap();
    assert_eq!(
        operations.iter().map(|op| op.rank).collect::<Vec<_>>(),
        scope.iter().collect::<Vec<_>>()
    );
    assert_eq!(
        s.commit_decision(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    for operation in operations {
        s.complete(operation, FinalizeOutcome::Success).unwrap();
    }
    assert_eq!(s.drain(), 2);
    ready(&mut s, tx);
    assert_eq!(s.commit_decision(tx), Ok(Decision::Commit));
    s.finalize(tx, ParallelRankId::new(2), FinalizeOutcome::Success)
        .unwrap();
    s.finalize(tx, ParallelRankId::new(3), FinalizeOutcome::Failure)
        .unwrap();
    assert_eq!(s.pending_ranks(tx).unwrap(), vec![ParallelRankId::new(3)]);
    assert_eq!(
        s.publish(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    s.finalize(tx, ParallelRankId::new(3), FinalizeOutcome::Success)
        .unwrap();
    assert!(s.pending_ranks(tx).unwrap().is_empty());
    s.publish(tx).unwrap();
    assert_eq!(s.publication_count(), 1);
}

#[test]
fn scoped_abort_publication_only_waits_for_its_own_cleanup_acks() {
    let topology = mesh(2, 2);
    let mut s = DistributedTransaction::new(topology.clone(), 0);
    let tx = id(43);
    s.begin_scoped(tx, topology.tensor_participants(0).unwrap())
        .unwrap();
    s.prepare_vote(tx, ParallelRankId::new(0), FinalizeOutcome::Failure)
        .unwrap();
    s.prepare_vote(tx, ParallelRankId::new(1), FinalizeOutcome::Success)
        .unwrap();
    assert_eq!(s.commit_decision(tx), Ok(Decision::Abort));
    assert_eq!(
        s.pending_ranks(tx).unwrap(),
        vec![ParallelRankId::new(0), ParallelRankId::new(1)]
    );
    s.finalize(tx, ParallelRankId::new(0), FinalizeOutcome::Success)
        .unwrap();
    assert_eq!(s.pending_ranks(tx).unwrap(), vec![ParallelRankId::new(1)]);
    assert_eq!(
        s.publish(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    s.finalize(tx, ParallelRankId::new(1), FinalizeOutcome::Success)
        .unwrap();
    s.publish(tx).unwrap();
    assert_eq!(s.publication_count(), 1);
}

#[test]
fn invalid_scopes_are_rejected_without_consuming_transaction_ids() {
    use ferrule_common::{ParallelTopologyError, ParticipantSet};

    let topology = mesh(2, 2);
    assert_eq!(
        ParticipantSet::new(&topology, []),
        Err(ParallelTopologyError::EmptyParticipants)
    );
    assert_eq!(
        ParticipantSet::new(&topology, [1, 0, 1].map(ParallelRankId::new)),
        Err(ParallelTopologyError::DuplicateParticipant)
    );
    assert_eq!(
        ParticipantSet::new(&topology, [ParallelRankId::new(4)]),
        Err(ParallelTopologyError::RankOutOfBounds)
    );
    let mut s = DistributedTransaction::new(topology.clone(), 2);
    let tx = id(44);
    for foreign_world in [mesh(1, 2), mesh(1, 8)] {
        assert_eq!(
            s.begin_scoped(tx, foreign_world.participants()),
            Err(DistributedTransactionError::InvalidScope)
        );
        assert_eq!(
            s.state(tx),
            Err(DistributedTransactionError::InvalidTransaction)
        );
        assert_eq!(
            s.participants(tx),
            Err(DistributedTransactionError::InvalidTransaction)
        );
        assert_eq!(s.in_use_credits(), 0);
        assert!(s.pending_completions().is_empty());
    }
    let scope = topology.tensor_participants(1).unwrap();
    s.begin_scoped(tx, scope.clone()).unwrap();
    assert_eq!(
        s.begin_scoped(tx, topology.tensor_participants(0).unwrap()),
        Err(DistributedTransactionError::InvalidTransaction)
    );
    assert_eq!(s.participants(tx), Ok(&scope));
    s.cancel(tx).unwrap();
    assert_eq!(
        s.begin_scoped(tx, scope),
        Err(DistributedTransactionError::InvalidTransaction)
    );
}

#[test]
fn batch_credit_backpressure_is_atomic_and_does_not_skip_identities() {
    let topology = mesh(2, 2);
    let mut s = DistributedTransaction::new(topology.clone(), 3);
    let tx = id(45);
    let other = id(46);
    s.begin_scoped(tx, topology.tensor_participants(0).unwrap())
        .unwrap();
    s.begin_scoped(other, topology.tensor_participants(1).unwrap())
        .unwrap();
    s.reserve(other, 1).unwrap();
    let existing = s.communicate(other, ParallelRankId::new(2)).unwrap();
    let pending = s.pending_completions();
    // One free credit is not enough for a two-rank batch. Repeated retries are inert.
    for _ in 0..3 {
        assert_eq!(
            s.communicate_all(tx),
            Err(DistributedTransactionError::Backpressure)
        );
        assert_eq!(s.pending_completions(), pending);
        assert_eq!(s.in_use_credits(), 2);
        assert_eq!(s.state(tx), Ok(TransactionState::Preparing));
        assert_eq!(
            s.pending_ranks(tx).unwrap(),
            vec![ParallelRankId::new(0), ParallelRankId::new(1)]
        );
    }
    s.release(other, 1).unwrap();
    let operations = s.communicate_all(tx).unwrap();
    assert_eq!(operations.len(), 2);
    assert_eq!(operations[0].identity, existing.identity + 1);
    assert_eq!(operations[1].identity, existing.identity + 2);
    assert_eq!(s.in_use_credits(), 3);
    let pending = s.pending_completions();
    assert_eq!(
        s.communicate_all(tx),
        Err(DistributedTransactionError::Backpressure)
    );
    assert_eq!(s.pending_completions(), pending);
    for operation in &operations {
        assert_eq!(
            s.release(tx, 1),
            Err(DistributedTransactionError::InvalidState)
        );
        s.complete(*operation, FinalizeOutcome::Success).unwrap();
    }
    assert_eq!(s.in_use_credits(), 3);
    assert_eq!(s.drain(), 2);
    assert_eq!(s.in_use_credits(), 1);
    let retry = s.communicate_all(tx).unwrap();
    assert_eq!(retry[0].identity, existing.identity + 3);
    assert_eq!(retry[1].identity, existing.identity + 4);
    assert_eq!(s.in_use_credits(), 3);
}

#[test]
fn batch_rejects_zero_capacity_unknown_and_terminal_transactions_without_operations() {
    let topology = mesh(2, 2);
    let tx = id(47);
    let mut s = DistributedTransaction::new(topology.clone(), 0);
    assert_eq!(
        s.communicate_all(tx),
        Err(DistributedTransactionError::InvalidTransaction)
    );
    s.begin_scoped(tx, topology.tensor_participants(0).unwrap())
        .unwrap();
    assert_eq!(
        s.communicate_all(tx),
        Err(DistributedTransactionError::Backpressure)
    );
    s.cancel(tx).unwrap();
    assert_eq!(
        s.communicate_all(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    assert!(s.pending_completions().is_empty());
    assert_eq!(s.in_use_credits(), 0);

    let mut s = DistributedTransaction::new(topology.clone(), 4);
    s.begin_scoped(tx, topology.tensor_participants(1).unwrap())
        .unwrap();
    for rank in s.participants(tx).unwrap().iter() {
        s.prepare(tx, rank).unwrap();
    }
    ready(&mut s, tx);
    s.commit_decision(tx).unwrap();
    post_acked(&mut s, tx);
    assert_eq!(
        s.communicate_all(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    s.publish(tx).unwrap();
    assert_eq!(
        s.communicate_all(tx),
        Err(DistributedTransactionError::InvalidState)
    );
    assert!(s.pending_completions().is_empty());
    assert_eq!(s.in_use_credits(), 0);
    s.begin(id(48)).unwrap();
    assert_eq!(s.communicate_all(id(48)).unwrap()[0].identity, 1);
}

#[test]
fn default_begin_and_batch_cover_all_ranks_for_dp_tp_meshes() {
    for (dp, tp) in [(2, 2), (1, 8)] {
        let topology = mesh(dp, tp);
        let all = topology.participants();
        let mut s = DistributedTransaction::new(topology, all.len());
        let tx = id(49);
        s.begin(tx).unwrap();
        assert_eq!(s.participants(tx), Ok(&all));
        ready(&mut s, tx);
        let operations = s.communicate_all(tx).unwrap();
        assert_eq!(
            operations.iter().map(|op| op.rank).collect::<Vec<_>>(),
            all.iter().collect::<Vec<_>>()
        );
        assert_eq!(s.in_use_credits(), all.len());
        for operation in operations {
            s.complete(operation, FinalizeOutcome::Success).unwrap();
        }
        assert_eq!(s.drain(), all.len());
        assert_eq!(s.commit_decision(tx), Ok(Decision::Commit));
        post_acked(&mut s, tx);
        s.publish(tx).unwrap();
        assert_eq!(s.publication_count(), 1);
        assert_eq!(s.in_use_credits(), 0);
    }
}

#[test]
fn scoped_begin_checks_epoch_and_mesh_without_consuming_ids_and_ignores_local_rank() {
    let topology = mesh(2, 2);
    let mut s = DistributedTransaction::new(topology.clone(), 4);
    for (index, epoch, plan) in [
        (1, 2, topology.plan()),
        (2, 1, ParallelismPlan::validated(1, 4, 1, 1, 1, 1).unwrap()),
        (3, 1, ParallelismPlan::validated(4, 1, 1, 1, 1, 1).unwrap()),
    ] {
        let foreign = ValidatedParallelTopology::new(
            ParallelTopologyId::new(epoch),
            4,
            ParallelRankId::new(0),
            plan,
        )
        .unwrap();
        let scope = ferrule_common::ParticipantSet::new(
            &foreign,
            topology.tensor_participants(0).unwrap().iter(),
        )
        .unwrap();
        assert_eq!(
            s.begin_scoped(id(100), scope),
            Err(DistributedTransactionError::InvalidScope)
        );
        assert_eq!(
            s.state(id(100)),
            Err(DistributedTransactionError::InvalidTransaction)
        );
        assert_eq!(s.retained_transaction_count(), (index - 1) as usize);
        assert_eq!(s.retained_operation_count(), 0);
        assert_eq!(s.in_use_credits(), 0);
        // The failed higher ID must not prevent this smaller, valid admission.
        s.begin_scoped(id(index), topology.tensor_participants(0).unwrap())
            .unwrap();
    }
    let peer = ValidatedParallelTopology::new(
        topology.topology_id(),
        4,
        ParallelRankId::new(3),
        topology.plan(),
    )
    .unwrap();
    s.begin_scoped(id(4), peer.tensor_participants(1).unwrap())
        .unwrap();
    s.begin_scoped(id(100), peer.tensor_participants(0).unwrap())
        .unwrap();
    assert_eq!(s.retained_transaction_count(), 5);
}

#[test]
fn metadata_defaults_and_zero_limits_are_explicit() {
    let topology = mesh(1, 2);
    let defaults = DistributedTransaction::new(topology.clone(), 2);
    assert_eq!(
        defaults.limits(),
        DistributedTransactionLimits {
            max_transactions: 1_024,
            max_operations: 65_536,
        }
    );
    let mut s = DistributedTransaction::new_with_limits(
        topology.clone(),
        2,
        DistributedTransactionLimits {
            max_transactions: 0,
            max_operations: 2,
        },
    );
    assert_eq!(
        s.begin(id(1)),
        Err(DistributedTransactionError::TransactionLimitReached)
    );
    assert_eq!(
        s.begin_scoped(id(1), topology.participants()),
        Err(DistributedTransactionError::TransactionLimitReached)
    );
    assert_eq!(s.retained_transaction_count(), 0);
    assert_eq!(s.retained_operation_count(), 0);
    assert_eq!(s.in_use_credits(), 0);
    assert!(s.pending_completions().is_empty());

    for (credits, operations, error) in [
        (2, 0, DistributedTransactionError::OperationLimitReached),
        (0, 2, DistributedTransactionError::Backpressure),
    ] {
        let mut s = DistributedTransaction::new_with_limits(
            topology.clone(),
            credits,
            DistributedTransactionLimits {
                max_transactions: 1,
                max_operations: operations,
            },
        );
        s.begin(id(1)).unwrap();
        for _ in 0..2 {
            assert_eq!(
                s.communicate(id(1), ParallelRankId::new(0)),
                Err(error.clone())
            );
            assert_eq!(s.communicate_all(id(1)), Err(error.clone()));
            assert_eq!(s.retained_transaction_count(), 1);
            assert_eq!(s.retained_operation_count(), 0);
            assert_eq!(s.in_use_credits(), 0);
            assert!(s.pending_completions().is_empty());
        }
        cancelled(&mut s, id(1));
        s.retire(id(1)).unwrap();
        assert_eq!(s.retained_transaction_count(), 0);
    }
}

#[test]
fn transaction_limit_counts_terminal_records_and_failed_begin_does_not_advance_highwater() {
    let topology = mesh(2, 2);
    let mut s = DistributedTransaction::new_with_limits(
        topology.clone(),
        4,
        DistributedTransactionLimits {
            max_transactions: 1,
            max_operations: 4,
        },
    );
    s.begin(id(10)).unwrap();
    assert_eq!(
        s.begin(id(9)),
        Err(DistributedTransactionError::InvalidTransaction)
    );
    assert_eq!(
        s.begin(id(100)),
        Err(DistributedTransactionError::TransactionLimitReached)
    );
    cancelled(&mut s, id(10));
    assert_eq!(
        s.begin_scoped(id(100), topology.tensor_participants(1).unwrap()),
        Err(DistributedTransactionError::TransactionLimitReached)
    );
    assert_eq!(s.retained_transaction_count(), 1);
    assert_eq!(s.retained_operation_count(), 0);
    assert_eq!(s.state(id(10)), Ok(TransactionState::Cancelled));
    s.retire(id(10)).unwrap();
    assert_eq!(s.retained_transaction_count(), 0);
    for old in [9, 10] {
        assert_eq!(
            s.begin(id(old)),
            Err(DistributedTransactionError::InvalidTransaction)
        );
    }
    s.begin_scoped(id(11), topology.tensor_participants(1).unwrap())
        .unwrap();
    cancelled(&mut s, id(11));
    s.retire(id(11)).unwrap();
    s.begin(id(100)).unwrap();
    cancelled(&mut s, id(100));
    s.retire(id(100)).unwrap();
    // Even IDs never admitted before are invalid if below the successful high-water mark.
    assert_eq!(
        s.begin(id(99)),
        Err(DistributedTransactionError::InvalidTransaction)
    );
    assert_eq!(
        s.begin(id(100)),
        Err(DistributedTransactionError::InvalidTransaction)
    );
    s.begin(id(u64::MAX)).unwrap();
    cancelled(&mut s, id(u64::MAX));
    s.retire(id(u64::MAX)).unwrap();
    assert_eq!(
        s.begin(id(u64::MAX)),
        Err(DistributedTransactionError::InvalidTransaction)
    );
    assert_eq!(
        s.begin(id(u64::MAX - 1)),
        Err(DistributedTransactionError::InvalidTransaction)
    );
    assert_eq!(s.retained_transaction_count(), 0);
}

#[test]
fn retained_operation_limits_are_atomic_count_drained_records_and_recover_after_retirement() {
    let topology = mesh(2, 2);
    let mut s = DistributedTransaction::new_with_limits(
        topology.clone(),
        8,
        DistributedTransactionLimits {
            max_transactions: 2,
            max_operations: 3,
        },
    );
    let first = id(1);
    let second = id(2);
    s.begin_scoped(first, topology.tensor_participants(0).unwrap())
        .unwrap();
    s.begin_scoped(second, topology.tensor_participants(1).unwrap())
        .unwrap();
    let first_ops = s.communicate_all(first).unwrap();
    for operation in &first_ops {
        s.complete(*operation, FinalizeOutcome::Success).unwrap();
    }
    assert_eq!(s.drain(), 2);
    assert_eq!(s.retained_operation_count(), 2);
    let pending = s.pending_completions();
    for _ in 0..3 {
        // One metadata slot remains, but the whole two-rank batch must fit.
        assert_eq!(
            s.communicate_all(second),
            Err(DistributedTransactionError::OperationLimitReached)
        );
        assert_eq!(s.pending_completions(), pending);
        assert_eq!(s.retained_operation_count(), 2);
        assert_eq!(s.in_use_credits(), 0);
    }
    let single = s.communicate(second, ParallelRankId::new(2)).unwrap();
    assert_eq!(single.identity, first_ops[1].identity + 1);
    s.complete(single, FinalizeOutcome::Success).unwrap();
    assert_eq!(s.drain(), 1);
    for _ in 0..2 {
        assert_eq!(
            s.communicate(second, ParallelRankId::new(3)),
            Err(DistributedTransactionError::OperationLimitReached)
        );
        assert_eq!(
            s.communicate_all(second),
            Err(DistributedTransactionError::OperationLimitReached)
        );
        assert_eq!(s.retained_operation_count(), 3);
        assert_eq!(s.in_use_credits(), 0);
        assert!(s.pending_completions().is_empty());
    }
    assert_eq!(
        s.complete(first_ops[0], FinalizeOutcome::Failure),
        Err(DistributedTransactionError::DuplicateOperation)
    );
    cancelled(&mut s, first);
    s.retire(first).unwrap();
    assert_eq!(s.retained_operation_count(), 1);
    assert_eq!(s.retained_transaction_count(), 1);
    assert_eq!(
        s.complete(first_ops[0], FinalizeOutcome::Failure),
        Err(DistributedTransactionError::StaleOperation)
    );
    let batch = s.communicate_all(second).unwrap();
    assert_eq!(batch[0].identity, single.identity + 1);
    assert_eq!(batch[1].identity, single.identity + 2);
    assert_eq!(s.retained_operation_count(), 3);
    for operation in batch {
        s.complete(operation, FinalizeOutcome::Success).unwrap();
    }
    assert_eq!(s.drain(), 2);
    for rank in s.participants(second).unwrap().iter() {
        s.prepare(second, rank).unwrap();
    }
    ready(&mut s, second);
    assert_eq!(s.commit_decision(second), Ok(Decision::Commit));
    post_acked(&mut s, second);
    s.publish(second).unwrap();
    s.retire(second).unwrap();
    assert_eq!(s.retained_transaction_count(), 0);
    assert_eq!(s.retained_operation_count(), 0);
}

#[test]
fn retirement_waits_for_drain_and_replay_becomes_stale_only_after_retirement() {
    let mut s = DistributedTransaction::new_with_limits(
        mesh(1, 2),
        3,
        DistributedTransactionLimits {
            max_transactions: 1,
            max_operations: 2,
        },
    );
    let tx = id(1);
    assert_eq!(
        s.retire(tx),
        Err(DistributedTransactionError::InvalidTransaction)
    );
    s.begin(tx).unwrap();
    s.reserve(tx, 1).unwrap();
    assert_eq!(s.retire(tx), Err(DistributedTransactionError::InvalidState));
    assert_eq!(s.in_use_credits(), 1);
    let operations = s.communicate_all(tx).unwrap();
    s.cancel(tx).unwrap();
    assert_eq!(s.in_use_credits(), 3);
    assert_eq!(s.retire(tx), Err(DistributedTransactionError::InvalidState));
    s.complete(operations[1], FinalizeOutcome::Failure).unwrap();
    assert_eq!(s.drain(), 1);
    assert_eq!(s.retire(tx), Err(DistributedTransactionError::InvalidState));
    s.complete(operations[0], FinalizeOutcome::Success).unwrap();
    let pending = s.pending_completions();
    assert_eq!(s.retire(tx), Err(DistributedTransactionError::InvalidState));
    assert_eq!(s.pending_completions(), pending);
    assert_eq!(s.retained_transaction_count(), 1);
    assert_eq!(s.retained_operation_count(), 2);
    assert_eq!(s.in_use_credits(), 2);
    assert_eq!(
        s.complete(operations[0], FinalizeOutcome::Failure),
        Err(DistributedTransactionError::DuplicateOperation)
    );
    assert_eq!(s.drain(), 1);
    assert_eq!(s.retire(tx), Err(DistributedTransactionError::InvalidState));
    assert_eq!(s.state(tx), Ok(TransactionState::Decided(Decision::Abort)));
    post_acked(&mut s, tx);
    s.cancel(tx).unwrap();
    s.retire(tx).unwrap();
    assert_eq!(s.retained_transaction_count(), 0);
    assert_eq!(s.retained_operation_count(), 0);
    assert_eq!(s.in_use_credits(), 0);
    assert_eq!(
        s.state(tx),
        Err(DistributedTransactionError::InvalidTransaction)
    );
    assert_eq!(
        s.pending_ranks(tx),
        Err(DistributedTransactionError::InvalidTransaction)
    );
    assert_eq!(
        s.retire(tx),
        Err(DistributedTransactionError::InvalidTransaction)
    );
    assert_eq!(
        s.begin(tx),
        Err(DistributedTransactionError::InvalidTransaction)
    );
    for operation in &operations {
        assert_eq!(
            s.complete(*operation, FinalizeOutcome::Success),
            Err(DistributedTransactionError::StaleOperation)
        );
    }
    s.begin(id(2)).unwrap();
    let next = s.communicate_all(id(2)).unwrap();
    assert_eq!(next[0].identity, operations[1].identity + 1);
    assert_eq!(
        s.complete(
            CommunicationOperation {
                transaction: id(2),
                ..operations[0]
            },
            FinalizeOutcome::Failure
        ),
        Err(DistributedTransactionError::StaleOperation)
    );
    assert_eq!(s.in_use_credits(), 2);
}

#[test]
fn retirement_cannot_discard_drained_failure_before_decision_or_unpublished_decision() {
    for (outcome, decision) in [
        (FinalizeOutcome::Success, Decision::Commit),
        (FinalizeOutcome::Failure, Decision::Abort),
    ] {
        let mut s = DistributedTransaction::new_with_limits(
            mesh(1, 2),
            3,
            DistributedTransactionLimits {
                max_transactions: 1,
                max_operations: 2,
            },
        );
        let tx = id(1);
        s.begin(tx).unwrap();
        s.reserve(tx, 1).unwrap();
        ready(&mut s, tx);
        let operations = s.communicate_all(tx).unwrap();
        for operation in operations {
            s.complete(operation, outcome).unwrap();
        }
        assert_eq!(s.drain(), 2);
        assert_eq!(s.retire(tx), Err(DistributedTransactionError::InvalidState));
        assert_eq!(s.state(tx), Ok(TransactionState::Preparing));
        assert_eq!(s.retained_operation_count(), 2);
        assert_eq!(s.commit_decision(tx), Ok(decision));
        assert_eq!(s.retire(tx), Err(DistributedTransactionError::InvalidState));
        assert_eq!(s.state(tx), Ok(TransactionState::Decided(decision)));
        assert_eq!(s.in_use_credits(), 1);
        for rank in s.participants(tx).unwrap().iter() {
            s.finalize(tx, rank, FinalizeOutcome::Success).unwrap();
        }
        assert_eq!(s.retire(tx), Err(DistributedTransactionError::InvalidState));
        s.publish(tx).unwrap();
        assert_eq!(s.in_use_credits(), 0);
        s.retire(tx).unwrap();
        assert_eq!(s.retained_transaction_count(), 0);
        assert_eq!(s.retained_operation_count(), 0);
        assert_eq!(s.publication_count(), 1);
    }
}

#[test]
fn dp_out_of_order_completion_and_retirement_preserve_other_scope_metadata() {
    let topology = mesh(2, 2);
    let mut s = DistributedTransaction::new_with_limits(
        topology.clone(),
        4,
        DistributedTransactionLimits {
            max_transactions: 2,
            max_operations: 4,
        },
    );
    let slow = id(1);
    let fast = id(2);
    s.begin_scoped(slow, topology.tensor_participants(0).unwrap())
        .unwrap();
    s.begin_scoped(fast, topology.tensor_participants(1).unwrap())
        .unwrap();
    let slow_ops = s.communicate_all(slow).unwrap();
    let fast_ops = s.communicate_all(fast).unwrap();
    s.complete(slow_ops[1], FinalizeOutcome::Failure).unwrap();
    s.complete(fast_ops[1], FinalizeOutcome::Success).unwrap();
    assert_eq!(s.drain(), 2);
    s.complete(fast_ops[0], FinalizeOutcome::Success).unwrap();
    assert_eq!(s.drain(), 1);
    for rank in s.participants(fast).unwrap().iter() {
        s.prepare(fast, rank).unwrap();
    }
    ready(&mut s, fast);
    assert_eq!(s.commit_decision(fast), Ok(Decision::Commit));
    post_acked(&mut s, fast);
    s.publish(fast).unwrap();
    s.retire(fast).unwrap();
    assert_eq!(s.retained_transaction_count(), 1);
    assert_eq!(s.retained_operation_count(), 2);
    assert_eq!(s.in_use_credits(), 1);
    assert_eq!(s.state(slow), Ok(TransactionState::Preparing));
    assert_eq!(
        s.pending_completions(),
        vec![PendingCompletion {
            operation: slow_ops[0],
            outcome: None
        }]
    );

    let next = id(3);
    s.begin_scoped(next, topology.tensor_participants(1).unwrap())
        .unwrap();
    let next_ops = s.communicate_all(next).unwrap();
    assert_eq!(next_ops[0].identity, fast_ops[1].identity + 1);
    s.cancel(slow).unwrap();
    assert_eq!(
        s.retire(slow),
        Err(DistributedTransactionError::InvalidState)
    );
    for operation in next_ops.into_iter().rev() {
        s.complete(operation, FinalizeOutcome::Success).unwrap();
    }
    assert_eq!(s.drain(), 2);
    for rank in s.participants(next).unwrap().iter() {
        s.prepare(next, rank).unwrap();
    }
    ready(&mut s, next);
    assert_eq!(s.commit_decision(next), Ok(Decision::Commit));
    post_acked(&mut s, next);
    s.publish(next).unwrap();
    s.retire(next).unwrap();
    assert_eq!(s.publication_count(), 2);
    assert_eq!(s.retained_operation_count(), 2);
    assert_eq!(s.in_use_credits(), 1);
    assert_eq!(
        s.complete(fast_ops[0], FinalizeOutcome::Failure),
        Err(DistributedTransactionError::StaleOperation)
    );
    assert_eq!(
        s.complete(slow_ops[1], FinalizeOutcome::Success),
        Err(DistributedTransactionError::DuplicateOperation)
    );
    s.complete(slow_ops[0], FinalizeOutcome::Success).unwrap();
    assert_eq!(s.drain(), 1);
    assert_eq!(
        s.retire(slow),
        Err(DistributedTransactionError::InvalidState)
    );
    post_acked(&mut s, slow);
    s.cancel(slow).unwrap();
    s.retire(slow).unwrap();
    assert_eq!(s.retained_transaction_count(), 0);
    assert_eq!(s.retained_operation_count(), 0);
    assert_eq!(s.in_use_credits(), 0);
    assert_eq!(
        s.begin(fast),
        Err(DistributedTransactionError::InvalidTransaction)
    );
}

#[test]
fn repeated_transactions_keep_retained_metadata_bounded_without_reusing_operation_ids() {
    let topology = mesh(2, 2);
    let mut s = DistributedTransaction::new_with_limits(
        topology.clone(),
        2,
        DistributedTransactionLimits {
            max_transactions: 1,
            max_operations: 2,
        },
    );
    let mut last_identity = 0;
    for sequence in 1..=2_048 {
        let tx = id(sequence);
        s.begin_scoped(
            tx,
            topology.tensor_participants((sequence % 2) as u32).unwrap(),
        )
        .unwrap();
        for rank in s.participants(tx).unwrap().iter() {
            s.prepare(tx, rank).unwrap();
        }
        let operations = s.communicate_all(tx).unwrap();
        assert_eq!(operations[0].identity, last_identity + 1);
        last_identity = operations[1].identity;
        assert_eq!(s.retained_transaction_count(), 1);
        assert_eq!(s.retained_operation_count(), 2);
        for operation in operations.iter().rev() {
            s.complete(*operation, FinalizeOutcome::Success).unwrap();
        }
        assert_eq!(s.drain(), 2);
        assert_eq!(s.retained_operation_count(), 2);
        ready(&mut s, tx);
        assert_eq!(s.commit_decision(tx), Ok(Decision::Commit));
        for rank in s.participants(tx).unwrap().iter() {
            s.finalize(tx, rank, FinalizeOutcome::Success).unwrap();
        }
        s.publish(tx).unwrap();
        s.retire(tx).unwrap();
        assert_eq!(s.retained_transaction_count(), 0);
        assert_eq!(s.retained_operation_count(), 0);
        assert_eq!(s.in_use_credits(), 0);
        assert!(s.pending_completions().is_empty());
        assert_eq!(
            s.complete(operations[0], FinalizeOutcome::Failure),
            Err(DistributedTransactionError::StaleOperation)
        );
        assert_eq!(
            s.begin(tx),
            Err(DistributedTransactionError::InvalidTransaction)
        );
    }
    assert_eq!(s.publication_count(), 2_048);
    assert_eq!(last_identity, 4_096);
}
