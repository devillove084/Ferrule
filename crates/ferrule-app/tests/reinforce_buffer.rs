use ferrule_runtime::{BaselineMode, OnPolicyBuffer, SequenceSample, batch_reinforce_stats};

#[test]
fn reinforce_buffer_stats_smoke() {
    let mut buf = OnPolicyBuffer::new();

    buf.push(SequenceSample {
        prompt_ids: vec![1, 2],
        completion_ids: vec![3, 4],
        completion_text: "ok".to_string(),
        token_logprobs: vec![-0.2, -0.3],
        reward: 1.0,
        finish_reason: "done".to_string(),
    });

    buf.push(SequenceSample {
        prompt_ids: vec![1, 2],
        completion_ids: vec![5],
        completion_text: "bad".to_string(),
        token_logprobs: vec![-1.2],
        reward: 0.0,
        finish_reason: "done".to_string(),
    });

    let stats = buf.stats();
    assert_eq!(stats.num_samples, 2);
    assert!(stats.mean_reward > 0.0);

    let train = batch_reinforce_stats(buf.samples(), BaselineMode::LeaveOneOut);
    assert!(train.mean_reward >= 0.0);
}
