use ferrule_core::{
    AgentAction, Environment, EpisodeId, Event, FerruleResult, ModelOutput, PolicyModel, RewardFn,
    SamplingParams, StopReason, Trajectory,
};
use metrics::{counter, histogram};
use std::time::Instant;

pub async fn run_episode<P, E, R>(
    policy: &P,
    env: &mut E,
    reward_fn: &R,
    prompt_token_ids: &[u32],
    params: SamplingParams,
    max_steps: usize,
    seed: u64,
) -> FerruleResult<Trajectory>
where
    P: PolicyModel,
    E: Environment,
    R: RewardFn<Trajectory>,
{
    counter!("ferrule_episode_started_total").increment(1);

    let mut traj = Trajectory::new(EpisodeId::new());
    let obs = env.reset(seed).await?;
    traj.events.push(Event::Reset {
        observation: obs.text,
    });

    let mut session = policy.new_session(prompt_token_ids, &params).await?;

    for step_idx in 0..max_steps {
        let started = Instant::now();
        let step = policy.step(&mut session).await?;
        histogram!("ferrule_rollout_step_latency_seconds").record(started.elapsed());

        match step.action {
            ModelOutput::Text { content } => {
                traj.events.push(Event::ModelText { content });
            }
            ModelOutput::CallTool {
                name,
                arguments_json,
            } => {
                traj.events.push(Event::ToolCall {
                    name: name.clone(),
                    arguments_json: arguments_json.clone(),
                });

                let env_out = env
                    .step(AgentAction::CallTool {
                        name,
                        arguments_json,
                    })
                    .await?;

                traj.events.push(Event::ToolResult {
                    content: env_out.observation.text.clone(),
                });

                if env_out.terminated {
                    traj.stop_reason = Some(StopReason::EnvTerminated);
                    traj.done = true;
                    break;
                }
            }
            ModelOutput::Finish { reason, final_text } => {
                traj.events.push(Event::Finish {
                    reason: reason.clone(),
                });

                let env_out = env
                    .step(AgentAction::Finish {
                        final_text: final_text.unwrap_or_default(),
                    })
                    .await?;

                if !env_out.observation.text.is_empty() {
                    traj.events.push(Event::ToolResult {
                        content: env_out.observation.text,
                    });
                }

                traj.stop_reason = Some(StopReason::ModelFinish);
                traj.done = true;
                break;
            }
        }

        if step_idx + 1 == max_steps {
            traj.stop_reason = Some(StopReason::MaxSteps);
        }
    }

    let reward = reward_fn.evaluate(&traj).await?;
    traj.total_reward = reward.total_reward;
    traj.reward = Some(reward);

    if traj.done {
        counter!("ferrule_episode_finished_total").increment(1);
    } else {
        counter!("ferrule_episode_truncated_total").increment(1);
    }

    histogram!("ferrule_episode_reward").record(traj.total_reward as f64);

    Ok(traj)
}
