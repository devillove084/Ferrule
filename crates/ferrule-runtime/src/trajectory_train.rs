use crate::trajectory::AgentTrajectory;
use ferrule_core::{FerruleError, FerruleResult};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredAgentStep {
    pub step_idx: usize,
    pub prompt_text: String,
    pub action_text: String,
    pub action_token_ids: Vec<u32>,
    pub token_logprobs: Vec<f32>,
    pub logprob_sum: f32,
    pub reward_delta: f32,
    pub return_to_go: f32,
    pub done: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredAgentTrajectory {
    pub initial_observation: String,
    pub steps: Vec<ScoredAgentStep>,
    pub total_reward: f32,
    pub finished: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentBatchStats {
    pub num_trajectories: usize,
    pub num_steps: usize,
    pub mean_total_reward: f32,
    pub mean_step_return: f32,
    pub mean_step_logprob_sum: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentBaselineMode {
    Zero,
    TrajectoryMean,
    TrajectoryLeaveOneOut,
}

impl AgentBaselineMode {
    pub fn parse(s: &str) -> FerruleResult<Self> {
        match s {
            "zero" => Ok(Self::Zero),
            "trajectory_mean" => Ok(Self::TrajectoryMean),
            "trajectory_leave_one_out" => Ok(Self::TrajectoryLeaveOneOut),
            other => Err(FerruleError::Config(format!(
                "unsupported agent baseline mode: {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentObjectiveStep {
    pub trajectory_idx: usize,
    pub step_idx: usize,
    pub reward_delta: f32,
    pub return_to_go: f32,
    pub baseline: f32,
    pub advantage: f32,
    pub logprob_sum: f32,
    pub objective: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentObjectiveStats {
    pub num_trajectories: usize,
    pub num_steps: usize,
    pub mean_total_reward: f32,
    pub mean_return_to_go: f32,
    pub mean_advantage: f32,
    pub mean_logprob_sum: f32,
    pub mean_objective: f32,
}

pub fn compute_returns_to_go(reward_deltas: &[f32], gamma: f32) -> Vec<f32> {
    let mut out = vec![0.0; reward_deltas.len()];
    let mut running = 0.0;

    for i in (0..reward_deltas.len()).rev() {
        running = reward_deltas[i] + gamma * running;
        out[i] = running;
    }

    out
}

pub fn build_scored_trajectory(
    traj: &AgentTrajectory,
    per_step_logprobs: Vec<Vec<f32>>,
    gamma: f32,
) -> FerruleResult<ScoredAgentTrajectory> {
    if traj.steps.len() != per_step_logprobs.len() {
        return Err(FerruleError::Runtime(format!(
            "trajectory steps len {} != logprob groups len {}",
            traj.steps.len(),
            per_step_logprobs.len()
        )));
    }

    let reward_deltas = traj
        .steps
        .iter()
        .map(|s| s.reward_delta)
        .collect::<Vec<_>>();
    let returns = compute_returns_to_go(&reward_deltas, gamma);

    let mut steps = Vec::with_capacity(traj.steps.len());

    for ((step, lps), ret) in traj
        .steps
        .iter()
        .zip(per_step_logprobs.into_iter())
        .zip(returns.into_iter())
    {
        if step.action_token_ids.len() != lps.len() {
            return Err(FerruleError::Runtime(format!(
                "step {} action_token_ids len {} != token_logprobs len {}",
                step.step_idx,
                step.action_token_ids.len(),
                lps.len()
            )));
        }

        let logprob_sum = lps.iter().copied().sum::<f32>();

        steps.push(ScoredAgentStep {
            step_idx: step.step_idx,
            prompt_text: step.prompt_text.clone(),
            action_text: step.action_text.clone(),
            action_token_ids: step.action_token_ids.clone(),
            token_logprobs: lps,
            logprob_sum,
            reward_delta: step.reward_delta,
            return_to_go: ret,
            done: step.done,
        });
    }

    Ok(ScoredAgentTrajectory {
        initial_observation: traj.initial_observation.clone(),
        steps,
        total_reward: traj.total_reward,
        finished: traj.finished,
    })
}

pub fn agent_batch_stats(trajs: &[ScoredAgentTrajectory]) -> AgentBatchStats {
    if trajs.is_empty() {
        return AgentBatchStats {
            num_trajectories: 0,
            num_steps: 0,
            mean_total_reward: 0.0,
            mean_step_return: 0.0,
            mean_step_logprob_sum: 0.0,
        };
    }

    let num_trajectories = trajs.len();
    let num_steps = trajs.iter().map(|t| t.steps.len()).sum::<usize>();

    if num_steps == 0 {
        return AgentBatchStats {
            num_trajectories,
            num_steps,
            mean_total_reward: trajs.iter().map(|t| t.total_reward).sum::<f32>()
                / num_trajectories as f32,
            mean_step_return: 0.0,
            mean_step_logprob_sum: 0.0,
        };
    }

    let mean_total_reward =
        trajs.iter().map(|t| t.total_reward).sum::<f32>() / num_trajectories as f32;

    let mean_step_return = trajs
        .iter()
        .flat_map(|t| t.steps.iter().map(|s| s.return_to_go))
        .sum::<f32>()
        / num_steps as f32;

    let mean_step_logprob_sum = trajs
        .iter()
        .flat_map(|t| t.steps.iter().map(|s| s.logprob_sum))
        .sum::<f32>()
        / num_steps as f32;

    AgentBatchStats {
        num_trajectories,
        num_steps,
        mean_total_reward,
        mean_step_return,
        mean_step_logprob_sum,
    }
}

pub fn compute_agent_objectives(
    trajs: &[ScoredAgentTrajectory],
    baseline_mode: AgentBaselineMode,
) -> (Vec<AgentObjectiveStep>, AgentObjectiveStats) {
    if trajs.is_empty() {
        return (
            vec![],
            AgentObjectiveStats {
                num_trajectories: 0,
                num_steps: 0,
                mean_total_reward: 0.0,
                mean_return_to_go: 0.0,
                mean_advantage: 0.0,
                mean_logprob_sum: 0.0,
                mean_objective: 0.0,
            },
        );
    }

    let num_trajectories = trajs.len();
    let num_steps = trajs.iter().map(|t| t.steps.len()).sum::<usize>();
    let total_reward_sum = trajs.iter().map(|t| t.total_reward).sum::<f32>();

    let mut rows = Vec::new();

    for (traj_idx, traj) in trajs.iter().enumerate() {
        let baseline = match baseline_mode {
            AgentBaselineMode::Zero => 0.0,
            AgentBaselineMode::TrajectoryMean => total_reward_sum / num_trajectories as f32,
            AgentBaselineMode::TrajectoryLeaveOneOut => {
                if num_trajectories == 1 {
                    0.0
                } else {
                    (total_reward_sum - traj.total_reward) / ((num_trajectories - 1) as f32)
                }
            }
        };

        for step in &traj.steps {
            let advantage = step.return_to_go - baseline;
            let objective = advantage * step.logprob_sum;

            rows.push(AgentObjectiveStep {
                trajectory_idx: traj_idx,
                step_idx: step.step_idx,
                reward_delta: step.reward_delta,
                return_to_go: step.return_to_go,
                baseline,
                advantage,
                logprob_sum: step.logprob_sum,
                objective,
            });
        }
    }

    if rows.is_empty() {
        return (
            rows,
            AgentObjectiveStats {
                num_trajectories,
                num_steps,
                mean_total_reward: total_reward_sum / num_trajectories as f32,
                mean_return_to_go: 0.0,
                mean_advantage: 0.0,
                mean_logprob_sum: 0.0,
                mean_objective: 0.0,
            },
        );
    }

    let n = rows.len() as f32;

    let stats = AgentObjectiveStats {
        num_trajectories,
        num_steps,
        mean_total_reward: total_reward_sum / num_trajectories as f32,
        mean_return_to_go: rows.iter().map(|r| r.return_to_go).sum::<f32>() / n,
        mean_advantage: rows.iter().map(|r| r.advantage).sum::<f32>() / n,
        mean_logprob_sum: rows.iter().map(|r| r.logprob_sum).sum::<f32>() / n,
        mean_objective: rows.iter().map(|r| r.objective).sum::<f32>() / n,
    };

    (rows, stats)
}
