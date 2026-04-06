use crate::{
    AgentActionKind, AgentStepRecord, AgentTrajectory, EpisodeContext, EpisodeStatus, ToolRegistry,
    parse_action,
};
use ferrule_core::{FerruleError, FerruleResult};

pub async fn run_agent_episode<FGen, FReward>(
    initial_observation: &str,
    max_steps: usize,
    tools: &ToolRegistry,
    mut generate: FGen,
    mut reward_fn: FReward,
) -> FerruleResult<(EpisodeContext, f32)>
where
    FGen: FnMut(&str) -> FerruleResult<String>,
    FReward: FnMut(&EpisodeContext) -> FerruleResult<f32>,
{
    let mut ctx = EpisodeContext::new(max_steps);
    let mut total_reward = 0.0;
    let mut prev_reward = 0.0;
    let mut current_observation = initial_observation.to_string();

    ctx.append(format!("OBS: {current_observation}"));

    while !ctx.is_terminal() {
        if ctx.step_idx >= ctx.max_steps {
            ctx.status = EpisodeStatus::Truncated;
            break;
        }

        let prompt = build_agent_prompt(&ctx, tools, &current_observation);
        let raw_model_text = generate(&prompt)?;
        let parsed = parse_action(&raw_model_text);

        ctx.append(format!("RAW_MODEL_OUTPUT: {}", raw_model_text.trim()));

        match parsed.kind {
            AgentActionKind::CallTool => {
                let tool_name = parsed.tool_name.unwrap_or_default();
                let tool_input = parsed.tool_input.unwrap_or_default();
                ctx.append(format!("TOOL_CALL {}({})", tool_name, tool_input));

                let tool = tools
                    .get(&tool_name)
                    .ok_or_else(|| FerruleError::Runtime(format!("tool not found: {tool_name}")))?;

                let tool_out = tool.call(&tool_input).await?;
                current_observation = tool_out.content.clone();
                ctx.append(format!("TOOL_RESULT: {}", tool_out.content));
                ctx.status = EpisodeStatus::Running;
            }
            AgentActionKind::Finish => {
                let text = parsed.finish_message.unwrap_or_default();
                ctx.append(format!("FINAL: {text}"));
                ctx.status = EpisodeStatus::Finished;
            }
            AgentActionKind::RespondText => {
                let text = parsed.response_text.unwrap_or_default();
                ctx.append(format!("ASSISTANT: {text}"));
                ctx.status = EpisodeStatus::Failed;
            }
            AgentActionKind::Invalid => {
                ctx.append(format!(
                    "INVALID_ACTION: {}",
                    parsed
                        .error
                        .unwrap_or_else(|| "unknown parse error".to_string())
                ));
                ctx.status = EpisodeStatus::Failed;
            }
        }

        let cumulative_reward = reward_fn(&ctx)?;
        let reward_delta = cumulative_reward - prev_reward;
        total_reward += reward_delta;
        prev_reward = cumulative_reward;

        ctx.step_idx += 1;
    }

    Ok((ctx, total_reward))
}

#[derive(Debug, Clone)]
pub struct AgentGeneration {
    pub text: String,
    pub token_ids: Vec<u32>,
}

pub async fn run_agent_episode_with_trace<FGen, FReward>(
    initial_observation: &str,
    max_steps: usize,
    tools: &ToolRegistry,
    mut generate: FGen,
    mut reward_fn: FReward,
) -> FerruleResult<(EpisodeContext, AgentTrajectory)>
where
    FGen: FnMut(&str) -> FerruleResult<AgentGeneration>,
    FReward: FnMut(&EpisodeContext) -> FerruleResult<f32>,
{
    let mut ctx = EpisodeContext::new(max_steps);
    let mut prev_reward = 0.0;
    let mut current_observation = initial_observation.to_string();
    let mut traj = AgentTrajectory {
        initial_observation: initial_observation.to_string(),
        steps: Vec::new(),
        total_reward: 0.0,
        finished: false,
    };

    ctx.append(format!("OBS: {current_observation}"));

    while !ctx.is_terminal() {
        if ctx.step_idx >= ctx.max_steps {
            ctx.status = EpisodeStatus::Truncated;
            break;
        }

        let prompt = build_agent_prompt(&ctx, tools, &current_observation);
        let agent_gen = generate(&prompt)?;
        let parsed = parse_action(&agent_gen.text);

        ctx.append(format!("RAW_MODEL_OUTPUT: {}", agent_gen.text.trim()));

        match parsed.kind {
            AgentActionKind::CallTool => {
                let tool_name = parsed.tool_name.unwrap_or_default();
                let tool_input = parsed.tool_input.unwrap_or_default();
                ctx.append(format!("TOOL_CALL {}({})", tool_name, tool_input));

                let tool = tools
                    .get(&tool_name)
                    .ok_or_else(|| FerruleError::Runtime(format!("tool not found: {tool_name}")))?;

                let tool_out = tool.call(&tool_input).await?;
                current_observation = tool_out.content.clone();
                ctx.append(format!("TOOL_RESULT: {}", tool_out.content));
                ctx.status = EpisodeStatus::Running;
            }
            AgentActionKind::Finish => {
                let text = parsed.finish_message.unwrap_or_default();
                ctx.append(format!("FINAL: {text}"));
                ctx.status = EpisodeStatus::Finished;
            }
            AgentActionKind::RespondText => {
                let text = parsed.response_text.unwrap_or_default();
                ctx.append(format!("ASSISTANT: {text}"));
                ctx.status = EpisodeStatus::Failed;
            }
            AgentActionKind::Invalid => {
                ctx.append(format!(
                    "INVALID_ACTION: {}",
                    parsed
                        .error
                        .unwrap_or_else(|| "unknown parse error".to_string())
                ));
                ctx.status = EpisodeStatus::Failed;
            }
        }

        let cumulative_reward = reward_fn(&ctx)?;
        let reward_delta = cumulative_reward - prev_reward;
        prev_reward = cumulative_reward;

        traj.steps.push(AgentStepRecord {
            step_idx: ctx.step_idx,
            prompt_text: prompt,
            action_text: agent_gen.text,
            action_token_ids: agent_gen.token_ids,
            reward_delta,
            cumulative_reward,
            done: matches!(
                ctx.status,
                EpisodeStatus::Finished | EpisodeStatus::Failed | EpisodeStatus::Truncated
            ),
        });

        ctx.step_idx += 1;
    }

    traj.total_reward = prev_reward;
    traj.finished = matches!(ctx.status, EpisodeStatus::Finished);

    Ok((ctx, traj))
}

fn build_agent_prompt(
    ctx: &EpisodeContext,
    tools: &ToolRegistry,
    current_observation: &str,
) -> String {
    let mut out = String::new();

    out.push_str("You are an action-taking agent.\n");
    out.push_str("Available tools: ");
    out.push_str(&tools.list_for_prompt());
    out.push_str("\n\n");

    out.push_str("You must output exactly one ACTION block.\n");
    out.push_str("Valid forms are:\n");
    out.push_str("<ACTION>\nTOOL: calc\nINPUT: 2+2\n</ACTION>\n");
    out.push_str("or\n");
    out.push_str("<ACTION>\nFINAL: 4\n</ACTION>\n\n");

    out.push_str("Rules:\n");
    out.push_str("- Do not explain.\n");
    out.push_str("- Do not output anything outside the ACTION block.\n");
    out.push_str("- If computation is needed, use calc.\n");
    out.push_str("- After receiving a tool result, output FINAL.\n\n");

    out.push_str("Example:\n");
    out.push_str("Observation: Please compute 2+2.\n");
    out.push_str("<ACTION>\nTOOL: calc\nINPUT: 2+2\n</ACTION>\n");
    out.push_str("Tool result: 4\n");
    out.push_str("<ACTION>\nFINAL: 4\n</ACTION>\n\n");

    out.push_str("Current observation:\n");
    out.push_str(current_observation);
    out.push_str("\n\nTranscript so far:\n");

    for line in &ctx.transcript {
        out.push_str(line);
        out.push('\n');
    }

    out.push_str("\nNow output exactly one ACTION block.\n");
    out
}
