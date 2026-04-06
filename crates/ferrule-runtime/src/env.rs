use ferrule_core::{AgentAction, Environment, FerruleResult, Observation, StepResult, async_trait};

#[derive(Default)]
pub struct EchoToolEnv;

#[async_trait]
impl Environment for EchoToolEnv {
    async fn reset(&mut self, seed: u64) -> FerruleResult<Observation> {
        Ok(Observation {
            text: format!("seed={seed}; available tools: echo"),
            done: false,
        })
    }

    async fn step(&mut self, action: AgentAction) -> FerruleResult<StepResult> {
        match action {
            AgentAction::CallTool {
                name,
                arguments_json,
            } => Ok(StepResult {
                observation: Observation {
                    text: format!("tool::{name} -> {arguments_json}"),
                    done: false,
                },
                reward: 0.0,
                terminated: false,
                truncated: false,
                info_json: serde_json::json!({
                    "tool_name": name
                }),
            }),
            AgentAction::Finish { final_text } => Ok(StepResult {
                observation: Observation {
                    text: format!("final_text_received={final_text}"),
                    done: true,
                },
                reward: 0.0,
                terminated: true,
                truncated: false,
                info_json: serde_json::json!({}),
            }),
        }
    }
}
