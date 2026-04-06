use ferrule_runtime::{EchoTool, ToolRegistry, run_agent_episode};

#[tokio::test]
async fn agent_loop_mock_tool_use_smoke() {
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let mut calls = 0usize;

    let (ctx, total_reward) = run_agent_episode(
        "Please compute 2+2 using the available tool.",
        4,
        &tools,
        |_prompt| {
            let out = match calls {
                0 => "<ACTION>\nTOOL: calc\nINPUT: 2+2\n</ACTION>".to_string(),
                _ => "<ACTION>\nFINAL: 4\n</ACTION>".to_string(),
            };
            calls += 1;
            Ok(out)
        },
        |ctx| {
            let mut reward = 0.0;
            for line in &ctx.transcript {
                if line.starts_with("TOOL_CALL ") {
                    reward += 0.1;
                }
                if let Some(rest) = line.strip_prefix("FINAL: ") {
                    if rest.trim() == "4" {
                        reward += 1.0;
                    }
                }
            }
            Ok(reward)
        },
    )
    .await
    .unwrap();

    assert!(ctx.is_terminal());
    assert!(total_reward >= 1.0);
    assert!(ctx.transcript.iter().any(|x| x.contains("TOOL_CALL")));
    assert!(ctx.transcript.iter().any(|x| x.contains("TOOL_RESULT")));
    assert!(ctx.transcript.iter().any(|x| x.contains("FINAL: 4")));
}
