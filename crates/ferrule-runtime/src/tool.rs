use ferrule_core::{FerruleError, FerruleResult, async_trait};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub success: bool,
    pub content: String,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;

    async fn call(&self, input: &str) -> FerruleResult<ToolResult>;
}

#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        self.tools.insert(tool.name().to_string(), Arc::new(tool));
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn list_for_prompt(&self) -> String {
        let mut names = self.tools.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names.join(", ")
    }
}

pub struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &'static str {
        "echo"
    }

    fn description(&self) -> &'static str {
        "Returns the input text directly."
    }

    async fn call(&self, input: &str) -> FerruleResult<ToolResult> {
        Ok(ToolResult {
            success: true,
            content: format!("echo:{input}"),
        })
    }
}

pub struct CalcTool;

#[async_trait]
impl Tool for CalcTool {
    fn name(&self) -> &'static str {
        "calc"
    }

    fn description(&self) -> &'static str {
        "Evaluates a simple arithmetic expression like 2+2, 10-3, 6/2, 3*4."
    }

    async fn call(&self, input: &str) -> FerruleResult<ToolResult> {
        let value = eval_simple_expr(input)?;
        Ok(ToolResult {
            success: true,
            content: value,
        })
    }
}

fn eval_simple_expr(input: &str) -> FerruleResult<String> {
    let expr = input.replace(' ', "");

    for op in ['+', '-', '*', '/'] {
        if let Some(idx) = expr.find(op) {
            let lhs = expr[..idx]
                .parse::<f64>()
                .map_err(|e| FerruleError::Runtime(format!("invalid lhs in calc expr: {e}")))?;
            let rhs = expr[idx + 1..]
                .parse::<f64>()
                .map_err(|e| FerruleError::Runtime(format!("invalid rhs in calc expr: {e}")))?;

            let out = match op {
                '+' => lhs + rhs,
                '-' => lhs - rhs,
                '*' => lhs * rhs,
                '/' => {
                    if rhs == 0.0 {
                        return Err(FerruleError::Runtime("division by zero".to_string()));
                    }
                    lhs / rhs
                }
                _ => unreachable!(),
            };

            if (out.fract()).abs() < 1e-9 {
                return Ok(format!("{}", out as i64));
            }
            return Ok(format!("{out}"));
        }
    }

    Err(FerruleError::Runtime(
        "unsupported calc expression; expected one binary op among + - * /".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn calc_tool_basic() {
        let tool = CalcTool;
        let out = tool.call("2+2").await.unwrap();
        assert_eq!(out.content, "4");
    }
}
