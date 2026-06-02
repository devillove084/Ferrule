Ferrule 架构全览

## 1. 四层架构

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: ferrule-app (CLI 入口)                            │
│  main.rs — 命令分发、训练循环、结果输出                       │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: ferrule-runtime (运行时)                          │
│  Agent 循环 / 工具调用 / Trajectory / REINFORCE 算法         │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: ferrule-candle (模型后端)                          │
│  Llama模型加载 / 推理 / 可训练PolicyBias / Logprobs打分       │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: ferrule-core (核心抽象)                            │
│  Traits / 协议类型 / 配置 / 错误处理 / 可观测性               │
└─────────────────────────────────────────────────────────────┘
```

## 2. 逐层详解

### Layer 1 — ferrule-core：一切都是 trait

这是整个框架的"宪法"，定义了所有组件必须遵守的接口契约。

**核心 Trait（`traits.rs`）**：

| Trait | 方法 | 职责 |
|---|---|---|
| `PolicyModel` | `new_session()` → `step()` | 模型的抽象：创建对话session，每一步生成输出 |
| `Environment` | `reset()` → `step()` | 环境的抽象：重置环境，执行一步 |
| `RewardFn<T>` | `evaluate()` | 奖励函数的抽象：给一个东西打分 |

关键设计：`PolicyModel` 用 **Session 模式**：
```rust
// 1. 创建 session（传入 prompt token ids + 采样参数）
let mut session = policy.new_session(&prompt_ids, &params).await?;

// 2. 循环 step（每一步返回 ModelStep）
loop {
    let step = policy.step(&mut session).await?;
    // 处理 step.action ...
}
```

**协议类型（`protocol.rs`）**：

| 类型 | 含义 |
|---|---|
| `SamplingParams` | 采样参数：温度、top_p、top_k、stop_strings 等 |
| `ModelOutput` | 模型输出枚举：`Text / CallTool / Finish` |
| `ModelStep` | 每步的完整输出 = `ModelOutput` + `TokenUsage` |
| `AgentAction` | Agent 发出的动作：`CallTool` 或 `Finish` |
| `Observation` | 环境返回的观察：文本 + 是否结束 |
| `StepResult` | 环境步结果：观察 + 奖励 + 终止标记 |
| `Trajectory` | 完整轨迹：episode_id + events + 总奖励 + 停止原因 |
| `Event` | 轨迹中的事件：Reset / ModelText / ToolCall / ToolResult / Finish / Error |

**配置系统（`config.rs`）**：
- `AppConfig` 是顶层配置，包含 5 个 section：`[observability]`, `[model]`, `[rollout]`, `[train]`, `[agent]`
- 支持从 `.toml` 文件反序列化

---

### Layer 2 — ferrule-candle：模型如何工作

**核心数据结构**：

```
CandlePolicy  (实现了 PolicyModel trait)
├── RealBackend (enum: 目前只有 Llama)
│   └── LlamaBackend
│       ├── model: Arc<Llama>        ← candle-transformers 的 Llama 模型
│       ├── config: LlamaConfig
│       ├── device
│       └── eos_token_id
├── PolicyBiasHead                    ← 可训练参数！
│   ├── bias: Tensor [vocab_size]    ← 每个 token 的可学习偏置
│   └── varmap: VarMap               ← 管理变量的容器
├── FerruleTokenizer                  ← tokenizers crate 包装
└── ResolvedModelPaths                ← 模型文件路径
```

**PolicyBiasHead 是 RL 训练的核心**：
```
原始 logits ──→ [vocab_size]  加上 bias ──→ 调整后的 logits
bias = [0, 0.3, -0.1, ...]    ← 这是 AdamW 要更新的参数
```

**推理流程**：

```
prompt string
    │
    ▼
tokenizer.encode() ──→ prompt_ids: [u32]
    │
    ▼
LlamaBackend.new_session()  ← 创建 KV Cache
    │
    ▼
LlamaBackend.forward_prefill(session, prompt_ids)  ← 一次性灌入所有 prompt tokens
    │
    ▼  logits [1, vocab_size]
    │
apply_policy_bias()   ← 加上可训练偏置
    │
    ▼
sample_next_token()   ← 按温度/top_p/top_k 采样
    │
    ▼  next_token_id
    │
│ 循环直到 eos / max_tokens / stop_strings │
    │
    ▼
LlamaBackend.forward_decode_one(session, next_token_id)  ← 逐个token解码
```

**Logprobs 打分（scoring.rs / diff_scoring.rs）**：

| 函数 | 用途 |
|---|---|
| `token_logprob_from_logits(logits, token_id)` | 从 logits 计算单个 token 的 log 概率 |
| `score_completion_ids(prompt_ids, completion_ids)` | 用 teacher-forcing 给一段 completion 打分 |
| `sequence_logprob_sum_tensor(logits_per_step, target_ids)` | **可微分版本**，保留计算图用于反向传播 |

两者的区别：
- **普通版**：`log_softmax(logits)[token_id]`，结果拉回 CPU 变成 `f32`，计算图断开
- **可微分版**：返回 `Tensor`，计算图保留，可以 `.backward()`

**当前支持的模型家族（`family/`）**：
- `family/llama.rs` — 完整的 Llama 后端封装

---

### Layer 3 — ferrule-runtime：Agent 如何行动和学习

这是框架最丰富的一层，包含 Agent 执行的完整逻辑和 RL 算法。

**Agent 循环（`agent_loop.rs`）**：

```
┌──────────────────────────────────────────────────┐
│  run_agent_episode_with_trace()                  │
│                                                  │
│  1. 初始化 EpisodeContext (transcript + status)  │
│  2. 循环（直到 finished / failed / truncated）：  │
│     a. 构建 prompt (系统提示 + 工具列表 + 历史)   │
│     b. 调用 generate(prompt) → 模型生成文本       │
│     c. parse_action(text) → 解析 ACTION 块       │
│     d. 如果是 TOOL → 调用工具，收到工具结果       │
│     e. 如果是 FINAL → 标记 finished              │
│     f. 如果是 INVALID → 标记 failed              │
│     g. 调用 reward_fn(ctx) → 计算奖励增量        │
│     h. 记录 AgentStepRecord                      │
│  3. 返回 (EpisodeContext, AgentTrajectory)       │
└──────────────────────────────────────────────────┘
```

**类型层次**：

```
AgentTrajectory                           ← 一条完整轨迹
  ├── initial_observation: String
  ├── steps: Vec<AgentStepRecord>          ← 每一步的记录
  │     ├── step_idx
  │     ├── prompt_text                    ← 该步的 prompt
  │     ├── action_text                    ← 模型生成的文字
  │     ├── action_token_ids               ← 模型生成的 token IDs
  │     ├── reward_delta                   ← 该步获得的增量奖励
  │     └── cumulative_reward              ← 累积奖励
  └── total_reward, finished

    ↓ (打分: score_completion_ids)

ScoredAgentTrajectory                     ← 打分后的轨迹
  └── steps: Vec<ScoredAgentStep>
        ├── token_logprobs: Vec<f32>       ← 每个 action token 的 log 概率
        ├── logprob_sum                    ← logprob 之和
        └── return_to_go                   ← 从这步开始到结束的累积折扣奖励

    ↓ (目标函数计算)

AgentObjectiveStep                        ← REINFORCE 目标
  ├── advantage                           ← return_to_go - baseline
  ├── logprob_sum
  └── objective = advantage * logprob_sum  ← 策略梯度项
```

**REINFORCE 算法（`trajectory_train.rs`）**：

核心公式：

$$\text{objective} = \text{advantage} \times \sum_t \log \pi(a_t | s_t)$$

$$\text{loss} = -\text{objective}$$

其中：
- $\sum_t \log \pi(a_t | s_t)$ = 该 step 所有 action tokens 的 log 概率之和
- $\text{advantage} = \text{return\_to\_go} - \text{baseline}$

Baseline 的作用是**减小方差**，三种模式：

| BaselineMode | 计算方式 |
|---|---|
| `Zero` | baseline = 0 |
| `TrajectoryMean` | baseline = 所有轨迹总奖励的平均 |
| `TrajectoryLeaveOneOut` | baseline = (总奖励 - 当前轨迹奖励) / (轨迹数 - 1) |

**工具系统**：

```
Tool trait:
  - name()      → 工具名
  - description() → 工具描述
  - call(input) → 异步调用工具

ToolRegistry:
  - register(tool)    → 注册工具
  - get(name)         → 获取工具
  - list_for_prompt() → 生成 prompt 中的工具列表

已有的工具：
  - CalcTool:   计算简单算术表达式 (2+2=4)
  - EchoTool:   直接回显输入
```

**Reward 系统**：

在 `main.rs` 里 `agent_reward_from_ctx` 函数：
```rust
reward = -0.05 * tool_calls       // 惩罚过多的工具调用
         - 0.20 * invalid_actions  // 惩罚无效 action
         + 1.00 * correct_final    // 奖励正确答案
```

---

### Layer 4 — ferrule-app：把所有东西串起来

**CLI 命令**：

| 命令 | 功能 |
|---|---|
| `doctor` | 检查模型是否能加载，做 prefill 验证 |
| `generate` | 单次文本生成 |
| `score` | 给 prompt + completion 打分 |
| `rollout` | 跑一个完整的 episode（用 Environment trait） |
| `train` | 简单 RL 训练（生成→打分→REINFORCE 统计，不更新参数） |
| `agent-demo` | 跑 agent 多步循环（支持 mock 和 real 模式） |
| `agent-train-step` | **核心命令**：完整的 RL 训练一步 |

**agent-train-step 的完整闭环**：

```
1. 加载模型，创建 PolicyBiasHead + AdamW optimizer
2. eval_before：跑 4 条轨迹，记录评估指标

3. 收集阶段（Rollout）：
   for episode in 0..num_episodes:
       a. 用 generate_text_once_with_bias() 执行 agent 循环
          （bias 影响采样，使模型倾向于高奖励行为）
       b. 收集 AgentTrajectory（包含 action_token_ids）
       c. 用 score_completion_ids_with_bias() 给每个 step 打分
          （计算带 bias 的 token logprobs）
       d. build_scored_trajectory() → 加入 return_to_go
          └── 依赖结构：
              每个 step 的 logprobs 和 token_ids 必须长度一致

4. 目标计算阶段：
   compute_agent_objectives() → 每条轨迹每条 step 的 objective
   （基于 return_to_go 和 baseline 计算 advantage）

5. 损失计算阶段：
   for each step:
       differentiable_action_logprob_sum() → 可微分 logprob 和
       loss_term = -advantage * logprob_sum
   loss = mean(loss_terms)

6. 参数更新：
   opt.backward_step(&loss)  ← 更新 PolicyBiasHead 的 bias 参数

7. eval_after：再次跑 4 条轨迹，对比行为变化
```

---

## 3. 关键设计模式

### 3.1 Trait 抽象 + Family 分发

```rust
// family 只是字符串，通过 match 分发到具体实现
match family {
    "llama"  => LlamaBackend::load(...),
    "deepseek" => DeepSeekBackend::load(...),  // ← 我们要加的
}
```

### 3.2 普通打分 vs 可微打分

这是 RL 训练的关键区别：

```
score_completion_ids()          → Vec<f32>      (CPU上，用于统计/记录)
differentiable_action_logprob_sum() → Tensor    (GPU上，用于反向传播)
```

两者计算逻辑相同，但后者保留计算图，允许梯度流过 PolicyBiasHead。

### 3.3 Session 模式

模型推理不是一次性完成的，而是通过 session 保持状态（KV Cache），每一步生成一部分输出。

---

## 4. 数据流全景图

```
                     ┌──────────────┐
                     │  Config.toml │
                     └──────┬───────┘
                            │
              ┌─────────────▼─────────────┐
              │     main.rs: run()        │
              │  Command::AgentTrainStep  │
              └─────────────┬─────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       │                    │                    │
       ▼                    ▼                    ▼
┌──────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ CandlePolicy │  │  ToolRegistry   │  │  AgentConfig    │
│ (LLaMA模型)  │  │  (calc工具)     │  │  (采样参数等)    │
└──────┬───────┘  └────────┬────────┘  └────────┬────────┘
       │                   │                    │
       └───────────────────┼────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ run_agent_episode_      │
              │ with_trace()            │
              │                         │
              │ prompt → generate →     │
              │ parse → tool/final →    │
              │ reward → record         │
              └────────────┬────────────┘
                           │
                           ▼ AgentTrajectory
              ┌─────────────────────────┐
              │ score_completion_ids_   │
              │ with_bias()             │
              └────────────┬────────────┘
                           │
                           ▼ ScoredAgentTrajectory
              ┌─────────────────────────┐
              │ compute_agent_          │
              │ objectives()            │
              │ REINFORCE: adv·logprob  │
              └────────────┬────────────┘
                           │
                           ▼ loss Tensor
              ┌─────────────────────────┐
              │ opt.backward_step()     │
              │ 更新 PolicyBias         │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  eval_after()           │
              │  验证行为是否改善       │
              └─────────────────────────┘
```

---

## 5. 对接 DeepSeek V4 需要改什么
