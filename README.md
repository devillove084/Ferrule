# Ferrule

Ferrule is a minimal Agentic RL training-and-inference framework built on top of Candle.

Its purpose is to validate a real closed loop:

- run a multi-step agent episode
- call tools during execution
- record trajectories
- score action tokens
- compute returns and objectives
- apply a small policy update
- run inference again and observe behavior changes

Ferrule is intentionally small. The current design focuses on making the full loop work end to end, not on maximizing model size, benchmark coverage, or training scale.

## Core idea

Ferrule treats agent behavior as a sequence of structured actions inside an episode.

A model produces action blocks such as:

```text
<ACTION>
TOOL: calc
INPUT: 2+2
</ACTION>

<ACTION>
FINAL: 4
</ACTION>

```

The runtime executes the tool call, appends the result to the transcript, and continues until the agent finishes, fails, or reaches the step limit.

## Architecture

``` mermaid
flowchart LR
    A[Config / CLI] --> B[Ferrule]
    B --> C[Agent Runtime]
    B --> D[Candle Model]
    C --> E[Episode Loop]
    C --> F[Tool Execution]
    C --> G[Trajectory Trace]
    D --> H[Tokenizer]
    D --> I[Llama Backend]
    D --> J[Trainable Policy Bias]
    I --> K[Base Logits]
    J --> L[Bias]
    K --> M[Adjusted Logits]
    L --> M
    M --> E
    G --> N[Scoring / Returns / Objective]
    N --> O[Optimizer Step]
    O --> E
```

## Train-infer Loop

``` mermaid
flowchart TD
    A[Initial Observation] --> B[Generate Action]
    B --> C{Action Type}
    C -->|TOOL| D[Run Tool]
    D --> E[Append Tool Result]
    E --> B
    C -->|FINAL| F[Finish Episode]
    C -->|INVALID| G[Fail Episode]
    F --> H[Build Trajectory]
    G --> H
    H --> I[Score Action Tokens]
    I --> J[Compute Returns / Objective]
    J --> K[Update Policy]
    K --> L[Run Agent Again]
```

## What Ferrule currently proves

Ferrule already demonstrates that:

- a local model can run a structured multi-step agent loop
- tool calls can affect later actions
- trajectories can be collected and scored step by step
- a training update can change later inference behavior

That is the main milestone of the project: the training and inference loop is no longer theoretical.

## Core test

Run the end-to-end agent training step:

``` bash
# bash scripts/download_model.sh HuggingFaceTB/SmolLM2-135M-Instruct models/SmolLM2-135M-Instruct
cargo run --release -p ferrule-app --features metal -- agent-train-step --config configs/agent.toml
```

A successful run should produce:

- objective statistics
- eval_before
- eval_after

The key signal is that eval_after shows improved behavior, such as fewer unnecessary tool calls, fewer steps, or higher reward.
