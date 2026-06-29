use clap::{Parser, Subcommand};
use std::io::Write;
use std::path::Path;

#[derive(Parser)]
#[command(name = "ferrule", version = "0.2")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Info {
        model: String,
    },
    Run {
        model: String,
        #[arg(short = 'p', long, default_value = "The capital of France is")]
        prompt: String,
        #[arg(short = 'n', long, default_value = "16")]
        max_tokens: usize,
    },
    /// Verify CUDA and benchmark GEMV.
    Cuda,
    /// Run inference on GPU.
    GpuRun {
        model: String,
        #[arg(short = 'p', long, default_value = "Hello")]
        prompt: String,
        #[arg(short = 'n', long, default_value = "4")]
        max_tokens: usize,
        #[arg(short = 'q', long, default_value = "q4")]
        quant: String,
    },
    /// Interactive chat REPL.
    Chat {
        model: String,
        #[arg(short = 'n', long, default_value = "256")]
        max_tokens: usize,
        #[arg(short = 'q', long, default_value = "q4")]
        quant: String,
    },
    Bench {
        #[arg(long, default_value = "2048")]
        hidden: usize,
        #[arg(long, default_value = "12")]
        layers: usize,
        #[arg(long, default_value = "2")]
        warmup: usize,
        #[arg(long, default_value = "8")]
        iters: usize,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Info { model } => cmd_info(&model),
        Command::Run {
            model,
            prompt,
            max_tokens,
        } => cmd_run(&model, &prompt, max_tokens),
        Command::Cuda => cmd_cuda(),
        Command::GpuRun {
            model,
            prompt,
            max_tokens,
            quant,
        } => cmd_gpu_run(&model, &prompt, max_tokens, &quant),
        Command::Chat {
            model,
            max_tokens,
            quant,
        } => cmd_chat(&model, max_tokens, &quant),
        Command::Bench {
            hidden,
            layers,
            warmup,
            iters,
        } => cmd_bench(hidden, layers, warmup, iters),
    }
}

fn cmd_info(model_dir: &str) -> anyhow::Result<()> {
    let model = ferrule_model::OlmoeModel::load(Path::new(model_dir))?;
    let c = &model.config;
    println!(
        "OLMoE: {}d×{}L, {}e top-{}, vocab={}",
        c.hidden_size, c.num_layers, c.num_experts, c.num_experts_per_tok, c.vocab_size
    );
    Ok(())
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold(
            (0usize, logits[0]),
            |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
        )
        .0 as u32
}

#[cfg(feature = "cuda")]
fn parse_quant(quant: &str) -> ferrule_quant::QuantType {
    match quant.to_ascii_lowercase().as_str() {
        "q2" | "q2s" => ferrule_quant::QuantType::Q2S,
        "t1" | "t1s" => ferrule_quant::QuantType::T1S,
        "q8" | "q8_0" => ferrule_quant::QuantType::Q8_0,
        _ => ferrule_quant::QuantType::Q4_0,
    }
}

#[derive(Clone, Copy)]
enum ChatFormat {
    OlmoeInstruct,
    Plain,
}

impl ChatFormat {
    fn name(self) -> &'static str {
        match self {
            Self::OlmoeInstruct => "olmoe-instruct",
            Self::Plain => "plain",
        }
    }
}

fn detect_chat_format(model_dir: &str) -> ChatFormat {
    let path = Path::new(model_dir).join("tokenizer_config.json");
    let Ok(text) = std::fs::read_to_string(path) else {
        return ChatFormat::Plain;
    };
    if text.contains("<|user|>") && text.contains("<|assistant|>") {
        ChatFormat::OlmoeInstruct
    } else {
        ChatFormat::Plain
    }
}

fn chat_prompt(format: ChatFormat, turn: &str, first_turn: bool) -> String {
    match format {
        ChatFormat::OlmoeInstruct => {
            if first_turn {
                format!("<|endoftext|>\n<|user|>\n{turn}\n<|assistant|>")
            } else {
                format!("\n<|user|>\n{turn}\n<|assistant|>")
            }
        }
        ChatFormat::Plain => {
            if first_turn {
                format!("User: {turn}\nAssistant:")
            } else {
                format!("\nUser: {turn}\nAssistant:")
            }
        }
    }
}

fn is_eos(model: &ferrule_model::OlmoeModel, token: u32) -> bool {
    model.eos_token_id() == Some(token)
}

fn cmd_run(model_dir: &str, prompt: &str, max_tokens: usize) -> anyhow::Result<()> {
    let model = ferrule_model::OlmoeModel::load(Path::new(model_dir))?;
    let c = &model.config;
    println!(
        "OLMoE: {}d×{}L, {}e top-{}",
        c.hidden_size, c.num_layers, c.num_experts, c.num_experts_per_tok
    );

    let tokens = model.encode(prompt)?;
    println!("Prompt: \"{prompt}\" → {} tokens", tokens.len());

    let nl = c.num_layers;
    let mut k_cache: Vec<Vec<f32>> = (0..nl).map(|_| Vec::new()).collect();
    let mut v_cache: Vec<Vec<f32>> = (0..nl).map(|_| Vec::new()).collect();
    let mut pos = 0usize;

    let t0 = std::time::Instant::now();

    // Prefill: process all prompt tokens, sample first token from last prefill output
    let mut gen = Vec::new();
    let mut logits = Vec::new();
    for &tid in &tokens {
        let (_, l) = model.forward(&[tid], &mut k_cache, &mut v_cache, pos)?;
        logits = l;
        pos += 1;
    }
    // First generated token comes from prefill output, not by re-feeding prompt
    let first = argmax(&logits);
    if !is_eos(&model, first) {
        gen.push(first);
        let txt = model.decode(&[first])?;
        println!(
            "  [{:>3}] {txt:30} {:.0}ms",
            0,
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }

    // Generate remaining tokens
    for step in 1..max_tokens {
        if gen.is_empty() {
            break;
        }
        let last = *gen.last().unwrap_or(&0);
        let (_, l) = model.forward(&[last], &mut k_cache, &mut v_cache, pos)?;
        logits = l;
        pos += 1;
        let next = argmax(&logits);
        if is_eos(&model, next) {
            break;
        }
        gen.push(next);
        let txt = model.decode(&[next])?;
        println!(
            "  [{step:>3}] {txt:30} {:.0}ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }

    let full = model.decode(&gen)?;
    let t = t0.elapsed();
    println!("\n{full}");
    println!(
        "{:.1}s, {:.1} tok/s",
        t.as_secs_f64(),
        max_tokens as f64 / t.as_secs_f64()
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn cmd_cuda() -> anyhow::Result<()> {
    println!("=== CUDA Probe ===");
    ferrule_cuda::forward::cuda_probe()?;

    println!("\n=== GEMV Benchmark (2048×2048) ===");
    let d = 2048usize;
    let x: Vec<f32> = (0..d).map(|i| (i as f32).sin()).collect();
    let w: Vec<f32> = (0..d * d).map(|i| (i as f32).cos()).collect();

    // Proper benchmark: create context once
    use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
    let ctx = CudaContext::new(0).map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    ctx.bind_to_thread()
        .map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    let module =
        ferrule_cuda::kernels::kernels::load(&ctx).map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    let s = ctx.default_stream();
    let xd = DeviceBuffer::from_host(&s, &x).map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    let wd = DeviceBuffer::from_host(&s, &w).map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    let mut yd = DeviceBuffer::<f32>::zeroed(&s, d).map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;

    // Warmup
    for _ in 0..10 {
        module
            .gemv_f32(
                &s,
                LaunchConfig::for_num_elems(d as u32),
                &xd,
                &wd,
                &mut yd,
                d as u32,
            )
            .map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    }

    let t0 = std::time::Instant::now();
    let n_iter = 2000;
    for _ in 0..n_iter {
        module
            .gemv_f32(
                &s,
                LaunchConfig::for_num_elems(d as u32),
                &xd,
                &wd,
                &mut yd,
                d as u32,
            )
            .map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    }
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iter as f64;

    // Measure empty kernel overhead: launch compute_rms with 1 thread, 1 element
    let mut rms_buf =
        DeviceBuffer::<f32>::zeroed(&s, 1).map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    let mut dummy =
        DeviceBuffer::<f32>::zeroed(&s, 1).map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    let t0 = std::time::Instant::now();
    let n_empty = 5000;
    for _ in 0..n_empty {
        module
            .compute_rms(
                &s,
                LaunchConfig::for_num_elems(1u32),
                &dummy,
                &mut rms_buf,
                1u32,
                1e-5f32,
            )
            .map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    }
    let empty_us = t0.elapsed().as_secs_f64() * 1e6 / n_empty as f64;

    // Measure compute_rms with real size (d=2048, 1 thread)
    let mut hidden_buf =
        DeviceBuffer::<f32>::zeroed(&s, d).map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    let t0 = std::time::Instant::now();
    let n_rms = 1000;
    for _ in 0..n_rms {
        module
            .compute_rms(
                &s,
                LaunchConfig::for_num_elems(1u32),
                &hidden_buf,
                &mut rms_buf,
                d as u32,
                1e-5f32,
            )
            .map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    }
    let rms_us = t0.elapsed().as_secs_f64() * 1e6 / n_rms as f64;

    // CPU comparison
    let t0 = std::time::Instant::now();
    for _ in 0..n_iter {
        let mut out = vec![0f32; d];
        for j in 0..d {
            let row = &w[j * d..(j + 1) * d];
            out[j] = row.iter().zip(x.iter()).map(|(r, xi)| r * xi).sum();
        }
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iter as f64;

    println!("  CPU: {cpu_ms:.2} ms");
    println!("  GPU GEMV (kernel only): {gpu_ms:.3} ms");
    println!("  Kernel launch overhead: {empty_us:.0} µs");
    println!("  compute_rms(d=2048): {rms_us:.0} µs");
    println!("  Speedup: {:.0}x", cpu_ms / gpu_ms);
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn cmd_cuda() -> anyhow::Result<()> {
    println!("CUDA not available. Rebuild with: cargo build --release --features cuda");
    Ok(())
}

fn cmd_chat_cpu(model_dir: &str, max_tokens: usize) -> anyhow::Result<()> {
    use console::style;
    use rustyline::error::ReadlineError;

    let model = ferrule_model::OlmoeModel::load(Path::new(model_dir))?;
    let c = &model.config;
    println!(
        "OLMoE: {}d×{}L, {}e top-{} (CPU FP32)",
        c.hidden_size, c.num_layers, c.num_experts, c.num_experts_per_tok
    );
    let chat_format = detect_chat_format(model_dir);
    println!(
        "{} Type /exit or Ctrl-D to quit. Template: {}.",
        style("Chat ready.").cyan(),
        chat_format.name()
    );

    let nl = c.num_layers;
    let mut k_cache: Vec<Vec<f32>> = (0..nl).map(|_| Vec::new()).collect();
    let mut v_cache: Vec<Vec<f32>> = (0..nl).map(|_| Vec::new()).collect();
    let mut pos = 0usize;
    let mut first_turn = true;
    let mut rl = rustyline::DefaultEditor::new()?;

    loop {
        let line = match rl.readline(&format!("{} ", style("You>").green().bold())) {
            Ok(line) => line,
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
            Err(err) => return Err(err.into()),
        };
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if matches!(input, "/exit" | "/quit") {
            break;
        }
        let _ = rl.add_history_entry(input);

        let prompt = chat_prompt(chat_format, input, first_turn);
        first_turn = false;
        let tokens = model.encode(&prompt)?;
        let mut logits = Vec::new();
        for tid in tokens {
            let (_, l) = model.forward(&[tid], &mut k_cache, &mut v_cache, pos)?;
            logits = l;
            pos += 1;
        }
        if logits.is_empty() {
            continue;
        }

        print!("{} ", style("Ferrule>").cyan().bold());
        std::io::stdout().flush()?;
        for _ in 0..max_tokens {
            let next = argmax(&logits);
            if is_eos(&model, next) {
                let _ = model.forward(&[next], &mut k_cache, &mut v_cache, pos)?;
                pos += 1;
                break;
            }
            let txt = model.decode(&[next])?;
            if txt.is_empty() {
                break;
            }
            print!("{txt}");
            std::io::stdout().flush()?;
            let (_, l) = model.forward(&[next], &mut k_cache, &mut v_cache, pos)?;
            logits = l;
            pos += 1;
        }
        println!();
    }

    Ok(())
}

fn cmd_bench(hidden: usize, layers: usize, warmup: usize, iters: usize) -> anyhow::Result<()> {
    use ferrule_graph::{CGraph, CpuBackend, OpKind, Shape};
    let d = hidden;
    let mut g = CGraph::new();
    let w = vec![1f32; d * d];
    let n = vec![1f32; d];
    let mut wids = vec![];
    let mut nids = vec![];
    for _ in 0..layers {
        wids.push(g.add_tensor_f32(Shape::matrix(d, d), w.clone()));
        nids.push(g.add_tensor_f32(Shape::vector(d), n.clone()));
    }
    let i = g.add_tensor_f32(Shape::vector(d), vec![0.5f32; d]);
    g.mark_input(i);
    let mut h = i;
    for l in 0..layers {
        let a = g.add_op(
            OpKind::RmsNorm,
            vec![h, nids[l]],
            Shape::vector(d),
            Some(1e-5),
            None,
        );
        let b = g.add_op(
            OpKind::MatMul,
            vec![a, wids[l]],
            Shape::vector(d),
            None,
            None,
        );
        let c = g.add_op(OpKind::SiLU, vec![b], Shape::vector(d), None, None);
        h = g.add_op(OpKind::Add, vec![h, c], Shape::vector(d), None, None);
    }
    g.set_output(h);
    let mut be = CpuBackend::new(rayon::current_num_threads());
    for _ in 0..warmup {
        be.compute(&g)?;
    }
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        be.compute(&g)?;
    }
    println!(
        "{:.1} ms/iter",
        t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn cmd_gpu_run(
    model_dir: &str,
    prompt: &str,
    max_tokens: usize,
    quant: &str,
) -> anyhow::Result<()> {
    let model = ferrule_model::OlmoeModel::load(std::path::Path::new(model_dir))?;
    let qt = parse_quant(quant);
    println!("Uploading to GPU (quant: {qt:?})...");
    let mut gpu = ferrule_cuda::forward::GpuOlmoeModel::from_cpu(&model, qt)?;
    let tokens = model.encode(prompt)?;
    println!("Prompt: \"{prompt}\" → {} tokens", tokens.len());

    let t0 = std::time::Instant::now();

    // Prefill: process all prompt tokens, keep logits from last position
    let mut logits = Vec::new();
    for &tid in &tokens {
        logits = gpu.forward(tid)?;
    }

    // First generated token comes from prefill output (not re-feeding prompt)
    let first = argmax(&logits);
    let mut gen = Vec::new();
    if !is_eos(&model, first) {
        gen.push(first);
        let txt = model.decode(&[first])?;
        println!(
            "  [{:>3}] {txt:30} {:.0}ms",
            0,
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }

    // Generate remaining tokens
    for step in 1..max_tokens {
        if gen.is_empty() {
            break;
        }
        let logits = gpu.forward(*gen.last().unwrap_or(&0))?;
        let next = argmax(&logits);
        if is_eos(&model, next) {
            break;
        }
        gen.push(next);
        let txt = model.decode(&[next])?;
        println!(
            "  [{step:>3}] {txt:30} {:.0}ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
    let t = t0.elapsed();
    println!(
        "{:.1}s, {:.1} tok/s",
        t.as_secs_f64(),
        max_tokens as f64 / t.as_secs_f64()
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn cmd_chat(model_dir: &str, max_tokens: usize, quant: &str) -> anyhow::Result<()> {
    use console::style;
    use rustyline::error::ReadlineError;

    if matches!(quant.to_ascii_lowercase().as_str(), "cpu" | "f32" | "fp32") {
        return cmd_chat_cpu(model_dir, max_tokens);
    }

    let model = ferrule_model::OlmoeModel::load(std::path::Path::new(model_dir))?;
    let qt = parse_quant(quant);
    println!("Uploading to GPU (quant: {qt:?})...");
    let mut gpu = ferrule_cuda::forward::GpuOlmoeModel::from_cpu(&model, qt)?;
    let chat_format = detect_chat_format(model_dir);
    println!(
        "{} Type /exit or Ctrl-D to quit. Template: {}.",
        style("Chat ready.").cyan(),
        chat_format.name()
    );

    let mut first_turn = true;
    let mut rl = rustyline::DefaultEditor::new()?;
    loop {
        let line = match rl.readline(&format!("{} ", style("You>").green().bold())) {
            Ok(line) => line,
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
            Err(err) => return Err(err.into()),
        };
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if matches!(input, "/exit" | "/quit") {
            break;
        }
        let _ = rl.add_history_entry(input);

        let prompt = chat_prompt(chat_format, input, first_turn);
        first_turn = false;
        let tokens = model.encode(&prompt)?;
        let mut logits = Vec::new();
        for tid in tokens {
            logits = gpu.forward(tid)?;
        }
        if logits.is_empty() {
            continue;
        }

        print!("{} ", style("Ferrule>").cyan().bold());
        std::io::stdout().flush()?;
        for _ in 0..max_tokens {
            let next = argmax(&logits);
            if is_eos(&model, next) {
                let _ = gpu.forward(next)?;
                break;
            }
            let txt = model.decode(&[next])?;
            if txt.is_empty() {
                break;
            }
            print!("{txt}");
            std::io::stdout().flush()?;
            logits = gpu.forward(next)?;
        }
        println!();
    }

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn cmd_chat(model_dir: &str, max_tokens: usize, quant: &str) -> anyhow::Result<()> {
    if matches!(quant.to_ascii_lowercase().as_str(), "cpu" | "f32" | "fp32") {
        cmd_chat_cpu(model_dir, max_tokens)
    } else {
        anyhow::bail!("chat -q {quant} requires rebuilding with --features cuda")
    }
}

#[cfg(not(feature = "cuda"))]
fn cmd_gpu_run(
    _model_dir: &str,
    _prompt: &str,
    _max_tokens: usize,
    _quant: &str,
) -> anyhow::Result<()> {
    println!("Rebuild with --features cuda");
    Ok(())
}
