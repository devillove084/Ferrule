#[cfg(feature = "cuda")]
use ferrule_runtime::ModelRunner;
#[cfg(feature = "cuda")]
use std::path::Path;

// ── argmax ───────────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold(
            (0usize, logits[0]),
            |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
        )
        .0 as u32
}

// ── compare_step ─────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
pub fn compare_step(
    cpu: &[f32],
    gpu: &[f32],
    max_abs_err: &mut f32,
    sum_abs_err: &mut f64,
    n: &mut usize,
) {
    for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let err = (c - g).abs();
        if err > *max_abs_err {
            *max_abs_err = err;
        }
        *sum_abs_err += err as f64;
        *n += 1;
        if i >= 4095 {
            break;
        }
    }
}

// ── compare-logits ───────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
pub fn cmd_compare_logits(
    model_dir: &str,
    prompt: &str,
    max_tokens: usize,
    quant: &str,
    free_run: bool,
    ctx_size: usize,
) -> anyhow::Result<()> {
    let mut cpu = ferrule_runtime::CpuModelRunner::load(Path::new(model_dir))?;
    let qt = super::run::parse_quant(quant);
    tracing::info!("Uploading to GPU (quant: {qt:?})...");
    let mut gpu = ferrule_runtime::GpuModelRunner::load(Path::new(model_dir), qt)?;

    let tokens = cpu.encode(prompt)?;
    println!("Prompt: \"{prompt}\" → {} tokens", tokens.len());
    if tokens.len() > ctx_size {
        anyhow::bail!(
            "prompt ({} tokens) exceeds ctx_size ({ctx_size})",
            tokens.len()
        );
    }

    // Prefill
    let cpu_logits = cpu.prefill(&tokens)?;
    let gpu_logits = gpu.prefill(&tokens)?;

    let mut cpu_token = argmax(&cpu_logits);
    let mut gpu_token = argmax(&gpu_logits);

    let mut cpu_text = String::new();
    let mut gpu_text = String::new();
    let mut first_divergence = None;
    let mut step = 0;
    let mut max_abs_err = 0.0f32;
    let mut sum_abs_err = 0.0f64;
    let mut n_compared = 0usize;

    compare_step(
        &cpu_logits,
        &gpu_logits,
        &mut max_abs_err,
        &mut sum_abs_err,
        &mut n_compared,
    );

    let eos = cpu.eos_token_id();

    loop {
        if step >= max_tokens || tokens.len() + step >= ctx_size {
            break;
        }
        if Some(cpu_token) == eos || Some(gpu_token) == eos {
            break;
        }

        let cpu_piece = cpu.decode(&[cpu_token])?;
        let gpu_piece = gpu.decode(&[gpu_token])?;
        cpu_text.push_str(&cpu_piece);
        gpu_text.push_str(&gpu_piece);

        if first_divergence.is_none() && cpu_token != gpu_token {
            first_divergence = Some(step);
        }

        step += 1;

        if free_run {
            let cl = cpu.decode_token(cpu_token)?;
            let gl = gpu.decode_token(gpu_token)?;
            compare_step(
                &cl,
                &gl,
                &mut max_abs_err,
                &mut sum_abs_err,
                &mut n_compared,
            );
            cpu_token = argmax(&cl);
            gpu_token = argmax(&gl);
        } else {
            let cl = cpu.decode_token(cpu_token)?;
            let gl = gpu.decode_token(cpu_token)?;
            compare_step(
                &cl,
                &gl,
                &mut max_abs_err,
                &mut sum_abs_err,
                &mut n_compared,
            );
            cpu_token = argmax(&cl);
            gpu_token = argmax(&gl);
        }
    }

    let mean_abs_err = if n_compared > 0 {
        sum_abs_err / n_compared as f64
    } else {
        0.0
    };

    println!();
    tracing::info!("=== compare-logits report ===");
    tracing::info!("model:       {model_dir}");
    tracing::info!("quant:       {qt:?}");
    println!("prompt:      \"{prompt}\"");
    tracing::info!("tokens:      {step}");
    tracing::info!("max abs err: {max_abs_err:.6}");
    tracing::info!("mean abs err:{mean_abs_err:.6}");
    tracing::info!("n compared:  {n_compared}");
    if let Some(d) = first_divergence {
        println!("first div:   step {d}");
    } else {
        println!("first div:   none (all {step} steps matched)");
    }
    println!();
    println!("CPU output: {}", cpu_text);
    println!("GPU output: {}", gpu_text);

    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn cmd_compare_logits(
    _model_dir: &str,
    _prompt: &str,
    _max_tokens: usize,
    _quant: &str,
    _free_run: bool,
    _ctx_size: usize,
) -> anyhow::Result<()> {
    anyhow::bail!("compare-logits requires --features cuda")
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use super::*;

    #[cfg(feature = "cuda")]
    #[test]
    fn test_argmax() {
        let v = vec![0.1f32, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(argmax(&v), 3);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_argmax_single() {
        let v = vec![42.0f32];
        assert_eq!(argmax(&v), 0);
    }
}
