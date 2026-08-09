#[cfg(feature = "cuda")]
pub fn cmd_cuda() -> anyhow::Result<()> {
    println!("=== CUDA Probe ===");
    let probe = ferrule_backend::cuda::diagnostics::probe_device(0)?;
    println!("  Device: {}", probe.name);
    println!(
        "  Memory: {:.1} GB free / {:.1} GB total",
        probe.memory.free_bytes as f64 / 1e9,
        probe.memory.total_bytes as f64 / 1e9
    );

    println!("\n=== GEMV Benchmark (2048×2048) ===");
    let report = ferrule_backend::cuda::run_smoke_benchmark()?;
    println!("  CPU: {:.2} ms", report.cpu_ms);
    println!("  GPU GEMV (kernel only): {:.3} ms", report.gpu_gemv_ms);
    println!(
        "  Kernel launch overhead: {:.0} µs",
        report.kernel_launch_overhead_us
    );
    println!("  compute_rms(d={}): {:.0} µs", report.dim, report.rms_us);
    println!("  Speedup: {:.0}x", report.speedup());
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn cmd_cuda() -> anyhow::Result<()> {
    println!("cuda requires --features cuda");
    Ok(())
}
