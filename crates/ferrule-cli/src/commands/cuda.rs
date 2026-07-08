#[cfg(feature = "cuda")]
pub fn cmd_cuda() -> anyhow::Result<()> {
    println!("=== CUDA Probe ===");
    ferrule_cuda::cuda_probe()?;

    println!("\n=== GEMV Benchmark (2048×2048) ===");
    let report = ferrule_cuda::run_smoke_benchmark()?;
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
