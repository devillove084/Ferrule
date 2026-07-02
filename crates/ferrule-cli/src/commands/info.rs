use ferrule_model::{EnginePlan, ModelDescriptor, TensorClassCount, TransformerSpec};
use ferrule_runtime::ModelInfo;
use std::path::Path;

// ── info ─────────────────────────────────────────────────────────────────────

pub fn cmd_info(model_dir: &str) -> anyhow::Result<()> {
    let descriptor = ModelDescriptor::load(Path::new(model_dir))?;
    print_transformer_spec(&descriptor.spec);
    print_tensor_classes(&descriptor.tensor_classes);
    print_engine_plan(&descriptor.engine_plan());
    Ok(())
}

pub fn print_model_info(info: &ModelInfo) {
    let moe = if info.num_experts > 0 {
        format!(", {}e top-{}", info.num_experts, info.num_experts_per_tok)
    } else {
        String::new()
    };
    let arch = info
        .architecture
        .as_deref()
        .map(|a| format!(", arch={a}"))
        .unwrap_or_default();
    println!(
        "{}: {}d×{}L{}, vocab={} ({}, attention={}, source={}{})",
        info.family,
        info.hidden_size,
        info.num_layers,
        moe,
        info.vocab_size,
        info.backend,
        info.attention,
        info.weight_source,
        arch
    );
}

fn print_transformer_spec(spec: &TransformerSpec) {
    println!("Model family: {}", spec.family);
    if let Some(arch) = &spec.architecture {
        println!("architecture: {arch}");
    }
    println!("source:       {}", spec.weight_source);
    println!("attention:    {}", spec.attention);
    println!(
        "transformer:  {}d × {}L, vocab={}",
        fmt_opt(spec.hidden_size),
        fmt_opt(spec.num_layers),
        fmt_opt(spec.vocab_size)
    );
    if spec.num_heads.is_some() || spec.num_kv_heads.is_some() || spec.head_dim.is_some() {
        println!(
            "heads:        q={}, kv={}, head_dim={}",
            fmt_opt(spec.num_heads),
            fmt_opt(spec.num_kv_heads),
            fmt_opt(spec.head_dim)
        );
    }
    if spec.moe.is_moe() {
        println!(
            "moe:          experts={}, top_k={}, router={}, shared_experts={}",
            fmt_opt(spec.moe.num_experts),
            fmt_opt(spec.moe.num_experts_per_tok),
            spec.moe.router,
            spec.moe.has_shared_experts
        );
    }
    if let Some(count) = spec.tensor_count {
        println!("tensors:      {count}");
    }
    if !spec.quantization.is_empty() {
        println!("quantization:");
        for item in &spec.quantization {
            println!("  {:>12}: {} tensors", item.format, item.tensors);
        }
    }
    if !spec.supports_current_runtime() {
        println!("runtime:      metadata-only for now; see engine plan below");
    }
    for note in &spec.notes {
        println!("note:         {note}");
    }
}

fn print_tensor_classes(classes: &[TensorClassCount]) {
    if classes.is_empty() {
        return;
    }
    println!("tensor classes:");
    for item in classes {
        println!("  {:>24}: {} tensors", item.class, item.tensors);
    }
}

fn print_engine_plan(plan: &EnginePlan) {
    println!("engine plan:  {}", plan.status);
    println!(
        "  policies: attention={} kv={} router={} expert={} quant_source={} residency={} speculation={:?}",
        plan.policies.attention.kind,
        plan.policies.kv.shape,
        plan.policies.router.kind,
        plan.policies.expert.kind,
        plan.policies.quant.source,
        fmt_residency(plan.policies.residency.streaming_allowed, plan.policies.residency.all_resident_required),
        plan.policies.speculation.mode,
    );
    if plan.missing.is_empty() {
        println!("  missing:  none");
    } else {
        println!("  missing policies:");
        for item in &plan.missing {
            println!("    - {}: {}", item.area, item.reason);
        }
    }
}

fn fmt_opt<T: std::fmt::Display>(value: Option<T>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn fmt_residency(streaming_allowed: bool, all_resident_required: bool) -> &'static str {
    match (streaming_allowed, all_resident_required) {
        (true, false) => "streaming_allowed",
        (true, true) => "streaming_optional",
        (false, true) => "all_resident",
        (false, false) => "unspecified",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmd_info_nonexistent() {
        let result = cmd_info("/nonexistent/path/to/model");
        assert!(result.is_err());
    }

    #[test]
    fn fmt_opt_unknown() {
        assert_eq!(fmt_opt::<usize>(None), "unknown");
        assert_eq!(fmt_opt(Some(7usize)), "7");
    }
}
