//! CPU-only call-site guards for CUDA owners, not a substitute for GPU execution.
//! The real FFN hook/failure ordering and production custody ledger have separate
//! behavioral CPU tests. Here we check function-local wiring, ignoring formatting,
//! comments and literal contents rather than depending on a `let result = ...` shape.
const COMMON: &str = include_str!("../../../src/transformer/standard/mod.rs");
const ADAPTER: &str = include_str!("../../../src/transformer/standard/layer_execution.rs");
const SEGMENT: &str = include_str!("../../../src/transformer/segment.rs");
const HYBRID: &str = include_str!("../../../src/transformer/standard/cuda/hybrid.rs");
const STAGE: &str = include_str!("../../../src/transformer/standard/cuda/stage.rs");
const CUDA_SEGMENT: &str = include_str!("../../../src/transformer/standard/cuda/segment.rs");
const RANK: &str = include_str!("../../../src/transformer/standard/cuda/hybrid_rank/execution.rs");
const ROOT: &str = include_str!("../../../src/transformer/standard/cuda/root_execution.rs");
const CUDA: &str = include_str!("../../../src/transformer/standard/cuda.rs");

// A deliberately small lexical guard, not a Rust parser. Keep identifiers whole,
// preserve delimiters and skip comments/literals so prose cannot satisfy a call.
fn tokens(source: &str) -> Vec<&str> {
    let bytes = source.as_bytes();
    let mut result = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let start = i;
        if bytes[i].is_ascii_whitespace() {
            i += 1;
        } else if source[i..].starts_with("//") {
            i += source[i..].find('\n').unwrap_or(bytes.len() - i);
        } else if source[i..].starts_with("/*") {
            i += 2;
            let mut depth = 1;
            while depth > 0 && i < bytes.len() {
                if bytes[i..].starts_with(b"/*") {
                    depth += 1;
                    i += 2;
                } else if bytes[i..].starts_with(b"*/") {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            assert_eq!(depth, 0, "unterminated block comment");
        } else if bytes[i] == b'"' {
            i += 1;
            while i < bytes.len() && bytes[i] != b'"' {
                i += if bytes[i] == b'\\' { 2 } else { 1 };
            }
            assert!(i < bytes.len(), "unterminated string literal");
            i += 1;
            result.push("<literal>");
        } else if bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' {
            i += 1;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            result.push(&source[start..i]);
        } else {
            i += source[i..]
                .chars()
                .next()
                .expect("remaining token")
                .len_utf8();
            result.push(&source[start..i]);
        }
    }
    result
}

fn position(source: &[&str], pattern: &str) -> usize {
    let pattern_tokens = tokens(pattern);
    source
        .windows(pattern_tokens.len())
        .position(|window| window == pattern_tokens)
        .unwrap_or_else(|| panic!("missing code tokens: {pattern}"))
}

fn contains(source: &[&str], pattern: &str) -> bool {
    let pattern = tokens(pattern);
    source
        .windows(pattern.len())
        .any(|window| window == pattern)
}

fn group<'a>(source: &'a [&'a str], open: usize) -> (&'a [&'a str], &'a [&'a str]) {
    let close = match source[open] {
        "{" => "}",
        "(" => ")",
        other => panic!("expected a code group, got {other}"),
    };
    let mut depth = 0;
    for i in open..source.len() {
        if source[i] == source[open] {
            depth += 1;
        } else if source[i] == close {
            depth -= 1;
            if depth == 0 {
                return (&source[open + 1..i], &source[i + 1..]);
            }
        }
    }
    panic!("unclosed code group");
}

fn function<'a>(source: &'a [&'a str], name: &str) -> &'a [&'a str] {
    let start = position(source, &format!("fn {name}"));
    let source = &source[start..];
    group(source, position(source, "{")).0
}

fn call<'a>(source: &'a [&'a str], callee: &str) -> (&'a [&'a str], &'a [&'a str]) {
    let start = position(source, &format!("{callee}("));
    group(source, start + tokens(callee).len())
}

#[test]
fn cuda_owners_enter_the_private_layer_scope_without_replacing_default_dispatch() {
    let adapter = tokens(ADAPTER);
    let dispatch = function(&adapter, "execute_layer");
    assert!(contains(
        dispatch,
        "Self::Standard(operators) => execute_standard_layer_inner("
    ));
    assert!(contains(
        dispatch,
        "Self::Cuda(operators) => operators.execute_root_layer("
    ));

    let common = tokens(COMMON);
    call(
        function(&common, "execute_standard_layer"),
        "operators.execute_layer",
    );
    let segment = tokens(SEGMENT);
    assert!(contains(&segment, "LayerOperators::Standard(operators)"));
    assert!(contains(
        &segment,
        "LayerOperators::Standard(&mut operators)"
    ));
    assert!(!contains(&segment, "execute_standard_layer_inner("));
    call(
        function(&segment, "execute_inner"),
        "execute_standard_layer",
    );
    call(
        function(&segment, "execute_bound_inner"),
        "operators.execute_layer",
    );

    let hybrid = tokens(HYBRID);
    call(function(&hybrid, "start_layer"), "self.execute_layer");
    let (args, _) = call(
        function(&hybrid, "execute_layer"),
        "this.operators.execute_root_layer",
    );
    assert!(contains(args, "execution.as_mut()"));
    assert!(!contains(&hybrid, "execute_standard_layer_inner("));

    // Check each owner/call separately: a CUDA adapter elsewhere in the file
    // must not hide a standard-dispatch bypass in one external/local branch.
    for (source, owner, callee) in [
        (
            STAGE,
            "execute_cuda_bound",
            "self.segment.execute_with_layer_operators",
        ),
        (
            STAGE,
            "execute_cuda_bound_with_experts",
            "self.segment.execute_with_experts_and_layer_operators",
        ),
        (
            CUDA_SEGMENT,
            "execute_inner",
            "self.segment.execute_bound_inner",
        ),
        (CUDA_SEGMENT, "execute_inner", "self.segment.execute_bound"),
        (
            RANK,
            "execute_cuda_bound",
            "self.segment.execute_bound_inner",
        ),
    ] {
        let source = tokens(source);
        assert!(!contains(&source, "execute_standard_layer_inner("));
        let (args, _) = call(function(&source, owner), callee);
        assert!(
            contains(args, "LayerOperators::Cuda(&mut self.operators)"),
            "{owner}: {callee} must opt into root dispatch"
        );
        assert!(!contains(args, "LayerOperators::Standard("));
    }

    let root = tokens(ROOT);
    let layer = function(&root, "execute_root_layer");
    let (args, _) = call(layer, "super::super::execute_standard_layer_inner");
    assert!(contains(args, "routed_execution"));
    let (scope_args, _) = call(layer, "self.with_root_submission");
    assert!(contains(scope_args, "run"));
    assert!(!contains(layer, "routed_execution.is_some()"));
    assert!(!contains(layer, "synchronous("));
    // Arbitrary TP/observer callbacks retain their original synchronous path.
    let scope = function(&root, "with_root_submission");
    let fallback = position(
        scope,
        "if self.bindings.tensor.is_some() || self.diagnostic.is_enabled()",
    );
    let fallback = &scope[fallback..];
    assert!(contains(
        group(fallback, position(fallback, "{")).0,
        "return self.synchronous(run)"
    ));
}

#[test]
fn external_dispatch_and_post_layer_callbacks_keep_explicit_completion_boundaries() {
    let common = tokens(COMMON);
    let ffn = function(&common, "execute_feed_forward");
    let start = position(ffn, "if let Some(execution) = routed_execution");
    let branch = &ffn[start..];
    let branch = group(branch, position(branch, "{")).0;
    let (_, after_hook) = call(branch, "operators.before_external_routed_execution");
    assert_eq!(
        after_hook.first(),
        Some(&"?"),
        "failed producer proof must stop dispatch"
    );
    call(after_hook, "execution.executor.routed_swiglu");
    assert!(!contains(
        &branch[..position(branch, "operators.before_external_routed_execution")],
        "routed_swiglu("
    ));

    let cuda = tokens(CUDA);
    let (reason, _) = call(
        function(&cuda, "before_external_routed_execution"),
        "self.root_barrier",
    );
    assert_eq!(reason, tokens("BarrierReason::ExternalExperts"));

    for (source, owner, dispatch, drain) in [
        (
            HYBRID,
            "execute_layer",
            "this.operators.execute_root_layer",
            "this.require_expert_drain",
        ),
        (
            CUDA_SEGMENT,
            "execute_inner",
            "self.segment.execute_bound",
            "self.external.require_drain",
        ),
        (
            RANK,
            "execute_cuda_bound",
            "self.segment.execute_bound_inner",
            "self.external.require_drain",
        ),
    ] {
        let source = tokens(source);
        let (_, after_dispatch) = call(function(&source, owner), dispatch);
        let before_drain = &after_dispatch[..position(after_dispatch, &format!("{drain}("))];
        assert!(
            contains(before_drain, "?"),
            "{owner} must propagate layer failure before draining external custody"
        );
    }
    let root = tokens(ROOT);
    let scope = function(&root, "with_root_submission");
    assert!(contains(
        scope,
        "self.root_barrier(BarrierReason::SegmentExit)"
    ));
    assert!(contains(scope, "self.poison_root_submission()"));
}

#[test]
fn external_error_exit_requires_clean_handoff_and_does_not_prove_remote_completion() {
    let root = tokens(ROOT);
    let scope = function(&root, "with_root_submission");
    let arm = &scope[position(scope, "Ok(Err(error)) => {")..];
    let error = group(arm, position(arm, "{")).0;
    for required in [
        "!self.needs_quarantine()",
        "SubmissionSegment::can_return_external_error",
        "self.root_submission = None;",
        "self.finish_cache_operation()",
        "return Err(error);",
        "self.poison_root_submission()",
    ] {
        assert!(
            contains(error, required),
            "external error must retain {required}"
        );
    }
    for forbidden in [
        "sync_stream",
        "consumer_proof",
        "root_barrier",
        "drain_experts",
    ] {
        assert!(
            !contains(error, forbidden),
            "external error cannot establish completion via {forbidden}"
        );
    }
    let barrier = function(&root, "root_barrier");
    let unknown = position(
        barrier,
        "if segment.is_unknown() || self.needs_quarantine()",
    );
    let idle = position(barrier, "if !segment.has_pending()");
    assert!(unknown < idle, "Unknown must reject even an idle barrier");
    let branch = &barrier[unknown..];
    assert!(contains(
        group(branch, position(branch, "{")).0,
        "return Err("
    ));
    assert!(idle < position(barrier, ".note_idle_barrier(reason)"));
}

#[test]
fn wiring_guards_ignore_layout_and_decoys_but_detect_a_dispatch_bypass() {
    let source = tokens(
        r#"
        // fn execute_layer() { operators.execute_root_layer(); }
        fn execute_layer() {
            let message = "operators.execute_root_layer()";
            /* outer /* nested */ operators.execute_root_layer(); */
            let renamed = with_binding(|binding| {
                operators
                    .execute_root_layer(binding)
            });
            renamed?;
            external.drain()
        }
        fn unrelated() { operators.synchronous(); }
    "#,
    );
    let layer = function(&source, "execute_layer");
    let (_, after) = call(layer, "operators.execute_root_layer");
    assert!(contains(after, "renamed ? ;"));
    assert!(!contains(layer, "operators.synchronous("));
    let bypass = tokens(
        "fn execute_layer() { /* operators.execute_root_layer() */ operators.execute_standard_layer_inner(); }",
    );
    assert!(!contains(
        function(&bypass, "execute_layer"),
        "operators.execute_root_layer("
    ));
}
