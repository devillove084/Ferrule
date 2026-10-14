#!/usr/bin/env python3
"""Targeted inherited-CUDA and pure-doctest source guards; stdlib only."""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

# These are canonical ownership/module paths. `graph.rs` is intentionally
# listed separately: CUDA graph capture remains an independent logical module.
CANONICAL_MODULES = (
    "crates/ferrule-backend/src/cuda/operator_owner.rs",
    "crates/ferrule-backend/src/cuda/operators/resource_views.rs",
    "crates/ferrule-backend/src/cuda/graph.rs",
    "crates/ferrule-model/src/moe/resource_manager.rs",
    "crates/ferrule-model/src/tensor_semantics.rs",
    "crates/ferrule-model/src/models/deepseek_v4/config.rs",
    "crates/ferrule-model/src/models/deepseek_v4/config/cuda.rs",
    "crates/ferrule-composition/src/model_factory/parallel/plan.rs",
    "crates/ferrule-composition/src/model_factory/parallel/bindings.rs",
    "crates/ferrule-composition/src/model_factory/parallel/budget.rs",
    "crates/ferrule-composition/src/model_factory/parallel/experts.rs",
    "crates/ferrule-composition/src/model_factory/parallel/provider.rs",
    "crates/ferrule-composition/src/model_factory/parallel/assembly.rs",
    "crates/ferrule-composition/src/model_factory/resident_f32_bound.rs",
)

# Preserve the inherited-CUDA guard, including #[path] children outside the
# parent's directory. Independent platform/test/oracle conditions stay legal.
CUDA_ONLY = (
    "crates/ferrule-backend/src/cuda",
    "crates/ferrule-model/src/decoder/hybrid_cuda.rs",
    "crates/ferrule-model/src/decoder/kv/cuda.rs",
    "crates/ferrule-model/src/moe/resource_manager.rs",
    "crates/ferrule-model/src/moe/cuda_materialization.rs",
    "crates/ferrule-model/src/moe/cuda_install.rs",
    "crates/ferrule-model/src/moe/cuda_payload.rs",
    "crates/ferrule-model/src/moe/cuda_upload.rs",
    "crates/ferrule-model/src/moe/cuda_materialization_test_support.rs",
    "crates/ferrule-model/src/moe/routed/cuda.rs",
    "crates/ferrule-model/src/models/deepseek_v4/config/cuda.rs",
    "crates/ferrule-model/src/transformer/cuda.rs",
    "crates/ferrule-model/src/transformer/standard/cuda.rs",
    "crates/ferrule-model/src/transformer/standard/cuda",
    "crates/ferrule-runtime/src/parallel/expert_process/child_cuda.rs",
    "crates/ferrule-runtime/src/parallel/pipeline/cuda.rs",
    "crates/ferrule-runtime/src/parallel/pipeline/hybrid_cuda.rs",
    "crates/ferrule-runtime/src/parallel/pipeline/hybrid_cuda_tensor.rs",
    "crates/ferrule-runtime/src/parallel/pipeline/tensor.rs",
)

# Keep these paths aligned with ci-cuda-compile's symbol filters. They contain
# only compile_fail/no_run examples; no positive GPU doctest is admitted here.
PURE_DOCS = (
    "crates/ferrule-backend/src/cuda/operators/linear.rs",
    "crates/ferrule-backend/src/cuda/operators/standard_bf16.rs",
    "crates/ferrule-model/src/decoder/hybrid_cuda.rs",
    "crates/ferrule-model/tests/compile/cuda_hybrid_state.md",
    "crates/ferrule-model/src/transformer/standard/cuda/segment.rs",
    "crates/ferrule-model/src/transformer/standard/numeric_precision.rs",
    "crates/ferrule-model/tests/compile/numeric_w8a16.md",
)

CFG = re.compile(r"#!?\s*\[\s*cfg(?:_attr)?\s*\([^\]]*\]", re.S)
CUDA = re.compile(r'\bfeature\s*=\s*"cuda"')
MODULE = re.compile(
    r"(?m)((?:^[ \t]*#\[[^\n]*\][ \t]*\n)+)"
    r"^[ \t]*(?:pub(?:\([^)]+\))?[ \t]+)?mod[ \t]+(\w+)[ \t]*(?:;|\{)"
)
CUDA_GATE = ('#[cfg(feature="cuda")]',)
NON_CUDA_GATE = ('#[cfg(not(feature="cuda"))]',)
DECLARATIONS = (
    ("crates/ferrule-model/src/moe/mod.rs", "resource_manager", [CUDA_GATE]),
    ("crates/ferrule-model/src/models/deepseek_v4/config.rs", "cuda", [CUDA_GATE]),
    ("crates/ferrule-runtime/src/parallel/expert.rs", "cuda", [CUDA_GATE]),
    ("crates/ferrule-runtime/src/parallel/pipeline.rs", "cuda", [CUDA_GATE]),
    ("crates/ferrule-runtime/src/parallel/pipeline.rs", "hybrid_cuda", [CUDA_GATE]),
    ("crates/ferrule-runtime/src/parallel/pipeline.rs", "tensor", [CUDA_GATE]),
    ("crates/ferrule-runtime/src/parallel/pipeline.rs", "hybrid_cuda_tensor", [CUDA_GATE]),
    # A real CUDA/non-CUDA implementation pair, not duplicate declarations.
    ("crates/ferrule-runtime/src/parallel/expert_process.rs", "child", [CUDA_GATE, NON_CUDA_GATE]),
)


def module_cuda_variants(source, name):
    return sorted(
        tuple(re.sub(r"\s+", "", cfg[0]) for cfg in CFG.finditer(attrs) if CUDA.search(cfg[0]))
        for attrs, module in MODULE.findall(source) if module == name
    )


def redundant_cuda_cfgs(source):
    # Ignore comments/doctests, preserving diagnostic line numbers.
    source = re.sub(r"//[^\n]*", lambda match: " " * len(match[0]), source)
    return [source.count("\n", 0, match.start()) + 1
            for match in CFG.finditer(source) if CUDA.search(match[0])]


def doc_fence_errors(source, markdown=False):
    if not markdown:
        source = "\n".join(re.findall(r"^\s*//[/!] ?(.*)$", source, re.M))
    in_fence = False
    invalid = []
    for number, line in enumerate(source.splitlines(), 1):
        if not line.strip().startswith("```"):
            continue
        if not in_fence:
            tags = {tag.strip() for tag in line.strip()[3:].split(",")}
            if not {"compile_fail", "no_run"}.intersection(tags):
                invalid.append(number)
        in_fence = not in_fence
    return invalid


def check():
    errors = []
    for relative in CANONICAL_MODULES:
        if not (ROOT / relative).is_file():
            errors.append(f"{relative}: missing canonical module path")
    for relative, name, expected in DECLARATIONS:
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"{relative}: missing audited module owner")
        elif module_cuda_variants(path.read_text(), name) != sorted(expected):
            errors.append(f"{relative}: mod {name} has missing, duplicate or unexpected CUDA gates")
    for relative in CUDA_ONLY:
        path = ROOT / relative
        if not path.exists():
            errors.append(f"{relative}: missing audited CUDA-only module")
            continue
        files = sorted(path.rglob("*.rs")) if path.is_dir() else [path]
        for file in files:
            for line in redundant_cuda_cfgs(file.read_text()):
                errors.append(f"{file.relative_to(ROOT)}:{line}: redundant inherited cuda cfg")
    for relative in PURE_DOCS:
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"{relative}: missing audited pure doctest source")
            continue
        if doc_fence_errors(path.read_text(), path.suffix == ".md"):
            errors.append(f"{relative}: selected doctest source contains a runtime example")
    return errors


def self_test():
    declaration = '#[cfg(feature = "cuda")]\nmod cuda;'
    assert module_cuda_variants(declaration, "cuda") == [CUDA_GATE]
    assert module_cuda_variants('#[cfg(feature="cuda")]\n' + declaration, "cuda") != [CUDA_GATE]
    assert module_cuda_variants(declaration + '\n' + declaration, "cuda") != [CUDA_GATE]
    variants = '#[cfg(feature="cuda")]\n#[path="child_cuda.rs"]\nmod child;\n' + (
        '#[cfg(not(feature="cuda"))]\n#[path="child_unavailable.rs"]\nmod child;'
    )
    assert module_cuda_variants(variants, "child") == sorted([CUDA_GATE, NON_CUDA_GATE])
    for cfg in (
        '#[cfg(feature = "cuda")]', '#![cfg(not(feature = "cuda"))]',
        '#[cfg(all(\n feature = "cuda", unix))]',
        '#[cfg(any(test, feature = "cuda"))]',
        '#[cfg_attr(feature = "cuda", allow(dead_code))]',
    ):
        assert redundant_cuda_cfgs(cfg), cfg
    for cfg in (
        '#[cfg(test)]', '#[cfg(not(test))]', '#[cfg(unix)]',
        '#[cfg(all(test, target_os = "linux"))]',
        '#[cfg(ferrule_cuda_test_oracle)]', '#[cfg(not(ferrule_cuda_test_oracle))]',
        '#[cfg(all(feature = "cuda-test-support", target_os = "linux"))]',
        '/// #[cfg(feature = "cuda")]',
    ):
        assert not redundant_cuda_cfgs(cfg), cfg
    assert not doc_fence_errors("/// ```compile_fail,E0499\n/// code\n/// ```")
    assert not doc_fence_errors("```no_run\ncode\n```", markdown=True)
    assert doc_fence_errors("/// ```rust\n/// code\n/// ```")
    assert doc_fence_errors("```\ncode\n```", markdown=True)


if __name__ == "__main__":
    self_test()
    if sys.argv[1:] == ["--self-test"]:
        print("feature boundary guard self-tests passed")
        sys.exit(0)
    if sys.argv[1:]:
        sys.exit("usage: check_feature_boundaries.py [--self-test]")
    errors = check()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        sys.exit(1)
    print("feature boundaries and pure CUDA doctest sources passed")
