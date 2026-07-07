#!/usr/bin/env python3
"""Wrap cuda-oxide kernel-launch calls in `unsafe { }`.

The fork's cuda-macros generates `unsafe fn` launchers, so every
`<receiver>.<kernel_method>(...)` call must be wrapped. Only wraps calls
whose receiver is a known LoadedModule alias and whose method is a known
kernel name, to avoid touching safe buffer/context ops.
"""
import re
import sys

# Kernel method names that became unsafe (from cargo check E0133 list).
KERNEL_METHODS = {
    "embed_lookup", "gemv_f32", "grouped_matvec_f32", "f32_bytes_to_f32",
    "bf16_pair_to_f32", "gemv_f32_bytes", "gemv_bf16_bytes", "gemv_q4",
    "gemv_q8", "gemv_q8_off", "gemv_q4_off", "gemv_fp4_e2m1_e8m0",
    "gemv_fp4_e2m1_e8m0_off", "gemv_dual_fp4_e2m1_e8m0_off",
    "swiglu_weighted_clamped", "fp8_e4m3fn_e8m0_quantize_f32_inplace",
    "gemv_fp8_e4m3fn_e8m0_2d", "gemm_fp4_e2m1_e8m0", "gemv_q2",
    "gemv_q2_off", "gemv_t1", "gemv_t1_off", "rms_norm_apply", "silu",
    "silu_mul", "gemv_dual_q4", "gemv_triple_q4", "gemv_dual_q4_off",
    "mul", "add", "saxpy", "copy_f32_slot", "compute_rms",
    "rms_norm_fused", "rms_norm_heads_fused", "router_topk", "rope",
    "sparse_attn_tiled_sink_f32", "attn_scores", "attn_combine_softmax",
    "topk_vocab", "mla_q_projection_f32", "rope_yarn", "rope_tail_yaarn",
    "swiglu_down_accumulate", "moe_gemv_dual_fp4_batched",
    "moe_swiglu_fp8_batched", "moe_gemv_down_fp4_batched",
    "moe_gemv_dual_fp4_batched_reduce", "moe_gemv_down_fp4_batched_reduce",
    "hc_pre_f32", "hc_post_f32", "hc_head_f32", "hc_pre_single_f32",
    "hc_post_single_f32", "hc_head_single_f32",
}

# Receivers that are LoadedModule aliases.
RECEIVERS = ["self.model.module", "self.module", "module", "m"]

def find_call_end(src, open_paren_idx):
    """Given index of '(', return index of matching ')' (balanced, string-aware)."""
    depth = 0
    i = open_paren_idx
    n = len(src)
    in_str = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    while i < n:
        c = src[i]
        nxt = src[i+1] if i+1 < n else ''
        if in_line_comment:
            if c == '\n':
                in_line_comment = False
            i += 1; continue
        if in_block_comment:
            if c == '*' and nxt == '/':
                in_block_comment = False; i += 2; continue
            i += 1; continue
        if in_str:
            if c == '\\':
                i += 2; continue
            if c == '"':
                in_str = False
            i += 1; continue
        if in_char:
            if c == '\\':
                i += 2; continue
            if c == "'":
                in_char = False
            i += 1; continue
        if c == '/' and nxt == '/':
            in_line_comment = True; i += 2; continue
        if c == '/' and nxt == '*':
            in_block_comment = True; i += 2; continue
        if c == '"':
            in_str = True; i += 1; continue
        if c == "'":
            # could be lifetime 'a or char 'x' — heuristic: skip, treat as char only if looks like char
            # To be safe, treat as char literal start only if followed by \ or non-alpha
            # Lifetimes won't contain parens, so ignoring is safe for paren matching.
            in_char = True; i += 1; continue
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1

def wrap_file(path):
    with open(path) as f:
        src = f.read()
    # Build a regex that matches receiver.method( at word boundaries.
    # Order receivers longest-first so self.model.module beats self.module.
    rec_alt = "|".join(re.escape(r) for r in sorted(RECEIVERS, key=len, reverse=True))
    method_alt = "|".join(re.escape(m) for m in KERNEL_METHODS)
    # Allow optional whitespace (incl. newline) between receiver and .method(
    # to cover chained calls like `module\n.gemv_f32(`.
    pat = re.compile(r'(?<![A-Za-z0-9_.])(' + rec_alt + r')(\s*)\.\s*(' + method_alt + r')\(')
    matches = list(pat.finditer(src))
    if not matches:
        return 0
    # Process from last to first so indices stay valid.
    count = 0
    for mt in reversed(matches):
        open_paren = mt.end() - 1  # index of '('
        # If there was whitespace between receiver and '.', the call starts
        # at the receiver; find the actual start of `receiver`.
        call_start = mt.start(1)
        close = find_call_end(src, open_paren)
        if close == -1:
            print(f"  WARN: unbalanced parens at {path}:{src[:open_paren].count(chr(10))+1}", file=sys.stderr)
            continue
        call = src[call_start:close+1]
        # Avoid double-wrapping
        prefix = src[call_start-9:call_start] if call_start >= 9 else src[:call_start]
        if 'unsafe {' in prefix:
            continue
        replacement = "unsafe { " + call + " }"
        src = src[:call_start] + replacement + src[close+1:]
        count += 1
    with open(path, 'w') as f:
        f.write(src)
    return count

if __name__ == "__main__":
    total = 0
    for p in sys.argv[1:]:
        n = wrap_file(p)
        print(f"{p}: wrapped {n}")
        total += n
    print(f"TOTAL: {total}")
