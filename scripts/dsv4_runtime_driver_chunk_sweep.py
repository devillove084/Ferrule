#!/usr/bin/env python3
"""Sweep DSV4 ResidentTopKDriver prefill chunk sizes.

This script intentionally stays outside the Rust crates: it is benchmark plumbing,
not runtime architecture. It runs `ferrule bench-interactive --runtime-driver --json`
for each chunk size, keeps the full JSON report for every run, and writes a compact
CSV that is easy to paste into perf notes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "models/DeepSeek-V4-Flash-DSpark"
DEFAULT_PROMPTS = ["Hello", "Explain Ferrule in one sentence."]
DEFAULT_CHUNKS = "1,2,4,8,16,4096"

SUMMARY_FIELDS = [
    "chunk_size",
    "status",
    "elapsed_s",
    "model",
    "runtime_path",
    "dsv4_profile_sync",
    "max_layers",
    "max_new_tokens",
    "total_turns",
    "total_prompt_tokens",
    "total_prefill_s",
    "total_generated",
    "final_position",
    "artifact_load_s",
    "warmup_s",
    "time_to_first_token_s",
    "aggregate_prefill_tok_per_s",
    "aggregate_decode_tok_per_s",
    "runtime_actions",
    "runtime_prefill_chunks",
    "runtime_prefill_tokens",
    "runtime_decode_steps",
    "runtime_emitted_tokens",
    "runtime_staged_tokens",
    "runtime_finished_sequences",
    "operator_kernel_launches",
    "operator_h2d_copies",
    "operator_h2d_bytes",
    "operator_d2h_copies",
    "operator_d2h_bytes",
    "operator_artifact_uploads",
    "operator_artifact_upload_bytes",
    "operator_moe_calls",
    "operator_moe_total_s",
    "operator_moe_tc_calls",
    "operator_moe_scalar_calls",
    "operator_moe_router_s",
    "operator_moe_routing_s",
    "operator_moe_plan_s",
    "operator_moe_cache_lookup_s",
    "operator_moe_expert_read_s",
    "operator_moe_expert_upload_s",
    "operator_moe_shared_s",
    "operator_moe_workspace_s",
    "operator_moe_compute_submit_s",
    "operator_moe_commit_s",
    "operator_output_head_calls",
    "operator_output_head_chunks",
    "operator_output_head_rows",
    "operator_output_head_cache_hits",
    "operator_output_head_cache_misses",
    "operator_output_head_hidden_uploads",
    "operator_output_head_hidden_upload_s",
    "operator_output_head_read_s",
    "operator_output_head_upload_s",
    "operator_output_head_topk_s",
    "operator_output_head_merge_s",
    "operator_expert_loads",
    "operator_expert_load_bytes",
    "runtime_step_count",
    "runtime_prefill_step_s",
    "runtime_decode_step_s",
    "profile_layer_total_s",
    "profile_bind_s",
    "profile_state_init_s",
    "profile_decode_total_s",
    "profile_prefill_total_s",
    "profile_attention_s",
    "profile_moe_s",
    "profile_attn_hc_pre_s",
    "profile_attn_norm_s",
    "profile_attn_hc_post_s",
    "profile_ffn_hc_pre_s",
    "profile_ffn_norm_s",
    "profile_ffn_hc_post_s",
    "profile_output_hc_head_s",
    "profile_output_norm_s",
    "profile_output_topk_s",
    "attention_q_a_s",
    "attention_q_norm_s",
    "attention_q_b_s",
    "attention_q_head_norm_s",
    "attention_q_rope_s",
    "attention_q_latent_download_s",
    "attention_kv_proj_s",
    "attention_kv_norm_s",
    "attention_kv_rope_quant_s",
    "attention_kv_cache_append_s",
    "attention_hidden_download_s",
    "attention_indexer_compress_s",
    "attention_main_compress_s",
    "attention_compressed_kv_upload_s",
    "attention_topk_build_s",
    "attention_sparse_attention_s",
    "attention_context_rope_s",
    "attention_output_a_s",
    "attention_output_b_s",
    "slowest_step_kind",
    "slowest_step_rows",
    "slowest_step_s",
    "dsv4_logits_calls",
    "dsv4_logits_tokens",
    "dsv4_no_logits_calls",
    "dsv4_no_logits_tokens",
    "dsv4_interactive_calls",
    "dsv4_interactive_tokens",
    "dsv4_batched_calls",
    "dsv4_batched_tokens",
    "dsv4_start_segment_calls",
    "dsv4_start_segment_tokens",
    "dsv4_append_segment_calls",
    "dsv4_append_segment_tokens",
    "dsv4_token_fallback_calls",
    "dsv4_token_fallback_tokens",
    "finish_reasons_json",
    "generated_token_ids_json",
    "error",
]


def parse_chunks(raw: str) -> list[int]:
    chunks: list[int] = []
    seen: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            chunk = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid chunk size {part!r}") from exc
        if chunk <= 0:
            raise argparse.ArgumentTypeError("chunk sizes must be positive")
        if chunk not in seen:
            chunks.append(chunk)
            seen.add(chunk)
    if not chunks:
        raise argparse.ArgumentTypeError("at least one chunk size is required")
    return chunks


def parse_report(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("bench output JSON root must be an object")
    return value


def f64(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def i64(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def summarize(report: dict[str, Any], chunk: int, elapsed_s: float) -> dict[str, Any]:
    runtime = report.get("runtime_driver_stats") or {}
    dsv4 = report.get("dsv4_prefill_stats") or {}
    ops = report.get("dsv4_operator_counters") or {}
    profile = report.get("dsv4_layer_profile_summary") or {}
    output_profile = report.get("dsv4_output_profile") or {}
    attention = report.get("dsv4_attention_profile_summary") or {}
    turns = report.get("turns") or []

    runtime_steps: list[dict[str, Any]] = []
    finish_reasons: list[str] = []
    generated_token_ids: list[list[int]] = []
    for turn in turns:
        if isinstance(turn, dict):
            finish_reasons.append(str(turn.get("finish_reason", "unknown")))
            ids = turn.get("generated_token_ids") or []
            generated_token_ids.append(ids if isinstance(ids, list) else [])
            steps = turn.get("runtime_steps") or []
            runtime_steps.extend(step for step in steps if isinstance(step, dict))

    prefill_step_s = sum(
        f64(step.get("elapsed_s"))
        for step in runtime_steps
        if step.get("action_kind") == "prefill"
    )
    decode_step_s = sum(
        f64(step.get("elapsed_s"))
        for step in runtime_steps
        if step.get("action_kind") == "decode"
    )
    slowest = max(runtime_steps, key=lambda step: f64(step.get("elapsed_s")), default={})

    return {
        "chunk_size": chunk,
        "status": "ok",
        "elapsed_s": elapsed_s,
        "model": report.get("model", ""),
        "runtime_path": report.get("runtime_path", ""),
        "dsv4_profile_sync": bool(report.get("dsv4_profile_sync", False)),
        "max_layers": i64(report.get("max_layers")),
        "max_new_tokens": i64(report.get("max_new_tokens")),
        "total_turns": i64(report.get("total_turns")),
        "total_prompt_tokens": i64(report.get("total_prompt_tokens")),
        "total_prefill_s": f64(report.get("total_prefill_s")),
        "total_generated": i64(report.get("total_generated")),
        "final_position": i64(report.get("final_position")),
        "artifact_load_s": f64(report.get("artifact_load_s")),
        "warmup_s": f64(report.get("warmup_s")),
        "time_to_first_token_s": f64(report.get("time_to_first_token_s")),
        "aggregate_prefill_tok_per_s": f64(report.get("aggregate_prefill_tok_per_s")),
        "aggregate_decode_tok_per_s": f64(report.get("aggregate_decode_tok_per_s")),
        "runtime_actions": i64(runtime.get("actions")),
        "runtime_prefill_chunks": i64(runtime.get("prefill_chunks")),
        "runtime_prefill_tokens": i64(runtime.get("prefill_tokens")),
        "runtime_decode_steps": i64(runtime.get("decode_steps")),
        "runtime_emitted_tokens": i64(runtime.get("emitted_tokens")),
        "runtime_staged_tokens": i64(runtime.get("staged_tokens")),
        "runtime_finished_sequences": i64(runtime.get("finished_sequences")),
        "operator_kernel_launches": i64(ops.get("kernel_launches")),
        "operator_h2d_copies": i64(ops.get("host_to_device_copies")),
        "operator_h2d_bytes": i64(ops.get("host_to_device_bytes")),
        "operator_d2h_copies": i64(ops.get("device_to_host_copies")),
        "operator_d2h_bytes": i64(ops.get("device_to_host_bytes")),
        "operator_artifact_uploads": i64(ops.get("artifact_uploads")),
        "operator_artifact_upload_bytes": i64(ops.get("artifact_upload_bytes")),
        "operator_moe_calls": i64(ops.get("moe_calls")),
        "operator_moe_total_s": f64(ops.get("moe_total_s")),
        "operator_moe_tc_calls": i64(ops.get("moe_tc_calls")),
        "operator_moe_scalar_calls": i64(ops.get("moe_scalar_calls")),
        "operator_moe_router_s": f64(ops.get("moe_router_s")),
        "operator_moe_routing_s": f64(ops.get("moe_routing_s")),
        "operator_moe_plan_s": f64(ops.get("moe_plan_s")),
        "operator_moe_cache_lookup_s": f64(ops.get("moe_cache_lookup_s")),
        "operator_moe_expert_read_s": f64(ops.get("moe_expert_read_s")),
        "operator_moe_expert_upload_s": f64(ops.get("moe_expert_upload_s")),
        "operator_moe_shared_s": f64(ops.get("moe_shared_s")),
        "operator_moe_workspace_s": f64(ops.get("moe_workspace_s")),
        "operator_moe_compute_submit_s": f64(ops.get("moe_compute_submit_s")),
        "operator_moe_commit_s": f64(ops.get("moe_commit_s")),
        "operator_output_head_calls": i64(ops.get("output_head_calls")),
        "operator_output_head_chunks": i64(ops.get("output_head_chunks")),
        "operator_output_head_rows": i64(ops.get("output_head_rows")),
        "operator_output_head_cache_hits": i64(ops.get("output_head_cache_hits")),
        "operator_output_head_cache_misses": i64(ops.get("output_head_cache_misses")),
        "operator_output_head_hidden_uploads": i64(ops.get("output_head_hidden_uploads")),
        "operator_output_head_hidden_upload_s": f64(ops.get("output_head_hidden_upload_s")),
        "operator_output_head_read_s": f64(ops.get("output_head_read_s")),
        "operator_output_head_upload_s": f64(ops.get("output_head_upload_s")),
        "operator_output_head_topk_s": f64(ops.get("output_head_topk_s")),
        "operator_output_head_merge_s": f64(ops.get("output_head_merge_s")),
        "operator_expert_loads": i64(ops.get("expert_loads")),
        "operator_expert_load_bytes": i64(ops.get("expert_load_bytes")),
        "runtime_step_count": len(runtime_steps),
        "runtime_prefill_step_s": prefill_step_s,
        "runtime_decode_step_s": decode_step_s,
        "profile_layer_total_s": f64(profile.get("decode_total_s")) + f64(profile.get("prefill_total_s")),
        "profile_bind_s": f64(profile.get("bind_s")),
        "profile_state_init_s": f64(profile.get("state_init_s")),
        "profile_decode_total_s": f64(profile.get("decode_total_s")),
        "profile_prefill_total_s": f64(profile.get("prefill_total_s")),
        "profile_attention_s": f64(profile.get("attention_s")),
        "profile_moe_s": f64(profile.get("moe_s")),
        "profile_attn_hc_pre_s": f64(profile.get("attn_hc_pre_s")),
        "profile_attn_norm_s": f64(profile.get("attn_norm_s")),
        "profile_attn_hc_post_s": f64(profile.get("attn_hc_post_s")),
        "profile_ffn_hc_pre_s": f64(profile.get("ffn_hc_pre_s")),
        "profile_ffn_norm_s": f64(profile.get("ffn_norm_s")),
        "profile_ffn_hc_post_s": f64(profile.get("ffn_hc_post_s")),
        "profile_output_hc_head_s": f64(output_profile.get("final_hc_head_s")),
        "profile_output_norm_s": f64(output_profile.get("final_norm_s")),
        "profile_output_topk_s": f64(output_profile.get("lm_head_topk_s")),
        "attention_q_a_s": f64(attention.get("q_a_s")),
        "attention_q_norm_s": f64(attention.get("q_norm_s")),
        "attention_q_b_s": f64(attention.get("q_b_s")),
        "attention_q_head_norm_s": f64(attention.get("q_head_norm_s")),
        "attention_q_rope_s": f64(attention.get("q_rope_s")),
        "attention_q_latent_download_s": f64(attention.get("q_latent_download_s")),
        "attention_kv_proj_s": f64(attention.get("kv_proj_s")),
        "attention_kv_norm_s": f64(attention.get("kv_norm_s")),
        "attention_kv_rope_quant_s": f64(attention.get("kv_rope_quant_s")),
        "attention_kv_cache_append_s": f64(attention.get("kv_cache_append_s")),
        "attention_hidden_download_s": f64(attention.get("hidden_download_s")),
        "attention_indexer_compress_s": f64(attention.get("indexer_compress_s")),
        "attention_main_compress_s": f64(attention.get("main_compress_s")),
        "attention_compressed_kv_upload_s": f64(attention.get("compressed_kv_upload_s")),
        "attention_topk_build_s": f64(attention.get("topk_build_s")),
        "attention_sparse_attention_s": f64(attention.get("sparse_attention_s")),
        "attention_context_rope_s": f64(attention.get("context_rope_s")),
        "attention_output_a_s": f64(attention.get("output_a_s")),
        "attention_output_b_s": f64(attention.get("output_b_s")),
        "slowest_step_kind": slowest.get("action_kind", ""),
        "slowest_step_rows": i64(slowest.get("rows")),
        "slowest_step_s": f64(slowest.get("elapsed_s")),
        "dsv4_logits_calls": i64(dsv4.get("logits_calls")),
        "dsv4_logits_tokens": i64(dsv4.get("logits_tokens")),
        "dsv4_no_logits_calls": i64(dsv4.get("no_logits_calls")),
        "dsv4_no_logits_tokens": i64(dsv4.get("no_logits_tokens")),
        "dsv4_interactive_calls": i64(dsv4.get("interactive_calls")),
        "dsv4_interactive_tokens": i64(dsv4.get("interactive_tokens")),
        "dsv4_batched_calls": i64(dsv4.get("batched_calls")),
        "dsv4_batched_tokens": i64(dsv4.get("batched_tokens")),
        "dsv4_start_segment_calls": i64(dsv4.get("start_segment_calls")),
        "dsv4_start_segment_tokens": i64(dsv4.get("start_segment_tokens")),
        "dsv4_append_segment_calls": i64(dsv4.get("append_segment_calls")),
        "dsv4_append_segment_tokens": i64(dsv4.get("append_segment_tokens")),
        "dsv4_token_fallback_calls": i64(dsv4.get("token_fallback_calls")),
        "dsv4_token_fallback_tokens": i64(dsv4.get("token_fallback_tokens")),
        "finish_reasons_json": json.dumps(finish_reasons, separators=(",", ":")),
        "generated_token_ids_json": json.dumps(generated_token_ids, separators=(",", ":")),
        "error": "",
    }


def failure_row(chunk: int, elapsed_s: float, error: str) -> dict[str, Any]:
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update({
        "chunk_size": chunk,
        "status": "error",
        "elapsed_s": elapsed_s,
        "error": error,
    })
    return row


def build_cuda(arch: str) -> None:
    cmd = [
        "cargo",
        "oxide",
        "build",
        "--features",
        "cuda",
        "--arch",
        arch,
        "--",
        "--release",
        "-p",
        "ferrule-cli",
    ]
    print("[sweep] building CUDA binary:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


def run_chunk(args: argparse.Namespace, chunk: int) -> tuple[dict[str, Any], str, str, float]:
    cmd = [args.bin, "bench-interactive", args.model]
    for prompt in args.prompt:
        cmd.extend(["-p", prompt])
    cmd.extend(
        [
            "-n",
            str(args.max_tokens),
            "--runtime-driver",
            "--warmup-tokens",
            str(args.warmup_tokens),
            "--prefill-chunk-size",
            str(chunk),
            "--max-layers",
            str(args.max_layers),
            "--json",
        ]
    )
    cmd.extend(args.extra)

    start = time.perf_counter()
    env = os.environ.copy()
    if args.profile_sync:
        env["FERRULE_DSV4_PROFILE_SYNC"] = "1"
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    elapsed_s = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(
            f"chunk={chunk} failed with exit code {proc.returncode}\n"
            f"command: {' '.join(cmd)}\n"
            f"stderr:\n{proc.stderr.strip()}"
        )
    report = parse_report(proc.stdout)
    return report, proc.stdout, proc.stderr, elapsed_s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("-p", "--prompt", action="append", default=None)
    parser.add_argument("-n", "--max-tokens", type=int, default=1)
    parser.add_argument("--warmup-tokens", type=int, default=0)
    parser.add_argument("--max-layers", type=int, default=43)
    parser.add_argument("--chunks", type=parse_chunks, default=parse_chunks(DEFAULT_CHUNKS))
    parser.add_argument("--bin", default="./target/release/ferrule")
    parser.add_argument("--output-dir", default="target/dsv4-runtime-driver-chunk-sweep")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--profile-sync", action="store_true")
    parser.add_argument("--build-cuda", action="store_true")
    parser.add_argument("--cuda-arch", default=os.environ.get("FERRULE_CUDA_ARCH", "sm_121a"))
    args, extra = parser.parse_known_args()
    args.extra = extra
    if args.prompt is None:
        args.prompt = list(DEFAULT_PROMPTS)

    if args.max_tokens < 0 or args.warmup_tokens < 0 or args.max_layers <= 0:
        parser.error("--max-tokens/--warmup-tokens must be >= 0 and --max-layers must be > 0")

    if args.build_cuda:
        build_cuda(args.cuda_arch)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"summary_layers{args.max_layers}_tokens{args.max_tokens}.csv"
    jsonl_path = out_dir / f"reports_layers{args.max_layers}_tokens{args.max_tokens}.jsonl"

    rows: list[dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for chunk in args.chunks:
            print(f"[sweep] running chunk={chunk}", file=sys.stderr)
            start = time.perf_counter()
            try:
                report, stdout, stderr, elapsed_s = run_chunk(args, chunk)
                (out_dir / f"chunk_{chunk}.json").write_text(stdout, encoding="utf-8")
                if stderr.strip():
                    (out_dir / f"chunk_{chunk}.stderr.log").write_text(stderr, encoding="utf-8")
                report["sweep"] = {"chunk_size": chunk, "elapsed_s": elapsed_s}
                jsonl.write(json.dumps(report, separators=(",", ":")) + "\n")
                row = summarize(report, chunk, elapsed_s)
                rows.append(row)
                print(
                    "[sweep] chunk={chunk} ttft={ttft:.3f}s prefill={prefill:.2f} tok/s "
                    "no_logits={no_logits} append_seg={append_seg} fallback={fallback}".format(
                        chunk=chunk,
                        ttft=row["time_to_first_token_s"],
                        prefill=row["aggregate_prefill_tok_per_s"],
                        no_logits=row["dsv4_no_logits_tokens"],
                        append_seg=row["dsv4_append_segment_tokens"],
                        fallback=row["dsv4_token_fallback_tokens"],
                    ),
                    file=sys.stderr,
                )
            except Exception as exc:  # noqa: BLE001 - benchmark harness should report all failure detail.
                elapsed_s = time.perf_counter() - start
                row = failure_row(chunk, elapsed_s, str(exc))
                rows.append(row)
                print(f"[sweep] chunk={chunk} ERROR: {exc}", file=sys.stderr)
                if not args.keep_going:
                    break

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[sweep] wrote {csv_path}", file=sys.stderr)
    print(f"[sweep] wrote {jsonl_path}", file=sys.stderr)
    ok = all(row.get("status") == "ok" for row in rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
