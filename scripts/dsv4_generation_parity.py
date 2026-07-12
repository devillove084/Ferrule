#!/usr/bin/env python3
"""Generate DeepSeek-V4 official prompt/token parity JSON.

This intentionally stays lightweight: it uses the model's official
`encoding/encoding_dsv4.py` plus HuggingFace `AutoTokenizer` to produce the
same prompt strings/token ids that official `inference/generate.py` feeds into
`Transformer.forward(..., start_pos=0)` during prefill. It does not load model
weights by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_official_modules(model_dir: Path):
    encoding_dir = model_dir / "encoding"
    if not (encoding_dir / "encoding_dsv4.py").exists():
        raise SystemExit(f"missing official encoding_dsv4.py under {encoding_dir}")
    sys.path.insert(0, str(encoding_dir))
    from encoding_dsv4 import encode_messages  # type: ignore

    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "transformers is required for DSV4 parity JSON; install the model requirements first"
        ) from exc

    return encode_messages, AutoTokenizer


def token_texts(tokenizer: Any, token_ids: list[int]) -> list[str]:
    out: list[str] = []
    for token_id in token_ids:
        try:
            out.append(tokenizer.decode([token_id]))
        except Exception:
            out.append("")
    return out


def build_case(
    tokenizer: Any,
    encode_messages: Any,
    name: str,
    messages: list[dict[str, Any]],
    thinking_mode: str,
    context: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prompt = encode_messages(
        messages,
        thinking_mode=thinking_mode,
        context=context,
    )
    tokens = tokenizer.encode(prompt)
    return {
        "name": name,
        "thinking_mode": thinking_mode,
        "context": context or [],
        "messages": messages,
        "prompt": prompt,
        "tokens": tokens,
        "token_texts": token_texts(tokenizer, tokens),
        "prefill": {
            "official_start_pos": 0,
            "official_forward_slice": [0, len(tokens)],
            "note": "Official generate.py calls model.forward(tokens[:, prev_pos:cur_pos], prev_pos) with prev_pos=0 for the first prompt segment.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_dir",
        nargs="?",
        default="models/DeepSeek-V4-Flash-DSpark",
        help="DeepSeek-V4 HF model directory",
    )
    parser.add_argument("--prompt", default="Hello", help="single-turn user prompt")
    parser.add_argument("--assistant", default="Hi!", help="assistant text for two-turn case")
    parser.add_argument("--follow-up", default="How are you?", help="second user turn")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", "-o", default="-", help="output JSON path or '-' for stdout")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    encode_messages, AutoTokenizer = load_official_modules(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)

    single_messages = [{"role": "user", "content": args.prompt}]
    two_turn_context = [
        {"role": "user", "content": args.prompt},
        {"role": "assistant", "content": args.assistant},
    ]
    follow_up_messages = [{"role": "user", "content": args.follow_up}]
    full_multi_turn_messages = two_turn_context + follow_up_messages

    cases = [
        build_case(
            tokenizer,
            encode_messages,
            "single_turn_chat",
            single_messages,
            "chat",
        ),
        build_case(
            tokenizer,
            encode_messages,
            "multi_turn_chat_full_prompt",
            full_multi_turn_messages,
            "chat",
        ),
        build_case(
            tokenizer,
            encode_messages,
            "incremental_followup_chat_turn",
            follow_up_messages,
            "chat",
            context=two_turn_context,
        ),
    ]

    data = {
        "schema": "ferrule.dsv4.generation_parity.v1",
        "model_dir": str(model_dir),
        "source": {
            "encoding": str(model_dir / "encoding" / "encoding_dsv4.py"),
            "official_generate": str(model_dir / "inference" / "generate.py"),
            "tokenizer": "transformers.AutoTokenizer.from_pretrained(model_dir)",
        },
        "special_tokens": {
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "bos_token": getattr(tokenizer, "bos_token", None),
            "eos_token": getattr(tokenizer, "eos_token", None),
            "pad_token": getattr(tokenizer, "pad_token", None),
        },
        "generation_contract": {
            "max_new_tokens": args.max_new_tokens,
            "prefill_start_pos": 0,
            "decode_start": "after the first forward over the full prompt segment",
            "official_generate_loop": "for cur_pos in range(min_prompt_len, total_len): next = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)[0]",
            "ferrule_expected_path": "DeepSeekV4Runner::prefill_tokens_*_batched followed by decode_token_*",
        },
        "cases": cases,
    }

    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if args.output == "-":
        print(text, end="")
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
