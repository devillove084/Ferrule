"""CPU-only retained-turn oracle using native HF BF16 checkpoint loading.

Reads the existing independent two-layer model fixture; does not use Ferrule
forward or round nonpersistent inverse frequencies with model.to(BF16).
"""
import json
from pathlib import Path
import tempfile
import torch
import transformers
from safetensors.torch import save_file
from transformers import Qwen3ForCausalLM


def main():
    torch.set_num_threads(1)
    here = Path(__file__).resolve().parent
    fixture = json.loads((here / "../../../ferrule-model/tests/fixtures/standard_bf16_hf.json").read_text())
    with tempfile.TemporaryDirectory(prefix="ferrule-public-bf16-hf-") as directory:
        checkpoint = Path(directory)
        (checkpoint / "config.json").write_text(json.dumps(fixture["config"]))
        weights = {w["name"]: torch.tensor(w["values"], dtype=torch.bfloat16).reshape(w["shape"]) for w in fixture["weights"]}
        save_file(weights, checkpoint / "model.safetensors", metadata={"format": "pt"})
        model = Qwen3ForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True).eval()
    assert model.model.rotary_emb.inv_freq.dtype == torch.float32
    assert all(torch.equal(p, weights[n]) for n, p in model.state_dict().items())
    turns = []
    cache = None
    position = 0
    with torch.inference_mode():
        for prompt, reset in [([1, 2, 3], True), ([4, 5], False), ([1, 2, 3, 7], True)]:
            if reset:
                cache, position = None, 0
            output = model(torch.tensor([prompt]), past_key_values=cache, use_cache=True)
            position += len(prompt)
            generated, logits = [], []
            for _ in range(2):
                row = output.logits[0, -1].float()
                token = int(row.argmax())
                generated.append(token)
                logits.append(float(row[token]))
                output = model(torch.tensor([[token]]), past_key_values=output.past_key_values, use_cache=True)
                position += 1
            cache = output.past_key_values
            turns.append(dict(prompt=prompt, reset=reset, generated=generated, logits=logits, retained_position=position))
    result = dict(schema="ferrule.public-dense-bf16-hf.v1", torch=torch.__version__, transformers=transformers.__version__, device="cpu", inverse_frequency_dtype="float32", turns=turns)
    (here / "hf_serving.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
