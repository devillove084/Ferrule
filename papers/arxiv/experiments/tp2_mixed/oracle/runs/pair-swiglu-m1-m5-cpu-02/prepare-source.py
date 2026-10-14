import copy
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
HERE = OUT.parents[1]
sys.path.insert(0, str(HERE))
import routed_down_extract as e
import routed_down_contract as rc
require = rc.require
m = copy.deepcopy(rc.bundle())
require(not (OUT / "manifest.json").exists(), "Fresh preparation required")
require(not e.o.torch.cuda.is_initialized(), "CPU-only preparation")
require(e.o.initialize() == m["native_source"], "Pinned HF sources differ")
ref = HERE / "runs/full40-20261009-03"
reference = e.c.load_manifest(ref, rc.REFERENCE_PIN, True)
metadata = json.loads(e.c.checked_file(ref, reference["checkpoint_metadata"]).read_text())
used = json.loads(e.c.checked_file(ref, reference["source_payloads"]).read_text())["used_payloads"]
model = Path(reference["checkpoint"]["directory"])
for name, field in (("config.json", "config_sha256"), ("model.safetensors.index.json", "index_sha256")):
    require(rc.sha(model / name) == reference["checkpoint"][field], "Checkpoint identity differs")
for value in m["original_payloads"].values():
    value["source_directory"] = str(rc.BUNDLE)
for fixture in m["fixtures"]:
    fixture["rows"] = 1
    folder = OUT / fixture["directory"]
    folder.mkdir()
    for name in ("preparation.json", "observed.safetensors"):
        shutil.copyfile(rc.BUNDLE / fixture["directory"] / name, folder / name)
case = next(c for c in reference["cases"] if c["id"] == "capital")
case_path = e.c.checked_file(ref, case)
name = "capital-prefill-layer09-expert095"
folder = OUT / name
folder.mkdir()
input_key = "prefill.layers.08.hidden"
keys = [input_key, "prefill.layers.09.attention", "prefill.layers.09.moe"]
for rank in (0, 1):
    keys += ["prefill.rank.%d.layers.09." % rank + field for field in
             ("routed_partial", "selected_expert_ids", "routing_weights", "raw_router_logits", "router_probabilities")]
with e.safe_open(case_path, framework="pt", device="cpu") as archive:
    observed = {k: archive.get_tensor(k).contiguous() for k in keys}
ids = observed["prefill.rank.0.layers.09.selected_expert_ids"]
require(tuple(ids.shape) == (5, 8) and e.o.torch.equal(ids, observed["prefill.rank.1.layers.09.selected_expert_ids"]), "M5 routes differ")
require(all(95 in row for row in ids.tolist()), "Expert95 must actually receive all five tokens")
experts = sorted(set(ids.flatten().tolist()))
require(len(experts) == 24, "M5 active expert set differs")
e.save_file(observed, str(folder / "observed.safetensors"))
base = "model.language_model.layers.9."
names = [base + "post_attention_layernorm.weight", base + "mlp.gate.weight"]
for expert in experts:
    for part in ("gate_proj", "up_proj", "down_proj"):
        weight = base + "mlp.experts.%d.%s.weight" % (expert, part)
        names.extend([weight, weight + "_scale_inv"])
payloads = e.PinnedPayloads(model, reference["checkpoint"], metadata, used)
(OUT / "payloads").mkdir()
for name_key in names:
    raw = payloads.read(name_key)
    filename = "payloads/" + name_key + ".bin"
    (OUT / filename).write_bytes(raw)
    entry = metadata["tensors"][name_key]
    m["original_payloads"][name_key] = dict(file=filename, bytes=len(raw), sha256=e.digest(raw),
        original_header=entry, original_used_payload=used[name_key],
        historical_whole_shard_sha256=reference["checkpoint"]["full_shards"][entry["shard"]])
payloads.finish()
require(payloads.bytes == 76559360 and len(names) == 146, "Bounded M5 read inventory differs")
report = json.loads(e.c.checked_file(ref, case["report"]).read_text())
prep = dict(case="capital", stage="prefill", layer=9, expert=95, rows=5,
    selected_expert_ids=experts, selected_expert_ids_by_token=ids.tolist(),
    selection="First layer with a five-token expert; minimum all-five expert ID", reference_case=case,
    call=report["calls"][0], observed_tensor_keys=keys, checkpoint_sha256=rc.REFERENCE_PIN,
    checkpoint_shard_set_sha256=reference["checkpoint"]["full_shard_set_sha256"])
e.write_json(folder / "preparation.json", prep)
m["fixtures"].append(dict(directory=name, case="capital", stage="prefill", layer=9, expert=95, rows=5, n=2048, local_k=256))
m["gpu_replays"].append(dict(fixture=name, upstream_input_key=input_key, payload_names=names,
    calls="two unchanged native HF Experts.forward calls, 24 actual active experts per rank"))
m.update(schema="ferrule.paper.pair-swiglu.cpu-replay-preparation.v1", status="cpu_prepared_pending_authorized_cuda",
    parent_bundle_sha256=rc.BUNDLE_PIN, cuda_executed=False, model_fixture_ready=False,
    new_payload_bytes=payloads.bytes, new_payload_count=len(names),
    cpu_generation_source_sha256=rc.sha(Path(__file__)),
    input_warning="Only archived residual/attention boundaries; no CPU normalized input or expected activation",
    future_cuda_calls=dict(norm=5, router=10, experts=10, gate_up_linear=112, down_linear=112))
require(not e.o.torch.cuda.is_initialized(), "CPU extraction initialized CUDA")
e.write_json(OUT / "manifest.json", m)
with (OUT / "SHA256SUMS.cpu-extract").open("x") as f:
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.cpu-extract":
            f.write(rc.sha(path) + "  " + path.relative_to(OUT).as_posix() + "\n")
print(json.dumps(dict(bundle=str(OUT), sha256=rc.sha(OUT / "SHA256SUMS.cpu-extract"), new_payload_bytes=payloads.bytes, cases=len(m["fixtures"]))))
