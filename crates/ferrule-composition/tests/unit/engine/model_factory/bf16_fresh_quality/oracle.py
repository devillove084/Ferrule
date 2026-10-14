"""Bounded fresh prompts using the maintained independent BF16 HF producer."""
import gc, json, sys, time
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'scripts'));import qwen35_fp8_bf16_reference as ref

def main(directory):
    spec=json.loads((directory/'INPUT.json').read_text())
    assert spec['decode_steps']==8 and len(spec['cases'])==3
    original=ROOT/'target/validation/numeric-bf16-gate-20261004-v2/profile-reference'
    assert ref.sha256(original/'manifest.json')==spec['original_reference_manifest_sha256']
    baseline=json.loads((original/'manifest.json').read_text())
    pins=ref.initialize();import torch
    from transformers import AutoTokenizer
    from safetensors.torch import save_file
    model_dir=Path('/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8')
    for name,pin in spec['tokenizer']['files'].items():assert ref.sha256(model_dir/name)==pin
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    torch.manual_seed(0)
    output=directory/'oracle';output.mkdir(exist_ok=False)
    config=ref.read_config(model_dir);checkpoint=ref.legacy.Checkpoint(model_dir)
    devices=[f'cuda:{i}'for i in range(8)];weights=[ref.DeviceWeights(checkpoint,d,256<<20)for d in devices]
    for d in devices:
        free,total=torch.cuda.mem_get_info(d)
        assert free >= (2*1024+256)*1024**2
        torch.cuda.reset_peak_memory_stats(d)
    model,placement=ref.build_model(config,weights)
    manifest=dict(schema='ferrule.bf16-fresh-quality-reference.v1',status='incomplete',profile=ref.PROFILE,router=ref.router_contract(ref.ROUTER_PROFILE),native_source=pins,versions=dict(ref.VERSIONS),precision_contract=dict(ref.CONTRACT),input_sha256=ref.sha256(directory/'INPUT.json'),original_reference=dict(directory=str(original),sha256=spec['original_reference_manifest_sha256']),cases=[],devices=[])
    started=time.monotonic()
    try:
        for case in spec['cases']:
            assert tokenizer.encode(case['prompt'],add_special_tokens=False)==case['input_ids']
            capture=ref.Capture(1024<<20,ref.ROUTER_PROFILE)
            cache=ref.f32_ssm_cache(config,placement)
            for i,layer in enumerate(model.model.layers):capture.layer(layer,i)
            calls=[];fed=case['input_ids'];position=0;generated=[]
            try:
                with torch.inference_mode():
                    for step in range(9):
                        stage=f'step.{step:02}';capture.stage=stage
                        x=torch.tensor([fed],device=placement[0],dtype=torch.int64)
                        pos=torch.arange(position,position+len(fed),device=placement[0])
                        mask=torch.ones((1,position+len(fed)),device=placement[0],dtype=torch.int64)
                        y=model(input_ids=x,position_ids=pos[None],attention_mask=mask,past_key_values=cache,use_cache=True,logits_to_keep=0,return_dict=True)
                        assert y.past_key_values is cache
                        capture.put('logits',y.logits);capture.cache(cache,range(40))
                        token=int(y.logits[0,-1].argmax());generated.append(token)
                        calls.append(dict(stage=stage,input_token_ids=fed,position_start=position,sequence_length_after=position+len(fed),next_token_id=token))
                        position+=len(fed);assert cache.get_seq_length()==position;fed=[token]
                path=output/(case['id']+'.safetensors')
                save_file(capture.tensors,str(path),metadata=dict(schema=ref.PROFILE_SCHEMA,profile=ref.PROFILE,model_boundary='bf16',router_tie_policy=ref.ROUTER_PROFILE))
                report=dict(id=case['id'],file=path.name,sha256=ref.sha256(path),calls=calls,profile=ref.PROFILE,router=ref.router_contract(ref.ROUTER_PROFILE),generated_ids=generated,generated_text=tokenizer.decode(generated,skip_special_tokens=False),prompt=case['prompt'],capture_bytes=capture.bytes)
                ref.write_json(output/(case['id']+'.json'),report)
                manifest['cases'].append(dict(id=case['id'],file=path.name,sha256=report['sha256'],report=case['id']+'.json',report_sha256=ref.sha256(output/(case['id']+'.json'))))
                print(json.dumps(report,ensure_ascii=False),flush=True)
            finally:capture.close();del capture,cache;gc.collect()
        manifest.update(status='complete',wall_seconds=time.monotonic()-started)
    finally:
        ref.write_json(output/'source-payloads.json',checkpoint.provenance())
        manifest['source_payloads']=dict(file='source-payloads.json',sha256=ref.sha256(output/'source-payloads.json'))
        for w in weights:
            manifest['devices'].append(dict(device=str(w.device),peak_allocated=torch.cuda.max_memory_allocated(w.device),peak_reserved=torch.cuda.max_memory_reserved(w.device),peak_operand_cache=w.peak_bytes));w.clear()
        del model;gc.collect()
        for d in devices:
            with torch.cuda.device(d):torch.cuda.synchronize();torch.cuda.empty_cache()
        ref.write_json(output/'manifest.json',manifest)

if __name__=='__main__':main(Path(sys.argv[1]).resolve())
