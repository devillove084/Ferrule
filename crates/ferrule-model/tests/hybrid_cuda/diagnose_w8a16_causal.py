"""CPU-only same-input diagnosis, not an acceptance oracle or a model forward.

CUDA layer boundaries are authoritative; CPU recomputations are labelled as such.
CPU Torch, cuBLAS and Ferrule do NOT have identical reduction orders.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

if '--execute-cuda' not in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import torch
import torch.nn.functional as F
from safetensors import safe_open
from analyze_numeric_trace import tensor
from diagnose_numeric_first import ref, packed_vector_reduction

ATOL = RTOL = 2e-4
NATIVE = 'b3001ab72dbddce268207a5771365e09961505e5e9fd092244fa17d362f57da5'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def metric(actual, expected):
    a, e = actual.detach().cpu().reshape(-1).double(), expected.detach().cpu().reshape(-1).double()
    require(a.shape == e.shape and a.numel() > 0, 'Shape mismatch or empty comparison')
    require(bool(torch.isfinite(a).all() and torch.isfinite(e).all()), 'Nonfinite comparison')
    error = (a-e).abs()
    return dict(n=a.numel(), max_abs=float(error.max()), outside=int((error > ATOL+RTOL*e.abs()).sum()),
                unequal=int((a != e).sum()), crossed_bf16_bins=int((a.float().bfloat16() != e.float().bfloat16()).sum()))


def operator_bound(actual, x, w):
    """Existing triangle bound; never promoted to a full-model acceptance rule."""
    precise = F.linear(x.double(), w.double())
    bound = 1e-5 + 2e-6 * F.linear(x.double().abs(), w.double().abs())
    error = (actual.double()-precise).abs()
    return dict(outside=int((error > bound).sum()), max_fraction=float((error/bound).max()),
                max_abs=float(error.max()))


def combine(table, ids, weights, expert_order=False):
    """Independent weighted sum with explicit F32 multiply then add boundaries."""
    require(table.shape[:2] == ids.shape == weights.shape, 'Invalid route table')
    out = torch.zeros_like(table[:, 0])
    order = ids.argsort(dim=-1) if expert_order else torch.arange(ids.shape[1]).expand_as(ids)
    for row in range(len(ids)):
        for slot in order[row].tolist():
            out[row] = out[row] + table[row, slot] * weights[row, slot]
    return out


def route_metrics(logits, ids, weights):
    require(ids.ndim==weights.ndim==logits.ndim==2 and ids.shape==weights.shape
            and len(ids)==len(logits) and 0<ids.shape[1]<logits.shape[1], 'Invalid routing shape')
    require(bool(((ids>=0)&(ids<logits.shape[1])).all()), 'Invalid expert ID')
    require(bool((ids.sort(-1).values[:,1:]!=ids.sort(-1).values[:,:-1]).all()), 'Duplicate expert ID')
    logits = logits.detach()
    probs = logits.softmax(-1)
    ids_device = ids.to(logits.device)
    values, selected = probs.topk(ids.shape[1], dim=-1)
    normalized = values/values.sum(-1, keepdim=True)
    # Align by expert ID. Different slots are not different routing semantics.
    selected = selected.cpu()
    same_set = ids.sort(-1).values.eq(selected.sort(-1).values).all(-1)
    aligned = probs.gather(1, ids_device)
    aligned /= aligned.sum(-1, keepdim=True)
    sorted_probs = probs.sort(-1, descending=True).values
    k = ids.shape[1]
    return dict(same_set=bool(same_set.all()), set_mismatch_rows=int((~same_set).sum()),
                slot_mismatch_rows=int((ids != selected).any(-1).sum()),
                topk_boundary_ties=int((sorted_probs[:, k-1] == sorted_probs[:, k]).sum()),
                same_selected_weights=metric(weights, aligned),
                native_weights=metric(weights, normalized) if torch.equal(ids, selected) else None,
                raw_topk_set_equal=bool(torch.equal(ids.sort(-1).values, logits.topk(k,-1).indices.sort(-1).values.cpu())))


def read_event(directory, event):
    path = directory / event['file']
    require(path.resolve().parent == directory.resolve(), 'Trace path escapes case directory')
    require(path.stat().st_size == 4 * event['shape'][0] * event['shape'][1], 'Truncated trace payload')
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.float32).reshape(event['shape'])


def validate_reference(directory, pin, case):
    require(sha256(directory/'manifest.json') == pin, 'CUDA manifest pin mismatch')
    manifest = json.loads((directory/'manifest.json').read_text())
    require(manifest['schema'] == 'ferrule.qwen35-fp8-cuda-reference.v1'
            and manifest['status'] == 'complete' and manifest['full_reference_complete'] is True, 'Incomplete CUDA reference')
    require(manifest['execution']['device'] == 'cuda' and manifest['execution']['tf32'] is False
            and manifest['precision_contract']['profile'] == 'bf16_rne', 'Wrong CUDA precision contract')
    from qwen35_fp8_cuda_reference import CONTRACT
    require(manifest['precision_contract'] == CONTRACT, 'Different semantic rounding contract')
    require(manifest['native_source']['sha256'] == NATIVE, 'Unaudited native source')
    entry = next(e for e in manifest['cases'] if e['id'] == case)
    for name, pin_key in (('file', 'sha256'), ('report', 'report_sha256')):
        path = directory/entry[name]
        require(path.resolve().parent == directory.resolve() and sha256(path) == entry[pin_key], 'CUDA payload pin mismatch')
    report = json.loads((directory/entry['report']).read_text())
    require(report['sha256'] == entry['sha256'], 'CUDA report/payload mismatch')
    return manifest, report, directory/entry['file']



class NativeTrace:
    def __init__(self,directory,entry):
        self.events={}
        if entry is None:
            return
        self.directory=directory/entry['directory']
        require(self.directory.resolve().parent==directory.resolve(),'Native trace escaped directory')
        index=self.directory/'events.jsonl'
        require(sha256(index)==entry['events_sha256'],'Native trace index mismatch')
        for line in index.read_text().splitlines():
            event=json.loads(line)
            require(event['key'] not in self.events,'Duplicate native trace key')
            self.events[event['key']]=event
        require(len(self.events)==entry['tensors'],'Incomplete native trace')

    def get(self,key):
        if key not in self.events:
            return None
        event=self.events[key]
        path=self.directory/event['file']
        require(path.resolve().parent==self.directory.resolve() and sha256(path)==event['sha256'],'Native tensor hash mismatch')
        dtype={'torch.float32':torch.float32,'torch.int64':torch.int64}[event['dtype']]
        return torch.frombuffer(bytearray(path.read_bytes()),dtype=dtype).reshape(event['shape'])

    def compare(self,module,x,actual,quantized):
        native_x,native_y=self.get(module+'.input.0'),self.get(module+'.output.0')
        if native_x is None or native_y is None:
            return None
        require(native_x.numel()==x.numel() and native_y.numel()==actual.numel(),'Native module layout mismatch')
        native_x=native_x.reshape_as(x)
        operand=ref.independent_bf16_rne(x) if quantized else x
        native_operand=ref.independent_bf16_rne(native_x) if quantized else native_x
        identical=torch.equal(operand,native_operand)
        return dict(trace_id=module,inputs=metric(x,native_x),rounded_operands_identical=identical,
                    output=metric(actual,native_y),
                    classification='same-input CUDA operator' if identical else 'propagated-input difference, NOT same-input')


def native_module(manifest):
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q
    require(sha256(q.__file__) == manifest['native_source']['sha256'], 'Imported native source differs')
    return q


def attention_same_input(q, config, get, start, history=None):
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    c = Qwen3_5MoeTextConfig(**config)
    rows = get('norm.output', 'AttentionQueryNorm').shape[0]
    hd, nh, nk = c.head_dim, c.num_attention_heads, c.num_key_value_heads
    query = get('norm.output', 'AttentionQueryNorm').reshape(1,rows,nh,hd).transpose(1,2)
    key = get('norm.output', 'AttentionKeyNorm').reshape(1,rows,nk,hd).transpose(1,2)
    value = get('linear.output', 'AttentionValue').reshape(1,rows,nk,hd).transpose(1,2)
    device=query.device
    rope = q.Qwen3_5MoeTextRotaryEmbedding(c, device=device)
    require(rope.inv_freq.dtype==torch.float32, 'RoPE inv_freq was narrowed')
    cos, sin = rope(torch.zeros(1,rows,c.hidden_size,device=device), torch.arange(start,start+rows,device=device)[None])
    query, key = q.apply_rotary_pos_emb(query,key,cos,sin)
    if history is not None:
        key, value = torch.cat((history[0],key),dim=2), torch.cat((history[1],value),dim=2)
    keys, values = q.repeat_kv(key,nh//nk), q.repeat_kv(value,nh//nk)
    scores = (query @ keys.transpose(-1,-2)) * hd**-0.5
    causal = torch.arange(key.shape[2],device=device)[None,:] > torch.arange(start,start+rows,device=device)[:,None]
    scores = scores.masked_fill(causal[None,None], torch.finfo(torch.float32).min)
    out = (scores.softmax(-1) @ values).transpose(1,2).reshape(rows,-1)
    gate = get('linear.output','AttentionQuery').reshape(rows,nh,2*hd)[...,hd:].reshape(rows,-1)
    return out*gate.sigmoid(), (key,value)


def gdn_same_input(q, config, get, weight, previous=None):
    kh, vh = config['linear_num_key_heads'], config['linear_num_value_heads']
    kd, vd = config['linear_key_head_dim'], config['linear_value_head_dim']
    raw = get('linear.output','LinearAttentionQkv')
    rows, channels = raw.shape
    conv_w = weight('conv1d.weight')
    history = raw.new_zeros(1,channels,conv_w.shape[-1]) if previous is None else previous[0]
    joined = torch.cat((history,raw.T[None]),dim=-1)
    conv = F.silu(F.conv1d(joined,conv_w,groups=channels))[:,:,-rows:].transpose(1,2)
    observed = get('gdn.convolved').reshape(1,rows,channels)
    query,key,value = observed.split([kh*kd,kh*kd,vh*vd],-1)
    query = query.reshape(1,rows,kh,kd).repeat_interleave(vh//kh,2)
    key = key.reshape(1,rows,kh,kd).repeat_interleave(vh//kh,2)
    value = value.reshape(1,rows,vh,vd)
    a,b = get('linear.output','LinearAttentionA'),get('linear.output','LinearAttentionBeta')
    g = -weight('A_log').exp()*F.softplus(a+weight('dt_bias'))
    initial = None if previous is None else previous[1].clone()
    fn = q.torch_chunk_gated_delta_rule if previous is None else q.torch_recurrent_gated_delta_rule
    output,state = fn(query,key,value,g.reshape(1,rows,vh),b.sigmoid().reshape(1,rows,vh),
                      initial_state=initial,output_final_state=True,use_qk_l2norm_in_kernel=True)
    rec = get('gdn.recurrent_output').reshape(-1,vd)
    norm = rec*torch.rsqrt(rec.square().mean(-1,keepdim=True)+config['rms_norm_eps'])*weight('norm.weight')
    gated = get('gdn.normalized').reshape(-1,vd)*F.silu(get('linear.output','LinearAttentionZ').reshape(-1,vd))
    return [('conv+silu',observed,conv), ('conv_state',get('gdn.conv_state'),joined[:,:,-conv_w.shape[-1]:]),
            ('recurrent_output',get('gdn.recurrent_output'),output),('recurrent_state',get('gdn.recurrent_state'),state),
            ('norm',get('gdn.normalized'),norm),('gate',get('gdn.gated'),gated)]



class SameInputReplay:
    """Independent Torch math on captured operands; never a Ferrule kernel call."""
    def __init__(self, device='cpu', authorized=False):
        require(device=='cpu' or (authorized and device.startswith('cuda:')), 'CUDA replay requires --execute-cuda')
        self.device=torch.device(device)
        if self.device.type=='cuda':
            os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False
            torch.use_deterministic_algorithms(True)

    def tensor(self,value):
        return value.to(self.device)

    def linear(self,x,w):
        return F.linear(self.tensor(x),self.tensor(w))

    def norm(self,x,w,epsilon,offset):
        x,w=self.tensor(x),self.tensor(w)
        return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+epsilon)*(w+int(offset))

    def silu_mul(self,x,y):
        return F.silu(self.tensor(x))*self.tensor(y)

    def sigmoid_mul(self,x,y):
        return self.tensor(x)*self.tensor(y).sigmoid()

    def add(self,x,y):
        return self.tensor(x)+self.tensor(y)

    def native_combine(self,table,ids,weights):
        # HF iterates experts in ID order, materializes weighted outputs, then
        # index_add_. Keep that formula; do not copy production route-slot fold.
        table,weights=self.tensor(table),self.tensor(weights)
        output=torch.zeros_like(table[:,0])
        for expert in sorted(set(ids.flatten().tolist())):
            row,slot=torch.where(ids==expert)
            row,slot=row.to(self.device),slot.to(self.device)
            value=table[row,slot]*weights[row,slot,None]
            output.index_add_(0,row,value)
        return output


def require_full40_capture(events):
    expected={(stage,layer) for stage in ('prefill','decode.0') for layer in range(40)}
    actual={(e['stage'],e['layer']) for e in events if e['layer'] is not None}
    require(actual==expected,'Expected all 80 layer-stage pairs; partial capture is not complete')
    for stage,layer in sorted(expected):
        group=[e for e in events if (e['stage'],e['layer'])==(stage,layer)]
        require(sum(e['event']=='router' for e in group)==1,'Expected one router per layer-stage')
        require(sum(e['event']=='residual.output' for e in group)==3,'Missing layer residuals')
        roles={e['role'] for e in group if e['event']=='norm.output'}
        require({'AttentionNorm','FeedForwardNorm'}<=roles,'Missing layer norms')
        if layer%4!=3:
            names={e['event'] for e in group}
            require({'gdn.conv_state','gdn.recurrent_state','gdn.convolved','gdn.recurrent_output','gdn.normalized','gdn.gated'}<=names,'Missing GDN state/operator capture')
        else:
            require({'AttentionQueryNorm','AttentionKeyNorm'}<=roles,'Missing attention head norms')
    for stage in ('prefill','decode.0'):
        root=[e for e in events if e['stage']==stage and e['layer'] is None]
        require(any(e['event']=='embedding' for e in root),'Missing embedding')
        for role in ('OutputNorm','OutputHead'):
            require(any(e['role']==role and e['event'].endswith('.output') for e in root),'Missing root operator')


def cause_locator(report):
    def first(rows,key):
        rows=sorted((r for r in rows if r.get('sequence') is not None),key=lambda r:r['sequence'])
        return next((r for r in rows if r[key]),None)
    return dict(first_strict_captured_boundary=first(report['boundaries'],'outside'),
                first_strict_same_input_operator=first(report['same_input'],'outside'),
                first_native_bitwise_output_difference=next((r for r in sorted(report['native_module_comparisons'],key=lambda r:r.get('sequence',10**12)) if r['output']['unequal']),None),
                native_changed_input=next((r for r in sorted(report['native_module_comparisons'],key=lambda r:r.get('sequence',10**12)) if not r['rounded_operands_identical']),None),
                first_same_input_route_failure=next((r for r in sorted(report['routes'],key=lambda r:r['sequence']) if not r['same_set'] or r['same_selected_weights']['outside']),None),
                route_watch=[r for r in report['routes'] if r['layer'] in (13,20,29,30)],
                decode_state=[r for r in report['same_input'] if r['stage']=='decode.0' and r['operation'] in ('gdn.conv_state','gdn.recurrent_state')],
                state_publication_failures=[r for r in report.get('state_publication',[]) if r['unequal']],
                physical_kv_captured=False)


def analyze(args):
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    replay=SameInputReplay(getattr(args,'device','cpu'),getattr(args,'execute_cuda',False))
    manifest, native_report, oracle_path = validate_reference(args.reference,args.reference_manifest_sha256,args.case)
    q = native_module(manifest)
    native_trace = NativeTrace(args.reference, next(e for e in manifest['cases'] if e['id']==args.case).get('operator_trace'))
    config = manifest['checkpoint']['effective_text_config']
    provenance = args.reference/manifest['source_payloads']['file']
    require(sha256(provenance) == manifest['source_payloads']['sha256'], 'Source provenance pin mismatch')
    source = json.loads(provenance.read_text())
    events = [json.loads(line) for line in (args.trace/'events.jsonl').read_text().splitlines()]
    require(events and all(e['precision']=='bf16_rne' for e in events),'Wrong trace precision')
    capture_path = args.trace/'capture.json'
    if capture_path.exists():
        capture = json.loads(capture_path.read_text())
        require(capture['status']=='complete','Incomplete production capture')
        require(capture['independent_cuda_reference']['manifest_sha256']==args.reference_manifest_sha256,'Capture used different CUDA reference')
        require(len(capture['calls'])==len(native_report['calls'])==2,'Incomplete capture calls')
        require(sha256(args.trace/'events.jsonl')==capture['events_sha256'],'Production trace index changed')
        for filename,digest in capture['payload_sha256'].items():
            path=args.trace/filename
            require(path.resolve().parent==args.trace.resolve() and sha256(path)==digest,'Production payload changed')
        for a,b in zip(capture['calls'],native_report['calls']):
            require(a['input_token_ids']==b['input_token_ids'] and a['stage']==b['stage'],'Token trajectory mismatch')
    if getattr(args,'require_full40',False):
        require(capture_path.exists(),'Full40 gate requires a completed pinned production capture')
        require_full40_capture(events)
        require(bool(native_trace.events),'Full40 gate requires pinned native --diagnostic-modules trace')
    groups = {}
    require([e['sequence'] for e in events]==list(range(len(events))),'Noncontiguous trace sequence')
    for event in events:
        groups.setdefault((event['stage'],event['layer']),[]).append(event)
    report = dict(schema='ferrule.w8a16-causal-diagnosis.v1',atol=ATOL,rtol=RTOL,case=args.case,
                  reference_manifest_sha256=args.reference_manifest_sha256,trace_events_sha256=sha256(args.trace/'events.jsonl'),
                  scope='Captured-operand independent Torch replay + pinned native boundaries; not model quality acceptance',
                  replay_device=str(replay.device),cpu_replay_is_not_cuda=replay.device.type=='cpu',
                  boundaries=[],same_input=[],routes=[],state_publication=[],rounding=[],combine=[],coverage=[],weight_hashes={},operand_attribution=[],native_module_comparisons=[])
    store = ref.TensorStore(manifest['checkpoint']['directory'])
    require(sha256(store.index_path)==source['index_sha256'], 'Checkpoint index mismatch')
    require(sha256(Path(manifest['checkpoint']['directory'])/'config.json')==manifest['checkpoint']['config_sha256'],'Config changed')
    previous_gdn,previous_kv = {},{}
    def read(e):
        return read_event(args.trace,e)
    def record(stage,layer,op,a,e,**extra):
        report['same_input'].append(dict(stage=stage,layer=layer,operation=op,replay_device=str(replay.device),**metric(a,e),**extra))
    def bound(stage,layer,key,a,oracle,sequence):
        if key in oracle.keys():
            report['boundaries'].append(dict(stage=stage,layer=layer,key=key,sequence=sequence,**metric(a,oracle.get_tensor(key))))
    def verified_tensor(meta):
        value=tensor(meta)
        name=meta['name']
        digest=hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        known=source['used_payloads'].get(name)
        if known is not None:
            require(known['sha256']==digest, 'Checkpoint payload differs from CUDA reference: '+name)
        else:
            # Changed routing may use an expert absent from the independent run.
            # Validate against the checkpoint index, and disclose additional payloads.
            current=store.read(name)
            require(current.shape==value.shape and current.dtype==value.dtype and store.reads[name]['sha256']==digest,
                    'Production trace/checkpoint slice mismatch: '+name)
            report.setdefault('additional_checkpoint_payloads',{})[name]=digest
        report['weight_hashes'][name]=digest
        return value
    try:
        with safe_open(oracle_path,framework='pt',device='cpu') as oracle:
            for stage in ('prefill','decode.0'):
                call=next(c for c in native_report['calls'] if c['stage']==stage)
                start=call['position_start']
                for e in groups.get((stage,None),[]):
                    if e['event']=='embedding':
                        bound(stage,None,stage+'.embedding',read(e),oracle,e['sequence'])
                    elif e['event']=='norm.output' and e['role']=='OutputNorm':
                        bound(stage,None,stage+'.final_norm',read(e),oracle,e['sequence'])
                for layer in range(40):
                    es=groups.get((stage,layer),[])
                    if not es:
                        continue
                    prefix=f'{stage}.layers.{layer:02}'
                    def one(name,role=None):
                        matches=[e for e in es if e['event']==name and (role is None or e['role']==role)]
                        require(len(matches)==1,f'Expected unique {stage}/{layer}/{name}/{role}, got {len(matches)}')
                        return matches[0]
                    def get(name,role=None):
                        return read(one(name,role))
                    residuals=[e for e in es if e['event']=='residual.output']
                    updates=[e for e in es if e['event']=='residual.update']
                    require(len(residuals)==len(updates)==3,'Unexpected residual topology')
                    for key,e in [('attention',updates[0]),('moe',updates[2]),('hidden',residuals[2]),
                                  ('shared_gate_logit',one('linear.output','SharedExpertOutputGate'))]:
                        bound(stage,layer,prefix+'.'+key,read(e),oracle,e['sequence'])
                    inputs={}
                    native_router_logits=None
                    last_residual=None
                    gate_output=up_output=down_output=None
                    route=one('router')
                    ids=torch.tensor(route['ids']).reshape(len(get('router')),-1)
                    weights=torch.tensor(route['weights']).reshape_as(ids)
                    table=torch.zeros(*ids.shape,config['hidden_size'])
                    installed=torch.zeros_like(ids,dtype=torch.bool)
                    for e in es:
                        name=e['event']; a=read(e)
                        if name in ('linear.input','norm.input'):
                            inputs[e['parameter']]=a
                        elif name=='linear.output':
                            x=inputs[e['parameter']]
                            raw=verified_tensor(e['weight'])
                            require(e['numeric']==(raw.dtype==torch.float8_e4m3fn),'Numeric/nonquantized activation contract mismatch')
                            w=ref.independent_dequantize(raw,verified_tensor(e['scale'])) if e['numeric'] else raw.float()
                            operand=ref.independent_bf16_rne(x) if e['numeric'] else x
                            expected=replay.linear(operand,w)
                            record(stage,layer,e['parameter'],a,expected,numeric=e['numeric'],sequence=e['sequence'],
                                   f64=metric(a,F.linear(operand.double(),w.double()).float()),
                                   operator_bound=operator_bound(a,operand,w) if e['numeric'] else None)
                            if replay.device.type=='cpu' and e['role'] in ('LinearAttentionQkv','SharedExpertGate') and e['numeric'] and layer<3:
                                order_replay=packed_vector_reduction(operand,w)
                                report['rounding'].append(dict(stage=stage,layer=layer,parameter=e['parameter'],
                                    production_order_replay=metric(a,order_replay),cpu_vs_production=metric(expected,a),
                                    note='replay is diagnostic, never an independent reference'))
                            module=e['weight']['name'].replace('model.language_model.','model.').removesuffix('.weight')
                            if e['role']=='RouterLogits':
                                native_router_input=native_trace.get(stage+'.trace.'+module+'.input.0')
                                if native_router_input is not None:
                                    native_router_logits=replay.linear(native_router_input.reshape_as(x),w).detach().cpu()
                            if e['role']!='RouterLogits':
                                comparison=native_trace.compare(stage+'.trace.'+module,x,a,e['numeric'])
                                if comparison is not None:
                                    report['native_module_comparisons'].append(dict(stage=stage,layer=layer,sequence=e['sequence'],parameter=e['parameter'],**comparison))
                            if stage=='prefill' and e['role']=='LinearAttentionQkv':
                                ne=one('norm.output','AttentionNorm')
                                hidden=oracle.get_tensor('prefill.embedding' if layer==0 else f'prefill.layers.{layer-1:02}.hidden').reshape(-1,x.shape[-1])
                                affine=verified_tensor(ne['weight']).float()+int(ne['one_plus_weight'])
                                native_x=hidden*torch.rsqrt(hidden.square().mean(-1,keepdim=True)+ne['epsilon'])*affine
                                delta=F.linear((operand-ref.independent_bf16_rne(native_x)).double(),w.double())
                                rows=min(len(a),config['linear_conv_kernel_dim'])
                                target=oracle.get_tensor(f'prefill.cache.conv_states.{layer:02}')[0,:,-rows:].T
                                report['operand_attribution'].append(dict(stage=stage,layer=layer,
                                    provenance='CPU norm reconstruction of captured CUDA hidden; NOT native CUDA norm capture',
                                    rows_compared=list(range(len(a)-rows,len(a))),
                                    changed_bf16_inputs=int((operand!=ref.independent_bf16_rne(native_x)).sum()),
                                    actual_vs_cuda=metric(a[-rows:],target),
                                    after_subtracting_operand_delta=metric(a[-rows:].double()-delta[-rows:],target)))
                            if e['role'] in ('SharedExpertGate','RoutedExpertGate'): gate_output=a
                            if e['role'] in ('SharedExpertUp','RoutedExpertUp'): up_output=a
                            if e['role'] in ('SharedExpertDown','RoutedExpertDown'):
                                down_output=a
                                record(stage,layer,e['role']+'.silu_mul',x,replay.silu_mul(gate_output,up_output),sequence=e['sequence'])
                            if e['role']=='RoutedExpertDown':
                                expert=int(e['weight']['name'].split('.experts.')[1].split('.')[0])
                                locations=torch.nonzero(ids==expert)
                                require(len(locations)==len(a),'Expert bucket/route cardinality mismatch')
                                for j,(row,slot) in enumerate(locations.tolist()):
                                    require(not installed[row,slot],'Duplicate expert route output')
                                    table[row,slot]=a[j]; installed[row,slot]=True
                        elif name=='swiglu.output':
                            require(down_output is not None,'SwiGLU output without down projection')
                            record(stage,layer,'swiglu.output_identity',a,replay.tensor(down_output),sequence=e['sequence'])
                        elif name=='norm.output':
                            w=verified_tensor(e['weight']).float()
                            x=inputs[e['parameter']].reshape(-1,w.numel())
                            n=replay.norm(x,w,e['epsilon'],e['one_plus_weight'])
                            record(stage,layer,e['parameter'],a,n,sequence=e['sequence'])
                            module=e['weight']['name'].replace('model.language_model.','model.').removesuffix('.weight')
                            comparison=native_trace.compare(stage+'.trace.'+module,x,a,False)
                            if comparison is not None:
                                report['native_module_comparisons'].append(dict(stage=stage,layer=layer,sequence=e['sequence'],parameter=e['parameter'],**comparison))
                        elif name=='residual.input': last_residual=a
                        elif name=='residual.update': update=a
                        elif name=='residual.output': record(stage,layer,'residual.add',a,replay.add(last_residual,update),sequence=e['sequence'])
                    require(bool(installed.all()),'Missing expert outputs; cannot diagnose combine')
                    actual=get('routed.output')
                    slot_sum=combine(table,ids,weights)
                    expert_sum=combine(table,ids,weights,expert_order=True)
                    report['combine'].append(dict(stage=stage,layer=layer,production_slot_order=metric(actual,slot_sum),
                        native_expert_id_order=metric(actual,expert_sum),
                        f64_weighted_sum=metric(actual,(table.double()*weights.double()[...,None]).sum(1).float())))
                    record(stage,layer,'moe.combine.native_expert_order',actual,replay.native_combine(table,ids,weights),sequence=one('routed.output')['sequence'])
                    record(stage,layer,'shared.sigmoid_gate',get('shared_gate.output'),
                           replay.sigmoid_mul(get('linear.output','SharedExpertDown'),get('linear.output','SharedExpertOutputGate')),sequence=one('shared_gate.output')['sequence'])
                    r=route_metrics(replay.tensor(get('router')),ids,weights)
                    ref_ids=oracle.get_tensor(prefix+'.selected_expert_ids')
                    top9=get('router').topk(9,dim=-1)
                    r.update(sequence=route['sequence'],replay_device=str(replay.device),raw_top9_ids=top9.indices.tolist(),raw_top9_logits=top9.values.tolist(),raw_boundary_gap=(top9.values[:,7]-top9.values[:,8]).tolist(),stage=stage,layer=layer,vs_cuda_set_mismatch_rows=int((ids.sort(-1).values!=ref_ids.sort(-1).values).any(-1).sum()))
                    if native_router_logits is not None:
                        ntop=native_router_logits.topk(9,dim=-1)
                        r['native_raw_logits_replay']=dict(provenance='independent replay on captured native router input; raw logits are not a router hook output',
                            replay_device=str(replay.device),vs_production=metric(get('router'),native_router_logits),
                            top9_ids=ntop.indices.tolist(),top9_logits=ntop.values.tolist(),
                            boundary_gap=(ntop.values[:,7]-ntop.values[:,8]).tolist())
                    report['routes'].append(r)
                    if config['layer_types'][layer]=='linear_attention':
                        def weight(suffix):
                            name=f'{ref.PREFIX}layers.{layer}.linear_attn.{suffix}'
                            value=store.read(name).float()
                            require(store.reads[name]['sha256']==source['used_payloads'][name]['sha256'],'GDN weight pin mismatch')
                            report['weight_hashes'][name]=store.reads[name]['sha256']
                            return value
                        previous=previous_gdn.get(layer) if stage=='decode.0' else None
                        require(stage=='prefill' or previous is not None,'Decode GDN lacks captured prefill state')
                        gpu_previous=None if previous is None else tuple(replay.tensor(v) for v in previous)
                        gdn_events={'conv+silu':'gdn.convolved','conv_state':'gdn.conv_state','recurrent_output':'gdn.recurrent_output','recurrent_state':'gdn.recurrent_state','norm':'gdn.normalized','gate':'gdn.gated'}
                        for name,a,expected_gdn in gdn_same_input(q,config,lambda *v:replay.tensor(get(*v)),lambda n:replay.tensor(weight(n)),gpu_previous):
                            record(stage,layer,'gdn.'+name,a,expected_gdn,sequence=one(gdn_events[name])['sequence'],initial_state='zero' if previous is None else 'captured production prefill state, not reference state')
                        shape=(1,config['linear_num_value_heads'],config['linear_key_head_dim'],config['linear_value_head_dim'])
                        published=[]
                        for field,name,target_shape in [('conv_states','gdn.conv_state',(1,-1,config['linear_conv_kernel_dim'])),('recurrent_states','gdn.recurrent_state',shape)]:
                            key=f'{stage}.cache.{field}.{layer:02}'
                            observed=get(name)
                            snapshot=args.trace/(key+'.f32')
                            if snapshot.exists():
                                committed=torch.frombuffer(bytearray(snapshot.read_bytes()),dtype=torch.float32).reshape_as(observed)
                                report['state_publication'].append(dict(stage=stage,layer=layer,key=key,sequence=one(name)['sequence'],**metric(committed,observed)))
                            else:
                                require(not getattr(args,'require_full40',False),'Missing committed GDN snapshot')
                                committed=observed
                            published.append(committed.reshape(target_shape))
                            bound(stage,layer,key,committed,oracle,one(name)['sequence'])
                        previous_gdn[layer]=tuple(published)
                    else:
                        history=previous_kv.get(layer) if stage=='decode.0' else None
                        require(stage=='prefill' or history is not None,'Decode attention lacks prefill operands')
                        expected,kv=attention_same_input(q,config,lambda *v:replay.tensor(get(*v)),start,history)
                        record(stage,layer,'attention.rope_gqa_sigmoid',get('linear.input','AttentionOutput'),expected,sequence=one('linear.input','AttentionOutput')['sequence'],
                               history='none' if history is None else 'Replay-device RoPE reconstruction from captured prefill K/V, NOT physical KV')
                        previous_kv[layer]=kv
                        for field,value in zip(('key_cache','value_cache'),kv):
                            report['coverage'].append(dict(stage=stage,layer=layer,field=field,
                                kind='Replay-device reconstructed operands, NOT captured physical KV',
                                vs_cuda=metric(value,oracle.get_tensor(f'{stage}.cache.{field}.{layer:02}'))))
                root_inputs={}
                for e in groups.get((stage,None),[]):
                    role,name=e.get('role'),e['event']
                    if role not in ('OutputNorm','OutputHead'):
                        continue
                    value=read(e)
                    if name.endswith('.input'):
                        root_inputs[e['parameter']]=value
                        continue
                    if not name.endswith('.output'):
                        continue
                    x=root_inputs[e['parameter']]
                    if role=='OutputNorm':
                        w=verified_tensor(e['weight']).float()
                        expected=replay.norm(x,w,e['epsilon'],e['one_plus_weight'])
                        record(stage,None,role,value,expected,sequence=e['sequence'])
                    else:
                        # Stream the output head by checkpoint rows: never allocate a
                        # vocabulary-sized F64 weight copy for diagnostic summation.
                        raw_meta=e['weight']; scale=verified_tensor(e['scale']) if e['numeric'] else None
                        require(raw_meta['shape'][1]==x.shape[1],'LM head width mismatch')
                        operand=ref.independent_bf16_rne(x) if e['numeric'] else x
                        dtype_bytes={'F8E4M3':1,'Bf16':2,'F32':4}[raw_meta['dtype']]
                        digest=hashlib.sha256()
                        full_weight=torch.empty(raw_meta['shape'],dtype=torch.float32,device=replay.device) if replay.device.type=='cuda' else None
                        for row in range(0,raw_meta['shape'][0],256):
                            count=min(256,raw_meta['shape'][0]-row)
                            meta=dict(raw_meta,shape=[count,x.shape[1]],offset=raw_meta['offset']+row*x.shape[1]*dtype_bytes,bytes=count*x.shape[1]*dtype_bytes)
                            raw=tensor(meta)
                            digest.update(raw.contiguous().view(torch.uint8).numpy().tobytes())
                            w=ref.independent_dequantize(raw,scale[row//128:(row+count+127)//128]) if e['numeric'] else raw.float()
                            actual=value[:,row:row+count]
                            if full_weight is not None:
                                full_weight[row:row+count].copy_(replay.tensor(w))
                            else:
                                record(stage,None,role+f'.rows.{row}:{row+count}',actual,replay.linear(operand,w),sequence=e['sequence'],
                                       f64=metric(actual,F.linear(operand.double(),w.double()).float()),
                                       operator_bound=operator_bound(actual,operand,w) if e['numeric'] else None)
                        if full_weight is not None:
                            record(stage,None,role,value,F.linear(replay.tensor(operand),full_weight),sequence=e['sequence'],
                                   shape_policy='full captured M/N/K on CUDA; F32 weights streamed to bounded full head allocation')
                            del full_weight
                        require(digest.hexdigest()==source['used_payloads'][raw_meta['name']]['sha256'],'LM head hash mismatch')
                        report['weight_hashes'][raw_meta['name']]=digest.hexdigest()
                    module=e['weight']['name'].replace('model.language_model.','model.').removesuffix('.weight')
                    comparison=native_trace.compare(stage+'.trace.'+module,x,value,e['numeric'])
                    if comparison is not None:
                        report['native_module_comparisons'].append(dict(stage=stage,layer=None,sequence=e['sequence'],parameter=e['parameter'],**comparison))
                logits_path=args.trace/f'{stage}.logits.f32'
                if logits_path.exists():
                    logits=torch.frombuffer(bytearray(logits_path.read_bytes()),dtype=torch.float32)
                    bound(stage,None,stage+'.logits',logits,oracle,max(e['sequence'] for e in events if e['stage']==stage)+0.5)
        report['boundaries'].sort(key=lambda value:value['sequence'])
        report['first_strict_cuda_boundary']=next((v for v in report['boundaries'] if v['outside']),None)
        report['summary']=dict(layer_stage_pairs_observed=len([k for k in groups if k[1] is not None]),
             same_input_checks=len(report['same_input']),same_input_failed_checks=sum(v['outside']>0 for v in report['same_input']),
             combine_slot_order_failed=sum(v['production_slot_order']['outside']>0 for v in report['combine']),
             route_same_input_failed=sum(not v['same_set'] or v['same_selected_weights']['outside']>0 for v in report['routes']),
             physical_kv_captured=False, all_operators_bitwise_certified=False,
             capture_scope='first available captured boundaries; not unobserved primitive first-cause proof')
        report['trace_payload_hashes']={e['file']:sha256(args.trace/e['file']) for e in events}
        report['diagnostic_script_sha256']=sha256(__file__)
        report['native_operator_trace_available']=bool(native_trace.events)
        required_events={'linear.output','norm.output','residual.output','swiglu.output','shared_gate.output','routed.output',
                         'gdn.convolved','gdn.recurrent_output','gdn.conv_state','gdn.recurrent_state','gdn.normalized','gdn.gated'}
        expected_sequences={e['sequence'] for e in events if e['event'] in required_events}
        replayed_sequences={r['sequence'] for r in report['same_input'] if r.get('sequence') is not None}
        report['replay_coverage']=dict(required_events=len(expected_sequences),covered_events=len(expected_sequences & replayed_sequences),
            missing_sequences=sorted(expected_sequences-replayed_sequences),router_layer_stages=len(report['routes']))
        if getattr(args,'require_full40',False):
            require(not report['replay_coverage']['missing_sequences'] and len(report['routes'])==80,'Incomplete per-operator replay')
        report['cause_locator']=cause_locator(report)
        report['cuda_initialized']=torch.cuda.is_initialized()
        require(replay.device.type=='cuda' or not report['cuda_initialized'],'CPU diagnosis initialized CUDA')
        return report
    finally:
        store.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--reference-manifest-sha256',required=True)
    parser.add_argument('--case',choices=('hello','capital'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--execute-cuda',action='store_true')
    parser.add_argument('--require-full40',action='store_true')
    args=parser.parse_args()
    require(not args.output.exists(),'Refuse to overwrite evidence')
    report=analyze(args)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(summary=report['summary'],first_strict_cuda_boundary=report['first_strict_cuda_boundary']),indent=2))


if __name__=='__main__':
    main()
