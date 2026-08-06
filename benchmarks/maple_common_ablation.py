#!/usr/bin/env python3
"""Diagnose which strict component first changes a fixed common-slice decode."""
import argparse
import json
from pathlib import Path

from maple_common_slice_benchmark import prompt_for, run

from mlx_lm import load
from mlx_lm.models import maple


def configure(model, *, add=False, qk=False, lhs=False):
    model.model._fused_add_norm = add
    for layer in model.model.layers:
        layer.self_attn._fused_qk = qk
        layer.mlp.gate._fused = False
    maple._use_cached_decode_lhs = lhs
    maple._cuda_router_indices_uint32 = False
    maple._use_cuda_ternary_up_gate = False
    maple._use_approximate_router = False
    # `add=True` deliberately enters the semantic path for diagnosis.
    maple._use_approximate_add_rms = add
    maple._decode_lhs_cache.clear()


def mismatch(a, b):
    if a == b:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-tokens',type=int,default=512)
    args=p.parse_args()
    case=json.loads(args.manifest.read_text())['cases'][0]
    model,tokenizer,_config=load(
        str(args.model), return_config=True,
        model_config={'model_file':None,'use_flash_head':False},
        tokenizer_config={'trust_remote_code':True}, trust_remote_code=True,
    )
    prompt=tokenizer.apply_chat_template(prompt_for(case),tokenize=True,add_generation_prompt=True)
    modes=[
        ('reference-1',False,False,False),
        ('reference-2',False,False,False),
        ('lhs',False,False,True),
        ('add',True,False,False),
        ('qk',False,True,False),
        ('add-qk',True,True,False),
        ('all-strict',True,True,True),
        ('reference-after',False,False,False),
    ]
    records=[]; baseline=None
    for name,add,qk,lhs in modes:
        configure(model,add=add,qk=qk,lhs=lhs)
        result=run(model,tokenizer,prompt,args.max_tokens)
        if baseline is None: baseline=result['tokens']
        record={
            'mode':name,'add':add,'qk':qk,'lhs':lhs,
            'tokens_equal':result['tokens']==baseline,
            'first_token_mismatch':mismatch(baseline,result['tokens']),
            'generated_tokens':len(result['tokens']),
            **{k:v for k,v in result.items() if k not in ('tokens','text')},
            'tokens':result['tokens'],
        }
        records.append(record); print(json.dumps({k:v for k,v in record.items() if k!='tokens'},sort_keys=True),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records))

if __name__=='__main__': main()
