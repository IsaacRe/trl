"""Micro-benchmark: isolate the prefill-once-then-expand speedup vs the naive
n-copies approach (prefill n identical context copies). Same decode work for both;
the only difference is prefilling B contexts vs B*n contexts per draft call."""
import sys, time, itertools
sys.path.insert(0, "scripts")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import speculative_eval as se
from train_sft import remap_roles
from trl.data_utils import maybe_convert_to_chatml
from datasets import load_dataset

CKPT = "/home/isaac/trl/outputs/Qwen3_1.7B-kimi_agent_sft-r256-lr2e-4/checkpoint-1"
PARQUET = "/home/isaac/trl/data/kimi-k2.6-claude-code-traces.parquet"
N, D, B = 4, 8, 4

tok = AutoTokenizer.from_pretrained(CKPT)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16, attn_implementation="sdpa")
model = PeftModel.from_pretrained(base, CKPT).to("cuda").eval()
dev = "cuda"

# B real contexts of differing length.
ds = load_dataset("parquet", data_files=PARQUET, streaming=True, split="train")
samples = [maybe_convert_to_chatml(remap_roles(dict(s))) for s in itertools.islice(ds, 0, B)]
contexts = []
for s in samples:
    msgs = s["messages"]
    for m in msgs:
        if m["role"] == "assistant":
            m["content"] = se._qwen3_convert_glm_think_blocks(m["content"])
    turns = se._assistant_turns(msgs, tok)
    # deepest turn -> longest realistic context (includes all prior turns),
    # capped to keep the naive n-copies variant from OOMing for the comparison.
    ctx = max((tok(t[0]).input_ids for t in turns), key=len)
    contexts.append(ctx[-4000:])
print("context lengths:", [len(c) for c in contexts])

def reqs():
    return [{"context_ids": c, "n": N, "d": D, "temperature": 0.6, "seed": 1000 + i} for i, c in enumerate(contexts)]

def n_copies_draft(requests):
    """Naive: prefill n identical copies of each context (the pre-fix behavior)."""
    row_ctx, offs, o = [], [], 0
    for req in requests:
        offs.append((o, o + req["n"])); row_ctx += [req["context_ids"]] * req["n"]; o += req["n"]
    rows = o; d = requests[0]["d"]
    L = [len(c) for c in row_ctx]; mx = max(L)
    pad = tok.pad_token_id
    ids = torch.full((rows, mx), pad, dtype=torch.long, device=dev)
    att = torch.zeros((rows, mx), dtype=torch.long, device=dev)
    for r, (c, l) in enumerate(zip(row_ctx, L)):
        ids[r, mx-l:] = torch.tensor(c, device=dev); att[r, mx-l:] = 1
    pos = att.long().cumsum(-1).sub_(1).clamp_(min=0)
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=True, logits_to_keep=1)
        past = out.past_key_values; logits = out.logits[:, -1, :]; cp = pos[:, -1]
        for t in range(d):
            tk = logits.argmax(-1)
            if t == d-1: break
            cp = cp + 1; att = torch.cat([att, torch.ones((rows,1),dtype=torch.long,device=dev)],1)
            out = model(input_ids=tk[:,None], attention_mask=att, position_ids=cp[:,None], past_key_values=past, use_cache=True, logits_to_keep=1)
            past = out.past_key_values; logits = out.logits[:,-1,:]
    return

def timed(fn, iters=5):
    fn(); torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.time()-t)/iters*1000

t_new = timed(lambda: se._draft_batch(model, tok, reqs(), dev))
t_old = timed(lambda: n_copies_draft(reqs()))
print(f"\nper draft-call (B={B}, n={N}, d={D}):")
print(f"  n-copies (prefill {B*N} rows): {t_old:7.1f} ms")
print(f"  prefill-once (prefill {B} rows + expand): {t_new:7.1f} ms")
print(f"  speedup: {t_old/t_new:.2f}x")
