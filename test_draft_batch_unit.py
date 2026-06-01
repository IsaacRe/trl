"""Isolate _draft_batch correctness: does padding/position handling introduce only
float noise, or a logic bug? Draft the SAME context greedily, solo vs batched
behind a longer context. With correct masking, the short context's drafts should
be identical (or differ only by tiny bf16 batched-matmul noise)."""
import sys
sys.path.insert(0, "scripts")
import itertools
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import speculative_eval as se
from train_sft import remap_roles
from trl.data_utils import maybe_convert_to_chatml
from datasets import load_dataset

CKPT = "/home/isaac/trl/outputs/Qwen3_1.7B-kimi_agent_sft-r256-lr2e-4/checkpoint-1"
PARQUET = "/home/isaac/trl/data/kimi-k2.6-claude-code-traces.parquet"

tok = AutoTokenizer.from_pretrained(CKPT)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16, attn_implementation="sdpa")
model = PeftModel.from_pretrained(base, CKPT).to("cuda").eval()
dev = "cuda"

# Two real contexts of very different lengths.
ds = load_dataset("parquet", data_files=PARQUET, streaming=True, split="train")
samples = [maybe_convert_to_chatml(remap_roles(dict(s))) for s in itertools.islice(ds, 0, 2)]
def first_ctx(sample):
    msgs = sample["messages"]
    for m in msgs:
        if m["role"] == "assistant":
            m["content"] = se._qwen3_convert_glm_think_blocks(m["content"])
    turns = se._assistant_turns(msgs, tok)
    return tok(turns[0][0]).input_ids

ctx_short = first_ctx(samples[0])
ctx_long  = first_ctx(samples[1])
if len(ctx_short) > len(ctx_long):
    ctx_short, ctx_long = ctx_long, ctx_short
print(f"short ctx len={len(ctx_short)}  long ctx len={len(ctx_long)}  (pad gap={len(ctx_long)-len(ctx_short)})")

def req(ctx, seed):
    return {"context_ids": ctx, "n": 4, "d": 8, "temperature": 0.0, "seed": seed}

torch.manual_seed(0)
solo    = se._draft_batch(model, tok, [req(ctx_short, 123)], dev)[0]
batched = se._draft_batch(model, tok, [req(ctx_long, 999), req(ctx_short, 123)], dev)[1]

mismatch = sum(1 for a, b in zip(solo, batched) for x, y in zip(a, b) if x != y)
total = sum(len(a) for a in solo)
print(f"greedy short-ctx drafts: {mismatch}/{total} token positions differ solo-vs-batched")
for di, (a, b) in enumerate(zip(solo, batched)):
    if a != b:
        print(f"  draft {di}: solo={a}\n            batched={b}")

# Logit-level: how big is the per-token logit perturbation from padding/batching?
def last_logits(ctx, batch_ctxs):
    ids = [c for c in batch_ctxs]
    maxL = max(len(c) for c in ids)
    inp = torch.full((len(ids), maxL), tok.pad_token_id, dtype=torch.long, device=dev)
    att = torch.zeros((len(ids), maxL), dtype=torch.long, device=dev)
    for r, c in enumerate(ids):
        inp[r, maxL-len(c):] = torch.tensor(c, device=dev)
        att[r, maxL-len(c):] = 1
    pos = att.long().cumsum(-1).sub_(1).clamp_(min=0)
    with torch.inference_mode():
        out = model(input_ids=inp, attention_mask=att, position_ids=pos, use_cache=True, logits_to_keep=1)
    idx = ids.index(ctx)
    return out.logits[idx, -1, :].float()

lg_solo = last_logits(ctx_short, [ctx_short])
lg_batch = last_logits(ctx_short, [ctx_long, ctx_short])
print(f"max|Δ logit| short ctx, solo vs padded-batched = {(lg_solo - lg_batch).abs().max().item():.3e}")
print(f"argmax solo={lg_solo.argmax().item()} batched={lg_batch.argmax().item()}")
