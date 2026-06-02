"""
Validation harness for batched speculative-decoding eval.

Two levels:

  --mode mock   Pure-Python drafter (no GPU). Proves the async batcher computes
                the *same logical result* at any batch size: the per-request seed
                depends only on (sample, turn, step), so drafts — and therefore
                every metric — must be bit-identical across batch sizes. This
                isolates orchestration correctness from GPU float nondeterminism.

  --mode gpu    Real Qwen3-1.7B + LoRA adapter. Establishes the deterministic
                unbatched baseline (batch_size=1) and compares batched runs to it.
                At batch_size=1 the computation is identical to the baseline
                (exact); at batch_size>1 only batched-matmul float differences
                remain, which we measure and report.

Usage:
  PY=/home/isaac/trl/.venv/bin/python
  $PY test_batched_specdec.py --mode mock
  $PY test_batched_specdec.py --mode gpu --ckpt outputs/.../checkpoint-1 \
       --n-samples 3 --batch-sizes 1,4 --temperature 0.6
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import speculative_eval as se
from speculative_eval import SpecDecEvalEntry, SpeculativeAcceptanceCallback


def load_samples(dataset: str, n: int, dataset_config: str | None = None) -> list[dict]:
    import os
    from datasets import load_dataset
    from train_sft import remap_roles
    from trl.data_utils import maybe_convert_to_chatml

    if os.path.exists(dataset):
        ds = load_dataset("parquet", data_files=dataset, streaming=True, split="train")
    else:
        ds = load_dataset(dataset, name=dataset_config, streaming=True, split="train")
    return [maybe_convert_to_chatml(remap_roles(dict(s))) for s in itertools.islice(ds, 0, n)]


def max_abs_diff(a: dict, b: dict) -> tuple[float, str]:
    keys = set(a) | set(b)
    worst, worst_key = 0.0, ""
    for k in sorted(keys):
        if k not in a or k not in b:
            return float("inf"), f"missing key {k}"
        diff = abs(a[k] - b[k])
        if diff > worst:
            worst, worst_key = diff, k
    return worst, worst_key


def run_entry(tokenizer, model, device, entry, batch_size, seed):
    cb = SpeculativeAcceptanceCallback(tokenizer, [entry], [], batch_size=batch_size)
    return cb._run(model, device, entry, batch_size=batch_size, seed=seed)


# --------------------------------------------------------------------------
# Mock drafter — pure function of (seed, committed_len). No GPU. Stubs both the
# per-sample cache extension and the batched draft so the orchestration can be
# checked for batch-invariance on CPU. Drafts depend only on per-sample state
# (seed, committed_len), never on batch composition, so any batch size must give
# bit-identical results.
# --------------------------------------------------------------------------
class _FakeKV:
    def crop(self, n):  # cross-step rollback is a no-op for the stub
        pass


def install_mock_drafter(vocab_size: int):
    def _mock_extend(model, kv, token_ids, start_pos, device):
        return None, _FakeKV()

    def _mock_draft(model, tokenizer, requests, device):
        results = []
        for req in requests:
            seed, cl = req["seed"], req["committed_len"]
            drafts = []
            for draft_idx in range(req["n"]):
                toks = [
                    (seed * 1009 + draft_idx * 31 + t * 7 + cl) % vocab_size
                    for t in range(req["d"])
                ]
                drafts.append(toks)
            results.append(drafts)
        return results

    se._forward_extend = _mock_extend
    se._draft_from_caches = _mock_draft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock", "gpu"], required=True)
    ap.add_argument("--parquet", default="data/kimi-k2.6-claude-code-traces.parquet")
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--ckpt", default="outputs/Qwen3_1.7B-kimi_agent_sft-r256-lr2e-4/checkpoint-1")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--n-drafts", type=int, default=4)
    ap.add_argument("--d-tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-characters", type=int, default=4000)
    ap.add_argument("--batch-sizes", default="1,4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    if args.mode == "mock":
        import types
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base)
        install_mock_drafter(tokenizer.vocab_size)
        samples = load_samples(args.parquet, args.n_samples, args.dataset_config)
        # _run reads model.generation_config for top_k/top_p; the mock drafter never
        # uses the model otherwise, so a tiny stub suffices on CPU.
        model = types.SimpleNamespace(generation_config=types.SimpleNamespace(top_k=20, top_p=0.95))
        device = "cpu"
        print(f"[mock] {len(samples)} samples, drafter=pure-python (no GPU)")
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        device = "cuda"
        tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
        base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, attn_implementation="sdpa")
        model = PeftModel.from_pretrained(base, args.ckpt).to(device)
        model.eval()
        samples = load_samples(args.parquet, args.n_samples, args.dataset_config)
        print(f"[gpu] {len(samples)} samples, model={args.base}+LoRA, dtype=bf16, temp={args.temperature}")

    def make_entry():
        return SpecDecEvalEntry(
            name="test",
            eval_samples=[dict(s) for s in samples],  # fresh copy (entries mutate content)
            n_drafts=args.n_drafts,
            d_tokens=args.d_tokens,
            temperature=args.temperature,
            max_characters=args.max_characters,
        )

    # Baseline = batch_size 1.
    base_metrics = run_entry(tokenizer, model, device, make_entry(), batch_size=1, seed=args.seed)
    # Determinism: run b=1 again, must be identical.
    base_metrics2 = run_entry(tokenizer, model, device, make_entry(), batch_size=1, seed=args.seed)
    d0, k0 = max_abs_diff(base_metrics, base_metrics2)
    print(f"\nbatch=1 reproducibility: max|Δ|={d0:.3e} ({'OK' if d0 == 0.0 else 'NONDETERMINISTIC @ '+k0})")
    print(f"baseline metrics ({len(base_metrics)} keys), e.g. avg@{args.n_drafts}_pos1 = "
          f"{base_metrics.get(f'spec_acc/test/avg@{args.n_drafts}_pos1')}")
    import json
    print("BASELINE_JSON=" + json.dumps(base_metrics))

    ok = (d0 == 0.0)
    for bs in batch_sizes:
        if bs == 1:
            continue
        m = run_entry(tokenizer, model, device, make_entry(), batch_size=bs, seed=args.seed)
        diff, key = max_abs_diff(base_metrics, m)
        exact = "EXACT" if diff == 0.0 else ("within tol" if diff <= args.tol else "OVER TOL")
        print(f"batch={bs} vs baseline: max|Δ|={diff:.3e} @ {key}  [{exact}]")
        if args.mode == "mock":
            ok = ok and (diff == 0.0)
        else:
            ok = ok and (diff <= args.tol)

    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
