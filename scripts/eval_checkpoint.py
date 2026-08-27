"""
Standalone speculative-decoding + full-eval on a saved checkpoint, logged to Weights & Biases.

Runs the same [`SpeculativeAcceptanceCallback`] used during training (`scripts/train_sft.py`)
against a checkpoint on disk, over the held-out eval sets, and logs the resulting
`spec_acc/*` and `full_eval/*` metrics to wandb — so a re-eval overlays directly on the
training run's plots (same metric names). Works for any checkpoint the training script
produces; for Laneformer (inline-reasoning) checkpoints the reasoning/response split fills
the `_think`/`_nothink` buckets automatically.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval_checkpoint.py \
        outputs/laneformer_2b_it-glm_mixed_sft-full-lr1e-5 --tag reeval

    # skip wandb, just print:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval_checkpoint.py <ckpt> --no-wandb

The wandb API key is read from the environment (e.g. `set -a; . .env; set +a`).
"""

import argparse
import itertools
import os
import sys
import types

sys.path.insert(0, os.path.dirname(__file__))

# The held-out eval sets: the first `skip_samples` rows of each training source, which the
# training configs exclude from training. (name, path, dataset_config)
DATASETS = [
    ("reasoning-heldout", "Jackrong/GLM-5.1-Reasoning-1M-Cleaned", "main"),
    ("hermes-heldout", "lambda/hermes-agent-reasoning-traces", "glm-5.1"),
    ("nemotron-hermes", "isaacrehg/Nemotron_Terminal_Synthetic_Tasks-GLM5.1-multiharness", "hermes"),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="Path to the checkpoint / saved-model directory to evaluate.")
    p.add_argument("--run-name", default=None,
                   help="wandb run name. Defaults to '<checkpoint basename>-<tag>'.")
    p.add_argument("--tag", default="reeval", help="wandb tag and run-name suffix (default: reeval).")
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "spec-dec"),
                   help="wandb project (default: $WANDB_PROJECT or 'spec-dec').")
    p.add_argument("--no-wandb", action="store_true", help="Skip wandb; just print the metrics.")
    p.add_argument("--global-step", type=int, default=0,
                   help="Step to log the metrics at (e.g. the checkpoint's train step).")
    p.add_argument("--n-spec", type=int, default=8, help="Spec-dec samples per dataset (0 to skip).")
    p.add_argument("--n-full", type=int, default=10, help="Full-eval samples per dataset (0 to skip).")
    p.add_argument("--max-length", type=int, default=4096, help="Token cap for eval contexts.")
    p.add_argument("--max-characters", type=int, default=3500, help="Per-turn spec-dec generation budget.")
    p.add_argument("--d-tokens", type=int, default=16, help="Draft tokens per speculative step.")
    p.add_argument("--n-drafts", type=int, default=1, help="Number of drafts per step.")
    p.add_argument("--temperature", type=float, default=0.0, help="Draft sampling temperature.")
    p.add_argument("--batch-size", type=int, default=4, help="Spec-dec draft batch size.")
    p.add_argument("--concurrency", type=int, default=4, help="Spec-dec samples in flight at once.")
    p.add_argument("--seed", type=int, default=42, help="Draft RNG seed.")
    return p.parse_args(argv)


def main(args):
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    from trl.data_utils import maybe_convert_to_chatml
    from trl.chat_template_utils import get_training_chat_template, laneformer_training_chat_template
    from train_sft import remap_roles
    from speculative_eval import (
        to_reasoning_format, reasoning_char_boundary,
        SpecDecEvalEntry, FullEvalEntry, SpeculativeAcceptanceCallback,
    )

    run_name = args.run_name or f"{os.path.basename(args.checkpoint.rstrip('/'))}-{args.tag}"

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    # Inline-reasoning format (Laneformer): its template renders reasoning inline (no <think>
    # tags) and emits <s> itself, so preprocess assistant targets the same way training did
    # and stop the tokenizer double-prepending bos in the eval helpers' plain encode() calls.
    reasoning_inline = get_training_chat_template(tokenizer) == laneformer_training_chat_template
    if reasoning_inline:
        tokenizer.add_bos_token = False

    def preprocess(ex):
        ex = maybe_convert_to_chatml(remap_roles(ex))
        if reasoning_inline:
            for m in ex.get("messages") or []:
                if m.get("role") == "assistant":
                    content = m.get("content") or ""
                    m["reasoning_chars"] = reasoning_char_boundary(content)
                    m["content"] = to_reasoning_format(content)
        return ex

    def samples(path, name, n):
        # The held-out set = the first n rows of each source (skipped during training).
        ds = load_dataset(path, name=name, split="train", streaming=True)
        return [preprocess(dict(s)) for s in itertools.islice(ds, n)]

    print(f"Loading checkpoint {args.checkpoint} (reasoning_inline={reasoning_inline})…", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=True, dtype=torch.bfloat16
    ).to("cuda").eval()

    spec_entries = [
        SpecDecEvalEntry(name=nm, eval_samples=samples(path, cfg, args.n_spec), n_drafts=args.n_drafts,
                         d_tokens=args.d_tokens, temperature=args.temperature,
                         max_characters=args.max_characters, max_length=args.max_length)
        for nm, path, cfg in DATASETS
    ] if args.n_spec > 0 else []
    full_entries = [
        FullEvalEntry(name=nm, eval_samples=samples(path, cfg, args.n_full), max_length=args.max_length)
        for nm, path, cfg in DATASETS
    ] if args.n_full > 0 else []

    callback = SpeculativeAcceptanceCallback(
        tokenizer=tokenizer, eval_entries=spec_entries, full_eval_entries=full_entries,
        batch_size=args.batch_size, eval_on_start=False, concurrency=args.concurrency,
    )

    report_to = []
    if not args.no_wandb:
        import wandb

        wandb.init(project=args.project, name=run_name, tags=[args.tag], job_type="eval",
                   config={"checkpoint": args.checkpoint, **vars(args)})
        report_to = ["wandb"]

    fake_args = types.SimpleNamespace(seed=args.seed, report_to=report_to)
    fake_state = types.SimpleNamespace(global_step=args.global_step)
    # _run_evals logs to wandb (when report_to includes it) and returns None; capture the
    # printed metrics from the logger. Call the two phases and print explicitly too.
    callback._run_evals(fake_args, fake_state, model, full_entries, spec_entries)

    if not args.no_wandb:
        import wandb

        wandb.finish()
    print("✅ Checkpoint eval complete.", flush=True)


if __name__ == "__main__":
    main(parse_args())
