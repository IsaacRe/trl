"""
Speculative decoding acceptance rate evaluation callback.

Uses the validation set assistant messages as the target responses (no external
model or endpoint required). For each sample:

1. Draft n proposals of d tokens from the model under evaluation.
2. Decode the full proposal text; also record the decoded char length at every
   token boundary (so we know exactly which characters each token "owns").
3. Find the longest common text prefix between the full decoded proposal and the
   remaining assistant message.
4. Count how many complete draft tokens fall within that prefix (i.e. whose
   cumulative decoded length <= prefix length).
5. Advance by that prefix length in the target text; resample and repeat until
   the full assistant message is covered.
6. Aggregate per-position acceptance rates:
     spec_acc/avg@{n}_pos{k}  – mean fraction of n drafts accepted at position k
     spec_acc/best@{n}_pos{k} – mean indicator: was the best draft accepted at k

All positions k are 1-indexed relative to the start of each draft iteration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import re

import torch
from tqdm.auto import tqdm
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


logger = logging.getLogger(__name__)


@dataclass
class SpecDecConfig:
    """
    Configuration for speculative decoding acceptance evaluation.

    Args:
        spec_dec_n_drafts (`int`, *optional*, defaults to `4`):
            Number of draft proposals per speculative iteration.
        spec_dec_d_tokens (`int`, *optional*, defaults to `8`):
            New tokens generated per draft proposal.
        spec_dec_n_eval_samples (`int`, *optional*, defaults to `10`):
            Validation samples used for both speculative eval and standard val
            loss. Keep small — each sample requires O(target_len / d) generate
            calls, so wall time scales linearly.
        spec_dec_draft_temperature (`float`, *optional*, defaults to `1.0`):
            Sampling temperature for draft proposals. ``0.0`` = greedy (all n
            drafts identical).
        spec_dec_eval_only (`bool`, *optional*, defaults to `False`):
            Skip training and only run ``trainer.evaluate()``.
    """

    spec_dec_n_drafts: int = field(
        default=4,
        metadata={"help": "Number of draft proposals per speculative iteration."},
    )
    spec_dec_d_tokens: int = field(
        default=8,
        metadata={"help": "New tokens per draft proposal."},
    )
    spec_dec_n_eval_samples: int = field(
        default=10,
        metadata={"help": "Validation samples for speculative eval and val loss."},
    )
    spec_dec_draft_temperature: float = field(
        default=1.0,
        metadata={"help": "Draft sampling temperature (0.0 = greedy)."},
    )
    spec_dec_eval_only: bool = field(
        default=False,
        metadata={"help": "Skip training and run evaluation only."},
    )
    spec_dec_eval_dataset: str = field(
        default="",
        metadata={"help": "HF dataset to use as the held-out eval set (repo id). If empty, falls back to carving from the training stream."},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qwen3_convert_glm_think_blocks(prompt: str) -> str:
    # add second newline inside empty <think> blocks
    converted = re.sub(r'<think>\n</think>', '<think>\n\n</think>', prompt)
    # add second newline after </think>
    converted = re.sub(r'</think>\n(?!\n)', '</think>\n\n', converted)
    return converted


def _assistant_turns(messages: list[dict], tokenizer) -> list[tuple[str, str, list[dict]]]:
    """
    Return ``(context, target, turn_messages)`` for every assistant turn.

    ``context`` is the templated prompt ending with the generation prefix (``<think>\\n``).
    ``target`` is the assistant content with the leading ``<think>\\n`` stripped.
    ``turn_messages`` is ``messages[:i+1]``, used for the parity check.
    """
    turns = []
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        content = msg.get("content") or ""
        if not content.startswith("<think>\n"):
            logger.error("no leading <think> block")
            raise ValueError("no leading <think> block in assistant message")
        target = content[len("<think>\n"):]
        if not target.strip():
            continue
        context = tokenizer.apply_chat_template(
            messages[:i], tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        turns.append((context, target, messages[:i + 1]))
    return turns


def _draft_completions(
    model,
    tokenizer,
    context: str,
    n: int,
    d: int,
    temperature: float,
    device: torch.device,
) -> list[list[int]]:
    """
    Sample n continuations of up to d tokens each.  Returns a list of token-id
    lists (EOS stripped), one per draft.
    """
    enc = tokenizer(context, return_tensors="pt").to(device)
    input_len = enc["input_ids"].shape[1]

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=d,
            num_return_sequences=n,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )

    eos_id = tokenizer.eos_token_id
    results = []
    for i in range(n):
        ids = out[i, input_len:].tolist()
        if eos_id is not None and eos_id in ids:
            ids = ids[: ids.index(eos_id)]
        results.append(ids)
    return results


def _lcp_len(a: str, b: str) -> int:
    """Character length of the longest common prefix of strings a and b."""
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def _accepted_tokens(draft_ids: list[int], target_remaining: str, tokenizer) -> tuple[int, int]:
    """
    Returns ``(n_accepted, lcp_chars)``.

    Decodes each token-prefix of ``draft_ids`` to build per-token char offsets,
    finds the longest common text prefix with ``target_remaining``, then counts
    how many complete draft tokens are contained within that prefix.

    A token at position k (1-indexed) is "complete" iff the cumulative decoded
    length after k tokens is <= the LCP length.
    """
    if not draft_ids:
        return 0, 0

    # Cumulative decoded length at each token boundary
    offsets: list[int] = []
    for k in range(1, len(draft_ids) + 1):
        offsets.append(len(tokenizer.decode(draft_ids[:k], skip_special_tokens=True)))

    full_decoded = tokenizer.decode(draft_ids, skip_special_tokens=True)
    lcp = _lcp_len(full_decoded, target_remaining)

    n_accepted = sum(1 for off in offsets if off <= lcp)
    return n_accepted, lcp


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class SpeculativeAcceptanceCallback(TrainerCallback):
    """
    Appends ``spec_acc/*`` metrics to the trainer's eval metrics dict.

    The target for acceptance checking is the assistant message already present
    in each validation sample — no external model or endpoint is required.

    Args:
        config (`SpecDecConfig`): speculative eval configuration.
        tokenizer: tokenizer of the model under training/evaluation.
        eval_samples (`list[dict]`): pre-collected samples, each with a
            ``"messages"`` key containing role/content dicts (post remap_roles
            and maybe_convert_to_chatml).
    """

    def __init__(self, config: SpecDecConfig, tokenizer, eval_samples: list[dict]):
        self.config = config
        self.tokenizer = tokenizer
        self.eval_samples = eval_samples[: config.spec_dec_n_eval_samples]

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        metrics: dict,
        **kwargs,
    ):
        if not self.eval_samples:
            return

        logger.info("Running speculative acceptance eval on %d samples…", len(self.eval_samples))
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device

        try:
            spec_metrics = self._run(model, device)
        except Exception:
            logger.exception("Speculative eval failed")
            spec_metrics = {}
        finally:
            if was_training:
                model.train()
            torch.cuda.empty_cache()

        metrics.update(spec_metrics)
        if spec_metrics:
            logger.info("Speculative metrics: %s", spec_metrics)

    def _run(self, model, device) -> dict[str, float]:
        cfg = self.config
        d, n = cfg.spec_dec_d_tokens, cfg.spec_dec_n_drafts

        pos_avg:  list[list[float]] = [[] for _ in range(d)]
        pos_best: list[list[float]] = [[] for _ in range(d)]

        n_samples = len(self.eval_samples)

        for i, sample in enumerate(self.eval_samples):
            messages = sample.get("messages") or sample.get("conversations") or []
            if not messages:
                continue

            # convert GLM-style <think> blocks to Qwen3 if needed
            for m in messages:
                if m["role"] == "assistant":
                    m["content"] = _qwen3_convert_glm_think_blocks(m["content"])

            turns = _assistant_turns(messages, self.tokenizer)
            n_turns = len(turns)
            total_chars = sum(len(target) for _, target, _ in turns)
            pbar = tqdm(total=total_chars, desc=f"sample {i + 1}/{n_samples} turn 1/{n_turns}", unit="char", leave=False)

            for j, (prompt, target_text, turn_messages) in enumerate(turns):
                # Parity check: prompt + target_text must be a prefix of the fully-templated turn.
                full = self.tokenizer.apply_chat_template(turn_messages, tokenize=False)
                reconstructed = prompt + target_text
                if not full.startswith(reconstructed):
                    logger.error("prompt+target is not a prefix of full template")
                    pbar.close()
                    raise ValueError("Prompt/target reconstruction failed parity check")

                pbar.set_description(f"sample {i + 1}/{n_samples} turn {j + 1}/{n_turns}")
                accepted_char_pos = 0

                for _ in range(200):  # hard cap on iterations per turn
                    remaining = target_text[accepted_char_pos:]
                    if not remaining.strip():
                        break

                    context = prompt + target_text[:accepted_char_pos]
                    all_ids = _draft_completions(model, self.tokenizer, context, n, d, cfg.spec_dec_draft_temperature, device)

                    results   = [_accepted_tokens(ids, remaining, self.tokenizer) for ids in all_ids]
                    k_values  = [r[0] for r in results]
                    lcp_chars = [r[1] for r in results]

                    best_k   = max(k_values)
                    best_idx = k_values.index(best_k)

                    # if j == 5:
                    #     import pdb; pdb.set_trace()

                    for pos in range(d):
                        pos_avg[pos].append(sum(1 for k in k_values if k > pos) / n)
                        pos_best[pos].append(1.0 if best_k > pos else 0.0)

                    best_lcp = lcp_chars[best_idx]
                    if best_lcp > 0:
                        accepted_char_pos += best_lcp
                        pbar.update(best_lcp)
                    else:
                        # Nothing matched — advance by minimum number of draft tokens that will
                        # get us to next clean utf-8 boundary in decoded text
                        skip_token_cnt = 0
                        skip_prefix = ""
                        remaining_ids = self.tokenizer.encode(remaining[:50])
                        while skip_token_cnt == 0 or remaining[:len(skip_prefix)] != skip_prefix:
                            skip_token_cnt += 1
                            skip_prefix = self.tokenizer.decode(remaining_ids[:skip_token_cnt])

                        accepted_char_pos += len(skip_prefix)
                        pbar.update(len(skip_prefix))

            pbar.close()

        out: dict[str, float] = {}
        for pos in range(d):
            if pos_avg[pos]:
                out[f"spec_acc/avg@{n}_pos{pos + 1}"]  = sum(pos_avg[pos])  / len(pos_avg[pos])
                out[f"spec_acc/best@{n}_pos{pos + 1}"] = sum(pos_best[pos]) / len(pos_best[pos])
        return out
