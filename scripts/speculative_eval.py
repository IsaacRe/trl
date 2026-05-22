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
     spec_acc/{name}/avg@{n}_pos{k}  – mean fraction of n drafts accepted at position k
     spec_acc/{name}/best@{n}_pos{k} – mean indicator: was the best draft accepted at k

All positions k are 1-indexed relative to the start of each draft iteration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import re
import wandb

import torch
from tqdm.auto import tqdm
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


logger = logging.getLogger(__name__)


@dataclass
class SpecDecConfig:
    """
    Configuration for speculative decoding acceptance evaluation.

    Args:
        spec_dec_eval_only (`bool`, *optional*, defaults to `False`):
            Skip training and only run ``trainer.evaluate()``.
    """

    spec_dec_eval_only: bool = field(
        default=False,
        metadata={"help": "Skip training and run evaluation only."},
    )


@dataclass
class SpecDecEvalEntry:
    """One eval dataset with its own drafting hyperparameters."""

    name: str
    eval_samples: list[dict]
    eval_steps: int | None = None
    n_drafts: int = 1
    d_tokens: int = 8
    temperature: float = 0.8
    max_characters: int | None = None


@dataclass
class FullEvalEntry:
    """One eval dataset for full forward-pass metric collection."""

    name: str
    eval_samples: list[dict]
    eval_steps: int | None = None
    max_length: int | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qwen3_convert_glm_think_blocks(prompt: str) -> str:
    # add second newline inside empty <think> blocks
    converted = re.sub(r'<think>\n</think>', '<think>\n\n</think>', prompt)
    # add second newline after </think>
    converted = re.sub(r'</think>\n(?!\n)', '</think>\n\n', converted)
    return converted


def _qwen3_open_think_block(prompt: str) -> str:
    # qwen3 doesnt automatically add <think>\n at the start of assistant messages, so add it if missing
    if prompt.endswith("<|im_start|>assistant\n"):
        return prompt + "<think>\n"
    return prompt


def _build_sequence_with_labels(messages: list[dict], tokenizer) -> tuple[list[int], list[int]]:
    """
    Tokenize the full conversation and return ``(input_ids, labels)`` where
    ``labels`` has ``-100`` for all non-assistant token positions.

    The context boundary for each assistant turn is ``<|im_start|>assistant\\n``
    (add_generation_prompt without enable_thinking), so the opening ``<think>\\n``
    token is included in the prediction target.
    """
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer.encode(full_text)
    labels = [-100] * len(full_ids)

    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        context_text = tokenizer.apply_chat_template(
            messages[:i], tokenize=False, add_generation_prompt=True
        )
        turn_end_text = tokenizer.apply_chat_template(
            messages[:i + 1], tokenize=False, add_generation_prompt=False
        )
        n_context = len(tokenizer.encode(context_text))
        n_end = min(len(tokenizer.encode(turn_end_text)), len(full_ids))
        for pos in range(n_context, n_end):
            labels[pos] = full_ids[pos]

    return full_ids, labels


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
        context = _qwen3_open_think_block(context)
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
    Appends ``spec_acc/{name}/*`` metrics to the trainer's eval metrics dict
    for each entry in ``eval_entries``.

    Args:
        tokenizer: tokenizer of the model under training/evaluation.
        eval_entries (`list[SpecDecEvalEntry]`): one entry per validation set,
            each carrying its own samples and drafting hyperparameters.
    """

    def __init__(self, tokenizer, eval_entries: list[SpecDecEvalEntry], full_eval_entries: list[FullEvalEntry]):
        self.tokenizer = tokenizer
        self.eval_entries = eval_entries
        self.full_eval_entries = full_eval_entries

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        **kwargs,
    ):
        if state.global_step == 0:
            return
        full_eval_entries = [
            e for e in self.full_eval_entries
            if e.eval_samples and e.eval_steps and state.global_step % e.eval_steps == 0
        ]
        spec_dec_entries = [
            e for e in self.eval_entries
            if e.eval_samples and e.eval_steps and state.global_step % e.eval_steps == 0
        ]
        if not full_eval_entries and not spec_dec_entries:
            return

        was_training = model.training
        model.eval()
        device = next(model.parameters()).device

        all_metrics: dict[str, float] = {}
        try:
            for entry in full_eval_entries:
                logger.warning("Running full eval on %d samples [%s]…", len(entry.eval_samples), entry.name)
                try:
                    all_metrics.update(self._run_full_eval(model, device, entry))
                except Exception:
                    logger.exception("Full eval failed for entry %s", entry.name)
            for entry in spec_dec_entries:
                logger.warning("Running spec dec eval on %d samples [%s]…", len(entry.eval_samples), entry.name)
                try:
                    all_metrics.update(self._run(model, device, entry))
                except Exception:
                    logger.exception("Spec dec eval failed for entry %s", entry.name)
        finally:
            if was_training:
                model.train()
            torch.cuda.empty_cache()

        if all_metrics:
            logger.warning("Full eval metrics: %s", all_metrics)
            if "wandb" in (args.report_to or []):
                if wandb.run is not None:
                    wandb.log({**all_metrics, "train/global_step": state.global_step})

    def _run_full_eval(self, model, device, entry: FullEvalEntry) -> dict[str, float]:
        total_correct = 0
        total_tokens = 0
        chunk_size = 2048

        for sample in entry.eval_samples:
            messages = sample.get("messages") or sample.get("conversations") or []
            if not messages:
                continue
            for m in messages:
                if m["role"] == "assistant":
                    m["content"] = _qwen3_convert_glm_think_blocks(m["content"])

            input_ids, label_ids = _build_sequence_with_labels(messages, self.tokenizer)
            if entry.max_length is not None:
                input_ids = input_ids[:entry.max_length]
                label_ids = label_ids[:entry.max_length]

            seq_len = len(input_ids)
            ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
            shift_labels = torch.tensor(label_ids[1:], dtype=torch.long, device=device)

            past_key_values = None
            import pdb; pdb.set_trace()
            for chunk_start in range(0, seq_len, chunk_size):
                chunk_end = min(chunk_start + chunk_size, seq_len)
                with torch.inference_mode():
                    out = model(
                        input_ids=ids_t[:, chunk_start:chunk_end],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                past_key_values = out.past_key_values

                # logits[j] predicts position chunk_start+j+1; last token predicts past end
                pred_end = min(chunk_end, seq_len - 1)
                n_preds = pred_end - chunk_start
                if n_preds > 0:
                    preds = out.logits[0, :n_preds].argmax(dim=-1)
                    lbls = shift_labels[chunk_start:pred_end]
                    mask = lbls != -100
                    total_correct += ((preds == lbls) & mask).sum().item()
                    total_tokens += mask.sum().item()
                del out

        if total_tokens == 0:
            return {}

        return {
            f"full_eval/{entry.name}/mean_token_accuracy": total_correct / total_tokens,
        }

    def _run(self, model, device, entry: SpecDecEvalEntry) -> dict[str, float]:
        d, n = entry.d_tokens, entry.n_drafts

        pos_avg:  list[list[float]] = [[] for _ in range(d)]
        pos_best: list[list[float]] = [[] for _ in range(d)]

        n_samples = len(entry.eval_samples)

        for i, sample in enumerate(entry.eval_samples):
            messages = sample.get("messages") or sample.get("conversations") or []
            if not messages:
                continue

            # convert GLM-style <think> blocks to Qwen3 if needed
            for m in messages:
                if m["role"] == "assistant":
                    m["content"] = _qwen3_convert_glm_think_blocks(m["content"])

            turns = _assistant_turns(messages, self.tokenizer)
            n_turns = len(turns)
            pbar_total = sum(len(target) for _, target, _ in turns)
            if entry.max_characters is not None:
                pbar_total = min(pbar_total, entry.max_characters)
            pbar = tqdm(total=pbar_total, desc=f"[{entry.name}] sample {i + 1}/{n_samples} turn 1/{n_turns}", unit="char", leave=False)

            chars_consumed = 0
            budget_exhausted = False

            for j, (prompt, target_text, turn_messages) in enumerate(turns):
                if budget_exhausted:
                    break

                # Parity check: prompt + target_text must be a prefix of the fully-templated turn.
                full = self.tokenizer.apply_chat_template(turn_messages, tokenize=False)
                import pdb; pdb.set_trace()
                reconstructed = prompt + target_text
                if not full.startswith(reconstructed):
                    logger.error("prompt+target is not a prefix of full template")
                    pbar.close()
                    raise ValueError("Prompt/target reconstruction failed parity check")

                pbar.set_description(f"[{entry.name}] sample {i + 1}/{n_samples} turn {j + 1}/{n_turns}")
                accepted_char_pos = 0

                for _ in range(128_000):  # hard cap on iterations per turn
                    remaining = target_text[accepted_char_pos:]
                    if not remaining.strip():
                        break
                    if entry.max_characters is not None and chars_consumed >= entry.max_characters:
                        budget_exhausted = True
                        break

                    context = prompt + target_text[:accepted_char_pos]
                    all_ids = _draft_completions(model, self.tokenizer, context, n, d, entry.temperature, device)

                    results   = [_accepted_tokens(ids, remaining, self.tokenizer) for ids in all_ids]
                    k_values  = [r[0] for r in results]

                    best_k   = max(k_values)
                    best_idx = k_values.index(best_k)

                    for pos in range(d):
                        pos_avg[pos].append(sum(1 for k in k_values if k > pos) / n)
                        pos_best[pos].append(1.0 if best_k > pos else 0.0)

                    # advance to next fully accepted token boundary of best draft proposal
                    best_lcp = len(self.tokenizer.decode(all_ids[best_idx][:best_k]))
                    if best_lcp > 0:
                        accepted_char_pos += best_lcp
                        chars_consumed += best_lcp
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
                        chars_consumed += len(skip_prefix)
                        pbar.update(len(skip_prefix))

            pbar.close()

        out: dict[str, float] = {}
        for pos in range(d):
            if pos_avg[pos]:
                out[f"spec_acc/{entry.name}/avg@{n}_pos{pos + 1}"]  = sum(pos_avg[pos])  / len(pos_avg[pos])
                out[f"spec_acc/{entry.name}/best@{n}_pos{pos + 1}"] = sum(pos_best[pos]) / len(pos_best[pos])
        return out
