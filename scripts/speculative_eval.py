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


def _build_sequence_with_labels(messages: list[dict], tokenizer) -> tuple[list[int], list[int], list[bool]]:
    """
    Tokenize the full conversation and return ``(input_ids, labels, think_mask)``.

    ``messages[-1]`` MUST be an assistant message; raises ``ValueError`` otherwise.
    Labels are ``-100`` everywhere except the last assistant turn. ``think_mask``
    is ``True`` for tokens inside the ``<think>...</think>`` block of that turn.

    The think block is located by searching the formatted assistant turn text
    (not ``content`` directly) so it works regardless of whether the dataset
    stores reasoning in ``content`` or a separate field handled by the template.
    """
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("last message must be an assistant message")

    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer.encode(full_text)
    labels = [-100] * len(full_ids)
    think_mask = [False] * len(full_ids)

    context_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    n_context = len(tokenizer.encode(context_text))

    for pos in range(n_context, len(full_ids)):
        labels[pos] = full_ids[pos]

    # Search in the formatted turn text so we find the block even when content
    # doesn't include <think> tags (template injects them from a separate field).
    assistant_turn_text = full_text[len(context_text):]
    m = re.search(r"^<think>.*?</think>\n\n", assistant_turn_text, re.DOTALL)
    if m:
        n_think_end = min(
            len(tokenizer.encode(context_text + assistant_turn_text[:m.end()])),
            len(full_ids),
        )
        for pos in range(n_context, n_think_end):
            think_mask[pos] = True

    return full_ids, labels, think_mask


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
        """
        Evaluate next-token accuracy on every assistant turn in each sample.

        Each turn is evaluated independently: the KV cache is built from scratch
        for that turn's full prefix (all prior turns, with reasoning stripped by
        the chat template) and discarded between turns.
        """
        total_correct = total_tokens = 0
        think_correct = think_tokens = 0
        nothink_correct = nothink_tokens = 0
        chunk_size = 2048

        for sample in entry.eval_samples:
            messages = sample.get("messages") or sample.get("conversations") or []
            if not messages:
                continue
            for m in messages:
                if m["role"] == "assistant":
                    m["content"] = _qwen3_convert_glm_think_blocks(m["content"])

            asst_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
            for asst_idx in asst_indices:
                turn_messages = messages[:asst_idx + 1]
                try:
                    input_ids, label_ids, think_mask = _build_sequence_with_labels(
                        turn_messages, self.tokenizer
                    )
                except ValueError:
                    continue

                if entry.max_length is not None:
                    input_ids  = input_ids[:entry.max_length]
                    label_ids  = label_ids[:entry.max_length]
                    think_mask = think_mask[:entry.max_length]

                seq_len = len(input_ids)
                ids_t        = torch.tensor([input_ids], dtype=torch.long, device=device)
                shift_labels = torch.tensor(label_ids[1:],  dtype=torch.long, device=device)
                shift_think  = torch.tensor(think_mask[1:], dtype=torch.bool, device=device)

                past_key_values = None
                for chunk_start in range(0, seq_len, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, seq_len)
                    with torch.inference_mode():
                        out = model(
                            input_ids=ids_t[:, chunk_start:chunk_end],
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                    past_key_values = out.past_key_values

                    pred_end = min(chunk_end, seq_len - 1)
                    n_preds = pred_end - chunk_start
                    if n_preds > 0:
                        preds    = out.logits[0, :n_preds].argmax(dim=-1)
                        lbls     = shift_labels[chunk_start:pred_end]
                        tmask    = shift_think[chunk_start:pred_end]
                        lbl_mask = lbls != -100
                        correct  = (preds == lbls) & lbl_mask

                        total_correct   += correct.sum().item()
                        total_tokens    += lbl_mask.sum().item()
                        think_correct   += (correct &  tmask).sum().item()
                        think_tokens    += (lbl_mask &  tmask).sum().item()
                        nothink_correct += (correct & ~tmask).sum().item()
                        nothink_tokens  += (lbl_mask & ~tmask).sum().item()
                    del out
                del past_key_values

        if total_tokens == 0:
            return {}

        result = {f"full_eval/{entry.name}/mean_token_accuracy": total_correct / total_tokens}
        if think_tokens > 0:
            result[f"full_eval/{entry.name}/mean_token_accuracy_think"]   = think_correct / think_tokens
        if nothink_tokens > 0:
            result[f"full_eval/{entry.name}/mean_token_accuracy_nothink"] = nothink_correct / nothink_tokens
        return result

    def _run_full_eval_with_caching(self, model, device, entry: FullEvalEntry) -> dict[str, float]:
        """
        Like ``_run_full_eval`` but reuses KV cache across turns within a sample.

        For each assistant turn j the cached prefix covers all tokens before that
        turn's response (context tokens with no reasoning, since the chat template
        strips prior turns' think blocks). Moving to turn j+1 extends the cache
        with the stripped response of turn j plus the next user message, avoiding
        a full recompute of the growing prefix.

        Results are bit-identical to ``_run_full_eval``: the token sequence seen
        by the model for each evaluated turn is the same; only the forward-pass
        order differs.
        """
        total_correct = total_tokens = 0
        think_correct = think_tokens = 0
        nothink_correct = nothink_tokens = 0
        chunk_size = 2048

        for sample in entry.eval_samples:
            messages = sample.get("messages") or sample.get("conversations") or []
            if not messages:
                continue
            for m in messages:
                if m["role"] == "assistant":
                    m["content"] = _qwen3_convert_glm_think_blocks(m["content"])

            asst_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
            prefix_kv  = None
            prefix_len = 0  # tokens already in prefix_kv

            for turn_num, asst_idx in enumerate(asst_indices):
                turn_messages = messages[:asst_idx + 1]
                try:
                    input_ids, label_ids, think_mask = _build_sequence_with_labels(
                        turn_messages, self.tokenizer
                    )
                except ValueError:
                    prefix_kv  = None
                    prefix_len = 0
                    continue

                if entry.max_length is not None:
                    input_ids  = input_ids[:entry.max_length]
                    label_ids  = label_ids[:entry.max_length]
                    think_mask = think_mask[:entry.max_length]

                seq_len = len(input_ids)

                # First labeled position = start of this turn's assistant tokens.
                n_context = next((i for i, l in enumerate(label_ids) if l != -100), seq_len)

                ids_t        = torch.tensor([input_ids], dtype=torch.long, device=device)
                shift_labels = torch.tensor(label_ids[1:],  dtype=torch.long, device=device)
                shift_think  = torch.tensor(think_mask[1:], dtype=torch.bool, device=device)

                # Step 1 — extend prefix cache to n_context (no-op for turns > 0
                # because Step 3 of the previous turn already built up to here).
                if n_context > prefix_len:
                    with torch.inference_mode():
                        ctx_out = model(
                            input_ids=ids_t[:, prefix_len:n_context],
                            past_key_values=prefix_kv,
                            use_cache=True,
                        )
                    prefix_kv  = ctx_out.past_key_values
                    prefix_len = n_context
                    del ctx_out

                # Step 2 — evaluate the assistant turn using the cached prefix.
                eval_kv = prefix_kv
                for chunk_start in range(n_context, seq_len, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, seq_len)
                    with torch.inference_mode():
                        out = model(
                            input_ids=ids_t[:, chunk_start:chunk_end],
                            past_key_values=eval_kv,
                            use_cache=True,
                        )
                    eval_kv = out.past_key_values

                    pred_end = min(chunk_end, seq_len - 1)
                    n_preds  = pred_end - chunk_start
                    if n_preds > 0:
                        preds    = out.logits[0, :n_preds].argmax(dim=-1)
                        lbls     = shift_labels[chunk_start:pred_end]
                        tmask    = shift_think[chunk_start:pred_end]
                        lbl_mask = lbls != -100
                        correct  = (preds == lbls) & lbl_mask

                        total_correct   += correct.sum().item()
                        total_tokens    += lbl_mask.sum().item()
                        think_correct   += (correct &  tmask).sum().item()
                        think_tokens    += (lbl_mask &  tmask).sum().item()
                        nothink_correct += (correct & ~tmask).sum().item()
                        nothink_tokens  += (lbl_mask & ~tmask).sum().item()
                    del out
                del eval_kv

                # Step 3 — build the prefix cache for the next turn.
                # The next turn's context = apply_chat_template(messages[:next_asst_idx],
                # add_generation_prompt=True). This strips reasoning from the current
                # turn (now intermediate) and appends the next user message + prompt.
                # We only need to forward the new suffix tokens (from prefix_len on).
                next_turn = turn_num + 1
                if next_turn < len(asst_indices):
                    next_asst_idx  = asst_indices[next_turn]
                    next_ctx_text  = self.tokenizer.apply_chat_template(
                        messages[:next_asst_idx], tokenize=False, add_generation_prompt=True
                    )
                    next_ctx_ids = self.tokenizer.encode(next_ctx_text)
                    if len(next_ctx_ids) > prefix_len:
                        new_tokens = torch.tensor(
                            [next_ctx_ids[prefix_len:]], dtype=torch.long, device=device
                        )
                        with torch.inference_mode():
                            nxt_out = model(
                                input_ids=new_tokens,
                                past_key_values=prefix_kv,
                                use_cache=True,
                            )
                        prefix_kv  = nxt_out.past_key_values
                        prefix_len = len(next_ctx_ids)
                        del nxt_out

        if total_tokens == 0:
            return {}

        result = {f"full_eval/{entry.name}/mean_token_accuracy": total_correct / total_tokens}
        if think_tokens > 0:
            result[f"full_eval/{entry.name}/mean_token_accuracy_think"]   = think_correct / think_tokens
        if nothink_tokens > 0:
            result[f"full_eval/{entry.name}/mean_token_accuracy_nothink"] = nothink_correct / nothink_tokens
        return result

    def _run(self, model, device, entry: SpecDecEvalEntry) -> dict[str, float]:
        d, n = entry.d_tokens, entry.n_drafts

        pos_avg:          list[list[float]] = [[] for _ in range(d)]
        pos_best:         list[list[float]] = [[] for _ in range(d)]
        think_pos_avg:    list[list[float]] = [[] for _ in range(d)]
        think_pos_best:   list[list[float]] = [[] for _ in range(d)]
        nothink_pos_avg:  list[list[float]] = [[] for _ in range(d)]
        nothink_pos_best: list[list[float]] = [[] for _ in range(d)]

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
                reconstructed = prompt + target_text
                if not full.startswith(reconstructed):
                    logger.error("prompt+target is not a prefix of full template")
                    pbar.close()
                    raise ValueError("Prompt/target reconstruction failed parity check")

                pbar.set_description(f"[{entry.name}] sample {i + 1}/{n_samples} turn {j + 1}/{n_turns}")
                accepted_char_pos = 0

                _think_sentinel = "</think>\n\n"
                think_end_idx = (
                    target_text.index(_think_sentinel) + len(_think_sentinel)
                    if _think_sentinel in target_text else 0
                )

                for _ in range(128_000):  # hard cap on iterations per turn
                    remaining = target_text[accepted_char_pos:]
                    if not remaining.strip():
                        break
                    if entry.max_characters is not None and chars_consumed >= entry.max_characters:
                        budget_exhausted = True
                        break

                    in_think = accepted_char_pos < think_end_idx

                    context = prompt + target_text[:accepted_char_pos]
                    all_ids = _draft_completions(model, self.tokenizer, context, n, d, entry.temperature, device)

                    results   = [_accepted_tokens(ids, remaining, self.tokenizer) for ids in all_ids]
                    k_values  = [r[0] for r in results]

                    best_k   = max(k_values)
                    best_idx = k_values.index(best_k)

                    for pos in range(d):
                        avg_val  = sum(1 for k in k_values if k > pos) / n
                        best_val = 1.0 if best_k > pos else 0.0
                        pos_avg[pos].append(avg_val)
                        pos_best[pos].append(best_val)
                        if in_think:
                            think_pos_avg[pos].append(avg_val)
                            think_pos_best[pos].append(best_val)
                        else:
                            nothink_pos_avg[pos].append(avg_val)
                            nothink_pos_best[pos].append(best_val)

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
            if think_pos_avg[pos]:
                out[f"spec_acc/{entry.name}/avg@{n}_pos{pos + 1}_think"]  = sum(think_pos_avg[pos])  / len(think_pos_avg[pos])
                out[f"spec_acc/{entry.name}/best@{n}_pos{pos + 1}_think"] = sum(think_pos_best[pos]) / len(think_pos_best[pos])
            if nothink_pos_avg[pos]:
                out[f"spec_acc/{entry.name}/avg@{n}_pos{pos + 1}_nothink"]  = sum(nothink_pos_avg[pos])  / len(nothink_pos_avg[pos])
                out[f"spec_acc/{entry.name}/best@{n}_pos{pos + 1}_nothink"] = sum(nothink_pos_best[pos]) / len(nothink_pos_best[pos])
        return out
