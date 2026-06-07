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

import asyncio
import json
import logging
from dataclasses import dataclass, field
import re
import wandb

import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from tqdm.auto import tqdm
from transformers import DynamicCache, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from trl.chat_template_utils import get_training_chat_template


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
    spec_dec_batch_size: int = field(
        default=32,
        metadata={"help": "Number of samples whose draft requests are batched together during spec-dec eval."},
    )
    spec_dec_concurrency: int = field(
        default=4,
        metadata={
            "help": "Number of samples processed concurrently during spec-dec eval. Each in-flight sample "
            "holds its own per-turn KV cache, so long-context samples need a lower value to fit in memory."
        },
    )
    baseline_eval_on_start: bool = field(
        default=False,
        metadata={"help": "Run the full/spec-dec evals once before the first training step (step-0 baseline)."},
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
    max_length: int | None = None


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


def _tools_from_sample(sample: dict):
    """Tool schemas for the chat template, or ``None``.

    Mirrors TRL's ``SFTTrainer`` tokenize path: a ``tools`` column is either a
    list of JSON schemas or a JSON string encoding that list.
    """
    tools = sample.get("tools")
    return json.loads(tools) if isinstance(tools, str) else tools


def _build_sequence_with_labels(
    messages: list[dict], tokenizer, tools=None, chat_template=None
) -> tuple[list[int], list[int], list[bool]]:
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

    full_text = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=False, chat_template=chat_template
    )
    full_ids = tokenizer.encode(full_text)
    labels = [-100] * len(full_ids)
    think_mask = [False] * len(full_ids)

    context_text = tokenizer.apply_chat_template(
        messages[:-1], tools=tools, tokenize=False, add_generation_prompt=True, chat_template=chat_template
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


def _assistant_turns(messages: list[dict], tokenizer, tools=None, chat_template=None) -> list[tuple[str, str, list[dict]]]:
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
        if content.startswith("<think>\n"):
            target = content[len("<think>\n"):]
        else:
            # Turns without a reasoning block (e.g. non-reasoning harnesses in
            # multiharness traces) render as an empty think block under the
            # training chat template; the target mirrors that rendering.
            target = "\n</think>\n\n" + content.lstrip("\n")
        if not target.strip():
            continue
        context = tokenizer.apply_chat_template(
            messages[:i], tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=True,
            chat_template=chat_template,
        )
        context = _qwen3_open_think_block(context)
        turns.append((context, target, messages[:i + 1]))
    return turns


def _request_seed(base_seed: int, i: int, j: int, step: int) -> int:
    """
    Derive a per-draft-request RNG seed from the stable ``(sample, turn, step)``
    coordinates. Because the seed depends only on these coordinates — never on how
    requests are grouped into batches — a given request samples the same drafts
    whether it is decoded alone or alongside others.
    """
    h = base_seed & 0x7FFFFFFFFFFFFFFF
    for v in (i, j, step):
        h = (h * 1000003 + v) & 0x7FFFFFFFFFFFFFFF
    return h


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    """Length of the longest common prefix of two token-id lists."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _forward_extend(model, kv, token_ids: list[int], start_pos: int, device: torch.device):
    """
    Forward ``token_ids`` (one row) on top of cache ``kv`` (which already holds
    ``start_pos`` tokens) and return ``(last_logits, kv)`` — the logits at the final
    token (the distribution for the next token) and the extended cache.

    No attention mask is passed: a single row with a tight (unpadded) cache attends to
    everything, so the cache it produces is independent of any batch it later joins.
    Used both for the one-time prompt prefill and for the incremental extension by
    newly-accepted tokens between speculative steps.
    """
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    pos = torch.arange(start_pos, start_pos + len(token_ids), device=device).unsqueeze(0)
    with torch.inference_mode():
        out = model(input_ids=ids, position_ids=pos, past_key_values=kv, use_cache=True, logits_to_keep=1)
    return out.logits[0, -1, :], out.past_key_values


def _apply_warpers(logits: torch.Tensor, top_k: int | None, top_p: float | None) -> torch.Tensor:
    """
    Apply top-k then top-p filtering (HF's default warper order) to temperature-scaled
    ``logits`` (shape ``(rows, V)``), masking removed tokens to ``-inf``. Mirrors how
    the model samples under its ``generation_config`` so drafts match real generation.
    """
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        thresh = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < thresh, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=False, dim=-1)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cum <= (1.0 - top_p)  # keeps the smallest set with cumprob >= top_p
        remove = remove.scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(remove, float("-inf"))
    return logits


def _draft_from_caches(model, tokenizer, requests: list[dict], device: torch.device) -> list[list[list[int]]]:
    """
    Sample ``n`` drafts of ``d`` tokens for each request, starting from each request's
    persistent committed-context cache — without re-prefilling the context.

    Each request is a dict with keys:
        ``kv``: the sample's single-row committed-context [`~transformers.DynamicCache`].
        ``next_logits`` (`Tensor`, shape `(V,)`): logits at the last committed token,
            i.e. the distribution for the first draft token.
        ``committed_len`` (`int`): number of real tokens in ``kv``.
        ``n``, ``d``, ``temperature`` (`int/float`): drafting hyperparameters.
        ``seed`` (`int`): per-request RNG seed (see [`_request_seed`]).

    The persistent caches are read-only here: they are stacked (left-padded to a common
    length) into a throwaway batched cache, expanded to ``n`` drafts per request, and
    decoded for ``d`` steps. The committed caches are built unbatched (and so are
    batch-invariant), so only the short ``d``-token draft decode sees batched-matmul
    float differences.

    Returns a list aligned with ``requests``; each element is a list of ``n`` token-id
    lists with everything from the first EOS token onward removed.
    """
    B = len(requests)
    n = requests[0]["n"]
    d = requests[0]["d"]
    lengths = [req["committed_len"] for req in requests]
    Lmax = max(lengths)

    # Stack the per-sample committed caches into one (B, …, Lmax, …) batched cache,
    # left-padding shorter rows; then expand to n drafts per request.
    n_layers = len(requests[0]["kv"].layers)
    ddp_data = []
    for li in range(n_layers):
        keys, vals = [], []
        for req, L in zip(requests, lengths):
            k = req["kv"].layers[li].keys
            v = req["kv"].layers[li].values
            pad = Lmax - L
            if pad:
                k = torch.nn.functional.pad(k, (0, 0, pad, 0))
                v = torch.nn.functional.pad(v, (0, 0, pad, 0))
            keys.append(k)
            vals.append(v)
        ddp_data.append((torch.cat(keys, dim=0), torch.cat(vals, dim=0)))
    past = DynamicCache(ddp_cache_data=ddp_data)
    past.batch_repeat_interleave(n)

    # Attention mask over the cached context (0 on the left pad), the position of the
    # first draft token (= committed length), and the first-token logits — each
    # expanded to n rows per request.
    attn = torch.zeros((B, Lmax), dtype=torch.long, device=device)
    for r, L in enumerate(lengths):
        attn[r, Lmax - L:] = 1
    attn = attn.repeat_interleave(n, dim=0)
    cur_pos = torch.tensor(lengths, dtype=torch.long, device=device).repeat_interleave(n, dim=0)
    logits = torch.stack([req["next_logits"] for req in requests], dim=0).repeat_interleave(n, dim=0)

    # One generator per request, reused across the d steps (RNG depends only on seed).
    gens: list[torch.Generator | None] = []
    temps: list[float] = []
    top_ks: list[int | None] = []
    top_ps: list[float | None] = []
    for req in requests:
        temps.append(req["temperature"])
        top_ks.append(req.get("top_k"))
        top_ps.append(req.get("top_p"))
        if req["temperature"] > 0.0:
            g = torch.Generator(device=device)
            g.manual_seed(req["seed"])
            gens.append(g)
        else:
            gens.append(None)

    rows = B * n  # request r owns rows [r*n : (r+1)*n]

    def _sample(lg_all: torch.Tensor) -> torch.Tensor:
        tok = torch.empty(rows, dtype=torch.long, device=device)
        for r_idx in range(B):
            a, b = r_idx * n, r_idx * n + n
            lg = lg_all[a:b].float()
            if gens[r_idx] is None:
                tok[a:b] = lg.argmax(dim=-1)
            else:
                lg = _apply_warpers(lg / temps[r_idx], top_ks[r_idx], top_ps[r_idx])
                probs = torch.softmax(lg, dim=-1)
                tok[a:b] = torch.multinomial(probs, 1, generator=gens[r_idx]).squeeze(-1)
        return tok

    generated = torch.empty((rows, d), dtype=torch.long, device=device)
    with torch.inference_mode():
        for t in range(d):
            tok = _sample(logits)
            generated[:, t] = tok
            if t == d - 1:
                break
            attn = torch.cat([attn, torch.ones((rows, 1), dtype=torch.long, device=device)], dim=1)
            out = model(
                input_ids=tok.unsqueeze(1),
                attention_mask=attn,
                position_ids=cur_pos.unsqueeze(1),
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            )
            logits = out.logits[:, -1, :]
            cur_pos = cur_pos + 1

    eos_id = tokenizer.eos_token_id
    results: list[list[list[int]]] = []
    for r_idx in range(B):
        drafts = []
        for r in range(r_idx * n, r_idx * n + n):
            ids = generated[r].tolist()
            if eos_id is not None and eos_id in ids:
                ids = ids[: ids.index(eos_id)]
            drafts.append(ids)
        results.append(drafts)
    return results


def _lcp_len(a: str, b: str) -> int:
    """Character length of the longest common prefix of strings a and b."""
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def _first_token_chars(text: str, tokenizer) -> int:
    """Char length of the smallest leading token-run of ``text`` that ends on a clean
    (round-trippable) character boundary — normally a single token. Used to force-commit
    the speculative-decoding correction token at the mismatch position."""
    ids = tokenizer.encode(text[:50])
    cnt, prefix = 0, ""
    while (cnt == 0 or text[: len(prefix)] != prefix) and cnt < len(ids):
        cnt += 1
        prefix = tokenizer.decode(ids[:cnt])
    return len(prefix)


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

    def __init__(self, tokenizer, eval_entries: list[SpecDecEvalEntry], full_eval_entries: list[FullEvalEntry], batch_size: int = 1, eval_on_start: bool = False, concurrency: int = 4):
        self.tokenizer = tokenizer
        self.eval_entries = eval_entries
        self.full_eval_entries = full_eval_entries
        self.batch_size = batch_size
        self.eval_on_start = eval_on_start
        self.concurrency = concurrency
        # Render with the same chat template SFTTrainer tokenizes with: when the
        # tokenizer's own template lacks {% generation %} markers, the trainer swaps
        # in TRL's prefix-preserving training template (which keeps empty <think>
        # blocks on intermediate turns instead of stripping them). Using it here too
        # keeps eval contexts byte-identical to the training format. ``None`` falls
        # back to the tokenizer's own template, mirroring the trainer.
        self.chat_template = get_training_chat_template(tokenizer)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        **kwargs,
    ):
        # Optional step-0 baseline: evaluate once before the first training step.
        if not self.eval_on_start:
            return
        full_eval_entries = [e for e in self.full_eval_entries if e.eval_samples]
        spec_dec_entries = [e for e in self.eval_entries if e.eval_samples]
        if full_eval_entries or spec_dec_entries:
            if PartialState().is_main_process:
                logger.warning("Running step-0 baseline eval before training…")
            self._run_evals(args, state, model, full_eval_entries, spec_dec_entries)

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
        self._run_evals(args, state, model, full_eval_entries, spec_dec_entries)

    def _run_evals(self, args, state, model, full_eval_entries, spec_dec_entries):
        if not full_eval_entries and not spec_dec_entries:
            return

        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        # Every rank runs the evals (each on its sample shard) and reaches the same
        # gathers, but only the main process prints the progress/metric lines.
        is_main = PartialState().is_main_process

        all_metrics: dict[str, float] = {}
        try:
            # Each phase pools its samples across all its datasets into one set,
            # shards that set across ranks, and aggregates back per dataset — so every
            # rank stays busy even when an individual dataset has fewer samples than
            # there are GPUs (total samples across datasets only needs to reach world_size).
            if full_eval_entries:
                if is_main:
                    names = ", ".join(f"{e.name}×{len(e.eval_samples)}" for e in full_eval_entries)
                    logger.warning("Running full eval on [%s]…", names)
                try:
                    for entry in full_eval_entries:
                        all_metrics.update(self._run_full_eval_with_caching(model, device, entry))
                except Exception:
                    logger.exception("Full eval failed")
            if spec_dec_entries:
                if is_main:
                    names = ", ".join(f"{e.name}×{len(e.eval_samples)}" for e in spec_dec_entries)
                    logger.warning("Running spec dec eval on [%s]…", names)
                try:
                    all_metrics.update(self._run(model, device, spec_dec_entries, batch_size=self.batch_size, seed=args.seed))
                except Exception:
                    logger.exception("Spec dec eval failed")
        finally:
            if was_training:
                model.train()
            torch.cuda.empty_cache()

        if all_metrics and is_main:
            logger.warning("Eval metrics: %s", all_metrics)
            if "wandb" in (args.report_to or []):
                if wandb.run is not None:
                    wandb.log({**all_metrics, "train/global_step": state.global_step})

    def _run_full_eval_with_caching(self, model, device, entry: FullEvalEntry) -> dict[str, float]:
        """
        Full forward-pass eval of one entry, reusing KV cache across turns within
        a sample.

        For each assistant turn j the cached prefix covers all tokens before that
        turn's response. Moving to turn j+1 extends the cache with turn j's
        response plus the next user message (the training chat template is
        prefix-preserving, so the next context render extends the current one),
        avoiding a full recompute of the growing prefix.

        Samples are sharded across ranks (``eval_samples[rank::world_size]``);
        per-rank token counts are summed across ranks before the accuracy ratios
        are formed.
        """
        total_correct = total_tokens = 0
        think_correct = think_tokens = 0
        nothink_correct = nothink_tokens = 0
        chunk_size = 2048

        state = PartialState()
        pbar = tqdm(entry.eval_samples[state.process_index :: state.num_processes],
                    desc=f"[full-eval] {entry.name}", unit="sample", leave=False,
                    disable=not state.is_main_process)
        for sample in pbar:
            messages = sample.get("messages") or sample.get("conversations") or []
            if not messages:
                continue
            tools = _tools_from_sample(sample)
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
                        turn_messages, self.tokenizer, tools=tools, chat_template=self.chat_template
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
                            logits_to_keep=1,  # cache-building only; full-prefix logits would be huge
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
                # ``eval_kv`` aliases ``prefix_kv`` and DynamicCache mutates in place,
                # so the turn's tokens were appended to the persistent prefix. Crop
                # them back out: the prefix must contain exactly the context render.
                prefix_kv.crop(prefix_len)
                del eval_kv

                # Step 3 — build the prefix cache for the next turn.
                # The next turn's context = apply_chat_template(messages[:next_asst_idx],
                # add_generation_prompt=True). The training chat template is
                # prefix-preserving (the current turn keeps its <think> block when it
                # becomes intermediate), so this render extends the current one.
                # We only need to forward the new suffix tokens (from prefix_len on).
                next_turn = turn_num + 1
                if next_turn < len(asst_indices):
                    next_asst_idx  = asst_indices[next_turn]
                    next_ctx_text  = self.tokenizer.apply_chat_template(
                        messages[:next_asst_idx], tools=tools, tokenize=False, add_generation_prompt=True,
                        chat_template=self.chat_template,
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
                                logits_to_keep=1,  # cache-building only; full-prefix logits would be huge
                            )
                        prefix_kv  = nxt_out.past_key_values
                        prefix_len = len(next_ctx_ids)
                        del nxt_out

        # Sum each rank's accumulators into global totals. Every rank reaches this
        # gather — even one whose shard was empty — so it cannot deadlock.
        gathered = gather_object([[total_correct, total_tokens, think_correct, think_tokens,
                                   nothink_correct, nothink_tokens]])
        tc, tt, thc, tht, ntc, ntt = (sum(g[k] for g in gathered) for k in range(6))
        if tt == 0:
            return {}

        result = {f"full_eval/{entry.name}/mean_token_accuracy": tc / tt}
        if tht > 0:
            result[f"full_eval/{entry.name}/mean_token_accuracy_think"]   = thc / tht
        if ntt > 0:
            result[f"full_eval/{entry.name}/mean_token_accuracy_nothink"] = ntc / ntt
        return result

    def _run(self, model, device, entries: list[SpecDecEvalEntry], batch_size: int = 1, seed: int = 0) -> dict[str, float]:
        tokenizer = self.tokenizer
        # Sample drafts the way the model actually generates: temperature comes from
        # each entry, but top-k / top-p are taken from the model's generation config.
        top_k = model.generation_config.top_k
        top_p = model.generation_config.top_p

        # Pool every (dataset, sample) pair into one work set and shard it across ranks.
        # Each sample keeps its index within its own dataset, so the per-request RNG seed
        # (and thus the sampled drafts) is independent of how the pool is split — pooling
        # never changes a dataset's metrics versus running it alone. Records are tagged
        # with the dataset name and gathered below.
        state = PartialState()
        pool = [(entry, i, sample) for entry in entries for i, sample in enumerate(entry.eval_samples)]
        local_work = pool[state.process_index :: state.num_processes]

        # Per-step records: (name, i, j, step, in_think, k_values). Metrics are aggregated
        # from these afterwards in a fixed (i, j, step) order, so the result does not
        # depend on the order in which concurrently-batched samples complete.
        records: list[tuple[str, int, int, int, bool, list[int]]] = []

        pbar = tqdm(total=len(local_work), desc="[spec-dec]", unit="sample", leave=False,
                    disable=not state.is_main_process)

        # --- Async batcher -------------------------------------------------
        # Each sample runs as a coroutine. When it needs drafts it parks on a future;
        # parked requests are decoded together once a same-(n_drafts, d_tokens) group
        # reaches `batch_size` (or every still-active sample is parked). A single decode
        # batch must be uniform in (n, d), so requests are grouped by it before flushing.
        pending: list[tuple[dict, asyncio.Future]] = []
        active = {"n": 0}
        total_chars = {"n": 0}  # characters processed across all samples, shown live in the pbar

        def _flush(batch: list[tuple[dict, asyncio.Future]]) -> None:
            outs = _draft_from_caches(model, tokenizer, [r for r, _ in batch], device)
            for (_, fut), res in zip(batch, outs):
                fut.set_result(res)

        def _maybe_flush() -> None:
            # Flush any (n, d) group that has reached batch_size.
            while True:
                groups: dict[tuple[int, int], list[int]] = {}
                for idx, (req, _) in enumerate(pending):
                    groups.setdefault((req["n"], req["d"]), []).append(idx)
                ready = next((idxs for idxs in groups.values() if len(idxs) >= batch_size), None)
                if ready is None:
                    break
                sel = set(ready[:batch_size])
                batch = [pending[i] for i in ready[:batch_size]]
                pending[:] = [p for k, p in enumerate(pending) if k not in sel]
                _flush(batch)
            # Tail: when no further requests can arrive (every still-active sample is
            # already parked), decode whatever remains — still grouped by (n, d) so each
            # batch stays uniform — instead of deadlocking.
            if active["n"] > 0 and pending and len(pending) >= active["n"]:
                groups: dict[tuple[int, int], list[tuple[dict, asyncio.Future]]] = {}
                for req, fut in pending:
                    groups.setdefault((req["n"], req["d"]), []).append((req, fut))
                pending.clear()
                for batch in groups.values():
                    _flush(batch)

        async def _draft(kv, next_logits, committed_len, i, j, step, n, d, temperature) -> list[list[int]]:
            fut = asyncio.get_running_loop().create_future()
            req = {
                "kv": kv,
                "next_logits": next_logits,
                "committed_len": committed_len,
                "n": n,
                "d": d,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "seed": _request_seed(seed, i, j, step),
            }
            pending.append((req, fut))
            _maybe_flush()
            return await fut

        async def _process(entry: SpecDecEvalEntry, i: int, sample: dict) -> None:
            messages = sample.get("messages") or sample.get("conversations") or []
            if not messages:
                return

            # convert GLM-style <think> blocks to Qwen3 if needed
            tools = _tools_from_sample(sample)
            for m in messages:
                if m["role"] == "assistant":
                    m["content"] = _qwen3_convert_glm_think_blocks(m["content"])

            turns = _assistant_turns(messages, tokenizer, tools=tools, chat_template=self.chat_template)
            chars_consumed = 0
            budget_exhausted = False

            for j, (prompt, target_text, turn_messages) in enumerate(turns):
                if budget_exhausted:
                    break

                # Parity check: prompt + target_text must be a prefix of the fully-templated turn.
                full = tokenizer.apply_chat_template(
                    turn_messages, tools=tools, tokenize=False, chat_template=self.chat_template
                )
                if not full.startswith(prompt + target_text):
                    logger.error("prompt+target is not a prefix of full template")
                    raise ValueError("Prompt/target reconstruction failed parity check")

                accepted_char_pos = 0

                _think_sentinel = "</think>\n\n"
                think_end_idx = (
                    target_text.index(_think_sentinel) + len(_think_sentinel)
                    if _think_sentinel in target_text else 0
                )

                # Persistent KV cache for this turn's committed context. The prompt is
                # prefilled once; each step then forwards only the newly-accepted tokens
                # (cross-step reuse) instead of re-prefilling the whole context. Draft
                # tokens are never committed to this cache — only accepted text is.
                committed_ids = tokenizer(prompt).input_ids
                # Stop the sample once a turn's context exceeds max_length tokens — the
                # KV cache (and its padded copy in the draft batch) scales with context
                # length and can exhaust GPU memory on long agent traces. Later turns
                # only grow the context, so stop rather than skip.
                if entry.max_length is not None and len(committed_ids) > entry.max_length:
                    break
                kv = DynamicCache()
                next_logits, kv = _forward_extend(model, kv, committed_ids, 0, device)

                for step in range(128_000):  # hard cap on iterations per turn
                    remaining = target_text[accepted_char_pos:]
                    if not remaining.strip():
                        break
                    if entry.max_characters is not None and chars_consumed >= entry.max_characters:
                        budget_exhausted = True
                        break

                    in_think = accepted_char_pos < think_end_idx

                    all_ids = await _draft(kv, next_logits, len(committed_ids), i, j, step,
                                           entry.n_drafts, entry.d_tokens, entry.temperature)

                    results   = [_accepted_tokens(ids, remaining, tokenizer) for ids in all_ids]
                    k_values  = [r[0] for r in results]
                    records.append((entry.name, i, j, step, in_think, k_values))

                    best_k   = max(k_values)
                    best_idx = k_values.index(best_k)

                    # Accept the matching draft prefix, then force-commit one correction
                    # token: the ground-truth token at the mismatch position — the bonus/
                    # correction token a real spec-dec engine gets free from the target. So
                    # each step advances accepted+1, which guarantees ≥1 token of progress
                    # and stops a hard (run-breaking) token from reappearing as the next
                    # step's position-1. Without it, greedy re-drafts and misses that token
                    # every step, pinning pos-1 acceptance near zero and crawling ~1 tok/step.
                    best_lcp = len(tokenizer.decode(all_ids[best_idx][:best_k]))
                    rest = target_text[accepted_char_pos + best_lcp:]
                    advanced = best_lcp + (_first_token_chars(rest, tokenizer) if rest.strip() else 0)
                    accepted_char_pos += advanced
                    chars_consumed += advanced
                    total_chars["n"] += advanced

                    pbar.set_postfix(chars=total_chars["n"])

                    # Extend the committed cache to cover the newly-accepted text. The
                    # accepted prefix is re-tokenized; if a token boundary shifted, the
                    # cache is cropped back to the divergence point before appending.
                    if target_text[accepted_char_pos:].strip():
                        new_committed = tokenizer(prompt + target_text[:accepted_char_pos]).input_ids
                        P = _common_prefix_len(committed_ids, new_committed)
                        if P < len(committed_ids):
                            kv.crop(P)
                        if len(new_committed) > P:
                            next_logits, kv = _forward_extend(model, kv, new_committed[P:], P, device)
                        committed_ids = new_committed

        async def _worker(entry: SpecDecEvalEntry, i: int, sample: dict) -> None:
            try:
                await _process(entry, i, sample)
            finally:
                active["n"] -= 1
                pbar.update(1)
                _maybe_flush()

        async def _driver() -> None:
            # Process samples in waves of `concurrency` to bound peak memory: every
            # in-flight sample holds its own per-turn KV cache, and long-context
            # samples (e.g. agent traces) can each occupy several GB.
            for start in range(0, len(local_work), self.concurrency):
                wave = local_work[start:start + self.concurrency]
                active["n"] = len(wave)
                await asyncio.gather(*[_worker(entry, i, s) for entry, i, s in wave])

        asyncio.run(_driver())
        pbar.close()

        # Gather every rank's per-step records onto all ranks (each rank reaches this
        # call even with an empty shard, so it cannot deadlock). Metrics below are then
        # aggregated identically on every rank, per dataset, in a fixed (i, j, step) order;
        # the caller logs them on the main process only.
        records = gather_object(records)
        by_name: dict[str, list[tuple]] = {}
        for rec in records:
            by_name.setdefault(rec[0], []).append(rec)

        out: dict[str, float] = {}

        # Acceptance length: mean tokens committed per speculative step. That's the mean
        # accepted draft tokens — k = #{pos : k > pos}, so E[k] = Σ_pos P(k > pos), the
        # sum of the per-position rates over all d positions — plus 1 for the correction/
        # bonus token force-committed each step. So it's the throughput multiplier vs
        # one-token-per-step decoding.
        def _accept_len(rows: list[list[float]]) -> float:
            return sum(sum(p) / len(p) for p in rows) + 1

        for entry in entries:
            d, n = entry.d_tokens, entry.n_drafts
            pos_avg:          list[list[float]] = [[] for _ in range(d)]
            pos_best:         list[list[float]] = [[] for _ in range(d)]
            think_pos_avg:    list[list[float]] = [[] for _ in range(d)]
            think_pos_best:   list[list[float]] = [[] for _ in range(d)]
            nothink_pos_avg:  list[list[float]] = [[] for _ in range(d)]
            nothink_pos_best: list[list[float]] = [[] for _ in range(d)]

            for name, i, j, step, in_think, k_values in sorted(by_name.get(entry.name, []), key=lambda r: (r[1], r[2], r[3])):
                best_k = max(k_values)
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

            # Headline efficiency: mean tokens committed per step (acceptance length),
            # aggregated over all steps — accepted draft tokens + the correction/bonus
            # token. `best` is the best-of-n draft, `avg` is averaged over the n drafts
            # (identical when n_drafts == 1).
            if pos_best[0]:
                out[f"spec_acc/{entry.name}/mean_accepted_best@{n}"] = _accept_len(pos_best)
                out[f"spec_acc/{entry.name}/mean_accepted_avg@{n}"]  = _accept_len(pos_avg)
            if think_pos_best[0]:
                out[f"spec_acc/{entry.name}/mean_accepted_best@{n}_think"] = _accept_len(think_pos_best)
                out[f"spec_acc/{entry.name}/mean_accepted_avg@{n}_think"]  = _accept_len(think_pos_avg)
            if nothink_pos_best[0]:
                out[f"spec_acc/{entry.name}/mean_accepted_best@{n}_nothink"] = _accept_len(nothink_pos_best)
                out[f"spec_acc/{entry.name}/mean_accepted_avg@{n}_nothink"]  = _accept_len(nothink_pos_avg)
        return out
