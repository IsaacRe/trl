"""
SFT training wrapper that:
  - Remaps ShareGPT role names (human→user, gpt→assistant)
  - Carves a held-out eval set from the training stream for val loss + perplexity
  - Optionally attaches speculative decoding acceptance eval (see SpecDecConfig)

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_sft.py --config configs/sft_full.yaml
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_sft.py --config configs/sft_lora.yaml
"""

import dataclasses
import glob
import itertools
import json
import logging
import os
import random
import re
import sys
import threading
import time

from tqdm.auto import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))

from speculative_eval import FullEvalEntry, SpecDecConfig, SpecDecEvalEntry, SpeculativeAcceptanceCallback

logger = logging.getLogger(__name__)

_ROLE_MAP = {"human": "user", "gpt": "assistant"}


@dataclasses.dataclass
class BaseEvalConfig:
    dataset: str
    dataset_config: str | None = None
    name: str = ""
    n_samples: int = 10
    skip_samples: int = 0
    shuffle: bool = False
    max_turns: int | None = None
    eval_steps: int | None = None

    def __post_init__(self):
        if not self.name:
            self.name = self.dataset.split("/")[-1].split("-")[0]


@dataclasses.dataclass
class SpecDecEvalConfig(BaseEvalConfig):
    n_drafts: int = 1
    d_tokens: int = 8
    temperature: float = 0.8
    max_characters: int | None = None


@dataclasses.dataclass
class FullEvalConfig(BaseEvalConfig):
    max_length: int | None = None


def remap_roles(example):
    for msg in example.get("conversations") or example.get("messages") or []:
        if isinstance(msg, dict) and msg.get("from") in _ROLE_MAP:
            msg["from"] = _ROLE_MAP[msg["from"]]
    return example


def _load_yaml_eval_section(argv: list[str], key: str, cls: type) -> list:
    """Load a full_eval/spec_dec_eval section as typed config objects."""
    import yaml
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            with open(argv[i + 1]) as f:
                section = yaml.safe_load(f).get(key) or {}
            if isinstance(section, dict):
                eval_steps = section.get("eval_steps")
                return [cls(**{**e, "eval_steps": eval_steps}) for e in section.get("datasets", [])]
            return [cls(**e) for e in section]
    return []


def _read_yaml_scalar(argv: list[str], key: str, default=None):
    """Return a value from the YAML --config file, or ``default`` if absent."""
    import yaml
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            with open(argv[i + 1]) as f:
                return yaml.safe_load(f).get(key, default)
    return default


def _infinite_reshuffled(ds, base_seed: int):
    """Yield examples from ``ds`` forever, reshuffling with a new seed each pass."""
    epoch = 0
    while True:
        for ex in ds.shuffle(seed=base_seed + epoch):
            yield ex
        epoch += 1



def _make_turn_sampler(seed: int):
    """Return a dataset map function that randomly selects one assistant turn as the
    final turn and drops all subsequent messages.

    The RNG is seeded with ``seed`` (typically ``training_args.seed``) so runs are
    reproducible. For streaming datasets the map is applied lazily and sequentially,
    so the RNG advances in sample order.
    """
    rng = random.Random(seed)

    def _sample(example):
        msgs = example.get("messages") or []
        asst_indices = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
        if not asst_indices:
            return example
        chosen_msg_idx = rng.choice(asst_indices)
        return {**example, "messages": list(msgs[:chosen_msg_idx + 1])}

    return _sample



def _to_prompt_completion(example, tokenizer) -> dict:
    """Convert the last assistant turn to prompt-completion format.

    ``prompt``     = full context ending with the generation prompt
                     (all prior turns with reasoning stripped by the template)
    ``completion`` = the last assistant turn's formatted text
                     (including ``<think>`` block and ``<|im_end|>`` suffix)

    When the dataset sample has these keys, ``SFTTrainer`` auto-sets
    ``completion_only_loss=True`` and computes loss only on the completion tokens.
    """
    msgs = example.get("messages") or []
    if not msgs or msgs[-1].get("role") != "assistant":
        return example
    # `tools` column is a list of JSON schemas or a JSON string encoding that list.
    tools = example.get("tools")
    tools = json.loads(tools) if isinstance(tools, str) else tools
    full_text   = tokenizer.apply_chat_template(msgs,      tools=tools, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(msgs[:-1], tools=tools, tokenize=False, add_generation_prompt=True)
    if not full_text.startswith(prompt_text):
        raise ValueError("prompt text must be a prefix of the full text")
    return {**example, "prompt": prompt_text, "completion": full_text[len(prompt_text):]}


def _proc_rss_gb() -> float:
    """Resident set size of this process in GiB, read from /proc."""
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def _weight_bytes(model_name_or_path: str) -> float | None:
    """Total size (GiB) of the model's local *.safetensors shards, or None.

    Used as the target for a materialization progress bar: the "Loading weights"
    bar only mmaps the shards (instant); the real cost is faulting that many bytes
    into RAM afterwards, which has no HF progress bar of its own.
    """
    if os.path.isdir(model_name_or_path):
        shards = glob.glob(os.path.join(model_name_or_path, "*.safetensors"))
    else:
        from huggingface_hub import try_to_load_from_cache
        idx = try_to_load_from_cache(model_name_or_path, "model.safetensors.index.json")
        snap = os.path.dirname(idx) if isinstance(idx, str) else ""
        shards = glob.glob(os.path.join(snap, "*.safetensors")) if snap else []
    total = sum(os.path.getsize(f) for f in shards)
    return total / (1024 ** 3) if total else None


def _compose_run_name(training_args, model_args) -> str:
    model_short = model_args.model_name_or_path.split("/")[-1].replace("-", "_")
    rank_str = f"r{model_args.lora_r}" if getattr(model_args, "use_peft", False) else "full"
    lr_str = re.sub(r"e([+-])0*(\d)", r"e\1\2", f"{training_args.learning_rate:.0e}")
    parts = [model_short]
    if training_args.run_name:
        parts.append(training_args.run_name)
    parts += [rank_str, f"lr{lr_str}"]
    return "-".join(parts)


def main(script_args, training_args, model_args, dataset_args, spec_dec_args):
    from accelerate import logging
    from datasets import load_dataset
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    from trl import SFTTrainer, get_dataset, get_kbit_device_map, get_peft_config, get_quantization_config
    from trl.data_utils import maybe_convert_to_chatml

    logger = logging.get_logger(__name__)

    training_args.run_name = _compose_run_name(training_args, model_args)
    if not training_args.output_dir:
        training_args.output_dir = f"outputs/{training_args.run_name}"

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    # The "Loading weights" bar only mmaps the shards (instant). The real cost is
    # materializing those bytes into RAM, which has no HF bar — so poll RSS in a
    # background thread and show it against the on-disk shard size.
    weight_gb = _weight_bytes(model_args.model_name_or_path)
    logger.warning("Loading model (mmap is instant; materializing %s into RAM)…",
                   f"~{weight_gb:.0f} GB" if weight_gb else "weights")
    t_model = time.time()
    load_done = threading.Event()

    def _watch_rss():
        base = _proc_rss_gb()
        # Count bytes and let tqdm's unit_scale render GB (e.g. "42.0GB/61.0GB,
        # 0.6GB/s") — avoids a custom bar_format.
        bar = tqdm(total=int(weight_gb * 1024 ** 3) if weight_gb else None,
                   desc="Materializing weights", unit="B", unit_scale=True, unit_divisor=1024, leave=False)
        while not load_done.wait(1.0):
            bar.n = int(max(_proc_rss_gb() - base, 0.0) * 1024 ** 3)
            bar.refresh()
        bar.close()

    rss_thread = threading.Thread(target=_watch_rss, daemon=True)
    rss_thread.start()
    try:
        if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    finally:
        load_done.set()
        rss_thread.join(timeout=3)
    logger.warning("Model materialized in %.1fs (RSS now %.1f GB)", time.time() - t_model, _proc_rss_gb())

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    logger.info("Model + tokenizer ready. Loading training dataset…")
    _t_data = time.time()
    interleave_specs = _read_yaml_scalar(sys.argv, "interleave_datasets", []) or []
    if interleave_specs:
        from datasets import IterableDataset, IterableDatasetDict, interleave_datasets

        streams = []
        any_infinite = False
        for s in interleave_specs:
            streaming = s.get("streaming", True)
            ds = load_dataset(
                s["path"], name=s.get("name"), streaming=streaming, split=script_args.dataset_train_split
            )
            # Skip the first `skip_samples` rows up front — before any shuffling,
            # so the same rows are excluded on every epoch.
            skip = s.get("skip_samples", 0)
            if skip:
                ds = ds.skip(skip) if streaming else ds.select(range(skip, len(ds)))
            if not streaming and s.get("shuffle", False):
                # Wrap a map-style Dataset as an infinite IterableDataset that
                # reshuffles every pass — so each epoch sees a different order.
                ds = IterableDataset.from_generator(
                    _infinite_reshuffled,
                    gen_kwargs={"ds": ds, "base_seed": training_args.seed},
                )
                any_infinite = True
            streams.append(ds)
        n = len(streams)
        # Sampling ratios. A spec may set an explicit `ratio`; entries that omit
        # it share the leftover mass (1 - sum of explicit ratios) evenly. If every
        # entry sets a ratio, they must sum to 1.
        ratios = [s.get("ratio") for s in interleave_specs]
        n_unset = sum(r is None for r in ratios)
        set_sum = sum(r for r in ratios if r is not None)
        if n_unset == 0:
            if abs(set_sum - 1.0) > 1e-6:
                raise ValueError(f"interleave_datasets ratios are all set but sum to {set_sum}, must sum to 1.")
            probabilities = ratios
        else:
            if set_sum > 1.0 + 1e-6:
                raise ValueError(
                    f"interleave_datasets ratios sum to {set_sum} > 1, leaving no mass for the {n_unset} "
                    "entries without a ratio."
                )
            fill = (1.0 - set_sum) / n_unset
            probabilities = [fill if r is None else r for r in ratios]
        # With an infinite stream, "all_exhausted" would never terminate; use
        # "first_exhausted" and let max_steps cap the run.
        stopping_strategy = "first_exhausted" if any_infinite else "all_exhausted"
        interleaved = interleave_datasets(
            streams,
            probabilities=probabilities,
            stopping_strategy=stopping_strategy,
            seed=training_args.seed,
        )
        dataset = IterableDatasetDict({script_args.dataset_train_split: interleaved})
    elif dataset_args.datasets and script_args.dataset_name:
        logger.info(
            "Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used to load the "
            "dataset and `dataset_name` will be ignored."
        )
        dataset = get_dataset(dataset_args)
    elif dataset_args.datasets and not script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif not dataset_args.datasets and script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    else:
        raise ValueError("Either `datasets`, `dataset_name`, or `interleave_datasets` must be provided.")

    # Remap ShareGPT role names and convert to ChatML format before TRL processes them
    dataset = dataset.map(lambda ex: maybe_convert_to_chatml(remap_roles(ex)))
    logger.info("Training dataset ready in %.1fs. Building eval sets…", time.time() - _t_data)

    # ------------------------------------------------------------------
    # Eval datasets — built from the spec_dec_eval list in the YAML.
    # Each entry specifies a dataset, sample count, and drafting params.
    # ------------------------------------------------------------------

    def _load_samples(cfg: SpecDecEvalConfig | FullEvalConfig) -> list[dict]:
        # Skip the first `skip_samples`, then take the next `n_samples`. When
        # `shuffle` is set the dataset is loaded non-streaming and shuffled
        # before slicing — streaming and shuffling are mutually exclusive.
        t0 = time.time()
        streaming = not cfg.shuffle
        logger.info(
            "Loading %d eval sample(s) from %s [%s] (shuffle=%s, streaming=%s)…",
            cfg.n_samples, cfg.dataset, cfg.name, cfg.shuffle, streaming,
        )
        start, stop = cfg.skip_samples, cfg.skip_samples + cfg.n_samples
        if cfg.dataset == script_args.dataset_name:
            src = dataset[script_args.dataset_train_split]
        elif os.path.exists(cfg.dataset):
            src = load_dataset("parquet", data_files=cfg.dataset, streaming=streaming, split="train")
        else:
            src = load_dataset(cfg.dataset, name=cfg.dataset_config, streaming=streaming, split="train")
        if cfg.shuffle:
            # Non-streaming: the whole dataset is downloaded/generated above, then
            # shuffled in memory — this is the slow part for `shuffle: true` entries.
            logger.info("  [%s] materializing full dataset to shuffle (non-streaming)…", cfg.name)
            src = src.shuffle(seed=training_args.seed)
        raw = [
            maybe_convert_to_chatml(remap_roles(dict(s)))
            for s in tqdm(
                itertools.islice(src, start, stop),
                total=cfg.n_samples,
                desc=f"  [{cfg.name}] loading samples",
                unit="sample",
                leave=False,
            )
        ]
        logger.info("  [%s] loaded %d sample(s) in %.1fs", cfg.name, len(raw), time.time() - t0)
        if cfg.max_turns is not None:
            for s in raw:
                msgs = s.get("messages") or s.get("conversations") or []
                count = 0
                for idx, m in enumerate(msgs):
                    if m.get("role") == "assistant":
                        count += 1
                        if count == cfg.max_turns:
                            del msgs[idx + 1:]
                            break
        return raw

    # Every rank loads the same (small) eval sample lists. The eval callback then
    # shards them across ranks (`samples[rank::world_size]`) and gathers results
    # to the main process, so eval runs ~world_size× faster instead of redundantly.
    # The explicit eval datasets are loaded identically on every rank (a streamed
    # islice of a handful of rows), so the per-rank shards partition cleanly.
    eval_entries: list[SpecDecEvalEntry] = []
    for cfg in _load_yaml_eval_section(sys.argv, "spec_dec_eval", SpecDecEvalConfig):
        eval_entries.append(SpecDecEvalEntry(
            name=cfg.name,
            eval_steps=cfg.eval_steps,
            eval_samples=_load_samples(cfg),
            n_drafts=cfg.n_drafts,
            d_tokens=cfg.d_tokens,
            temperature=cfg.temperature,
            max_characters=cfg.max_characters,
        ))

    full_eval_entries: list[FullEvalEntry] = []
    for cfg in _load_yaml_eval_section(sys.argv, "full_eval", FullEvalConfig):
        full_eval_entries.append(FullEvalEntry(
            name=cfg.name,
            eval_steps=cfg.eval_steps,
            eval_samples=_load_samples(cfg),
            max_length=cfg.max_length,
        ))
    logger.info(
        "Eval sets ready: %d spec-dec + %d full-eval. Starting trainer…",
        len(eval_entries), len(full_eval_entries),
    )

    callbacks = [SpeculativeAcceptanceCallback(
        tokenizer=tokenizer,
        eval_entries=eval_entries,
        full_eval_entries=full_eval_entries,
        batch_size=spec_dec_args.spec_dec_batch_size,
        eval_on_start=spec_dec_args.baseline_eval_on_start,
    )]

    # ------------------------------------------------------------------
    # Training dataset — turn sampling + optional prompt-completion conversion
    #
    # Two loss modes (set in YAML):
    #   assistant_only_loss: true    → conversational format, loss on ALL assistant
    #                                   turns (intermediate turns have reasoning
    #                                   stripped by the chat template; final turn
    #                                   retains its <think> block)
    #   completion_only_loss: true   → prompt-completion format, loss ONLY on the
    #                                   selected final assistant turn (with reasoning)
    #
    # In both modes the turn sampler randomly selects which assistant turn is
    # treated as the final one, dropping all subsequent messages.
    # ------------------------------------------------------------------
    train_dataset = dataset[script_args.dataset_train_split]
    train_dataset = train_dataset.map(_make_turn_sampler(training_args.seed))

    if training_args.completion_only_loss is True:
        train_dataset = train_dataset.map(lambda ex: _to_prompt_completion(ex, tokenizer))

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=get_peft_config(model_args),
        callbacks=callbacks if callbacks else None,
    )

    if spec_dec_args.spec_dec_eval_only:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        return

    trainer.train()
    trainer.accelerator.print("✅ Training completed.")

    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


if __name__ == "__main__":
    from trl import DatasetMixtureConfig, ModelConfig, ScriptArguments, SFTConfig, TrlParser

    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, DatasetMixtureConfig, SpecDecConfig))
    script_args, training_args, model_args, dataset_args, spec_dec_args = parser.parse_args_and_config(
        fail_with_unknown_args=False
    )
    main(script_args, training_args, model_args, dataset_args, spec_dec_args)
