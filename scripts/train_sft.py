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
import itertools
import logging
import os
import random
import re
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))

from speculative_eval import FullEvalEntry, SpecDecConfig, SpecDecEvalEntry, SpeculativeAcceptanceCallback

logger = logging.getLogger(__name__)

_ROLE_MAP = {"human": "user", "gpt": "assistant"}


@dataclasses.dataclass
class BaseEvalConfig:
    dataset: str
    name: str = ""
    n_samples: int = 10
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
    full_text   = tokenizer.apply_chat_template(msgs,      tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    return {**example, "prompt": prompt_text, "completion": full_text[len(prompt_text):]}


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

    if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    if dataset_args.datasets and script_args.dataset_name:
        logger.warning(
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
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    # Remap ShareGPT role names and convert to ChatML format before TRL processes them
    dataset = dataset.map(lambda ex: maybe_convert_to_chatml(remap_roles(ex)))

    # ------------------------------------------------------------------
    # Eval datasets — built from the spec_dec_eval list in the YAML.
    # Each entry specifies a dataset, sample count, and drafting params.
    # ------------------------------------------------------------------

    def _load_samples(cfg: SpecDecEvalConfig | FullEvalConfig) -> list[dict]:
        if cfg.dataset == script_args.dataset_name:
            raw = [
                maybe_convert_to_chatml(remap_roles(dict(s)))
                for s in itertools.islice(dataset[script_args.dataset_train_split], cfg.n_samples)
            ]
        elif os.path.exists(cfg.dataset):
            _stream = load_dataset("parquet", data_files=cfg.dataset, streaming=True, split="train")
            raw = [
                maybe_convert_to_chatml(remap_roles(dict(s)))
                for s in itertools.islice(_stream, cfg.n_samples)
            ]
        else:
            _stream = load_dataset(cfg.dataset, streaming=True, split="train")
            raw = [
                maybe_convert_to_chatml(remap_roles(dict(s)))
                for s in itertools.islice(_stream, cfg.n_samples)
            ]
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

    callbacks = [SpeculativeAcceptanceCallback(
        tokenizer=tokenizer,
        eval_entries=eval_entries,
        full_eval_entries=full_eval_entries,
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
