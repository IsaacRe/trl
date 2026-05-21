"""
SFT training wrapper that:
  - Remaps ShareGPT role names (human→user, gpt→assistant)
  - Carves a held-out eval set from the training stream for val loss + perplexity
  - Optionally attaches speculative decoding acceptance eval (see SpecDecConfig)

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_sft.py --config configs/sft_full.yaml
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_sft.py --config configs/sft_lora.yaml
"""

import itertools
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))

from speculative_eval import SpecDecConfig, SpecDecEvalEntry, SpeculativeAcceptanceCallback

_ROLE_MAP = {"human": "user", "gpt": "assistant"}


def remap_roles(example):
    for msg in example.get("conversations") or example.get("messages") or []:
        if isinstance(msg, dict) and msg.get("from") in _ROLE_MAP:
            msg["from"] = _ROLE_MAP[msg["from"]]
    return example


def _load_spec_dec_eval(argv: list[str]) -> list[dict]:
    """Extract the spec_dec_eval list directly from the --config YAML."""
    import yaml
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            with open(argv[i + 1]) as f:
                return yaml.safe_load(f).get("spec_dec_eval", [])
    return []


def _compose_run_name(training_args, model_args) -> str:
    model_short = model_args.model_name_or_path.split("/")[-1].replace("-", "_")
    rank_str = f"r{model_args.lora_r}" if getattr(model_args, "use_peft", False) else "full"
    import re
    lr_str = re.sub(r"e([+-])0*(\d)", r"e\1\2", f"{training_args.learning_rate:.0e}")
    parts = [model_short]
    if training_args.run_name:
        parts.append(training_args.run_name)
    parts += [rank_str, f"lr{lr_str}"]
    return "-".join(parts)


def main(script_args, training_args, model_args, dataset_args, spec_dec_args):
    from accelerate import logging
    from datasets import Dataset, load_dataset
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

    spec_dec_eval_list = _load_spec_dec_eval(sys.argv)
    eval_entries: list[SpecDecEvalEntry] = []
    for entry_cfg in spec_dec_eval_list:
        dataset_id = entry_cfg["dataset"]
        n_samples  = entry_cfg.get("n_samples", 10)
        name       = entry_cfg.get("name") or dataset_id.split("/")[-1].split("-")[0]

        if dataset_id == script_args.dataset_name:
            raw = [
                maybe_convert_to_chatml(dict(s))
                for s in itertools.islice(dataset[script_args.dataset_train_split], n_samples)
            ]
        elif os.path.exists(dataset_id):
            _stream = load_dataset("parquet", data_files=dataset_id, streaming=True, split="train")
            raw = [
                maybe_convert_to_chatml(remap_roles(dict(s)))
                for s in itertools.islice(_stream, n_samples)
            ]
        else:
            _stream = load_dataset(dataset_id, streaming=True, split="train")
            raw = [
                maybe_convert_to_chatml(remap_roles(dict(s)))
                for s in itertools.islice(_stream, n_samples)
            ]

        eval_entries.append(SpecDecEvalEntry(
            name=name,
            n_drafts=entry_cfg.get("n_drafts", 4),
            d_tokens=entry_cfg.get("d_tokens", 8),
            temperature=entry_cfg.get("temperature", 1.0),
            eval_samples=raw,
            max_characters=entry_cfg.get("max_characters"),
        ))

    all_raw_eval = [s for e in eval_entries for s in e.eval_samples]
    eval_dataset = Dataset.from_list(all_raw_eval) if training_args.eval_strategy != "no" and all_raw_eval else None

    callbacks = [SpeculativeAcceptanceCallback(tokenizer=tokenizer, eval_entries=eval_entries)]

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset,
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
