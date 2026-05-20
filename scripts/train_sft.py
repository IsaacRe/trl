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

sys.path.insert(0, os.path.dirname(__file__))

from speculative_eval import SpecDecConfig, SpeculativeAcceptanceCallback

_ROLE_MAP = {"human": "user", "gpt": "assistant"}


def remap_roles(example):
    for msg in example.get("conversations") or example.get("messages") or []:
        if isinstance(msg, dict) and msg.get("from") in _ROLE_MAP:
            msg["from"] = _ROLE_MAP[msg["from"]]
    return example


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

    # Remap ShareGPT role names before TRL processes them
    dataset = dataset.map(remap_roles)

    # ------------------------------------------------------------------
    # Eval dataset — either a separate held-out HF dataset or (fallback)
    # the first N samples carved from the training stream.
    # ------------------------------------------------------------------
    n_eval = spec_dec_args.spec_dec_n_eval_samples
    if spec_dec_args.spec_dec_eval_dataset:
        _eval_stream = load_dataset(spec_dec_args.spec_dec_eval_dataset, streaming=True, split="train")
        raw_eval = [
            maybe_convert_to_chatml(remap_roles(dict(s)))
            for s in itertools.islice(_eval_stream, n_eval)
        ]
    else:
        raw_eval = [
            maybe_convert_to_chatml(dict(s))
            for s in itertools.islice(dataset[script_args.dataset_train_split], n_eval)
        ]
    eval_dataset = Dataset.from_list(raw_eval) if training_args.eval_strategy != "no" else None

    # ------------------------------------------------------------------
    # Speculative acceptance callback (runs whenever eval runs)
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    callbacks = [
        SpeculativeAcceptanceCallback(
            config=spec_dec_args,
            tokenizer=tokenizer,
            eval_samples=raw_eval,
        )
    ]

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
