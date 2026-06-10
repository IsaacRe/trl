import os

import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F402

BASE_MODEL = os.environ.get("BASE_MODEL", None)
assert (
    BASE_MODEL
), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=huggyllama/llama-7b`"  # noqa: E501

LORA_MODEL = os.environ.get("LORA_MODEL", None)
assert (
    BASE_MODEL
), "Please specify a value for LORA_MODEL environment variable, e.g. `export LORA_MODEL=huggyllama/llama-7b`"  # noqa: E501


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map={"": "cpu"},
)

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL,
    device_map={"": "cpu"},
    torch_dtype=torch.bfloat16,
)

lora_weight = lora_model.base_model.model.model.layers[
    0
].self_attn.q_proj.weight

assert torch.allclose(first_weight_old, first_weight)

# merge weights - new merging method from peft
lora_model = lora_model.merge_and_unload()

lora_model.train(False)

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./hf_ckpt")
base_model.save_pretrained(
    OUTPUT_DIR, state_dict=deloreanized_sd, max_shard_size="5000MB"
)
tokenizer.save_pretrained(OUTPUT_DIR)