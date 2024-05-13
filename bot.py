from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# model_id="mistralai/Mistral-7B-Instruct-v0.2"
model_id="mistralai/Mixtral-8x7B-v0.1"
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

import os
os.environ["HF_TOKEN"] = "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"

cache_dir = "/mnt/data1/viswaz/Project_K/huggingface_cache"
os.environ["HUGGINGFACE_CACHE_DIR"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = "/mnt/data1/viswaz/Project_K/huggingface_cache"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=nf4_config,
    use_cache=False,

)