from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
device = "cuda"
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)
import os
os.environ["HF_TOKEN"] = "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"
os.environ['HF_HOME'] = '/mnt/data1/backup/viswaz/Project_K/huggingface_cache/'

model = AutoModelForCausalLM.from_pretrained(
       "mistralai/Mistral-7B-Instruct-v0.2",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir = "/mnt/data1/backup/viswaz/Project_K/huggingface_cache/",
        trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
torch.save(model,"./mistral_inst.pt")
messages = [
    {"role": "user", "content": "who are you?"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])