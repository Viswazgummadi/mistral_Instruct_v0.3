import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import re
device = "cuda"
model_id = "Dobby091/KOKO"
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16
)
os.environ["HF_TOKEN"] = "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"
os.environ['HF_HOME'] = '/mnt/data1/backup/viswaz/Project_K/huggingface_cache/'

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    cache_dir="/mnt/data1/backup/viswaz/Project_K/huggingface_cache/",

)

tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def generate_response(prompt, model, max_output_tokens=256, num_beams=5, length_penalty=1.3, num_return_sequences=1):
    encoded_input = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=True)
    model_inputs = encoded_input.to(device)

    try:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_output_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )

        decoded_output = tokenizer.batch_decode(generated_ids)

        answer = decoded_output[0].split('<s>')[1].split('\n')
    # remove the last line, which is an empty line
        answer = '\n'.join(answer[:-1])
        lines = answer.split('\n')
        for i, line in enumerate(lines):
            if re.match(r'^\d+\s*$', line, re.MULTILINE):
                answer = '\n'.join(lines[:i])
                break

        return answer

    except Exception as e:
        print(f"Error during generation: {e}")
        return None


prompt = """
What does value represent in the 'Munsell' system?"""


response = generate_response(
    prompt, model, num_beams=5, length_penalty=0.8, num_return_sequences=3)

print(response)
