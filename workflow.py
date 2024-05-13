import os
import PyPDF2
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

def extract_text_from_file(file_path):
    with open(file_path, 'rb') as file:
        # Determine the type of the file based on its extension
        if os.path.splitext(file_path)[1] == '.pdf':
            # If the file is a PDF, use PyPDF2 to extract the text
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page_obj in pdf_reader.pages:
                text += page_obj.extract_text()
        else:
            # If the file is a text file, simply read the file
            file = open(file_path, 'r')
            text = file.read()
    return text

def prepare_qa_data(qa_prompt, file_paths):
    qa_data = []
    for file_path in file_paths:
        # Extract the base name of the file (without extension)
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        # Extract the text from the file
        text = extract_text_from_file(file_path)
        # Iterate over the questions in the prompt
        for question in qa_prompt.split('\n'):
            # Include the filename of the corresponding file
            file_name = base_filename + os.path.splitext(file_path)[1]
            # Since we don't have the answer in the prompt, we'll set it to None
            answer = None
            qa_data.append({'file_name': file_name, 'context': text, 'question': question, 'answer': answer })
    return qa_data

def save_as_jsonl(data, filename):
    with open(filename, 'w') as f:
        for i in data:
            f.write(json.dumps(i) + "\n")

qa_prompt = """
Question 1
Question 2
Question 3
"""

file_paths = [
    '/mnt/data1/backup/viswaz/Project_K/pdf/228_4.pdf',
    '/mnt/data1/backup/viswaz/Project_K/pdf/228_5.pdf',
    # Add more files here
]

# Limit the number of files
file_paths = file_paths[:10]

qa_data = prepare_qa_data(qa_prompt, file_paths)

# Save the data as a single JSONL file
save_as_jsonl(qa_data, "qa_data.jsonl")

# Load the dataset
dataset = load_dataset('json', data_files='qa_data.jsonl')

# Set up the embedding model and the retriever
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.chunk_size = 256
Settings.chunk_overlap = 25
documents = SimpleDirectoryReader(".").load_data()
index = VectorStoreIndex.from_documents(documents)
top_k = 3
retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])

# Load the tokenizer


# Define a function to create the prompt
def create_prompt(sample):
    bos_token = "<s>"
    base_prompt1 = "below context is from "
    base_prompt2 = ", answer the following questions based on the context given \n"
    file_name = sample['file_name']
    context = sample['context']
    question = sample['question']
    eos_token = "</s>"
    full_prompt = ""
    full_prompt += bos_token
    full_prompt += "[INST]"
    full_prompt += "###Instruction:\n"
    full_prompt += base_prompt1
    full_prompt += file_name
    full_prompt += base_prompt2
    full_prompt += "\n\n###file_name:\n" + file_name
    full_prompt += "\n\n###context:\n" + context
    full_prompt += "\n\n###question:\n" + question
    full_prompt += "[/INST]"
    full_prompt += "\n\n###answer:\n"
    full_prompt += eos_token
    return full_prompt


# Define a function to answer a question using the retriever
def answer_question(question, query_engine):
    response = query_engine.query(question)
    context = "Context:\n"
    for i in range(top_k):
        context = context + response.source_nodes[i].text + "\n\n"
    return context

# Define a function to create a prompt and answer a question
def create_prompt_and_answer(sample, query_engine):
    prompt = create_prompt(sample)
    question = sample['question']
    context = answer_question(question, query_engine)
    full_prompt = prompt + context
    return full_prompt

# Create a prompt and answer a question
sample = dataset['train'][0]
prompt_and_answer = create_prompt_and_answer(sample, query_engine)
print(prompt_and_answer)


# Load the language model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

device = "cuda"
# model_id="mistralai/Mistral-7B-Instruct-v0.2"
model_id="mistralai/Mistral-7B-v0.1"
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16
)

import os
os.environ["HF_TOKEN"] = "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"
os.environ['HF_HOME'] = '/mnt/data1/backup/viswaz/Project_K/huggingface_cache/'

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    cache_dir = "/mnt/data1/backup/viswaz/Project_K/huggingface_cache/",
    
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_size = "right"

# Define a function to generate a response using the language model
def generate_response(prompt, model):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)
    response = tokenizer.batch_decode(outputs)[0]
    return response

# Generate a response using the language model
sample = dataset['train'][0]
prompt_and_answer = create_prompt_and_answer(sample, query_engine)
response = generate_response(prompt_and_answer, model)
print(response)
