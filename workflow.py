from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from datasets import DatasetDict, Dataset
import re

# import any embedding model on HF hub (https://huggingface.co/spaces/mteb/leaderboard)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Settings.embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large") # alternative model

Settings.llm = None
Settings.chunk_size = 256
Settings.chunk_overlap = 25

# articles available here:  {add GitHub repo}
documents = SimpleDirectoryReader("pdf").load_data()

# some ad hoc document refinement
for doc in documents:
    if "Member-only story" in doc.text:
        documents.remove(doc)
        continue

    if "The Data Entrepreneurs" in doc.text:
        documents.remove(doc)

    if " min read" in doc.text:
        documents.remove(doc)

# store docs into vector DB
index = VectorStoreIndex.from_documents(documents)
# set number of docs to retrieve
top_k = 3

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=top_k,
)
# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)

# load questions
with open('questions.txt', 'r') as file:
    questions = file.readlines()

# strip newline characters from questions
questions = [q.strip() for q in questions]

# create a list to store the mapping of questions to their respective contexts
question_context_pairs = []

# iterate over each question and query the engine
for i, question in enumerate(questions):
    response = query_engine.query(question)

    # create a combined context from the top 3 chunks
    context = "Context:\n"
    for j in range(top_k):
        context = context + response.source_nodes[j].text + "\n\n"

    # store the mapping of the question to its respective context
    question_context_pairs.append({"question": question, "context": context})

# create a dictionary with two lists: questions and contexts
data = {
    "question": [pair["question"] for pair in question_context_pairs],
    "context": [pair["context"] for pair in question_context_pairs]
}

# create a Hugging Face Dataset from the dictionary
dataset = Dataset.from_dict(data)


from transformers import AutoTokenizer, AutoModelForCausalLM

model_id="mistralai/Mistral-7B-Instruct-v0.2"
# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    cache_dir = "/mnt/data1/backup/viswaz/Project_K/huggingface_cache/",
)


def generate_response(prompt, model):
  encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
  model_inputs = encoded_input.to('cuda')

  generated_ids = model.generate(**model_inputs,
                                 max_new_tokens=150,
                                 do_sample=True,
                                 pad_token_id=tokenizer.eos_token_id)

  return generated_ids



# prompt (context)
intstructions_string = f""" you are a textbot that helps in finding answers to questions in the research papers, blogs,pdf's or any text context.
,make your answers more meaningful and short"

please answer the following question
"""
prompt_template_w_context = lambda context, question: f'''[INST] {intstructions_string}

{context}

Please answer to the following question. Use the context above if it is helpful.

{question}

[/INST]'''

# Initialize an empty list to store the texts between [/INST] and </s> tags
texts_between_inst_and_eos = []

# Iterate over each question in the dataset
for i in range(len(dataset)):
    # Get the question and the context
    question = dataset[i]["question"]
    context = dataset[i]["context"]

    # Define the prompt as the concatenation of the context and the question
    prompt = prompt_template_w_context(context, question)

    outputs = generate_response(prompt, model)

    # Decode the generated IDs
    decoded_output = tokenizer.batch_decode(outputs)[0]

    # Extract texts between [/INST] and </s> tags
    inst_eos_texts = re.findall(r'\[\/INST\](.*?)\<\/s\>', decoded_output, re.DOTALL)
    texts_between_inst_and_eos.extend(inst_eos_texts)

# Print the texts between [/INST] and </s> tags
# print("Texts between [/INST] and </s> tags:")
for text in texts_between_inst_and_eos:
    print(text.strip())
    print("\n")
    
