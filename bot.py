import os
import PyPDF2
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as pdf_file_obj:
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
        text = ''
        for page_obj in pdf_reader.pages:
            text += page_obj.extract_text()
    return text

def prepare_qa_data(qa_directory, pdf_directory):
    qa_data = []
    for filename in os.listdir(qa_directory):
        # Check if the file is a CSV
        if filename.endswith('.csv'):
            # Construct the full file path
            csv_file_path = os.path.join(qa_directory, filename)
            # Extract the base name of the file (without extension)
            base_filename = os.path.splitext(filename)[0]
            # Construct the full file path of the corresponding PDF
            pdf_file_path = os.path.join(pdf_directory, f'{base_filename}.pdf')
            # Read the CSV file
            qa_df = pd.read_csv(csv_file_path, converters={'Question': str,'Answer': str})
            # Extract the text from the PDF file
            text = extract_text_from_pdf(pdf_file_path)
            # Preprocess the text
            # preprocessed_text = preprocess_text(text)
            # Iterate over the rows in the CSV file
            for _, row in qa_df.iterrows():
                question = row['Question']
                answer = row['Answer']
                # Include the filename of the corresponding PDF
                pdf_filename = base_filename + '.pdf'
                qa_data.append({'pdf_filename': pdf_filename, 'context': text, 'question': question, 'answer': answer })
    return qa_data

qa_directory = 'QNA'
pdf_directory = 'pdf'
qa_data = prepare_qa_data(qa_directory, pdf_directory)
print(len(qa_data))

import json

def save_as_jsonl(data, filename):
    with open(filename, 'w') as f:
        for i in data:
            json_str = json.dumps(i)
            print(json_str)
            f.write(json_str + "\n")

qa_directory = 'QNA'
pdf_directory = 'pdf'
qa_data = prepare_qa_data(qa_directory, pdf_directory)

train_size = int(len(qa_data) * 0.8)
train_data = qa_data[:train_size]
test_data = qa_data[train_size:]

save_as_jsonl(train_data, "train.jsonl")
save_as_jsonl(test_data, "test.jsonl")


data_files = {"train":"train.jsonl", "test":"test.jsonl"}
dataset = load_dataset("json", data_files=data_files)
dataset
print(dataset["train"][0])
dataset.push_to_hub("KOKO")