from data.process_data import load_documents, embed_and_store_documents, split_documents
from langchain.prompts import ChatPromptTemplate
from src.database.chroma_search_functions import get_relevant_data
from src.models.models import llama_groq
import os


def format_context(context):
    return "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(context)])


def check_and_process_documents():
    path = "data/processed/chroma"
    print(f"Checking if path exists: {path}")
    
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        
        documents = load_documents()
        print("Documents loaded")
        
        chunks = split_documents(documents)
        print("Documents split into chunks")
        
        embed_and_store_documents(chunks)
        print("Documents embedded and stored")
    else:
        print(f"Path already exists: {path}")



def reasoning(query, prompt):

    check_and_process_documents()

    print("#"*100 + "\n\n")
    
    results = get_relevant_data(query)

    prompt_template = ChatPromptTemplate.from_template(prompt)
    prompt = prompt_template.format(context=results, question=query)
    response = llama_groq(query, prompt)
    return response