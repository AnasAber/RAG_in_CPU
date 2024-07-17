# Loading our model and making a get_embeddings function

import os
import numpy as np
from transformers import AutoModel, AutoTokenizer, pipeline
from transformers import BertTokenizer, BertModel
from dotenv import load_dotenv
import torch
from langchain.embeddings.base import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings

        
load_dotenv()

api_key = os.getenv("HUGGINGFACE_API_KEY")


"""
if we want to manually load the model and tokenizer


tokenizer4 = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
m4 = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
model4 = pipeline("feature-extraction", model=m4, tokenizer=tokenizer4)

global embedding_model

embedding_model = model4

"""

def get_embeddings(text=None):
    embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1")
    return embeddings

__all__ = ['get_embeddings', 'get_embeddings_query']




# (1, 8, 1024)

