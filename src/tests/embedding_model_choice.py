from transformers import AutoModel, AutoTokenizer, pipeline
from transformers import BertTokenizer, BertModel
from dotenv import load_dotenv
import numpy as np
import torch
import os



        
load_dotenv()

api_key = os.getenv("HUGGINGFACE_API_KEY")

model1 = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2", api_key=api_key)

tokenizer2 = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")
m2 = AutoModel.from_pretrained("intfloat/e5-base-v2", use_auth_token=api_key)
model2 = pipeline("feature-extraction", model=m2, tokenizer=tokenizer2)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
m = BertModel.from_pretrained('bert-base-uncased')
model = pipeline("feature-extraction", model=m, tokenizer=tokenizer)

tokenizer3 = AutoTokenizer.from_pretrained("BAAI/bge-m3")
m3 = AutoModel.from_pretrained("BAAI/bge-m3", use_auth_token=api_key)
model3 = pipeline("feature-extraction", model=m3, tokenizer=tokenizer3)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


tokenizer4 = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
m4 = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
model4 = pipeline("feature-extraction", model=m4, tokenizer=tokenizer4)


# length of the embeddings
len_of_model1 = len(model1("This is a test")[0][0])
len_of_model2 = len(model2("This is a test")[0][0])   
len_of_model = len(model("This is a test")[0][0])
len_of_model3 = len(model3("This is a test")[0][0]) 
len_of_model4 = len(model4("This is a test")[0][0])


print(f"Length of model1 embeddings: {len_of_model1}")
print(f"Length of model2 embeddings: {len_of_model2}")
print(f"Length of model embeddings: {len_of_model}")
print(f"Length of model3 embeddings: {len_of_model3}")
print(f"Length of model4 embeddings: {len_of_model4}")



# now we want to test the embeddings of each model
# Using the cosine similarity function to compare the embeddings

def cosine_similarity(embeddings1, embeddings2):
    """
        A function to compute the cosine similarity between two embeddings.
        It returns the cosine similarity score.
    """
    dot_product = np.dot(embeddings1, embeddings2)
    norm1 = np.linalg.norm(embeddings1)
    norm2 = np.linalg.norm(embeddings2)
    return dot_product / (norm1 * norm2)

print("#"*100)

# Example of the same embedding
print("We are testing the same embeddings")
text1 = "That is a sad person"
text2 = text1

print(cosine_similarity(model1(text1)[0][0], model1(text2)[0][0]))
print("-----------------")
print(cosine_similarity(model2(text1)[0][0], model2(text2)[0][0]))
print("-----------------")
print(cosine_similarity(model(text1)[0][0], model(text2)[0][0]))
print("-----------------")
print(cosine_similarity(model3(text1)[0][0], model3(text2)[0][0]))
print("-----------------")
print(cosine_similarity(model4(text1)[0][0], model4(text2)[0][0]))



print("#"*50)
print("We are testing the same embedding but in different order")

text1 = "That is a sad person"
text2 = "That person is sad"

print(cosine_similarity(model1(text1)[0][0], model1(text2)[0][0]))
print("-----------------")
print(cosine_similarity(model2(text1)[0][0], model2(text2)[0][0]))
print("-----------------")
print(cosine_similarity(model(text1)[0][0], model(text2)[0][0]))
print("-----------------")
print(cosine_similarity(model3(text1)[0][0], model3(text2)[0][0]))
print("-----------------")
print(cosine_similarity(model4(text1)[0][0], model4(text2)[0][0]))



print("#"*50)
print("We are testing the opposite embeddings (opposite meaning)")

text1 = "That is a sad person"
text2 = "That person is happy"

print(cosine_similarity(model1(text1)[0][0], model1(text2)[0][0]))
print("-----------------")
print(cosine_similarity(model2(text1)[0][0], model2(text2)[0][0]))
print("-----------------")
print(cosine_similarity(model(text1)[0][0], model(text2)[0][0]))
print("-----------------")
print(cosine_similarity(model3(text1)[0][0], model3(text2)[0][0]))
print("-----------------")
print(cosine_similarity(model4(text1)[0][0], model4(text2)[0][0]))



print("#"*50)
print("We are testing completely different meanings")

text1 = "That is a sad person"
text2 = "I ate an apple this morning"

print(cosine_similarity(model1(text1)[0][0], model1(text2)[0][0]))
print("-----------------")
print(cosine_similarity(model2(text1)[0][0], model2(text2)[0][0]))
print("-----------------")
print(cosine_similarity(model(text1)[0][0], model(text2)[0][0]))
print("-----------------")
print(cosine_similarity(model3(text1)[0][0], model3(text2)[0][0]))
print("-----------------")
print(cosine_similarity(model4(text1)[0][0], model4(text2)[0][0]))

print("The best model is: model3")