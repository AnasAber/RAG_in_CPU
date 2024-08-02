import src.data_processing.get_embeddings
from data.process_data import load_documents, embed_and_store_documents, split_documents
from langchain.prompts import ChatPromptTemplate
from groq import Groq
import os
from src.database.chroma_search_functions import get_relevant_data

"""
    Importing the functions and setting up the environment variables

"""

CHROMA_PATH = "chroma/"
DATA_PATH = "data/raw"

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)


"""
Again, if we want to load a huggingFace model and tokenizer, we can do it like this:

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
from langchain.llms import HuggingFacePipeline

model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
model_name,
)

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=512,  # Reduced number of tokens
    device=-1  # Ensure it's running on CPU
)

GemmaLLM = HuggingFacePipeline(pipeline=text_generation_pipeline)

"""




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

def main():

    """
    loading documents should be performed only once, ti will take a bit of time at first.
    You can comment them out as chromaDB has the infos already
    
    """
    check_and_process_documents()

    if not os.path.exists("data/processed/chroma"):
        documents = load_documents()
        chunks = split_documents(documents)
        embed_and_store_documents(chunks)
        print("Documents loaded, split, and stored")
    



    query = "How to enter prison in Monopoly?"

    PROMPT_TEMPLATE = """
    Answer this question in a clear, unboring matter,  based on the follwing context:
    {context}

    -----

    Answer this question based on the above context, without siting the context in your answer:
    {question}

    Answer:
    """

    results = get_relevant_data(query)

    # context_text= "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=results, question=query)

    print("#"*100 + "\n\n")


    chat_completion = client.chat.completions.create(
        #
        # Required parameters
        #
        messages=[
                # Set an optional system message. This sets the behavior of the
                # assistant and can be used to provide specific instructions for
                # how it should behave throughout the conversation.
                {
                    "role": "system",
                    "content": prompt
                },
                # Set a user message for the assistant to respond to.
                {
                    "role": "user",
                    "content": query,
                }
            ],

            # The language model which will generate the completion.
            model="llama3-70b-8192",

            #
            # Optional parameters
            #

            # Controls randomness: lowering results in less random completions.
            # As the temperature approaches zero, the model will become deterministic
            # and repetitive.
            temperature=0.5,

            # The maximum number of tokens to generate. Requests can use up to
            # 2048 tokens shared between prompt and completion.
            max_tokens=1024,

            # Controls diversity via nucleus sampling: 0.5 means half of all
            # likelihood-weighted options are considered.
            top_p=1,

            # A stop sequence is a predefined or user-specified text string that
            # signals an AI to stop generating content, ensuring its responses
            # remain focused and concise. Examples include punctuation marks and
            # markers like "[end]".
            stop=None,

            # If set, partial message deltas will be sent.
            stream=False,
        )


    response = chat_completion.choices[0].message.content

    print("Response: ", response)


if __name__ == "__main__":
    main()




