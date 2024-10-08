from src.main_reasoning import reasoning


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


def main():
    


    query = "How to enter prison in Monopoly?"



    PROMPT_TEMPLATE = """
    Answer this question in a clear, unboring matter, based on the follwing context:
    {context}

    -----

    Answer this question based on the above context, without siting the context in your answer:
    {question}

    Answer:
    """

    response = reasoning(query, PROMPT_TEMPLATE)

    print("Response: ", response)


if __name__ == "__main__":
    main()




