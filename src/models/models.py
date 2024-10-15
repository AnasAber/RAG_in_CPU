from dotenv import load_dotenv
from groq import Groq
import cohere
import os


load_dotenv()


client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

# init client
co = cohere.Client(os.getenv("COHERE_API_KEY"))


def llama_groq(query, prompt):
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
    return response


def llama_groq_structured(prompt):
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
            ],

            model="llama3-70b-8192",

            temperature=0.5,

            max_tokens=1024,

            top_p=1,

            stop=None,

            stream=False,
        )

    response = chat_completion.choices[0].message.content
    return response

def cohere_reranker(query, valid_chunks, top_k=3):
    # Use the cohere rerank API
    rerank_docs = co.rerank(
        query=query,
        documents=valid_chunks,
        top_n=top_k,
        model="rerank-english-v2.0"
    )
    return rerank_docs