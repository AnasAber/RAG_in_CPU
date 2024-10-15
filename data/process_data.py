from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.prompts.prompts import STRUCTURED_CV_RESUME_TEMPLATE
from src.data_processing.get_embeddings import get_embeddings
from src.models.models import llama_groq_structured
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from langchain.prompts import PromptTemplate

CHROMA_PATH = "data/processed/chroma"
DATA_PATH = "data"


def load_documents():
    document_loader = PyPDFDirectoryLoader("data/test")
    print("Loading documents...")
    return document_loader.load()

def split_documents(documents):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    docs = []

    # to better structure the content
    structured_docs = []
    prompt = PromptTemplate(
        template=STRUCTURED_CV_RESUME_TEMPLATE,
        input_variables=["context"],
    ).format(context=documents)

    print("beginning of further structuring")
    structured_docs = llama_groq_structured(prompt)
    print("End of furthr structuring")
    # end of structuring

    print("Splitting documents...")
    for document in documents:
        for chunk in text_splitter.split_text(document.page_content):
            docs.append(Document(page_content=chunk, metadata={"source": document.metadata["source"]}))

    # Ensure structured_docs is either a string or a list of strings
    print("Splitting structured content...")
    try:
        if isinstance(structured_docs, str):
            # If structured_docs is a single string, split and append
            for chunk in text_splitter.split_text(structured_docs):
                docs.append(Document(page_content=chunk, metadata={"source": "structured_content"}))
        elif isinstance(structured_docs, list):
            # If structured_docs is a list of strings, handle each string separately
            for structured_doc in structured_docs:
                for chunk in text_splitter.split_text(structured_doc):
                    docs.append(Document(page_content=chunk, metadata={"source": "structured_content"}))
    except Exception as e:
        print(f"Unexpected document type: {type(document)}")

    print("Documents split successfully.")
    return docs




def embed_and_store_documents(chunks):
    chroma_db = Chroma(
    persist_directory=CHROMA_PATH, embedding_function=get_embeddings()
    )
    print("Storing documents...")
    print(f"document type: {type(chunks)}")
    # hna we can re structure the chunks or the content of the cv to make the retrieval more efficient
    chroma_db.add_documents(chunks, persist_directory=CHROMA_PATH,embeddings=get_embeddings())
    chroma_db.persist()
    print("Documents stored successfully.")

