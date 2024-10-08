from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema.document import Document
from src.data_processing.get_embeddings import get_embeddings

CHROMA_PATH = "data/processed/chroma"
DATA_PATH = "data"


def load_documents():
    document_loader = PyPDFDirectoryLoader("data/")
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
    print("Splitting documents...")
    for document in documents:
        for chunk in text_splitter.split_text(document.page_content):
            docs.append(Document(page_content=chunk, metadata={"source": document.metadata["source"]}))
    print("Documents split successfully.")
    return docs


def embed_and_store_documents(chunks):
    chroma_db = Chroma(
    persist_directory=CHROMA_PATH, embedding_function=get_embeddings()
    )
    print("Storing documents...")
    chroma_db.add_documents(chunks, persist_directory=CHROMA_PATH,embeddings=get_embeddings())
    chroma_db.persist()
    print("Documents stored successfully.")

