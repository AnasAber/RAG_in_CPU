from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema.document import Document
# from FlagEmbedding.flag_models import FlagModel
# from FlagEmbedding.flag_reranker import FlagReranker
from get_embeddings import get_embeddings
import uuid
import os
from dotenv import load_dotenv
import cohere

load_dotenv()

"""
    Initializating the APIs and setting up the environment variables

"""

api_key = os.getenv("COHERE_API_KEY")

CHROMA_PATH = "chroma"
DATA_PATH = "data"

# init client
co = cohere.Client(api_key=api_key)




# load the data
def get_chroma_db(get_embeddings=get_embeddings):
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())


def main():


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


    
    def retrieve_documents(query, top_k=5):
        chroma_db = get_chroma_db()
        print("#"*100 + "\n\n")

        print("Retrieving documents...")
        results = chroma_db.similarity_search_with_score(query, top_k)
        context_text= "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        print("Documents before reranking: ", context_text)

        return context_text
    

    """
    If you want to use the FlagReranker to rerank the retrieved documents, you can use the following code snippet:

        reranker = FlagModel("BAAI/bge-reranker-v2-m3", use_fp16=True)

        def reranked_documents(query, retrieved_chunks, top_k=3):
            reranked_chunks = reranker.predict(query, retrieved_chunks)
            return [chunk for chunk, _ in reranked_chunks[:top_k]]
        
        Initialize the FlagReranker
        reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        
        
    I'll personally use the cohere API to rerank the documents.
    """


    def format_context(context):
        return "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(context)])


    def reranked_documents(query, long_string, top_k=3):
        # Split the long string into individual chunks using '\n\n---\n\n' as the separator
        chunks = long_string.split("\n\n---\n\n")

        # Ensure all chunks are valid (non-empty) and strip leading/trailing whitespace
        valid_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        if not valid_chunks:
            print("No valid chunks to rerank.")
            return []

        # Use the cohere rerank API
        rerank_docs = co.rerank(
            query=query,
            documents=valid_chunks,
            top_n=top_k,
            model="rerank-english-v2.0"
        )

        print("#"*100 + "\n\n")
        # Extract and print reranked chunks using the indices from the rerank response
        reranked_chunks = [valid_chunks[result.index] for result in rerank_docs.results]
        print("Reranked Chunks:\n\n", format_context(reranked_chunks))

        return reranked_chunks
        

    def get_relevant_data(query):
        retrieved_chunks = retrieve_documents(query)
        reranked_chunks = reranked_documents(query, retrieved_chunks)
        return reranked_chunks
    

    
    def add_to_chroma_db(reranked_chunks):
        chroma_db = get_chroma_db()
        chroma_db.add_documents(reranked_chunks)
        chroma_db.persist()


    return load_documents, split_documents, embed_and_store_documents, retrieve_documents, get_relevant_data, add_to_chroma_db, get_chroma_db
    

if __name__ == "__main__":
    main()
    



