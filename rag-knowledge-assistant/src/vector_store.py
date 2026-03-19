from langchain_community.vectorstores import FAISS
from config import VECTOR_DB_PATH 
from embeddings import get_embedding_model
import os

def create_vector_store(chunks):
    embeddings = get_embedding_model()
    if os.path.exists(VECTOR_DB_PATH):
        print("Loading existing vector database...")
        vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating new vector database...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(VECTOR_DB_PATH)
    return vector_store