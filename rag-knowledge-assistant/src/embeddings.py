# import os
# from langchain_openai import OpenAIEmbeddings ##credit issue hence used HuggingFaceEmbeddings
# from dotenv import load_dotenv
# load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings