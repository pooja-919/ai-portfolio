# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader 
from config import DOCUMENTS_PATH
import os

def load_documents(path):
    textDocs = []
    for file in os.listdir(DOCUMENTS_PATH):
        if file.endswith(".pdf"): #if pdf files present
            loader = PyPDFLoader (os.path.join(DOCUMENTS_PATH, file)) 
            for doc in loader.load():
                doc.metadata["source"] = f"[{file}]" #to keep track of 'source file' for each document
            textDocs.extend(loader.load()) #Lazy load: doesn't load all the documents into memory at once

    return textDocs

if __name__ == "__main__":
    documents = load_documents(DOCUMENTS_PATH)
    print(f"Loaded {len(documents)} documents.")

