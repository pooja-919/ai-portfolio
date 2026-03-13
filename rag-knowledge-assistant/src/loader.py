# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader 
import os

def load_documents(folder_path):
    textDocs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"): #if textfiles present
            loader = PyPDFLoader (os.path.join(folder_path, file)) 
            textDocs.extend(loader.lazy_load()) #Lazy load: doesn't load all the documents into memory at once
    return textDocs

if __name__ == "__main__":
    documents = load_documents("C:\\Users\\pkashyap\\pooja-919Github\\ai-portfolio\\rag-knowledge-assistant\\data\\documents")
    print(f"Loaded {len(documents)} documents.")

