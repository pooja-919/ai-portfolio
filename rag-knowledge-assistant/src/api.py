from fastapi import FastAPI
from pydantic import BaseModel

from loader import load_documents
from chunking import split_documents
from vector_store import create_vector_store
from retriever import get_retriever, generate_answer
from config import DOCUMENTS_PATH


app = FastAPI()
print("Loading RAG system...")

docs = load_documents(DOCUMENTS_PATH)
chunks = split_documents(docs)
vector_store = create_vector_store(chunks)
retriever = get_retriever(vector_store, chunks)
print("RAG system ready!")

class QuestionRequest(BaseModel):
    question: str
@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        response,context,sources = generate_answer(retriever, request.question)
        return {"answer": response, "context": context, "sources": sources}
    except Exception as e:
        return {"error": str(e)}
    
