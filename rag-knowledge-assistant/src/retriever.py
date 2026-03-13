from loader import load_documents
from chunking import split_documents
from vector_store import create_vector_store
from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI ##credit issue hence used ChatOllama
from dotenv import load_dotenv
import os

load_dotenv() 

def get_retriever(vector_store):
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )
    return retriever

def generate_answer(retriever, query):
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
    Use the context below to answer the question and if you don't know the answer, say you don't know. Be concise.
    Context:{context}, Question:{query}, Answer:
    """
    llm = ChatOllama(model="llama3")
    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    docs = load_documents("C:\\Users\\pkashyap\\pooja-919Github\\ai-portfolio\\rag-knowledge-assistant\\data\\documents")
    chunks = split_documents(docs)
    vector_store = create_vector_store(chunks)
    retriever = get_retriever(vector_store)
    query = input("Enter your query: ")
    # query = "What are the documents about?"
    # results = retriever.invoke(query)

    # for i, doc in enumerate(results):
    #     print(f"\nResult {i+1}")
    #     print(doc.page_content)

    answer = generate_answer(retriever, query)
    print(f"\nQuery:\n{query}")
    print(f"\nResponse:\n{answer}")

        