from loader import load_documents
from chunking import split_documents
from vector_store import create_vector_store
from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI ##credit issue hence used ChatOllama
from dotenv import load_dotenv
import os
from langchain_community.retrievers import BM25Retriever
# from langchain_community.retrievers import EnsembleRetriever
from config import DOCUMENTS_PATH
load_dotenv() 

chat_history = []

def get_retriever(vector_store,chunks): #hybrid retrieval: combines vector search and keyword search for better results
    # vector_retriever = vector_store.as_retriever(
    #     search_kwargs={"k": 8}
    # )
    # keyword_retriever = BM25Retriever.from_documents(chunks) #some issue with EnsembleRetriever import not present in langchain_community
    # keyword_retriever.k = 4
    # retriever = EnsembleRetriever(
    #     retrievers=[vector_retriever, keyword_retriever],
    #     weights=[0.5, 0.5]
    # )
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 8} #to give more number of relevant contexts, if that many
    )
    return retriever

def generate_answer(retriever, query):
    docs = retriever.invoke(query) #docs is a list of contexts retrieved from vector store basing the query
    print(".......Retrieved docs:", len(docs)) #to remove
    for i, doc in enumerate(docs):
        print(f".....Doc {i} preview:", doc.page_content[:100]) #to remove
    context = "\n\n".join([doc.page_content for doc in docs])
    history_text = "\n".join(chat_history)
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
    prompt = prompt = f"""
    You are a helpful assistant.
    Use the provided context to answer the question.
    If the answer is partially in the context, explain it clearly.
    If the context is insufficient, say that clearly.
    Conversation History:
    {history_text}

    Context:
    {context}

    Question:
    {query}
    """
    llm = ChatOllama(model="llama3")
    response = llm.invoke(prompt)
    answer = response.content
    source_text = "\n".join(sources) #add sources in the final llm output
    final_output = f"{answer}\n\nSources:\n{source_text}"
    chat_history.append(f"user query: {query}")
    chat_history.append(f"AI answer: {answer}")
    return final_output,context,sources

if __name__ == "__main__":
    docs = load_documents(DOCUMENTS_PATH)
    chunks = split_documents(docs)
    vector_store = create_vector_store(chunks)
    retriever = get_retriever(vector_store,chunks)
    print("\nRAG Assistant Ready!")
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        final_output,context,sources = generate_answer(retriever, query)
        print(f"\nQuery:\n{query}")
        print(f"\nResponse:\n{final_output}")


        