from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        separators=[
            "\n\n",   # paragraph
            "\n",     # line
            ". ",     # sentence
            " ",      # word
        ]
    )
    chunks = splitter.split_documents(documents)
    return chunks

if __name__== "__main__":
    documents = load_documents("C:\\Users\\pkashyap\\pooja-919Github\\ai-portfolio\\rag-knowledge-assistant\\data\\documents")
    chunks = split_documents(documents)
    print(f"Number of chunks created: {len(chunks)}")
    print(chunks[0].page_content)
    print(".......")
    print(chunks[1].page_content)