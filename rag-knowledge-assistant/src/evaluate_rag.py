from loader import load_documents
from chunking import split_documents
from vector_store import create_vector_store
from retriever import get_retriever, generate_answer
from langchain_ollama import ChatOllama
from config import DOCUMENTS_PATH
# from langchain_openai import ChatOpenAI ##credit issue hence used ChatOllama
from dotenv import load_dotenv
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
sample_queries = [
    "Who introduced the theory of relativity?",
    "Who was the first computer programmer?",
    "What did Isaac Newton contribute to science?",
    "Who won two Nobel Prizes for research on radioactivity?",
    "What is the theory of evolution by natural selection?"
]

expected_responses = [
    "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
    "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'."
]

dataset = []
docs = load_documents(DOCUMENTS_PATH)
chunks = split_documents(docs)
vector_store = create_vector_store(chunks)
retriever = get_retriever(vector_store,chunks)
for query,reference in zip(sample_queries,expected_responses):
    response,context,sources = generate_answer(retriever, query)
    dataset.append(
        {
            "user_query":query,
            "retrieved_contexts":context,
            "response":response,
            "reference":reference
        }
    )
evaluation_dataset = EvaluationDataset.from_list(dataset)

evaluator_llm = LangchainLLMWrapper(llm)
result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
result