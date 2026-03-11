1. Load documents
2. Split documents into chunks
3. Convert chunks → embeddings
4. Store embeddings in vector DB
5. Retrieve relevant chunks for a query
6. Send chunks + question to LLM
7. Return grounded answer
8. Expose as API


Build an AI system that can answer questions using your own documents, not just general LLM knowledge.

Example:
User: What are the causes of shipment delays?

AI:
According to logistics_report.txt:
Weather disruptions and customs inspections are common causes.