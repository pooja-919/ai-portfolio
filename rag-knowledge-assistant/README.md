PROJECT: Build an AI system(RAG) with your context, not just general LLM knowledge.
Example:
User: What are the causes of shipment delays?
Response: Weather disruptions and customs inspections are common causes.


Actions:
1. Load documents
2. Split documents into chunks
3. Convert chunks → embeddings
4. Store embeddings in vector DB
5. Retrieve relevant chunks for a query
6. Send chunks + question to LLM
7. Return grounded answer
8. Expose as API



REQUIRED:
Except from the requirements.txt(pip install -r requirements.txt in terminal), also download OLLAMA for local LLM accessibility by doing the following:

    1. (Go to
    https://ollama.com/download > Download the Windows installer > Run the installer) 
    OR 
    (run "winget install Ollama.Ollama" in powershell)
    2. Restart PowerShell / terminal after installation (important).  
    3. verify installation: ollama --version
    4. download model llama3: ollama pull llama3