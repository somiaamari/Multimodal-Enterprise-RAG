\# Multimodal Enterprise RAG Agent



An enterprise-grade Retrieval-Augmented Generation (RAG) system that can answer questions from PDF documents, including text, tables, and charts.



\## Architecture



\- \*\*LLM\*\*: Groq (free, OpenAI-compatible API)

\- \*\*Embeddings\*\*: all-MiniLM-L6-v2 (local, offline)

\- \*\*Vector Database\*\*: Qdrant (Docker)

\- \*\*Framework\*\*: LlamaIndex

\- \*\*Frontend\*\*: Chainlit



\## Setup



```bash

\# Install dependencies

poetry install



\# Start Qdrant

docker-compose up -d



\# Run the environment test

poetry run python test\_env.py

