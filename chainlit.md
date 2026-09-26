"""

Chainlit frontend for EnterpriseMind AI RAG system.

"""

import sys

from pathlib import Path



\# ─────────────────────────────────────────────────────────────

\# Ensure the project root is on sys.path so `app.\*` imports work

\# when Chainlit is launched from any directory.

\# ─────────────────────────────────────────────────────────────

PROJECT\_ROOT = Path(\_\_file\_\_).resolve().parents\[2]  # .../multimodal-enterprise-rag

if str(PROJECT\_ROOT) not in sys.path:

&#x20;   sys.path.insert(0, str(PROJECT\_ROOT))



import os

import shutil

import asyncio

from concurrent.futures import ThreadPoolExecutor



import chainlit as cl



from app.retrieval.retriever import DocumentRetriever

from app.ingestion.pipeline import ingest\_document

from app.utils.config import Settings
# 🚀 EnterpriseMind AI



Welcome to \*\*Multimodal Enterprise RAG Agent\*\* — an intelligent assistant that answers questions about your PDF documents.



\## How to Use



1\. \*\*📎 Upload a PDF\*\* using the attachment button in the chat input.

2\. \*\*⏳ Wait\*\* for indexing to complete (30-90 seconds).

3\. \*\*💬 Ask questions\*\* about your document.

4\. \*\*📚 Get answers\*\* with source citations and page numbers.



\## Features



\- 🔍 \*\*Hybrid Retrieval\*\*: Vector + keyword search

\- 🎯 \*\*Cross-Encoder Reranking\*\*: Most relevant chunks first

\- 🤖 \*\*Self-RAG\*\*: Evaluates relevance, retries, or rejects

\- 📄 \*\*Source Citations\*\*: Every answer shows its sources

\- ⚡ \*\*Streaming Responses\*\*: Real-time answers



\## Example Questions



\- "What were Tesla's revenues in 2023?"

\- "What are the main risks mentioned in the report?"

\- "How does the company generate revenue?"



\---



\*\*Built with:\*\* LlamaIndex · Qdrant · Groq · HuggingFace Embeddings

