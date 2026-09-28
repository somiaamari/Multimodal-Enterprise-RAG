# 🚀 Multimodal Enterprise RAG Agent

An enterprise-grade **Retrieval-Augmented Generation (RAG)** system that answers questions about PDF documents with **source citations**, **reranking**, and **self-reflective retrieval**.

Built as a portfolio project demonstrating production-grade AI engineering.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **PDF Ingestion** | Upload any PDF → auto-parsed with LlamaParse |
| 🔍 **Hybrid Retrieval** | Vector search + reranking for precision |
| 🎯 **Cross-Encoder Reranking** | Most relevant chunks first |
| 🤖 **Self-RAG** | Evaluates retrieval quality, retries or rejects |
| 📚 **Source Citations** | Every answer cites page numbers and scores |
| 💬 **Chat Interface** | Chainlit UI with streaming answers |
| 📤 **Drag & Drop Upload** | Upload PDFs directly in the UI |
| 🖼️ **Multimodal Ready** | Extracts tables; image captioning via BLIP |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL RAG PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   PDF ──▶ LlamaParse ──▶ Text ──▶ Chunks ──▶ Embeddings        │
│             │                                    │             │
│             ▼                                    ▼             │
│        Tables &                             Qdrant DB          │
│        Figures                             (Vector Store)      │
│                                                  │             │
│                                                  ▼             │
│   Question ──▶ Retrieve ──▶ Rerank ──▶ Evaluate ──▶ Answer    │
│                                            │                   │
│                                     ┌──────┴──────┐            │
│                                     │             │            │
│                                  ✅ Good      ❌ Bad          │
│                                     │             │            │
│                                     ▼             ▼            │
│                                  Answer      Rewrite/Reject    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧰 Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **LLM** | Groq (`openai/gpt-oss-120b`) | Free, OpenAI-compatible, fast |
| **Embeddings** | `all-MiniLM-L6-v2` (local) | Free, offline, no API cost |
| **Vector DB** | Qdrant (Docker) | Fast, production-ready, self-hosted |
| **Framework** | LlamaIndex | Best-in-class RAG abstractions |
| **PDF Parser** | LlamaParse | Clean text + table extraction |
| **Reranker** | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) | Improves retrieval precision |
| **Frontend** | Chainlit | Beautiful chat UI in pure Python |

---

## 📁 Project Structure

```
multimodal-enterprise-rag/
├── app/
│   ├── ingestion/          # PDF parsing, chunking, embeddings
│   │   ├── pipeline.py     # Main ingestion flow
│   │   ├── embedding.py    # HuggingFace embedding setup
│   │   └── captioner.py    # Image captioning (BLIP)
│   ├── retrieval/          # Retrieval + reranking
│   │   ├── retriever.py    # DocumentRetriever class
│   │   └── reranker.py     # Cross-encoder reranker
│   ├── agents/             # Self-RAG
│   │   ├── evaluator.py    # Relevance evaluator
│   │   └── self_rag.py     # Self-RAG orchestrator
│   ├── frontend/           # Chainlit UI
│   │   └── chat.py         # Main chat app
│   └── utils/
│       └── config.py       # Settings & env loading
├── scripts/                # CLI utilities
│   ├── ingest.py
│   ├── query.py
│   └── self_rag_demo.py
├── data/                   # PDFs (gitignored)
├── docker-compose.yml      # Qdrant service
├── .chainlit/              # Chainlit config
├── chainlit.md             # Welcome screen
└── pyproject.toml          # Poetry dependencies
```

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.11**
- **Poetry** (2.x)
- **Docker Desktop** (for Qdrant)

### 2. Clone & Install

```bash
git clone https://github.com/somiaamari/Multimodal-Enterprise-RAG.git
cd Multimodal-Enterprise-RAG
poetry install
```

### 3. Configure Environment

Create a `.env` file:

```bash
QDRANT_HOST=localhost
QDRANT_PORT=6333
GROQ_API_KEY=your_groq_api_key_here
LLAMA_PARSE_API_KEY=your_llamaparse_api_key_here
```

Get free keys:
- **Groq**: https://console.groq.com
- **LlamaParse**: https://cloud.llamaindex.ai

### 4. Start Qdrant

```bash
docker-compose up -d
```

### 5. Launch the UI

```bash
py -m poetry run chainlit run app/frontend/chat.py -w
```

Open **http://localhost:8000** in your browser.

---

## 🖥️ Usage

### Via the Chat UI

1. Open **http://localhost:8000**
2. Click 📎 to upload a PDF (or drag & drop)
3. Wait for indexing (~30–90 seconds)
4. Ask questions in the chat box
5. Get answers with source citations

### Via the CLI

```bash
# Ingest a PDF
py -m poetry run python -m scripts.ingest

# Query the document
py -m poetry run python -m scripts.query

# Try Self-RAG
py -m poetry run python -m scripts.self_rag_demo
```

---

## 🧠 How It Works

### Ingestion Pipeline

1. **Parse** — LlamaParse extracts clean text and tables from the PDF
2. **Chunk** — SentenceSplitter creates 512-token chunks with 50-token overlap
3. **Embed** — `all-MiniLM-L6-v2` converts each chunk into a 384-dim vector
4. **Store** — Qdrant stores vectors + metadata (page number, source file)

### Query Pipeline

1. **Retrieve** — Top-k chunks fetched via vector similarity
2. **Rerank** — Cross-encoder scores each chunk against the question
3. **Evaluate** — Self-RAG checks if retrieval is good enough
4. **Answer** — Groq LLM generates an answer from the best chunks
5. **Cite** — Sources shown with page numbers and scores

### Self-RAG Logic

If the retrieval score is too low:
- **Retry** with a rewritten query (LLM-based rewriting)
- If still low after max retries: **Reject** with an honest "I don't know"

This dramatically reduces hallucinations.

---

## 📊 What Works Today

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Basic RAG (PDF → Answer) | ✅ |
| 2 | Cross-Encoder Reranking | ✅ |
| 3 | Self-RAG (evaluate, retry, reject) | ✅ |
| 4 | Multimodal (tables + image captions) | ⚠️ Partial |
| 5 | Chainlit Frontend | ✅ |
| 6 | Deployment | 🔜 |

---

## 🔮 Roadmap

- [ ] Deploy to Hugging Face Spaces
- [ ] Dockerize the full app
- [ ] GitHub Actions CI
- [ ] Multi-document support
- [ ] Full image understanding via VLM (LLaVA)

---






