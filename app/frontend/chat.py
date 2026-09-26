"""
Chainlit frontend for EnterpriseMind AI RAG system.
"""
import sys
from pathlib import Path

# Add project root to sys.path so `app.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"✅ PROJECT_ROOT added to sys.path: {PROJECT_ROOT}")

import os
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor

import chainlit as cl

from app.retrieval.retriever import DocumentRetriever
from app.ingestion.pipeline import ingest_document
from app.utils.config import Settings


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

_executor = ThreadPoolExecutor(max_workers=1)


@cl.on_chat_start
async def on_chat_start():
    """Initialize a new chat session."""
    cl.user_session.set("retriever", None)
    cl.user_session.set("document_name", None)

    await cl.Message(
        content=(
            "👋 **Welcome to EnterpriseMind AI!**\n\n"
            "I can answer questions about any PDF document you upload.\n\n"
            "**To get started:**\n"
            "1. Click the 📎 paperclip icon below\n"
            "2. Upload a PDF file\n"
            "3. Wait for indexing to complete\n"
            "4. Ask me anything about the document!"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages."""

    # Check for PDF attachments
    pdf_element = None
    for element in message.elements or []:
        name = getattr(element, "name", "")
        if name.lower().endswith(".pdf"):
            pdf_element = element
            break

    if pdf_element is not None:
        await handle_pdf_upload(pdf_element)
        return

    # Otherwise, treat as a question
    await handle_question(message.content)


async def handle_pdf_upload(element):
    """Process an uploaded PDF and build the index."""
    name = element.name
    dest_path = DATA_DIR / name

    # Copy the uploaded file to our data folder
    try:
        if getattr(element, "path", None):
            shutil.copy(element.path, dest_path)
        elif getattr(element, "content", None):
            with open(dest_path, "wb") as f:
                f.write(element.content)
        else:
            raise ValueError("Uploaded file has no path or content.")
    except Exception as e:
        await cl.Message(content=f"❌ Failed to save file: `{e}`").send()
        return

    status = cl.Message(
        content=f"📄 Received **{name}**. Parsing and indexing... this may take 30–90 seconds."
    )
    await status.send()

    loop = asyncio.get_event_loop()
    try:
        _, count = await loop.run_in_executor(
            _executor, _ingest_file, str(dest_path)
        )
    except Exception as e:
        await cl.Message(
            content=f"❌ **Ingestion failed:**\n```\n{e}\n```"
        ).send()
        return

    try:
        retriever = await loop.run_in_executor(_executor, _build_retriever)
    except Exception as e:
        await cl.Message(
            content=f"❌ **Retriever setup failed:**\n```\n{e}\n```"
        ).send()
        return

    cl.user_session.set("retriever", retriever)
    cl.user_session.set("document_name", name)

    await cl.Message(
        content=(
            f"✅ **Indexing complete!**\n\n"
            f"- 📄 Document: `{name}`\n"
            f"- 📦 Chunks indexed: **{count}**\n\n"
            f"You can now ask me questions about this document."
        )
    ).send()


def _ingest_file(file_path: str):
    """Blocking wrapper for ingestion (runs in thread pool)."""
    index, count = ingest_document(file_path)
    return index, count


def _build_retriever():
    """Blocking wrapper for retriever construction."""
    return DocumentRetriever(use_hybrid=False, use_reranker=True)


async def handle_question(question: str):
    """Answer a user question using the loaded retriever."""
    retriever = cl.user_session.get("retriever")

    if retriever is None:
        await cl.Message(
            content=(
                "⚠️ **No document loaded.**\n\n"
                "Please upload a PDF using the 📎 attachment button below."
            )
        ).send()
        return

    msg = cl.Message(content="")
    await msg.send()

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor, _run_query, retriever, question
        )
    except Exception as e:
        msg.content = f"❌ **Query failed:**\n```\n{e}\n```"
        await msg.update()
        return

    answer = result.get("answer", "").strip() or "(empty response)"
    for token in answer.split():
        await msg.stream_token(token + " ")

    sources_md = _format_sources(result.get("sources", []))
    if sources_md:
        await msg.stream_token("\n\n---\n\n" + sources_md)

    await msg.update()


def _run_query(retriever, question: str):
    """Blocking query wrapper."""
    return retriever.query(question, similarity_top_k=3)


def _format_sources(sources) -> str:
    """Render source citations as Markdown."""
    if not sources:
        return ""

    lines = ["### 📚 Sources"]
    for i, src in enumerate(sources, 1):
        score = src.get("score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"

        meta = src.get("metadata", {}) or {}
        page = (
            meta.get("page_label")
            or meta.get("page")
            or meta.get("page_number")
            or "N/A"
        )
        fname = meta.get("file_name", "document")

        preview = (src.get("text") or "").replace("\n", " ").strip()
        if len(preview) > 220:
            preview = preview[:220] + "..."

        lines.append(
            f"\n**[{i}]** `score={score_str}` · page `{page}` · `{fname}`\n\n> {preview}"
        )

    return "\n".join(lines)


@cl.on_chat_end
async def on_chat_end():
    """Cleanup when the session ends."""
    cl.user_session.set("retriever", None)
    cl.user_session.set("document_name", None)