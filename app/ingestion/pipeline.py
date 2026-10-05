import asyncio
import logging
import os

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_parse import LlamaParse
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.ingestion.captioner import ImageCaptioner
from app.ingestion.embedding import embed_model
from app.utils.config import Settings, get_qdrant_client

logger = logging.getLogger(__name__)


def create_collection(client, collection_name):
    """Create a Qdrant collection with the correct vector size"""
    collections = client.get_collections().collections
    if collection_name in [c.name for c in collections]:
        print(f"⚠️  Collection '{collection_name}' exists. Recreating...")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=384, distance=models.Distance.COSINE  # all-MiniLM-L6-v2 dimension
        ),
    )
    print(f"✅ Collection '{collection_name}' created")


def _create_parser(image_dir):
    return LlamaParse(
        api_key=os.getenv("LLAMA_PARSE_API_KEY"),
        result_type="markdown",
        verbose=True,
        images_extract_path=image_dir,
    )


def _index_documents(file_path, collection_name, documents, image_dir):
    print(f"📖 Loaded {len(documents)} pages")

    client = get_qdrant_client()
    create_collection(client, collection_name)

    # 4. Check for extracted images
    extracted_images = []
    if os.path.exists(image_dir):
        extracted_images = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        print(f"🖼️  Found {len(extracted_images)} extracted images")

    # 5. Caption images (Multimodal step!)
    if extracted_images:
        print("\n🔄 Captioning images...")
        captioner = ImageCaptioner()
        image_captions = captioner.caption_images_batch(extracted_images)

        # Create Document objects for each image caption
        image_docs = []
        for img_path, caption in image_captions.items():
            # Add image info to the text
            metadata = {
                "source": file_path,
                "type": "image",
                "image_path": img_path,
                "page": (
                    os.path.basename(img_path).split("_")[0]
                    if "_" in os.path.basename(img_path)
                    else "unknown"
                ),
            }
            # Store both caption and the fact that it's an image
            doc_text = f"[Image] {caption} (Image file: {os.path.basename(img_path)})"
            doc = Document(text=doc_text, metadata=metadata)
            image_docs.append(doc)

        print(f"✅ Created {len(image_docs)} image caption documents")
    else:
        image_docs = []

    # 6. Chunk text nodes
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    text_nodes = splitter.get_nodes_from_documents(documents)
    print(f"✂️  Created {len(text_nodes)} text chunks")

    # 7. Combine text nodes and image caption nodes
    all_nodes = text_nodes + image_docs
    print(
        f"📦 Total nodes: {len(all_nodes)} (text: {len(text_nodes)}, images: {len(image_docs)})"
    )

    # 8. Set up Qdrant vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 9. Build the index
    index = VectorStoreIndex(
        nodes=all_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    print(f"✅ Index complete! {len(all_nodes)} nodes stored in Qdrant.")
    return index, len(all_nodes)


async def ingest_document_async(file_path, collection_name=None):
    """Parse a PDF on the caller's event loop, then index it off-loop."""
    if collection_name is None:
        collection_name = Settings.COLLECTION_NAME

    print(f"📄 Ingesting: {file_path}")
    image_dir = os.path.join("data", "extracted_images")
    os.makedirs(image_dir, exist_ok=True)

    print("📖 Parsing PDF with LlamaParse (including images)...")
    parser = _create_parser(image_dir)
    async with parser.aclient:
        documents = await parser.aload_data(file_path)

    return await asyncio.to_thread(
        _index_documents, file_path, collection_name, documents, image_dir
    )


def ingest_document(file_path, collection_name=None):
    """Synchronous entrypoint for CLI callers."""
    return asyncio.run(ingest_document_async(file_path, collection_name))


def test_connection():
    """Quick test to verify Qdrant is reachable"""
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        print(f"✅ Qdrant connected. Collections: {collections}")
        return client
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return None
