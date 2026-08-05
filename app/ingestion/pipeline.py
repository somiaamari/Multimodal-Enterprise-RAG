import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_parse import LlamaParse
from app.utils.config import Settings
from app.ingestion.embedding import embed_model

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
            size=384,  # all-MiniLM-L6-v2 dimension
            distance=models.Distance.COSINE
        )
    )
    print(f"✅ Collection '{collection_name}' created")

def ingest_document(file_path, collection_name=None):
    """
    Ingest a PDF document into Qdrant using LlamaParse.
    """
    if collection_name is None:
        collection_name = Settings.COLLECTION_NAME
    
    print(f"📄 Ingesting: {file_path}")
    
    # 1. Connect to Qdrant
    client = QdrantClient(host=Settings.QDRANT_HOST, port=Settings.QDRANT_PORT)
    
    # 2. Create/Recreate the collection
    create_collection(client, collection_name)
    
    # 3. Parse PDF with LlamaParse
    print("📖 Parsing PDF with LlamaParse...")
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_PARSE_API_KEY"),
        result_type="markdown",  # Clean markdown output
        verbose=True,
    )
    
    documents = parser.load_data(file_path)
    print(f"📖 Loaded {len(documents)} pages")
    
    # 4. Chunk into nodes
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"✂️  Created {len(nodes)} chunks")
    
    # 5. Set up Qdrant vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 6. Build the index
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    
    print(f"✅ Index complete! {len(nodes)} chunks stored in Qdrant.")
    return index, len(nodes)