import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from app.utils.config import Settings
from app.ingestion.embedding import embed_model

logger = logging.getLogger(__name__)

def create_collection(client, collection_name):
    """Create a Qdrant collection with the correct vector size"""
    # Delete if exists to start fresh
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
    Ingest a PDF document into Qdrant.
    
    Args:
        file_path: Path to the PDF file
        collection_name: Optional override for collection name
    
    Returns:
        Tuple: (VectorStoreIndex, number_of_nodes)
    """
    if collection_name is None:
        collection_name = Settings.COLLECTION_NAME
    
    print(f"📄 Ingesting: {file_path}")
    
    # 1. Connect to Qdrant
    client = QdrantClient(host=Settings.QDRANT_HOST, port=Settings.QDRANT_PORT)
    
    # 2. Create/Recreate the collection
    create_collection(client, collection_name)
    
    # 3. Load the PDF
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()
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
    
    # 6. Build the index (this embeds and stores everything)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    
    print(f"✅ Index complete! {len(nodes)} chunks stored in Qdrant.")
    return index, len(nodes)

def test_connection():
    """Quick test to verify Qdrant is reachable"""
    try:
        client = QdrantClient(host=Settings.QDRANT_HOST, port=Settings.QDRANT_PORT)
        collections = client.get_collections()
        print(f"✅ Qdrant connected. Collections: {collections}")
        return client
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return None