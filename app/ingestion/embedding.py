from llama_index.core import Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from app.utils.config import Settings as AppSettings

# Initialize the embedding model (local, offline, free)
embed_model = HuggingFaceEmbedding(model_name=AppSettings.EMBED_MODEL)

# Set it as the global embedding model for LlamaIndex
Settings.embed_model = embed_model

def get_vector_store():
    """Get a Qdrant vector store instance configured for LlamaIndex"""
    client = QdrantClient(host=AppSettings.QDRANT_HOST, port=AppSettings.QDRANT_PORT)
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=AppSettings.COLLECTION_NAME,
    )
    return vector_store

def test_indexing():
    """Test: create a tiny document, embed it, store it in Qdrant, and retrieve it"""
    print("🧪 Testing LlamaIndex + Qdrant integration...")
    
    # Create a test document
    test_text = "Tesla's Q4 2023 revenue was $25.17 billion, a 3% increase year-over-year."
    doc = Document(text=test_text, metadata={"source": "test", "year": 2023})
    
    # Get vector store and index
    vector_store = get_vector_store()
    
    # We'll build a simple index (just for testing)
    from llama_index.core import VectorStoreIndex
    from llama_index.core.storage import StorageContext
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index (this will embed the document and store it in Qdrant)
    index = VectorStoreIndex.from_documents(
        [doc], 
        storage_context=storage_context,
        embed_model=embed_model
    )
    print(f"✅ Document stored in Qdrant. ID: {doc.doc_id}")
    
    # Now test retrieval
    query = "What was Tesla's revenue?"
    retriever = index.as_retriever(similarity_top_k=1)
    results = retriever.retrieve(query)
    
    print(f"\n🔍 Query: '{query}'")
    print(f"📄 Retrieved: {results[0].text}")
    print(f"📊 Score: {results[0].score:.4f}")
    
    return index

if __name__ == "__main__":
    test_indexing()