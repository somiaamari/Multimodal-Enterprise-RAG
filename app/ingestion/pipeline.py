from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.utils.config import Settings

def test_connection():
    """Test if we can connect to Qdrant"""
    try:
        client = QdrantClient(host=Settings.QDRANT_HOST, port=Settings.QDRANT_PORT)
        collections = client.get_collections()
        print(f"✅ Qdrant connected successfully!")
        print(f"   Collections: {collections}")
        return client
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        return None

def create_collection_if_not_exists(client):
    """Create the collection for our documents if it doesn't exist"""
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if Settings.COLLECTION_NAME not in collection_names:
        print(f"Creating collection: {Settings.COLLECTION_NAME}")
        client.create_collection(
            collection_name=Settings.COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384,  # Dimension for all-MiniLM-L6-v2
                distance=models.Distance.COSINE
            )
        )
        print(f"✅ Collection created!")
    else:
        print(f"✅ Collection '{Settings.COLLECTION_NAME}' already exists")

if __name__ == "__main__":
    client = test_connection()
    if client:
        create_collection_if_not_exists(client)