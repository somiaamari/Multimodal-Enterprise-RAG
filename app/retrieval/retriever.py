from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from app.utils.config import Settings
from app.ingestion.embedding import embed_model

class DocumentRetriever:
    """Handles retrieval from the Qdrant index"""
    
    def __init__(self, collection_name=None):
        self.collection_name = collection_name or Settings.COLLECTION_NAME
        self.client = QdrantClient(host=Settings.QDRANT_HOST, port=Settings.QDRANT_PORT)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
        )
        self.index = None
    
    def load_index(self):
        """Load the existing index from Qdrant"""
        from llama_index.core import VectorStoreIndex
        from llama_index.core.storage import StorageContext
        
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            embed_model=embed_model,
        )
        return self.index
    
    def query(self, question, similarity_top_k=3):
        """Retrieve relevant chunks for a question"""
        if self.index is None:
            self.load_index()
        
        retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
        nodes = retriever.retrieve(question)
        
        results = []
        for node in nodes:
            results.append({
                "text": node.text,
                "score": node.score,
                "metadata": node.metadata,
            })
        
        return results