import logging
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from app.utils.config import Settings as AppSettings
from app.ingestion.embedding import embed_model

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Handles retrieval and answer generation from the Qdrant index"""
    
    def __init__(self, collection_name=None):
        self.collection_name = collection_name or AppSettings.COLLECTION_NAME
        self.client = QdrantClient(host=AppSettings.QDRANT_HOST, port=AppSettings.QDRANT_PORT)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
        )
        self.index = None
        self.query_engine = None
        
        # Initialize Groq LLM (OpenAI-compatible)
        self.llm = OpenAILike(
          api_key=AppSettings.GROQ_API_KEY,
          api_base="https://api.groq.com/openai/v1",
          model="llama-3.3-70b-versatile",
          temperature=0.1,
          context_window=128000,
          max_tokens=4096,
          is_chat_model=True,  # Important for chat models like Llama
)
        # Set as global LLM for LlamaIndex
        Settings.llm = self.llm
    
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
    
    def create_query_engine(self, similarity_top_k=3):
        """Create a query engine with the index"""
        if self.index is None:
            self.load_index()
        
        # Configure retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k,
        )
        
        # Configure response synthesizer
        response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            response_mode="compact",
        )
        
        # Create query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        
        return self.query_engine
    
    def query(self, question, similarity_top_k=3):
        """
        Query the document with a question.
        Returns answer with source chunks.
        """
        if self.query_engine is None:
            self.create_query_engine(similarity_top_k=similarity_top_k)
        
        # Run the query
        response = self.query_engine.query(question)
        
        # Extract source nodes
        sources = []
        for node in response.source_nodes:
            sources.append({
                "text": node.text[:500] + "..." if len(node.text) > 500 else node.text,
                "score": node.score,
                "metadata": node.metadata,
            })
        
        return {
            "answer": str(response),
            "sources": sources,
            "source_count": len(sources),
        }
    
    def get_retriever(self, similarity_top_k=3):
        """Return just the retriever (for later agentic RAG)"""
        if self.index is None:
            self.load_index()
        return self.index.as_retriever(similarity_top_k=similarity_top_k)