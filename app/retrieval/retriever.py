import logging
from typing import List, Optional
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from app.utils.config import Settings as AppSettings
from app.ingestion.embedding import embed_model
from app.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Advanced retriever with Hybrid Search + Reranking"""
    
    def __init__(self, collection_name=None, use_hybrid=True, use_reranker=True):
        self.collection_name = collection_name or AppSettings.COLLECTION_NAME
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker
        
        # Connect to Qdrant
        self.client = QdrantClient(host=AppSettings.QDRANT_HOST, port=AppSettings.QDRANT_PORT)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
        )
        self.index = None
        self.query_engine = None
        
        # Initialize Groq LLM
        self.llm = OpenAILike(
            api_key=AppSettings.GROQ_API_KEY,
            api_base="https://api.groq.com/openai/v1",
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            context_window=128000,
            max_tokens=4096,
            is_chat_model=True,
        )
        Settings.llm = self.llm
        
        # Initialize reranker (if used)
        self.reranker = Reranker() if use_reranker else None
    
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
    
    def _get_hybrid_retriever(self, similarity_top_k=10):
        """Create a hybrid retriever (BM25 + Vector)"""
        if self.index is None:
            self.load_index()
        
        # 1. Vector Retriever (Semantic Search)
        vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k,
        )
        
        # 2. BM25 Retriever (Keyword Search)
        try:
            # Correct import for BM25
            from llama_index.retrievers.bm25 import BM25Retriever
            
            # Get nodes from the docstore
            docstore = self.index.docstore
            all_nodes = list(docstore.docs.values())
            
            if not all_nodes:
                logger.warning("No nodes in docstore, trying alternative method...")
                # Try to get nodes from the index's ref_doc_info
                all_nodes = []
                for doc_id in self.index.ref_doc_info.keys():
                    # Get the nodes for this document
                    doc_nodes = self.index.docstore.get_nodes([doc_id])
                    all_nodes.extend(doc_nodes)
            
            if not all_nodes:
                logger.warning("Still no nodes found. Falling back to vector-only search.")
                return vector_retriever
            
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=all_nodes,
                similarity_top_k=similarity_top_k,
            )
            
            # 3. Combine using QueryFusionRetriever
            from llama_index.core.retrievers import QueryFusionRetriever
            
            hybrid_retriever = QueryFusionRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                mode="reciprocal_rerank",
                similarity_top_k=similarity_top_k,
            )
            
            return hybrid_retriever
            
        except ImportError as e:
            logger.warning(f"BM25 not available: {e}. Using vector-only search.")
            return vector_retriever
        except Exception as e:
            logger.warning(f"Error setting up hybrid search: {e}. Using vector-only search.")
            return vector_retriever
    
    def _get_vector_retriever(self, similarity_top_k=10):
        """Fallback: Just vector search"""
        if self.index is None:
            self.load_index()
        return self.index.as_retriever(similarity_top_k=similarity_top_k)
    
    def create_query_engine(self, similarity_top_k=10):
        """Create the query engine with optional hybrid and reranking"""
        if self.index is None:
            self.load_index()
        
        # Choose retriever
        if self.use_hybrid:
            retriever = self._get_hybrid_retriever(similarity_top_k)
        else:
            retriever = self._get_vector_retriever(similarity_top_k)
        
        # If we have a reranker, wrap the retriever
        if self.use_reranker and self.reranker:
            try:
                retriever = self.reranker.wrap_retriever(retriever, top_k=3)
            except Exception as e:
                logger.warning(f"Reranker failed: {e}. Using unwrapped retriever.")
        
        # Create response synthesizer
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
    
    def query(self, question, similarity_top_k=10):
        """Query the document with a question."""
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